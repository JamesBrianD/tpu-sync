// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tpu_sync/frameworks/torch/torch_host_signal.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "csrc/api/tensor_buffer.h"
#include "torch/headeronly/core/DeviceType.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_sync/core/host_signal.h"
#include "tpu_sync/frameworks/torch/torch_tpu_utils.h"

namespace raiden {
namespace {

[[noreturn]] void Throw(absl::string_view context, const absl::Status& status) {
  throw std::runtime_error(absl::StrCat(context, ": ", status.message()));
}

// Channel buffers are addressed by byte offset from the host, so they must be
// whole allocations of plain int32 words spanning whole pages.
void CheckPageTensor(const at::Tensor& tensor, absl::string_view role) {
  if (tensor.device().type() != at::DeviceType::PrivateUse1 ||
      tensor.dim() != 1 || !tensor.is_contiguous() ||
      tensor.storage_offset() != 0 || tensor.scalar_type() != at::kInt) {
    throw std::invalid_argument(
        absl::StrCat(role, " must be a whole 1-D int32 TPU tensor"));
  }
  const int64_t bytes = static_cast<int64_t>(tensor.nbytes());
  if (bytes == 0 || bytes % kHostSignalPageBytes != 0) {
    throw std::invalid_argument(absl::StrCat(
        role, " must span a whole number of ", kHostSignalPageBytes,
        "-byte pages; got ", bytes, " bytes"));
  }
}

absl::string_view MemoryKind(xla::PjRtBuffer* buffer) {
  return buffer->memory_space() != nullptr ? buffer->memory_space()->kind()
                                           : absl::string_view();
}

torch_tpu::TensorBufferHandle ResolvePinnedHost(const at::Tensor& tensor) {
  CheckPageTensor(tensor, "Pinned host pages");
  auto handle = torch_tpu::GetBaseTensorBuffer(tensor);
  if (!handle.ok()) {
    Throw("Failed to resolve pinned host buffer", handle.status());
  }
  return std::move(handle).value();
}

tpu_raiden::torch::UnpackedTensor ResolveDevice(const at::Tensor& tensor) {
  CheckPageTensor(tensor, "Upload target");
  // No usage hold for the channel's lifetime: the server's copies and the
  // programs that read the buffer are ordered by the channel protocol, and the
  // unpacked ref keeps the buffer alive.
  return tpu_raiden::torch::UnpackTorchTensor(tensor,
                                              /*unsafe_skip_buffer_lock=*/true);
}

// Everything one channel's server-side state points into.
struct ChannelLeases {
  ChannelLeases(const at::Tensor& payload_tensor,
                const at::Tensor& request_tensor,
                const at::Tensor& response_tensor,
                const at::Tensor& staging_tensor,
                const at::Tensor& retained_tensor)
      : payload(payload_tensor),
        request(request_tensor),
        response(response_tensor),
        staging(staging_tensor),
        retained(retained_tensor) {}

  PinnedHostPages payload;
  PinnedHostPages request;
  PinnedHostPages response;
  PinnedHostPages staging;
  DeviceUploadTarget retained;
};

}  // namespace

PinnedHostPages::PinnedHostPages(const at::Tensor& tensor)
    : handle_(ResolvePinnedHost(tensor)) {
  if (handle_.size_bytes() != static_cast<int64_t>(tensor.nbytes())) {
    throw std::invalid_argument(
        "Pinned host pages must span their whole allocation");
  }
  if (absl::Status status = handle_.Materialize(); !status.ok()) {
    Throw("Failed to materialize pinned host buffer", status);
  }
  if (absl::Status status = handle_.Synchronize(); !status.ok()) {
    Throw("Failed to synchronize pinned host buffer", status);
  }
  auto buffer = handle_.AwaitBuffer();
  if (!buffer.ok()) Throw("Failed to fetch pinned host buffer", buffer.status());
  if (MemoryKind(*buffer) != "pinned_host") {
    throw std::invalid_argument(absl::StrCat(
        "Pinned host pages must be in pinned_host memory, not '",
        MemoryKind(*buffer), "'"));
  }
  auto reference = (*buffer)->AcquireExternalReference();
  if (!reference.ok()) {
    Throw("Failed to lease pinned host buffer", reference.status());
  }
  owner_ = std::move(reference).value();
  data_ = static_cast<int32_t*>(owner_->OpaqueDeviceMemoryDataPointer());
  bytes_ = static_cast<int64_t>(tensor.nbytes());
  if (data_ == nullptr ||
      reinterpret_cast<uintptr_t>(data_) % kHostSignalPageBytes != 0) {
    throw std::invalid_argument("Pinned host pages must be page-aligned");
  }
}

at::Tensor PinnedHostPages::View() {
  auto buffer = handle_.AwaitBuffer();
  if (!buffer.ok()) Throw("Failed to fetch pinned host buffer", buffer.status());
  xla::PjRtBuffer* source = *buffer;
  xla::PjRtClient* client = source->client();
  auto view = client->CreateViewOfDeviceBuffer(
      data_, source->on_device_shape(), source->memory_space(), [] {});
  if (!view.ok()) Throw("Failed to view pinned host buffer", view.status());
  if (absl::Status ready = (*view)->GetReadyFuture().Await(); !ready.ok()) {
    Throw("Pinned host view did not become ready", ready);
  }
  auto address = client->UnsafeBufferPointer(view->get());
  if (!address.ok() || *address != reinterpret_cast<uintptr_t>(data_)) {
    throw std::runtime_error("Pinned host view does not alias its allocation");
  }
  auto made = torch_tpu::MakeTensorBuffer(std::move(view).value());
  if (!made.ok()) Throw("Failed to wrap pinned host view", made.status());
  views_.push_back(std::make_unique<torch_tpu::TensorBufferHandle>(
      std::move(made).value()));
  auto tensor = views_.back()->ToTensor();
  if (!tensor.ok()) Throw("Failed to expose pinned host view", tensor.status());
  return std::move(tensor).value();
}

DeviceUploadTarget::DeviceUploadTarget(const at::Tensor& tensor)
    : unpacked_(ResolveDevice(tensor)) {
  xla::PjRtBuffer* buffer = unpacked_.buffer.buffer;
  if (buffer == nullptr) {
    throw std::runtime_error("Upload target has no device buffer");
  }
  if (MemoryKind(buffer) != "device") {
    throw std::invalid_argument(absl::StrCat(
        "Upload target must be in device memory, not '", MemoryKind(buffer),
        "'"));
  }
  // Its initial contents (e.g. a zero fill) must land before the first copy.
  if (absl::Status ready = buffer->GetReadyFuture().Await(); !ready.ok()) {
    Throw("Upload target did not become ready", ready);
  }
  auto size = buffer->GetOnDeviceSizeInBytes();
  if (!size.ok()) Throw("Failed to size upload target", size.status());
  bytes_ = static_cast<int64_t>(tensor.nbytes());
  if (static_cast<int64_t>(*size) != bytes_) {
    throw std::invalid_argument(absl::StrCat(
        "Upload target's device size (", *size, " bytes) differs from its ",
        bytes_, " logical bytes, so byte offsets would not address words"));
  }
}

absl::Status DeviceUploadTarget::Upload(const void* source,
                                        int64_t bytes) const {
  if (bytes < 0 || bytes > bytes_) {
    return absl::OutOfRangeError(absl::StrCat(
        "upload of ", bytes, " bytes into a ", bytes_, "-byte buffer"));
  }
  if (bytes == 0) return absl::OkStatus();
  // Issued directly, not through RawDmaPacer: a program is waiting for exactly
  // these bytes, and queueing them behind other raw copies only delays it.
  return unpacked_.buffer.CopyRawHostToDevice(source, 0, bytes).Await();
}

std::tuple<int64_t, at::Tensor, at::Tensor> TorchHostSignalServer::Add(
    const at::Tensor& payload, const at::Tensor& request,
    const at::Tensor& response, const at::Tensor& staging,
    const at::Tensor& retained, uintptr_t handler, intptr_t user,
    int64_t after) {
  if (handler == 0) {
    throw std::invalid_argument("handler must be a function address");
  }
  auto leases = std::make_shared<ChannelLeases>(payload, request, response,
                                                staging, retained);
  if (leases->staging.bytes() != leases->retained.bytes()) {
    throw std::invalid_argument(absl::StrCat(
        "staging (", leases->staging.bytes(), " bytes) and retained (",
        leases->retained.bytes(), " bytes) must match"));
  }
  // Programs donate the payload and request pages they post into.
  at::Tensor payload_view = leases->payload.View();
  at::Tensor request_view = leases->request.View();

  ChannelLeases* raw = leases.get();
  HostSignalService service;
  service.mailbox = std::make_shared<HostMailbox>(
      raw->payload.data(), raw->payload.words(), raw->request.data(),
      raw->response.data(), leases);
  service.handler = reinterpret_cast<HostSignalHandler>(handler);
  service.user = user;
  service.staging = reinterpret_cast<uint8_t*>(raw->staging.data());
  service.staging_bytes = raw->staging.bytes();
  service.upload = [raw](int64_t bytes) {
    return raw->retained.Upload(raw->staging.data(), bytes);
  };
  service.after = after;
  service.keep_alive = leases;
  auto index = server_.Add(std::move(service));
  if (!index.ok()) Throw("Failed to add host signal channel", index.status());
  return {static_cast<int64_t>(*index), std::move(payload_view),
          std::move(request_view)};
}

void TorchHostSignalServer::Start() {
  if (absl::Status status = server_.Start(); !status.ok()) {
    Throw("Failed to start host signal server", status);
  }
}

}  // namespace raiden
