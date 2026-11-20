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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_TORCH_HOST_SIGNAL_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_TORCH_HOST_SIGNAL_H_

// torch_tpu side of the host signal channel (tpu_sync/core/host_signal.h).
//
// The device writes and reads the channel's pages by DMA from compiled
// programs, so every page is a whole 1-D int32 TPU tensor in pinned host
// memory. The host server reads and writes the same pages through their
// addresses, which this layer leases for the channel's lifetime.

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "csrc/api/tensor_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_sync/core/host_signal.h"
#include "tpu_sync/frameworks/torch/torch_tpu_utils.h"

namespace raiden {

// The allocation behind a whole 1-D int32 TPU tensor in pinned host memory,
// leased together with its address for this object's lifetime.
class PinnedHostPages {
 public:
  explicit PinnedHostPages(const at::Tensor& tensor);

  PinnedHostPages(const PinnedHostPages&) = delete;
  PinnedHostPages& operator=(const PinnedHostPages&) = delete;

  int32_t* data() const { return data_; }
  int64_t bytes() const { return bytes_; }
  int64_t words() const { return bytes_ / 4; }

  // A tensor over a distinct PJRT buffer at the same address. Compiled
  // programs may donate it: a buffer holding an external reference must not
  // be donated, so the lease keeps the reference and hands out views.
  at::Tensor View();

 private:
  torch_tpu::TensorBufferHandle handle_;
  std::unique_ptr<xla::PjRtBuffer::ExternalReference> owner_;
  std::vector<std::unique_ptr<torch_tpu::TensorBufferHandle>> views_;
  int32_t* data_ = nullptr;
  int64_t bytes_ = 0;
};

// A whole 1-D int32 TPU tensor in device memory that the server writes with
// raw host-to-device copies. Compiled programs must not donate it.
class DeviceUploadTarget {
 public:
  explicit DeviceUploadTarget(const at::Tensor& tensor);

  DeviceUploadTarget(const DeviceUploadTarget&) = delete;
  DeviceUploadTarget& operator=(const DeviceUploadTarget&) = delete;

  int64_t bytes() const { return bytes_; }
  // Copies source[0, bytes) to the start of the buffer and waits until it
  // landed.
  absl::Status Upload(const void* source, int64_t bytes) const;

 private:
  tpu_raiden::torch::UnpackedTensor unpacked_;
  int64_t bytes_ = 0;
};

// HostSignalServer over torch_tpu tensors.
class TorchHostSignalServer {
 public:
  TorchHostSignalServer() = default;
  ~TorchHostSignalServer() { server_.Stop(); }

  TorchHostSignalServer(const TorchHostSignalServer&) = delete;
  TorchHostSignalServer& operator=(const TorchHostSignalServer&) = delete;

  // Adds a channel and returns (index, payload view, request view); compiled
  // programs post through the views. `payload`, `request`, `response` and
  // `staging` are pinned host tensors, `retained` a device tensor of
  // `staging`'s size. `handler` is the address of a HostSignalHandler called
  // with `user`; `after` is an earlier channel's index or -1.
  std::tuple<int64_t, at::Tensor, at::Tensor> Add(
      const at::Tensor& payload, const at::Tensor& request,
      const at::Tensor& response, const at::Tensor& staging,
      const at::Tensor& retained, uintptr_t handler, intptr_t user,
      int64_t after);

  void Start();
  void Stop() { server_.Stop(); }
  std::string Error() const { return server_.error(); }
  bool Running() const { return server_.running(); }
  int64_t Size() const { return static_cast<int64_t>(server_.size()); }

 private:
  HostSignalServer server_;
};

}  // namespace raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_TORCH_HOST_SIGNAL_H_
