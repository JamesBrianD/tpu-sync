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

#include "tpu_sync/core/host_signal.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"

namespace raiden {

HostMailbox::HostMailbox(int32_t* payload, int64_t payload_words,
                         int32_t* request, int32_t* response,
                         std::shared_ptr<void> owner)
    : payload_(payload),
      payload_words_(payload_words),
      request_(request),
      response_(response),
      owner_(std::move(owner)) {}

int32_t HostMailbox::Peek() const {
  // Acquire: the device writes the payload before the generation, so the
  // payload loads that follow observe the request this generation names.
  const int32_t generation = __atomic_load_n(request_, __ATOMIC_ACQUIRE);
  return generation > 0 && generation != last_taken_ ? generation : 0;
}

std::optional<int32_t> HostMailbox::TryTake() {
  const int32_t generation = Peek();
  if (generation == 0) return std::nullopt;
  Take(generation);
  return generation;
}

int32_t HostMailbox::After() const {
  return payload_words_ > 1 ? payload_[1] : 0;
}

absl::StatusOr<absl::Span<const int32_t>> HostMailbox::Message() const {
  const int64_t capacity = payload_words_ - kHostSignalHeaderWords;
  const int64_t count = payload_[0];
  if (capacity < 0 || count < 0 || count > capacity) {
    return absl::OutOfRangeError(absl::StrCat(
        "request of ", count, " words outside [0, ", capacity, "]"));
  }
  return absl::Span<const int32_t>(payload_ + kHostSignalHeaderWords,
                                   static_cast<size_t>(count));
}

void HostMailbox::Respond(int32_t generation, bool ok) {
  __atomic_store_n(response_, ok ? generation : -generation, __ATOMIC_RELEASE);
}

void HostMailbox::Stop() {
  __atomic_store_n(response_, kHostSignalStopped, __ATOMIC_RELEASE);
}

int32_t HostMailbox::response() const {
  return __atomic_load_n(response_, __ATOMIC_ACQUIRE);
}

namespace {

// Whether mailbox i's pending request may be served now: it names no
// predecessor generation, or the predecessor has answered that generation
// (successfully or not). A request naming a generation without a predecessor
// is ready too; Serve fails it.
bool Ready(absl::Span<HostMailbox* const> mailboxes,
           absl::Span<const int64_t> after, size_t i) {
  const int32_t needed = mailboxes[i]->After();
  if (needed == 0 || after[i] < 0) return true;
  const int32_t answer = mailboxes[after[i]]->response();
  return answer == needed || answer == -needed ||
         answer == kHostSignalStopped;
}

}  // namespace

std::optional<std::pair<size_t, int32_t>> NextRequest(
    absl::Span<HostMailbox* const> mailboxes, absl::Span<const int64_t> after) {
  for (size_t i = 0; i < mailboxes.size(); ++i) {
    int32_t generation = mailboxes[i]->Peek();
    if (generation == 0 || !Ready(mailboxes, after, i)) continue;
    // A lower-index request posted before this one may have landed after
    // that mailbox was checked; look again so it is still served first.
    size_t index = i;
    for (size_t j = 0; j < i; ++j) {
      const int32_t earlier = mailboxes[j]->Peek();
      if (earlier != 0 && Ready(mailboxes, after, j)) {
        index = j;
        generation = earlier;
        break;
      }
    }
    mailboxes[index]->Take(generation);
    return std::make_pair(index, generation);
  }
  return std::nullopt;
}

HostSignalServer::~HostSignalServer() { Stop(); }

absl::StatusOr<size_t> HostSignalServer::Add(HostSignalService service) {
  if (started_ || stopped_) {
    return absl::FailedPreconditionError(
        "host signal services must be added before Start");
  }
  if (service.mailbox == nullptr || service.handler == nullptr ||
      !service.upload) {
    return absl::InvalidArgumentError(
        "a host signal service needs a mailbox, a handler and an upload");
  }
  if (service.mailbox->payload_words() < kHostSignalHeaderWords) {
    return absl::InvalidArgumentError(absl::StrCat(
        "a payload page needs at least ", kHostSignalHeaderWords, " words"));
  }
  if (service.staging_bytes < 0 ||
      (service.staging_bytes > 0 && service.staging == nullptr)) {
    return absl::InvalidArgumentError("invalid host signal staging buffer");
  }
  if (service.after < -1 ||
      service.after >= static_cast<int64_t>(services_.size())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "predecessor ", service.after, " must name an earlier service (0..",
        static_cast<int64_t>(services_.size()) - 1, ") or be -1"));
  }
  for (const HostSignalService& existing : services_) {
    if (existing.mailbox == service.mailbox) {
      return absl::AlreadyExistsError("mailbox is already served");
    }
  }
  mailboxes_.push_back(service.mailbox.get());
  after_.push_back(service.after);
  services_.push_back(std::move(service));
  return services_.size() - 1;
}

absl::Status HostSignalServer::Start() {
  if (started_ || stopped_) {
    return absl::FailedPreconditionError("host signal server already started");
  }
  if (services_.empty()) {
    return absl::FailedPreconditionError("host signal server has no services");
  }
  started_ = true;
  running_.store(true, std::memory_order_release);
  worker_ = std::thread([this] { Run(); });
  return absl::OkStatus();
}

void HostSignalServer::Stop() {
  running_.store(false, std::memory_order_release);
  if (worker_.joinable()) worker_.join();
  if (stopped_) return;
  stopped_ = true;
  // After the join nothing else answers, so a waiting program sees kStopped
  // instead of polling forever.
  for (HostSignalService& service : services_) service.mailbox->Stop();
}

std::string HostSignalServer::error() const {
  return failed_.load(std::memory_order_acquire) ? error_ : std::string();
}

void HostSignalServer::Run() {
  while (running_.load(std::memory_order_acquire)) {
    const auto next = NextRequest(mailboxes_, after_);
    if (!next.has_value()) {
      // The device writes host memory by DMA; nothing to block on but time.
      std::this_thread::yield();
      continue;
    }
    const auto [index, generation] = *next;
    HostSignalService& service = services_[index];
    const bool ok =
        !failed_.load(std::memory_order_relaxed) && Serve(service, index);
    // Stop answers kStopped itself; do not overwrite it.
    if (!running_.load(std::memory_order_acquire)) return;
    service.mailbox->Respond(generation, ok);
  }
}

bool HostSignalServer::Serve(HostSignalService& service, size_t index) {
  HostMailbox& mailbox = *service.mailbox;
  if (const int32_t needed = mailbox.After(); needed != 0) {
    if (service.after < 0) {
      return Fail(absl::StrCat("mailbox ", index, ": request waits for ",
                               "predecessor generation ", needed,
                               " but the mailbox has no predecessor"));
    }
    if (mailboxes_[service.after]->response() != needed) {
      return Fail(absl::StrCat("mailbox ", index, ": predecessor mailbox ",
                               service.after, " failed generation ", needed));
    }
  }
  absl::StatusOr<absl::Span<const int32_t>> message = mailbox.Message();
  if (!message.ok()) {
    return Fail(absl::StrCat("mailbox ", index, ": ",
                             message.status().message()));
  }
  const int64_t bytes =
      service.handler(service.user, message->data(),
                      static_cast<int64_t>(message->size()), service.staging,
                      service.staging_bytes);
  if (bytes < 0) {
    return Fail(absl::StrCat("mailbox ", index, ": handler returned error ",
                             bytes, " for ", message->size(), " words"));
  }
  if (bytes > service.staging_bytes) {
    return Fail(absl::StrCat("mailbox ", index, ": handler wrote ", bytes,
                             " bytes into ", service.staging_bytes));
  }
  if (bytes == 0) return true;
  // Raw copies move whole pages; staging is a whole number of them.
  const int64_t padded =
      std::min(service.staging_bytes,
               (bytes + kHostSignalPageBytes - 1) / kHostSignalPageBytes *
                   kHostSignalPageBytes);
  absl::Status status;
  try {
    status = service.upload(padded);
  } catch (const std::exception& e) {
    status = absl::InternalError(e.what());
  }
  if (!status.ok()) {
    return Fail(absl::StrCat("mailbox ", index, ": upload of ", padded,
                             " bytes failed: ", status.message()));
  }
  return true;
}

bool HostSignalServer::Fail(std::string message) {
  if (!failed_.load(std::memory_order_relaxed)) {
    std::fprintf(stderr, "[tpu_sync] host signal server failed: %s\n",
                 message.c_str());
    error_ = std::move(message);
    failed_.store(true, std::memory_order_release);
  }
  return false;
}

}  // namespace raiden
