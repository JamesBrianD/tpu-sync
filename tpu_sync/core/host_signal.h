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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_HOST_SIGNAL_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_HOST_SIGNAL_H_

// Host side of a device-initiated request channel (see host_signal.md).
//
// A compiled TPU program posts a request into three pinned host pages. It
// writes `[count, after, word_0, ..., word_{count-1}]` to the payload page and
// only then a positive generation to word 0 of the request page. The host
// answers on word 0 of the response page: the generation once the request is
// served, its negation if serving failed, kHostSignalStopped once nobody will
// answer. The device polls the response page and never reads a generation's
// result before its answer lands.
//
// `after` is 0, or a generation of the channel's predecessor that must be
// answered before this request is served. It orders the requests of one step
// (e.g. a later layer's after an earlier layer's) even when they land out of
// order.
//
// This file only touches raw page memory, so it is framework-agnostic and
// testable without a device: the torch layer allocates the pages as pinned
// host buffers and supplies the upload.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace raiden {

inline constexpr int64_t kHostSignalPageWords = 1024;
inline constexpr int64_t kHostSignalPageBytes = 4096;
// Payload words before the request's own words: count, after.
inline constexpr int64_t kHostSignalHeaderWords = 2;
// The answer once a channel stops; never a valid generation (those are > 0).
inline constexpr int32_t kHostSignalStopped =
    std::numeric_limits<int32_t>::min();

// C ABI of a request handler, so it can come from any language that produces
// a C function pointer (a numba cfunc, ctypes, C++). Called on the server
// thread without the Python GIL. `words` holds the `count` words the device
// posted. The handler writes its result to `staging` and returns the number of
// bytes written (0 <= n <= staging_bytes), or a negative error code.
using HostSignalHandler = int64_t (*)(intptr_t user, const int32_t* words,
                                      int64_t count, uint8_t* staging,
                                      int64_t staging_bytes);

// One request channel over three caller-owned pages. Only one thread may take
// requests (TryTake/Take); answers may come from any thread.
class HostMailbox {
 public:
  // `payload` spans `payload_words` (>= kHostSignalHeaderWords) words;
  // `request` and `response` hold at least one word. `owner` keeps the page
  // memory alive.
  HostMailbox(int32_t* payload, int64_t payload_words, int32_t* request,
              int32_t* response, std::shared_ptr<void> owner = nullptr);

  HostMailbox(const HostMailbox&) = delete;
  HostMailbox& operator=(const HostMailbox&) = delete;

  // A posted generation newer than the last one taken, else 0.
  int32_t Peek() const;
  // Marks `generation` (from Peek) taken.
  void Take(int32_t generation) { last_taken_ = generation; }
  // Peek, then Take if a request is pending.
  std::optional<int32_t> TryTake();

  // The predecessor generation the pending request waits for, or 0. Read
  // only after Peek returned that request's generation.
  int32_t After() const;
  // The words of the pending request. Valid until it is answered.
  absl::StatusOr<absl::Span<const int32_t>> Message() const;

  // Answers `generation`: itself if `ok`, else its negation.
  void Respond(int32_t generation, bool ok);
  // Answers kHostSignalStopped: every current and later wait ends.
  void Stop();

  int32_t last_taken() const { return last_taken_; }
  int64_t payload_words() const { return payload_words_; }
  int32_t response() const;

 private:
  int32_t* payload_;
  int64_t payload_words_;
  int32_t* request_;
  int32_t* response_;
  std::shared_ptr<void> owner_;
  int32_t last_taken_ = 0;
};

// Takes the next request that may be served now, or returns nullopt.
// `after[i]` is mailbox i's predecessor index or -1; a request naming a
// predecessor generation waits until that mailbox answered it. Among ready
// requests the lowest index wins.
std::optional<std::pair<size_t, int32_t>> NextRequest(
    absl::Span<HostMailbox* const> mailboxes, absl::Span<const int64_t> after);

// What the server does for one mailbox: run `handler` into `staging`, then
// `upload` the bytes it wrote to the device and wait for them to land.
struct HostSignalService {
  std::shared_ptr<HostMailbox> mailbox;
  HostSignalHandler handler = nullptr;
  intptr_t user = 0;
  uint8_t* staging = nullptr;
  int64_t staging_bytes = 0;
  // Copies staging[0, bytes) to the start of the device buffer and returns
  // once it landed. `bytes` is the handler's result rounded up to a page.
  std::function<absl::Status(int64_t bytes)> upload;
  // Index of an earlier service whose answers this one's requests may wait
  // for, or -1.
  int64_t after = -1;
  std::shared_ptr<void> keep_alive;
};

// One thread serving a fixed, ordered set of mailboxes: a request is handled,
// uploaded and answered before the next one is taken, and lower-index
// mailboxes win when several are ready. The first failure is sticky: every
// later request is answered as failed, so the device never reads a stale or
// partial upload. Add/Start/Stop must be called from one thread.
class HostSignalServer {
 public:
  HostSignalServer() = default;
  ~HostSignalServer();

  HostSignalServer(const HostSignalServer&) = delete;
  HostSignalServer& operator=(const HostSignalServer&) = delete;

  // Returns the service's index.
  absl::StatusOr<size_t> Add(HostSignalService service);
  absl::Status Start();
  // Joins the thread and answers kHostSignalStopped on every mailbox.
  // Idempotent. The caller must ensure no program still needs an answer.
  void Stop();

  // Empty while healthy; the first failure otherwise.
  std::string error() const;
  bool running() const { return running_.load(std::memory_order_acquire); }
  size_t size() const { return services_.size(); }

 private:
  void Run();
  bool Serve(HostSignalService& service, size_t index);
  bool Fail(std::string message);

  std::vector<HostSignalService> services_;
  std::vector<HostMailbox*> mailboxes_;
  std::vector<int64_t> after_;
  std::atomic<bool> running_{false};
  std::atomic<bool> failed_{false};
  bool started_ = false;
  bool stopped_ = false;
  std::string error_;
  std::thread worker_;
};

}  // namespace raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_HOST_SIGNAL_H_
