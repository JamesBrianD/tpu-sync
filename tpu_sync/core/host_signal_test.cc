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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"

namespace raiden {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

// Stand-ins for the pinned host pages a device writes by DMA.
struct Pages {
  explicit Pages(int64_t payload_words = kHostSignalPageWords)
      : payload(payload_words, 0),
        request(kHostSignalPageWords, 0),
        response(kHostSignalPageWords, 0) {}

  std::shared_ptr<HostMailbox> Mailbox() {
    return std::make_shared<HostMailbox>(
        payload.data(), static_cast<int64_t>(payload.size()), request.data(),
        response.data());
  }

  // What the request kernel does: payload first, then the generation.
  void Post(int32_t generation, const std::vector<int32_t>& words,
            int32_t after = 0) {
    payload[0] = static_cast<int32_t>(words.size());
    payload[1] = after;
    std::memcpy(payload.data() + kHostSignalHeaderWords, words.data(),
                words.size() * 4);
    __atomic_store_n(request.data(), generation, __ATOMIC_RELEASE);
  }

  int32_t Answer() const {
    return __atomic_load_n(response.data(), __ATOMIC_ACQUIRE);
  }

  // What the receive kernel does: poll until this generation is answered.
  int32_t Await(int32_t generation) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
      const int32_t answer = Answer();
      if (answer == generation || answer == -generation ||
          answer == kHostSignalStopped) {
        return answer;
      }
      std::this_thread::yield();
    }
    ADD_FAILURE() << "no answer for generation " << generation;
    return 0;
  }

  std::vector<int32_t> payload, request, response;
};

// Handler state: copies each word, plus one, as little-endian int32 bytes.
struct Echo {
  std::atomic<int> calls{0};
  int64_t fail_with = 0;
  int64_t overrun_by = 0;
};

int64_t EchoHandler(intptr_t user, const int32_t* words, int64_t count,
                    uint8_t* staging, int64_t staging_bytes) {
  auto* echo = reinterpret_cast<Echo*>(user);
  echo->calls.fetch_add(1);
  if (echo->fail_with != 0) return echo->fail_with;
  if (count * 4 > staging_bytes) return -2;
  for (int64_t i = 0; i < count; ++i) {
    const int32_t value = words[i] + 1;
    std::memcpy(staging + i * 4, &value, 4);
  }
  return count * 4 + echo->overrun_by;
}

// The device side of an upload: records what landed.
struct Device {
  std::vector<int64_t> uploads;
  std::vector<int32_t> landed;
  absl::Status status;
  std::vector<int>* order = nullptr;
  int tag = 0;
};

HostSignalService Service(std::shared_ptr<HostMailbox> mailbox, Echo& echo,
                          std::vector<uint8_t>& staging, Device& device,
                          int64_t after = -1) {
  HostSignalService service;
  service.mailbox = std::move(mailbox);
  service.handler = &EchoHandler;
  service.user = reinterpret_cast<intptr_t>(&echo);
  service.staging = staging.data();
  service.staging_bytes = static_cast<int64_t>(staging.size());
  service.after = after;
  service.upload = [&device, &staging](int64_t bytes) {
    device.uploads.push_back(bytes);
    if (device.order != nullptr) device.order->push_back(device.tag);
    if (!device.status.ok()) return device.status;
    device.landed.assign(bytes / 4, 0);
    std::memcpy(device.landed.data(), staging.data(), bytes);
    return absl::OkStatus();
  };
  return service;
}

TEST(HostMailboxTest, TakesEachPositiveGenerationOnce) {
  Pages pages;
  auto mailbox = pages.Mailbox();
  EXPECT_FALSE(mailbox->TryTake().has_value());

  pages.Post(3, {7, 8}, /*after=*/5);
  ASSERT_EQ(mailbox->TryTake(), 3);
  EXPECT_FALSE(mailbox->TryTake().has_value()) << "same generation again";
  EXPECT_EQ(mailbox->After(), 5);
  auto message = mailbox->Message();
  ASSERT_TRUE(message.ok());
  EXPECT_THAT(*message, ElementsAre(7, 8));

  __atomic_store_n(pages.request.data(), -5, __ATOMIC_RELEASE);
  EXPECT_FALSE(mailbox->TryTake().has_value()) << "non-positive generation";
  pages.Post(4, {});
  EXPECT_EQ(mailbox->TryTake(), 4);
  EXPECT_EQ(mailbox->After(), 0);
  EXPECT_THAT(*mailbox->Message(), IsEmpty());
}

TEST(HostMailboxTest, RejectsCountOutsidePayload) {
  Pages pages(/*payload_words=*/8);
  auto mailbox = pages.Mailbox();
  pages.payload[0] = 7;  // The two header words leave room for 6.
  EXPECT_EQ(mailbox->Message().status().code(), absl::StatusCode::kOutOfRange);
  pages.payload[0] = -1;
  EXPECT_EQ(mailbox->Message().status().code(), absl::StatusCode::kOutOfRange);
  pages.payload[0] = 6;
  EXPECT_TRUE(mailbox->Message().ok());
}

TEST(HostMailboxTest, AnswersAndStops) {
  Pages pages;
  auto mailbox = pages.Mailbox();
  mailbox->Respond(9, /*ok=*/true);
  EXPECT_EQ(mailbox->response(), 9);
  mailbox->Respond(9, /*ok=*/false);
  EXPECT_EQ(mailbox->response(), -9);
  mailbox->Stop();
  EXPECT_EQ(mailbox->response(), kHostSignalStopped);
}

TEST(NextRequestTest, PrefersLowerIndex) {
  Pages first, second;
  auto a = first.Mailbox();
  auto b = second.Mailbox();
  std::vector<HostMailbox*> mailboxes = {a.get(), b.get()};
  std::vector<int64_t> after = {-1, -1};

  EXPECT_FALSE(NextRequest(mailboxes, after).has_value());
  second.Post(1, {});
  first.Post(1, {});
  auto next = NextRequest(mailboxes, after);
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(next->first, 0u);
  next = NextRequest(mailboxes, after);
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(next->first, 1u);
  EXPECT_FALSE(NextRequest(mailboxes, after).has_value());
}

TEST(NextRequestTest, WaitsForThePredecessorsAnswer) {
  Pages first, second;
  auto a = first.Mailbox();
  auto b = second.Mailbox();
  std::vector<HostMailbox*> mailboxes = {a.get(), b.get()};
  std::vector<int64_t> after = {-1, 0};

  // The later layer's request landed first and names generation 4.
  second.Post(1, {}, /*after=*/4);
  EXPECT_FALSE(NextRequest(mailboxes, after).has_value());
  a->Respond(3, /*ok=*/true);  // An older answer is not enough.
  EXPECT_FALSE(NextRequest(mailboxes, after).has_value());
  a->Respond(4, /*ok=*/false);  // A failed answer releases it, to fail.
  auto next = NextRequest(mailboxes, after);
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(next->first, 1u);

  second.Post(2, {}, /*after=*/5);
  a->Stop();  // Stopped releases it too.
  next = NextRequest(mailboxes, after);
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(next->first, 1u);
}

TEST(HostSignalServerTest, ServesUploadsThenAnswers) {
  Pages pages;
  Echo echo;
  Device device;
  std::vector<uint8_t> staging(2 * kHostSignalPageBytes);
  HostSignalServer server;
  ASSERT_TRUE(server.Add(Service(pages.Mailbox(), echo, staging, device)).ok());
  ASSERT_TRUE(server.Start().ok());

  pages.Post(1, {10, 20, 30});
  EXPECT_EQ(pages.Await(1), 1);
  // The answer is stored after the upload returned.
  ASSERT_THAT(device.uploads, ElementsAre(kHostSignalPageBytes));
  EXPECT_EQ(device.landed[0], 11);
  EXPECT_EQ(device.landed[2], 31);

  pages.Post(2, {});  // Nothing to upload, still answered.
  EXPECT_EQ(pages.Await(2), 2);
  EXPECT_EQ(device.uploads.size(), 1u);
  EXPECT_EQ(echo.calls.load(), 2);
  EXPECT_EQ(server.error(), "");

  server.Stop();
  EXPECT_EQ(pages.Await(3), kHostSignalStopped);
  server.Stop();  // Idempotent.
}

TEST(HostSignalServerTest, ServesALaterLayerOnlyAfterItsPredecessor) {
  Pages first, second;
  Echo echo_a, echo_b;
  std::vector<int> order;
  Device device_a, device_b;
  device_a.order = device_b.order = &order;
  device_a.tag = 1;
  device_b.tag = 14;
  std::vector<uint8_t> staging_a(kHostSignalPageBytes),
      staging_b(kHostSignalPageBytes);
  HostSignalServer server;
  auto index_a = server.Add(Service(first.Mailbox(), echo_a, staging_a,
                                    device_a));
  ASSERT_TRUE(index_a.ok());
  EXPECT_EQ(*index_a, 0u);
  ASSERT_TRUE(server
                  .Add(Service(second.Mailbox(), echo_b, staging_b, device_b,
                               /*after=*/0))
                  .ok());
  ASSERT_TRUE(server.Start().ok());

  // L14 lands first; it must not be served before L1's generation 1.
  second.Post(1, {2}, /*after=*/1);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_EQ(second.Answer(), 0);
  EXPECT_TRUE(order.empty());
  first.Post(1, {1});
  EXPECT_EQ(first.Await(1), 1);
  EXPECT_EQ(second.Await(1), 1);
  EXPECT_THAT(order, ElementsAre(1, 14));
  EXPECT_EQ(device_a.landed[0], 2);
  EXPECT_EQ(device_b.landed[0], 3);
  server.Stop();
}

TEST(HostSignalServerTest, PredecessorFailureFailsTheSuccessor) {
  Pages first, second;
  Echo echo_a, echo_b;
  echo_a.fail_with = -1;
  Device device_a, device_b;
  std::vector<uint8_t> staging_a(kHostSignalPageBytes),
      staging_b(kHostSignalPageBytes);
  HostSignalServer server;
  ASSERT_TRUE(
      server.Add(Service(first.Mailbox(), echo_a, staging_a, device_a)).ok());
  ASSERT_TRUE(server
                  .Add(Service(second.Mailbox(), echo_b, staging_b, device_b,
                               /*after=*/0))
                  .ok());
  ASSERT_TRUE(server.Start().ok());
  first.Post(1, {1});
  second.Post(1, {2}, /*after=*/1);
  EXPECT_EQ(first.Await(1), -1);
  EXPECT_EQ(second.Await(1), -1);
  EXPECT_EQ(echo_b.calls.load(), 0);
  server.Stop();
}

TEST(HostSignalServerTest, AfterWithoutPredecessorFails) {
  Pages pages;
  Echo echo;
  Device device;
  std::vector<uint8_t> staging(kHostSignalPageBytes);
  HostSignalServer server;
  ASSERT_TRUE(server.Add(Service(pages.Mailbox(), echo, staging, device)).ok());
  ASSERT_TRUE(server.Start().ok());
  pages.Post(1, {1}, /*after=*/3);
  EXPECT_EQ(pages.Await(1), -1);
  EXPECT_THAT(server.error(), HasSubstr("has no predecessor"));
  EXPECT_EQ(echo.calls.load(), 0);
  server.Stop();
}

TEST(HostSignalServerTest, HandlerFailureIsStickyAndAnsweredNegative) {
  Pages pages;
  Echo echo;
  echo.fail_with = -7;
  Device device;
  std::vector<uint8_t> staging(kHostSignalPageBytes);
  HostSignalServer server;
  ASSERT_TRUE(server.Add(Service(pages.Mailbox(), echo, staging, device)).ok());
  ASSERT_TRUE(server.Start().ok());

  pages.Post(1, {1});
  EXPECT_EQ(pages.Await(1), -1);
  EXPECT_THAT(server.error(), HasSubstr("handler returned error -7"));

  echo.fail_with = 0;
  pages.Post(2, {1});
  EXPECT_EQ(pages.Await(2), -2) << "failures stay sticky";
  EXPECT_EQ(echo.calls.load(), 1) << "no handler call after a failure";
  EXPECT_TRUE(device.uploads.empty());
  server.Stop();
}

TEST(HostSignalServerTest, UploadFailureAndOverrunFail) {
  {
    Pages pages;
    Echo echo;
    Device device;
    device.status = absl::InternalError("dma");
    std::vector<uint8_t> staging(kHostSignalPageBytes);
    HostSignalServer server;
    ASSERT_TRUE(
        server.Add(Service(pages.Mailbox(), echo, staging, device)).ok());
    ASSERT_TRUE(server.Start().ok());
    pages.Post(5, {1});
    EXPECT_EQ(pages.Await(5), -5);
    EXPECT_THAT(server.error(), HasSubstr("upload of 4096 bytes failed: dma"));
    server.Stop();
  }
  {
    Pages pages;
    Echo echo;
    echo.overrun_by = kHostSignalPageBytes;
    Device device;
    std::vector<uint8_t> staging(kHostSignalPageBytes);
    HostSignalServer server;
    ASSERT_TRUE(
        server.Add(Service(pages.Mailbox(), echo, staging, device)).ok());
    ASSERT_TRUE(server.Start().ok());
    pages.Post(1, {1});
    EXPECT_EQ(pages.Await(1), -1);
    EXPECT_THAT(server.error(), HasSubstr("handler wrote 4100 bytes"));
    server.Stop();
  }
}

TEST(HostSignalServerTest, BadCountFails) {
  Pages pages(/*payload_words=*/4);
  Echo echo;
  Device device;
  std::vector<uint8_t> staging(kHostSignalPageBytes);
  HostSignalServer server;
  ASSERT_TRUE(server.Add(Service(pages.Mailbox(), echo, staging, device)).ok());
  ASSERT_TRUE(server.Start().ok());
  pages.payload[0] = 100;
  __atomic_store_n(pages.request.data(), 1, __ATOMIC_RELEASE);
  EXPECT_EQ(pages.Await(1), -1);
  EXPECT_THAT(server.error(), HasSubstr("outside [0, 2]"));
  EXPECT_EQ(echo.calls.load(), 0);
  server.Stop();
}

TEST(HostSignalServerTest, StopEndsAPendingWait) {
  Pages pages;
  Echo echo;
  Device device;
  std::vector<uint8_t> staging(kHostSignalPageBytes);
  HostSignalServer server;
  ASSERT_TRUE(server.Add(Service(pages.Mailbox(), echo, staging, device)).ok());
  ASSERT_TRUE(server.Start().ok());
  std::thread stopper([&server] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    server.Stop();
  });
  EXPECT_EQ(pages.Await(1), kHostSignalStopped);
  stopper.join();
  EXPECT_FALSE(server.running());
}

TEST(HostSignalServerTest, StopWithoutStartStillAnswersStopped) {
  Pages pages;
  Echo echo;
  Device device;
  std::vector<uint8_t> staging(kHostSignalPageBytes);
  HostSignalServer server;
  ASSERT_TRUE(server.Add(Service(pages.Mailbox(), echo, staging, device)).ok());
  server.Stop();
  EXPECT_EQ(pages.response[0], kHostSignalStopped);
  EXPECT_EQ(server.Start().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(HostSignalServerTest, RejectsInvalidSetup) {
  Pages pages;
  Echo echo;
  Device device;
  std::vector<uint8_t> staging(kHostSignalPageBytes);
  HostSignalServer server;
  EXPECT_EQ(server.Start().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(server.Add(HostSignalService{}).status().code(),
            absl::StatusCode::kInvalidArgument);
  Pages tiny(/*payload_words=*/1);
  EXPECT_EQ(
      server.Add(Service(tiny.Mailbox(), echo, staging, device)).status().code(),
      absl::StatusCode::kInvalidArgument);
  Pages ahead;
  EXPECT_EQ(server
                .Add(Service(ahead.Mailbox(), echo, staging, device,
                             /*after=*/0))
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument)
      << "a predecessor must be added first";

  auto mailbox = pages.Mailbox();
  ASSERT_TRUE(server.Add(Service(mailbox, echo, staging, device)).ok());
  EXPECT_EQ(server.Add(Service(mailbox, echo, staging, device)).status().code(),
            absl::StatusCode::kAlreadyExists);
  ASSERT_TRUE(server.Start().ok());
  Pages other;
  EXPECT_EQ(
      server.Add(Service(other.Mailbox(), echo, staging, device)).status().code(),
      absl::StatusCode::kFailedPrecondition);
  server.Stop();
}

}  // namespace
}  // namespace raiden
