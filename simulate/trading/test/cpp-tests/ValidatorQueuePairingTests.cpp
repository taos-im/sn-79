// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT
//
// The engine publishes a state with one size_t on the request queue and takes the next size_t on
// the response queue as the validator's answer to it. Both queues outlive their processes, so a
// reply left by a validator that answered a previous (killed) engine would be read as the answer
// to the new engine's first state, and every later reply would pair one state late. These pin
// that a stale frame in either queue is discarded before a request is sent. Test-only queue names,
// never the production /taosim-* ones.

#include <gtest/gtest.h>

#include <taosim/ipc/PosixMessageQueue.hpp>

#include <unistd.h>

#include <bit>
#include <cstddef>
#include <optional>
#include <span>
#include <string>

using taosim::ipc::PosixMessageQueue;
using taosim::ipc::PosixMessageQueueDesc;
using taosim::ipc::sendFreshRequest;

namespace
{

std::string testQueueName(const char* role)
{
    return "/taosim-test-pairing-" + std::to_string(::getpid()) + "-" + role;
}

PosixMessageQueueDesc testDesc(const std::string& name)
{
    PosixMessageQueueDesc desc{.name = name};
    desc.timeout = std::optional<size_t>{100'000'000};
    return desc;
}

bool sendSize(PosixMessageQueue& q, size_t value)
{
    return q.send(std::span<const char>{std::bit_cast<const char*>(&value), sizeof(value)});
}

std::optional<size_t> receiveSize(PosixMessageQueue& q)
{
    size_t value{};
    if (q.receive(std::span<char>{std::bit_cast<char*>(&value), sizeof(value)}) == -1) {
        return std::nullopt;
    }
    return value;
}

}  // namespace

TEST(ValidatorQueuePairing, AStaleReplyIsDiscardedBeforeTheNextRequest)
{
    PosixMessageQueue req{testDesc(testQueueName("req"))};
    PosixMessageQueue res{testDesc(testQueueName("res"))};

    // The validator's answer to a state the previous engine published.
    ASSERT_TRUE(sendSize(res, 111));

    const size_t packedSize = 222;
    ASSERT_TRUE(sendFreshRequest(
        req, res, std::span<const char>{std::bit_cast<const char*>(&packedSize), sizeof(packedSize)}));

    EXPECT_EQ(res.size(), std::optional<size_t>{0});

    // The validator reads this request and answers it, and that answer is what the engine pairs.
    EXPECT_EQ(receiveSize(req), std::optional<size_t>{222});
    ASSERT_TRUE(sendSize(res, 333));
    EXPECT_EQ(receiveSize(res), std::optional<size_t>{333});
}

TEST(ValidatorQueuePairing, AStaleRequestIsDiscardedBeforeTheNextRequest)
{
    PosixMessageQueue req{testDesc(testQueueName("req"))};
    PosixMessageQueue res{testDesc(testQueueName("res"))};

    // A state the previous engine published that no validator read.
    ASSERT_TRUE(sendSize(req, 111));

    const size_t packedSize = 222;
    ASSERT_TRUE(sendFreshRequest(
        req, res, std::span<const char>{std::bit_cast<const char*>(&packedSize), sizeof(packedSize)}));

    EXPECT_EQ(receiveSize(req), std::optional<size_t>{222});
    EXPECT_EQ(req.size(), std::optional<size_t>{0});
}

TEST(ValidatorQueuePairing, AFlushRightAfterOpeningDropsWhatAPreviousProcessLeft)
{
    const std::string name = testQueueName("open");
    PosixMessageQueue previous{testDesc(name)};
    ASSERT_TRUE(sendSize(previous, 111));

    PosixMessageQueue reopened{testDesc(name)};
    ASSERT_EQ(reopened.size(), std::optional<size_t>{1});
    reopened.flush();
    EXPECT_EQ(reopened.size(), std::optional<size_t>{0});
}
