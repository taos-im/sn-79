/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

/*
 * Every wakeup handler reads its book id through `static_pointer_cast<WakeupPayload>`, which
 * cannot fail and cannot report failure. Handed a payload of another type it reads whatever
 * lies at that offset and returns a plausible-looking book id; handed a null it dereferences
 * one. Neither is an exception, so neither reaches the watchdog: the first indexes a container
 * with garbage and the second takes the process down with no message at all.
 *
 * The cast is safe only because a type invariant holds at every site that can produce one of
 * these messages, and that invariant is not enforced anywhere. It has two halves, and this
 * fixture pins the half a checkpoint restore owns.
 *
 *   1. Every dispatch of a WAKEUP-family type passes `MessagePayload::create<WakeupPayload>`.
 *      Verified by reading the dispatch sites; there is no runtime hook to assert it on.
 *   2. A restore rebuilds those payloads as the same type. That is `PayloadFactory`, and it is
 *      what these tests cover, because it is a lookup keyed by a STRING: adding a wakeup type
 *      without adding its branch compiles, runs, and fails only when someone restores a
 *      checkpoint that happened to have one in flight.
 *
 * `WAKEUP_ALGOTRADER` is the exception that proves the invariant is a real distinction rather
 * than a formality: it carries no payload and its handler never reads one.
 */

#include <taosim/message/PayloadFactory.hpp>
#include <taosim/message/MultiBookMessagePayloads.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>

//-------------------------------------------------------------------------

using namespace taosim;
using namespace testing;

//-------------------------------------------------------------------------

// The payload-carrying wakeup types, each of whose handler casts to WakeupPayload and reads
// `bookId`. A new one added to the engine belongs in this list.
class WakeupPayloadContractTest : public TestWithParam<const char*>
{
};

INSTANTIATE_TEST_SUITE_P(
    WakeupFamily,
    WakeupPayloadContractTest,
    Values("WAKEUP", "WAKEUP_HFT_FILL", "WAKEUP_ALGOTRADER_BOOK", "WAKEUP_ALGOTRADER_EXECUTE"));

//-------------------------------------------------------------------------

// A restored wakeup is a WakeupPayload and its book id survives. Were the branch missing, the
// factory would fall through to an EmptyPayload and the handler would read a book id off an
// object that has none, which is undefined rather than wrong: on a good day an out-of-range id
// and a thrown `.at()`, on a bad one a valid id for the wrong book.
TEST_P(WakeupPayloadContractTest, RestoresAsAWakeupPayloadCarryingItsBookId)
{
    static constexpr BookId kBookId = 5;

    // The factory keys off the message's own "type" member, which is why a missing branch is
    // a silent fallthrough rather than a compile error.
    rapidjson::Document json{rapidjson::kObjectType};
    auto& allocator = json.GetAllocator();
    json.AddMember("type", rapidjson::Value{GetParam(), allocator}, allocator);
    json.AddMember(
        "payload",
        [&] {
            rapidjson::Value payloadJson{rapidjson::kObjectType};
            payloadJson.AddMember("bookId", rapidjson::Value{kBookId}, allocator);
            return payloadJson;
        }().Move(),
        allocator);

    const auto payload = PayloadFactory::createFromJsonMessage(json);
    ASSERT_NE(payload, nullptr) << GetParam() << " restored as nothing at all";

    const auto wakeup = std::dynamic_pointer_cast<WakeupPayload>(payload);
    ASSERT_NE(wakeup, nullptr)
        << GetParam() << " restored as some other payload type. Every handler for it casts "
           "with static_pointer_cast, which will not notice";
    EXPECT_EQ(wakeup->bookId, kBookId)
        << GetParam() << " restored against the wrong book";
}

//-------------------------------------------------------------------------

// The one that carries nothing. Asserted explicitly so that giving it a payload later is a
// deliberate change rather than a silent one, and so the asymmetry with the family above is
// recorded somewhere other than a comment.
TEST(WakeupPayloadContractTest, TheAlgoTraderWakeupCarriesNoPayloadAndItsHandlerReadsNone)
{
    rapidjson::Document json{rapidjson::kObjectType};
    auto& allocator = json.GetAllocator();
    json.AddMember("type", rapidjson::Value{"WAKEUP_ALGOTRADER", allocator}, allocator);
    json.AddMember("payload", rapidjson::Value{rapidjson::kObjectType}.Move(), allocator);

    const auto payload = PayloadFactory::createFromJsonMessage(json);
    ASSERT_NE(payload, nullptr)
        << "restored as nothing, so a checkpoint holding one in the queue cannot be restored";
    EXPECT_NE(std::dynamic_pointer_cast<EmptyPayload>(payload), nullptr)
        << "WAKEUP_ALGOTRADER grew a payload; ALGOTraderAgent::handleWakeup ignores its "
           "message argument entirely and loops over every book, so whatever it now carries "
           "is being dropped";
}
