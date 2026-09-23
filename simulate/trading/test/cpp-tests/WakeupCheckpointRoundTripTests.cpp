// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT
//
// A WAKEUP must survive a checkpoint. It did not, and the engine could not restart.
//
// `packMessagePayload` (serialization/helpers.hpp) dispatches over a chain of
// `dynamic_pointer_cast` branches and has NO WakeupPayload branch, so every WakeupPayload fell
// through to `else { o.pack_nil(); }` and was written to the checkpoint as nil. In 0.6.1 that was
// invisible: PayloadFactory turned a restored WAKEUP into an EmptyPayload, and nil reads back as
// empty. 0.6.2 casts it to WakeupPayload instead, so nil throws:
//
//     PayloadFactory::createFromMessagePack(...): Error creating payload of type 'WAKEUP':
//     std::bad_cast
//
// The engine then crash-loops on its own checkpoint, the validator issues zero query rounds and the
// miners go quiet until someone cold-starts past it by hand. Seen ten times in one error log, every
// one of type WAKEUP. A restart test that exercises engine recovery catches it end to end.
//
// WakeupPayloadContractTests already pins that DISPATCH passes a WakeupPayload. It could not catch
// this, because nothing exercised the payload through the message serializer the checkpoint uses.
// That is the gap: the round trip, not the payload in isolation.

#include <taosim/message/PayloadFactory.hpp>
#include <taosim/message/MultiBookMessagePayloads.hpp>
#include <taosim/message/serialization/helpers.hpp>
#include <taosim/serialization/msgpack/common.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sstream>
#include <string>

using namespace taosim;
using namespace testing;

namespace
{

// Pack a payload exactly as a checkpoint does, then read it back exactly as a restore does.
MessagePayload::Ptr roundTrip(MessagePayload::Ptr payload, std::string_view type)
{
    // BinaryStream, not a bare sbuffer: the decimal adaptor static_asserts on any other stream
    // type, and a checkpoint is written through this one.
    taosim::serialization::BinaryStream stream;
    msgpack::packer<taosim::serialization::BinaryStream> packer{stream};
    taosim::message::serialization::packMessagePayload(packer, payload);
    auto handle = msgpack::unpack(stream.data(), stream.size());
    return PayloadFactory::createFromMessagePack(handle.get(), type);
}

}  // namespace

// The WAKEUP-family types a checkpoint can hold, each of which the factory casts to WakeupPayload.
class WakeupCheckpointRoundTrip : public TestWithParam<std::string> {};

TEST_P(WakeupCheckpointRoundTrip, SurvivesTheTripTheCheckpointPutsItThrough)
{
    constexpr BookId kBookId = 7;
    MessagePayload::Ptr restored;
    ASSERT_NO_THROW(
        restored = roundTrip(MessagePayload::create<WakeupPayload>(kBookId), GetParam()))
        << "a " << GetParam() << " written to a checkpoint cannot be read back, so an engine "
        << "restart dies here and pm2 restarts it into the same failure";

    const auto wakeup = std::dynamic_pointer_cast<WakeupPayload>(restored);
    ASSERT_NE(wakeup, nullptr)
        << GetParam() << " came back as something other than a WakeupPayload, so every handler that "
        << "reads its book id through static_pointer_cast is reading the wrong object";
    EXPECT_EQ(wakeup->bookId, kBookId)
        << "the book id did not survive, so the wakeup would fire against the wrong book";
}

INSTANTIATE_TEST_SUITE_P(
    WakeupFamily, WakeupCheckpointRoundTrip,
    Values("WAKEUP", "WAKEUP_ALGOTRADER_BOOK", "WAKEUP_ALGOTRADER_EXECUTE", "WAKEUP_HFT_FILL"));

// The payload-free member of the family. It must keep restoring, and must NOT start demanding a
// WakeupPayload: PayloadFactory gives it an EmptyPayload precisely because it carries nothing.
TEST(WakeupCheckpointRoundTripAlgoTrader, ThePayloadFreeMemberStillRestores)
{
    MessagePayload::Ptr restored;
    ASSERT_NO_THROW(
        restored = roundTrip(MessagePayload::create<EmptyPayload>(), "WAKEUP_ALGOTRADER"));
    EXPECT_NE(std::dynamic_pointer_cast<EmptyPayload>(restored), nullptr);
}


// THE CHECKPOINTS ALREADY ON DISK. Every one written before the packer branch existed stored its
// WAKEUPs as nil, and a validator resumes `-c latest`. If those are refused, the first restart after
// this ships is the same outage the fix removes -- so nil must still load.
TEST(WakeupCheckpointRoundTripLegacy, ANilPayloadFromAnOlderCheckpointStillLoads)
{
    taosim::serialization::BinaryStream stream;
    msgpack::packer<taosim::serialization::BinaryStream> packer{stream};
    packer.pack_nil();
    auto handle = msgpack::unpack(stream.data(), stream.size());

    MessagePayload::Ptr restored;
    ASSERT_NO_THROW(restored = PayloadFactory::createFromMessagePack(handle.get(), "WAKEUP"))
        << "an engine upgraded onto an older checkpoint cannot start";
    EXPECT_NE(std::dynamic_pointer_cast<WakeupPayload>(restored), nullptr)
        << "it must come back as a WakeupPayload, because every handler casts to one";
}
