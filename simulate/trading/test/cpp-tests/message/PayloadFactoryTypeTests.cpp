// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT
//
// The msgpack payload factory decodes the message types it models and refuses every other, as the JSON
// factory does. A checkpointed message queue also holds the engine's own events and responses, which the
// factory does not model; those still restore, with an empty payload.

#include <taosim/message/Message.hpp>
#include <taosim/message/MultiBookMessagePayloads.hpp>
#include <taosim/message/PayloadFactory.hpp>
#include <taosim/message/serialization/Message.hpp>
#include <taosim/message/serialization/helpers.hpp>
#include <taosim/serialization/msgpack/common.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <rapidjson/document.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

//-------------------------------------------------------------------------

using namespace taosim::literals;

using testing::Test;
using testing::TestWithParam;
using testing::Values;

//-------------------------------------------------------------------------

namespace
{

constexpr BookId kBookId = 3;
constexpr AgentId kAgentId = 5;

// Packs a payload as the engine writes it and decodes it as an inbound message of `type`.
MessagePayload::Ptr decode(const MessagePayload::Ptr& payload, std::string_view type)
{
    taosim::serialization::BinaryStream stream;
    msgpack::packer<taosim::serialization::BinaryStream> packer{stream};
    taosim::message::serialization::packMessagePayload(packer, payload);
    const auto handle = msgpack::unpack(stream.data(), stream.size());
    return PayloadFactory::createFromMessagePack(handle.get(), type);
}

// Packs a whole message as a checkpoint writes it into the queue, and reads it back as a restore does.
Message restore(const Message& msg)
{
    taosim::serialization::BinaryStream stream;
    msgpack::packer<taosim::serialization::BinaryStream> packer{stream};
    packer.pack(msg);
    const auto handle = msgpack::unpack(stream.data(), stream.size());
    return handle.get().as<Message>();
}

MessagePayload::Ptr marketBuy()
{
    return MessagePayload::create<PlaceOrderMarketPayload>(OrderDirection::BUY, 2_dec, kBookId);
}

MessagePayload::Ptr wrapped(MessagePayload::Ptr payload)
{
    return MessagePayload::create<DistributedAgentResponsePayload>(kAgentId, std::move(payload));
}

}  // namespace

//-------------------------------------------------------------------------

class UnrecognizedMessagePackType : public TestWithParam<std::string> {};

TEST_P(UnrecognizedMessagePackType, IsRefused)
{
    EXPECT_THROW(std::ignore = decode(marketBuy(), GetParam()), std::runtime_error);
}

TEST_P(UnrecognizedMessagePackType, IsRefusedInsideADistributedWrapper)
{
    EXPECT_THROW(
        std::ignore = decode(wrapped(marketBuy()), "DISTRIBUTED_" + GetParam()), std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(
    PayloadFactory,
    UnrecognizedMessagePackType,
    Values(
        "PLACE_ORDER_MARKET_LIMIT",
        "XPLACE_ORDER_MARKET",
        "place_order_market",
        "PLACE_ORDER",
        ""));

//-------------------------------------------------------------------------

TEST(PayloadFactoryTypes, TheJsonFactoryRefusesAnUnrecognizedTypeToo)
{
    rapidjson::Document json{rapidjson::kObjectType};
    auto& allocator = json.GetAllocator();
    json.AddMember("type", rapidjson::Value{"PLACE_ORDER_MARKET_LIMIT", allocator}, allocator);
    json.AddMember("payload", rapidjson::Value{rapidjson::kObjectType}, allocator);

    EXPECT_THROW(std::ignore = PayloadFactory::createFromJsonMessage(json), std::runtime_error);
}

//-------------------------------------------------------------------------

TEST(PayloadFactoryTypes, AMarginCallDecodesAsTheMarketOrderItCarries)
{
    const auto payload = std::dynamic_pointer_cast<PlaceOrderMarketPayload>(
        decode(marketBuy(), "PLACE_ORDER_MARKET_MC"));
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->bookId, kBookId);
    EXPECT_EQ(payload->volume, 2_dec);
    EXPECT_EQ(payload->direction, OrderDirection::BUY);

    const auto distributed = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(
        decode(wrapped(marketBuy()), "DISTRIBUTED_PLACE_ORDER_MARKET_MC"));
    ASSERT_NE(distributed, nullptr);
    EXPECT_EQ(distributed->agentId, kAgentId);
    const auto inner = std::dynamic_pointer_cast<PlaceOrderMarketPayload>(distributed->payload);
    ASSERT_NE(inner, nullptr);
    EXPECT_EQ(inner->bookId, kBookId);
}

//-------------------------------------------------------------------------

TEST(PayloadFactoryTypes, EveryInstructionTypeStillDecodesToItsPayload)
{
    const auto expectDecodes = [](const MessagePayload::Ptr& payload, const std::string& type, auto isA) {
        MessagePayload::Ptr decoded;
        ASSERT_NO_THROW(decoded = decode(payload, type)) << type;
        EXPECT_TRUE(isA(decoded)) << type;

        ASSERT_NO_THROW(decoded = decode(wrapped(payload), "DISTRIBUTED_" + type)) << type;
        const auto distributed = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(decoded);
        ASSERT_NE(distributed, nullptr) << type;
        EXPECT_EQ(distributed->agentId, kAgentId) << type;
        EXPECT_TRUE(isA(distributed->payload)) << type;
    };
    const auto is = []<typename T>(std::type_identity<T>) {
        return [](const MessagePayload::Ptr& p) { return std::dynamic_pointer_cast<T>(p) != nullptr; };
    };

    expectDecodes(marketBuy(), "PLACE_ORDER_MARKET", is(std::type_identity<PlaceOrderMarketPayload>{}));
    expectDecodes(
        MessagePayload::create<PlaceOrderLimitPayload>(OrderDirection::SELL, 1_dec, 10_dec, kBookId),
        "PLACE_ORDER_LIMIT",
        is(std::type_identity<PlaceOrderLimitPayload>{}));
    expectDecodes(
        MessagePayload::create<CancelOrdersPayload>(
            std::vector{taosim::event::Cancellation{OrderID{7}}}, kBookId),
        "CANCEL_ORDERS",
        is(std::type_identity<CancelOrdersPayload>{}));
    expectDecodes(
        MessagePayload::create<ClosePositionsPayload>(ClosePosition{OrderID{7}}, kBookId),
        "CLOSE_POSITIONS",
        is(std::type_identity<ClosePositionsPayload>{}));
    expectDecodes(
        MessagePayload::create<RetrieveOrdersPayload>(std::vector{OrderID{7}}, kBookId),
        "RETRIEVE_ORDERS",
        is(std::type_identity<RetrieveOrdersPayload>{}));
    expectDecodes(
        MessagePayload::create<ResetAgentsPayload>(std::vector{kAgentId}),
        "RESET_AGENT",
        is(std::type_identity<ResetAgentsPayload>{}));
}

//-------------------------------------------------------------------------

class CheckpointedEngineMessage : public TestWithParam<std::string> {};

TEST_P(CheckpointedEngineMessage, RestoresFromTheQueueWithAnEmptyPayload)
{
    const Message msg{
        Timestamp{10},
        Timestamp{11},
        "EXCHANGE",
        "DISTRIBUTED_PROXY_AGENT",
        GetParam(),
        wrapped(MessagePayload::create<CancelOrdersResponsePayload>(
            std::vector{OrderID{7}},
            MessagePayload::create<CancelOrdersPayload>(
                std::vector{taosim::event::Cancellation{OrderID{7}}}, kBookId)))};

    Message restored;
    ASSERT_NO_THROW(restored = restore(msg)) << "a checkpoint holding a " << GetParam() << " cannot be restored";
    EXPECT_EQ(restored.type, GetParam());
    EXPECT_EQ(restored.arrival, msg.arrival);
    EXPECT_NE(std::dynamic_pointer_cast<EmptyPayload>(restored.payload), nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    PayloadFactory,
    CheckpointedEngineMessage,
    Values(
        "RESPONSE_DISTRIBUTED_CANCEL_ORDERS",
        "EVENT_TRADE",
        "RESPONSE_RETRIEVE_ORDERS",
        "EVENT_ORDER_LIMIT"));

//-------------------------------------------------------------------------
