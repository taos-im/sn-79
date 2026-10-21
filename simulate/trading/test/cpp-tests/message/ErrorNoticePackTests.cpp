// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT
//
// Every error notice is packed from the payload it carries. A typed error names the request it refuses,
// as it always has. Any other payload is the refusal of an instruction that could not be decoded: it is
// packed as a refusal carrying the reason, if it has one, and names no book, order or agent it does not
// know.

#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/message/Message.hpp>
#include <taosim/message/MultiBookMessagePayloads.hpp>
#include <taosim/serialization/msgpack/common.hpp>
#include <taosim/simulation/serialization/NoticePack.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

//-------------------------------------------------------------------------

using namespace taosim::literals;

using testing::Test;
using testing::TestWithParam;
using testing::Values;

//-------------------------------------------------------------------------

namespace
{

constexpr AgentId kAgent{7};
constexpr BookId kBook{1};
constexpr OrderID kOrder{31};
const std::string kReason = "the instruction could not be decoded";

using Fields = std::map<std::string, msgpack::object>;

[[nodiscard]] Fields mapOf(const msgpack::object& o)
{
    Fields out;
    if (o.type != msgpack::type::MAP) return out;
    for (const auto& kv : o.via.map) {
        out.emplace(kv.key.as<std::string>(), kv.val);
    }
    return out;
}

[[nodiscard]] Message::Ptr notice(const std::string& type, MessagePayload::Ptr payload)
{
    return Message::create(
        Timestamp{5},
        Timestamp{5},
        "EXCHANGE",
        "DISTRIBUTED_PROXY_AGENT",
        type,
        MessagePayload::create<DistributedAgentResponsePayload>(kAgent, std::move(payload)));
}

class Packed
{
public:
    explicit Packed(const Message::Ptr& msg)
    {
        msgpack::packer packer{m_stream};
        taosim::simulation::serialization::packNotice(packer, msg, std::string{}, "ErrorNoticePackTests");
        m_handle = msgpack::unpack(m_stream.data(), m_stream.size());
        fields = mapOf(m_handle.get());
    }

    // The entries of the list under `key`, each as its fields.
    [[nodiscard]] std::vector<Fields> items(const std::string& key) const
    {
        std::vector<Fields> out;
        const auto& list = fields.at(key);
        if (list.type != msgpack::type::ARRAY) return out;
        for (const auto& item : list.via.array) out.push_back(mapOf(item));
        return out;
    }

    Fields fields;

private:
    taosim::serialization::HumanReadableStream m_stream;
    msgpack::object_handle m_handle;
};

[[nodiscard]] MessagePayload::Ptr plainError()
{
    return MessagePayload::create<ErrorResponsePayload>(kReason);
}

[[nodiscard]] MessagePayload::Ptr typedError(const std::string& type)
{
    const auto error = MessagePayload::create<ErrorResponsePayload>("Order IDs 31 do not exist.");
    if (type.ends_with("PLACE_ORDER_LIMIT")) {
        return MessagePayload::create<PlaceOrderLimitErrorResponsePayload>(
            MessagePayload::create<PlaceOrderLimitPayload>(OrderDirection::SELL, 2_dec, 10_dec, kBook), error);
    }
    if (type.ends_with("PLACE_ORDER_MARKET")) {
        return MessagePayload::create<PlaceOrderMarketErrorResponsePayload>(
            MessagePayload::create<PlaceOrderMarketPayload>(OrderDirection::SELL, 2_dec, kBook), error);
    }
    if (type.ends_with("CANCEL_ORDERS")) {
        return MessagePayload::create<CancelOrdersErrorResponsePayload>(
            std::vector{kOrder},
            MessagePayload::create<CancelOrdersPayload>(std::vector{taosim::event::Cancellation{kOrder}}, kBook),
            error);
    }
    if (type.ends_with("CLOSE_POSITIONS")) {
        return MessagePayload::create<ClosePositionsErrorResponsePayload>(
            std::vector{kOrder}, MessagePayload::create<ClosePositionsPayload>(ClosePosition{kOrder}, kBook), error);
    }
    return MessagePayload::create<ResetAgentsErrorResponsePayload>(
        std::vector{kAgent}, MessagePayload::create<ResetAgentsPayload>(std::vector{kAgent}), error);
}

// The fields of the refusal itself: the notice for a placement, its one entry otherwise.
[[nodiscard]] Fields refusalOf(const Packed& packed, const std::string& type)
{
    if (type.contains("PLACE_ORDER")) return packed.fields;
    const auto key = type.ends_with("CANCEL_ORDERS") ? "c" : type.ends_with("CLOSE_POSITIONS") ? "o" : "r";
    const auto entries = packed.items(key);
    EXPECT_EQ(entries.size(), 1u) << type;
    return entries.empty() ? Fields{} : entries.front();
}

}  // namespace

//-------------------------------------------------------------------------

class ErrorNotice : public TestWithParam<std::string> {};

TEST_P(ErrorNotice, AnUndecodedInstructionIsPackedAsARefusalCarryingItsReason)
{
    std::unique_ptr<Packed> packed;
    ASSERT_NO_THROW(packed = std::make_unique<Packed>(notice(GetParam(), plainError())));

    const auto refusal = refusalOf(*packed, GetParam());
    ASSERT_TRUE(refusal.contains("u") && refusal.contains("m")) << GetParam();
    EXPECT_FALSE(refusal.at("u").as<bool>());
    EXPECT_EQ(refusal.at("m").as<std::string>(), kReason);
    for (const auto* unnamed : {"b", "o"}) {
        if (refusal.contains(unnamed)) {
            EXPECT_EQ(refusal.at(unnamed).type, msgpack::type::NIL) << GetParam() << " named a " << unnamed;
        }
    }
    if (packed->fields.contains("b")) {
        EXPECT_EQ(packed->fields.at("b").type, msgpack::type::NIL) << GetParam() << " named a book";
    }
}

TEST_P(ErrorNotice, APayloadOfAnotherTypeIsPackedAsARefusalWithoutAReason)
{
    const auto wrong = MessagePayload::create<CancelOrdersResponsePayload>(
        std::vector{kOrder},
        MessagePayload::create<CancelOrdersPayload>(std::vector{taosim::event::Cancellation{kOrder}}, kBook));
    const auto payload = GetParam().ends_with("CANCEL_ORDERS") ? MessagePayload::create<EmptyPayload>() : wrong;

    std::unique_ptr<Packed> packed;
    ASSERT_NO_THROW(packed = std::make_unique<Packed>(notice(GetParam(), payload)));

    const auto refusal = refusalOf(*packed, GetParam());
    ASSERT_TRUE(refusal.contains("u") && refusal.contains("m")) << GetParam();
    EXPECT_FALSE(refusal.at("u").as<bool>());
    EXPECT_EQ(refusal.at("m").as<std::string>(), "");
}

TEST_P(ErrorNotice, ATypedErrorIsPackedFromTheRequestItRefuses)
{
    const Packed packed{notice(GetParam(), typedError(GetParam()))};

    const auto refusal = refusalOf(packed, GetParam());
    ASSERT_TRUE(refusal.contains("u") && refusal.contains("m")) << GetParam();
    EXPECT_FALSE(refusal.at("u").as<bool>());
    EXPECT_EQ(refusal.at("m").as<std::string>(), "Order IDs 31 do not exist.");
    if (GetParam().ends_with("RESET_AGENT")) {
        EXPECT_EQ(refusal.at("a").as<AgentId>(), kAgent);
        return;
    }
    EXPECT_EQ(packed.fields.at("b").as<BookId>(), kBook);
    if (!GetParam().contains("PLACE_ORDER")) {
        EXPECT_EQ(refusal.at("o").as<OrderID>(), kOrder);
        EXPECT_EQ(refusal.at("b").as<BookId>(), kBook);
    }
}

INSTANTIATE_TEST_SUITE_P(
    NoticePack,
    ErrorNotice,
    Values(
        "ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT",
        "ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET",
        "ERROR_RESPONSE_DISTRIBUTED_CANCEL_ORDERS",
        "ERROR_RESPONSE_DISTRIBUTED_CLOSE_POSITIONS",
        "ERROR_RESPONSE_DISTRIBUTED_RESET_AGENT"));

//-------------------------------------------------------------------------
