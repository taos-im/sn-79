/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

// A distributed message reaches the handler its type names exactly, and only when it carries the payload
// that type carries. An order, a retrieve or a close must name a book the exchange has; an order or a close
// must come from an agent whose account holds that book. A message that fails is answered with the error
// its family answers with, or, when it carries no agent payload or the wrong one, dropped; either way the
// step goes on. Every message the validator and the engine send still reaches its handler as before.

#include "MultiBookExchangeAgent.hpp"
#include "Simulation.hpp"
#include "util.hpp"
#include <taosim/agent/DistributedProxyAgent.hpp>
#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/message/MultiBookMessagePayloads.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <range/v3/algorithm/count_if.hpp>
#include <range/v3/algorithm/find.hpp>
#include <range/v3/algorithm/find_if.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

//-------------------------------------------------------------------------

using namespace taosim::literals;

using testing::Test;

namespace fs = std::filesystem;

//-------------------------------------------------------------------------

namespace
{

const auto kTestDataPath = fs::path{__FILE__}.parent_path().parent_path() / "data";

constexpr BookId kBookId{};
constexpr AgentId kTrader{0};
constexpr AgentId kCounterparty{1};
constexpr AgentId kUnregistered{7};

//-------------------------------------------------------------------------

class DistributedInboundCheckTest : public Test
{
protected:
    // MultiAgentFees.xml: one book, remote agents 0 and 1 registered at configure.
    void SetUp() override
    {
        nodes = taosim::util::parseSimulationFile(kTestDataPath / "MultiAgentFees.xml");
        simulation = std::make_unique<Simulation>();
        simulation->setDebug(false);
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        ASSERT_NE(exchange, nullptr);
        book = exchange->books().at(kBookId);
    }

    [[nodiscard]] BookId missingBook() const { return static_cast<BookId>(exchange->books().size()); }

    // A validator instruction as the engine receives it.
    [[nodiscard]] Message::Ptr instruction(std::string type, MessagePayload::Ptr payload) const
    {
        return Message::create(
            simulation->currentTimestamp(),
            simulation->currentTimestamp(),
            "DISTRIBUTED_PROXY_AGENT",
            "EXCHANGE",
            std::move(type),
            std::move(payload));
    }

    [[nodiscard]] Message::Ptr instruction(std::string type, AgentId agentId, MessagePayload::Ptr payload) const
    {
        return instruction(
            std::move(type), MessagePayload::create<DistributedAgentResponsePayload>(agentId, std::move(payload)));
    }

    void send(std::string type, AgentId agentId, MessagePayload::Ptr payload)
    {
        exchange->receiveMessage(instruction(std::move(type), agentId, std::move(payload)));
    }

    [[nodiscard]] static MessagePayload::Ptr limit(
        OrderDirection direction, taosim::decimal_t volume, taosim::decimal_t price, BookId bookId = kBookId)
    {
        return MessagePayload::create<PlaceOrderLimitPayload>(direction, volume, price, bookId);
    }

    [[nodiscard]] static MessagePayload::Ptr market(
        OrderDirection direction, taosim::decimal_t volume, BookId bookId = kBookId)
    {
        return MessagePayload::create<PlaceOrderMarketPayload>(direction, volume, bookId);
    }

    // The answers the exchange has queued for the proxy, in queue order.
    [[nodiscard]] std::vector<Message::Ptr> answers() const
    {
        std::vector<Message::Ptr> res;
        for (const auto& entry : simulation->messageQueue().queue().underlying()) {
            const auto& msg = entry.pmsg.msg;
            if (ranges::find(msg->targets, std::string{"DISTRIBUTED_PROXY_AGENT"}) != msg->targets.end()) {
                res.push_back(msg);
            }
        }
        return res;
    }

    [[nodiscard]] int answerCount(std::string_view type) const
    {
        return static_cast<int>(ranges::count_if(answers(), [&](const auto& msg) { return msg->type == type; }));
    }

    // The inner payload of the one answer of `type`, as T.
    template<typename T>
    [[nodiscard]] std::shared_ptr<T> answer(std::string_view type) const
    {
        const auto all = answers();
        const auto it = ranges::find_if(all, [&](const auto& msg) { return msg->type == type; });
        if (it == all.end()) return nullptr;
        const auto wrapped = std::dynamic_pointer_cast<DistributedAgentResponsePayload>((*it)->payload);
        return wrapped ? std::dynamic_pointer_cast<T>(wrapped->payload) : nullptr;
    }

    [[nodiscard]] taosim::decimal_t restingVolume(OrderDirection side) const
    {
        taosim::decimal_t total{};
        for (const auto& level : side == OrderDirection::BUY ? book->buyQueue() : book->sellQueue()) {
            total += level.volume();
        }
        return total;
    }

    OrderID restLimit(AgentId agentId, OrderDirection direction, taosim::decimal_t volume, taosim::decimal_t price)
    {
        send("DISTRIBUTED_PLACE_ORDER_LIMIT", agentId, limit(direction, volume, price));
        const auto ack = answer<PlaceOrderLimitResponsePayload>("RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT");
        EXPECT_NE(ack, nullptr) << "the resting order was not placed";
        const auto id = ack ? ack->id : OrderID{};
        simulation->messageQueue() = taosim::message::MessageQueue{};
        return id;
    }

    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange{};
    taosim::book::Book::Ptr book;
};

}  // namespace

//-------------------------------------------------------------------------
// Every handled type still reaches its handler.

TEST_F(DistributedInboundCheckTest, ALimitOrderIsPlacedAndAcknowledged)
{
    send("DISTRIBUTED_PLACE_ORDER_LIMIT", kTrader, limit(OrderDirection::BUY, 1_dec, 100_dec));

    EXPECT_EQ(answerCount("RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT"), 1);
    EXPECT_EQ(restingVolume(OrderDirection::BUY), 1_dec);
}

TEST_F(DistributedInboundCheckTest, AMarketOrderIsFilledAndAcknowledged)
{
    restLimit(kCounterparty, OrderDirection::SELL, 10_dec, 300_dec);

    send("DISTRIBUTED_PLACE_ORDER_MARKET", kTrader, market(OrderDirection::BUY, 1_dec));

    EXPECT_EQ(answerCount("RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET"), 1);
    EXPECT_EQ(restingVolume(OrderDirection::SELL), 9_dec);
}

TEST_F(DistributedInboundCheckTest, AMarginCallIsHandledAsAMarketOrder)
{
    restLimit(kCounterparty, OrderDirection::SELL, 10_dec, 300_dec);

    send("DISTRIBUTED_PLACE_ORDER_MARKET_MC", kTrader, market(OrderDirection::BUY, 1_dec));

    EXPECT_EQ(answerCount("RESPONSE_DISTRIBUTED_PLACE_MARKET_ORDER"), 1);
    EXPECT_EQ(restingVolume(OrderDirection::SELL), 9_dec);
}

TEST_F(DistributedInboundCheckTest, ACancelCancelsTheAgentsOrder)
{
    const auto orderId = restLimit(kTrader, OrderDirection::BUY, 1_dec, 100_dec);

    send(
        "DISTRIBUTED_CANCEL_ORDERS",
        kTrader,
        MessagePayload::create<CancelOrdersPayload>(std::vector{taosim::event::Cancellation{orderId}}, kBookId));

    EXPECT_EQ(answerCount("RESPONSE_DISTRIBUTED_CANCEL_ORDERS"), 1);
    EXPECT_EQ(restingVolume(OrderDirection::BUY), 0_dec);
}

TEST_F(DistributedInboundCheckTest, ARetrieveReturnsTheOrder)
{
    const auto orderId = restLimit(kTrader, OrderDirection::BUY, 1_dec, 100_dec);

    send(
        "DISTRIBUTED_RETRIEVE_ORDERS",
        kTrader,
        MessagePayload::create<RetrieveOrdersPayload>(std::vector{orderId}, kBookId));

    const auto res = answer<RetrieveOrdersResponsePayload>("RESPONSE_DISTRIBUTED_RETRIEVE_ORDERS");
    ASSERT_NE(res, nullptr);
    ASSERT_EQ(res->orders.size(), 1uz);
    EXPECT_EQ(res->orders.front().id(), orderId);
}

TEST_F(DistributedInboundCheckTest, AClosePositionsClosesTheAgentsLoan)
{
    restLimit(kCounterparty, OrderDirection::SELL, 10_dec, 300_dec);
    send(
        "DISTRIBUTED_PLACE_ORDER_MARKET",
        kTrader,
        MessagePayload::create<PlaceOrderMarketPayload>(OrderDirection::BUY, 2_dec, 1_dec, kBookId));
    const auto ack = answer<PlaceOrderMarketResponsePayload>("RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET");
    ASSERT_NE(ack, nullptr);
    const auto loanId = ack->id;
    ASSERT_TRUE(exchange->accounts().at(kTrader).at(kBookId).getLoan(loanId).has_value());
    simulation->messageQueue() = taosim::message::MessageQueue{};
    restLimit(kCounterparty, OrderDirection::BUY, 10_dec, 290_dec);

    send(
        "DISTRIBUTED_CLOSE_POSITIONS",
        kTrader,
        MessagePayload::create<ClosePositionsPayload>(ClosePosition{loanId}, kBookId));

    EXPECT_EQ(answerCount("RESPONSE_DISTRIBUTED_CLOSE_POSITIONS"), 1);
}

TEST_F(DistributedInboundCheckTest, AResetResetsTheListedAgents)
{
    restLimit(kTrader, OrderDirection::BUY, 1_dec, 100_dec);

    send(
        "DISTRIBUTED_RESET_AGENT",
        kTrader,
        MessagePayload::create<ResetAgentsPayload>(std::vector{kTrader}));

    EXPECT_EQ(answerCount("RESPONSE_DISTRIBUTED_RESET_AGENT"), 1);
    EXPECT_EQ(restingVolume(OrderDirection::BUY), 0_dec);
}

//-------------------------------------------------------------------------
// A book the exchange does not have.

TEST_F(DistributedInboundCheckTest, ALimitOrderOnABookThatDoesNotExistIsRefused)
{
    ASSERT_NO_THROW(send(
        "DISTRIBUTED_PLACE_ORDER_LIMIT", kTrader, limit(OrderDirection::BUY, 1_dec, 100_dec, missingBook())));

    const auto refusal = answer<PlaceOrderLimitErrorResponsePayload>("ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT");
    ASSERT_NE(refusal, nullptr);
    EXPECT_EQ(refusal->requestPayload->bookId, missingBook());
    EXPECT_EQ(refusal->errorPayload->message, "CONTRACT_VIOLATION");
    EXPECT_EQ(answers().size(), 1uz);
    EXPECT_EQ(restingVolume(OrderDirection::BUY), 0_dec);
}

TEST_F(DistributedInboundCheckTest, AMarketOrderOnABookThatDoesNotExistIsRefused)
{
    restLimit(kCounterparty, OrderDirection::SELL, 10_dec, 300_dec);

    ASSERT_NO_THROW(send(
        "DISTRIBUTED_PLACE_ORDER_MARKET", kTrader, market(OrderDirection::BUY, 1_dec, missingBook())));

    const auto refusal =
        answer<PlaceOrderMarketErrorResponsePayload>("ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET");
    ASSERT_NE(refusal, nullptr);
    EXPECT_EQ(refusal->errorPayload->message, "CONTRACT_VIOLATION");
    EXPECT_EQ(answers().size(), 1uz);
    EXPECT_EQ(restingVolume(OrderDirection::SELL), 10_dec);
}

TEST_F(DistributedInboundCheckTest, ARetrieveOnABookThatDoesNotExistIsRefused)
{
    ASSERT_NO_THROW(send(
        "DISTRIBUTED_RETRIEVE_ORDERS",
        kTrader,
        MessagePayload::create<RetrieveOrdersPayload>(std::vector{OrderID{}}, missingBook())));

    const auto refusal = answer<ErrorResponsePayload>("ERROR_RESPONSE_DISTRIBUTED_RETRIEVE_ORDERS");
    ASSERT_NE(refusal, nullptr);
    EXPECT_EQ(refusal->message, "CONTRACT_VIOLATION");
    EXPECT_EQ(answers().size(), 1uz);
}

TEST_F(DistributedInboundCheckTest, AClosePositionsOnABookThatDoesNotExistIsRefused)
{
    ASSERT_NO_THROW(send(
        "DISTRIBUTED_CLOSE_POSITIONS",
        kTrader,
        MessagePayload::create<ClosePositionsPayload>(ClosePosition{OrderID{3}}, missingBook())));

    const auto refusal =
        answer<ClosePositionsErrorResponsePayload>("ERROR_RESPONSE_DISTRIBUTED_CLOSE_POSITIONS");
    ASSERT_NE(refusal, nullptr);
    EXPECT_EQ(refusal->orderIds, std::vector{OrderID{3}});
    EXPECT_EQ(refusal->errorPayload->message, "CONTRACT_VIOLATION");
    EXPECT_EQ(answers().size(), 1uz);
}

TEST_F(DistributedInboundCheckTest, ACancelOnABookThatDoesNotExistIsRefusedAsBefore)
{
    ASSERT_NO_THROW(send(
        "DISTRIBUTED_CANCEL_ORDERS",
        kTrader,
        MessagePayload::create<CancelOrdersPayload>(
            std::vector{taosim::event::Cancellation{OrderID{3}}}, missingBook())));

    const auto refusal = answer<CancelOrdersErrorResponsePayload>("ERROR_RESPONSE_DISTRIBUTED_CANCEL_ORDERS");
    ASSERT_NE(refusal, nullptr);
    EXPECT_EQ(refusal->errorPayload->message, "Order IDs 3 do not exist.");
}

//-------------------------------------------------------------------------
// Exact types, and the payload each carries.

TEST_F(DistributedInboundCheckTest, ATypeThatOnlyContainsAHandledNameIsNotHandledAsIt)
{
    restLimit(kCounterparty, OrderDirection::SELL, 10_dec, 300_dec);

    for (const std::string type : {
             "DISTRIBUTED_RESPONSE_PLACE_ORDER_MARKET",
             "DISTRIBUTED_ERROR_RESPONSE_PLACE_ORDER_LIMIT",
             "DISTRIBUTED_RESPONSE_CANCEL_ORDERS"}) {
        const auto payload = type.contains("LIMIT") ? limit(OrderDirection::BUY, 1_dec, 300_dec)
            : type.contains("CANCEL")
                ? MessagePayload::create<CancelOrdersPayload>(
                    std::vector{taosim::event::Cancellation{OrderID{}}}, kBookId)
                : market(OrderDirection::BUY, 1_dec);

        ASSERT_NO_THROW(send(type, kTrader, payload)) << type;

        const auto refusal = answer<ErrorResponsePayload>("ERROR_RESPONSE_" + type);
        ASSERT_NE(refusal, nullptr) << type << " was not answered as an unhandled type";
        EXPECT_EQ(refusal->message, "Unknown message type: " + type);
        EXPECT_EQ(answers().size(), 1uz) << type;
        EXPECT_EQ(restingVolume(OrderDirection::SELL), 10_dec) << type << " was handled as an order";
        simulation->messageQueue() = taosim::message::MessageQueue{};
    }
}

TEST_F(DistributedInboundCheckTest, APayloadOtherThanTheOneItsTypeCarriesIsDropped)
{
    const auto cancel = MessagePayload::create<CancelOrdersPayload>(
        std::vector{taosim::event::Cancellation{OrderID{}}}, kBookId);
    const auto close = MessagePayload::create<ClosePositionsPayload>(ClosePosition{OrderID{}}, kBookId);

    const std::vector<std::pair<std::string, MessagePayload::Ptr>> mismatches{
        {"DISTRIBUTED_PLACE_ORDER_LIMIT", cancel},
        {"DISTRIBUTED_PLACE_ORDER_MARKET", limit(OrderDirection::BUY, 1_dec, 100_dec)},
        {"DISTRIBUTED_PLACE_ORDER_MARKET_MC", close},
        {"DISTRIBUTED_CANCEL_ORDERS", market(OrderDirection::BUY, 1_dec)},
        {"DISTRIBUTED_CLOSE_POSITIONS", cancel},
        {"DISTRIBUTED_RETRIEVE_ORDERS", close},
        {"DISTRIBUTED_RESET_AGENT", cancel},
        {"DISTRIBUTED_PLACE_ORDER_LIMIT", nullptr},
    };
    for (const auto& [type, payload] : mismatches) {
        ASSERT_NO_THROW(send(type, kTrader, payload)) << type;
        EXPECT_TRUE(answers().empty()) << type;
        EXPECT_TRUE(simulation->messageQueue().empty()) << type;
    }
    EXPECT_EQ(restingVolume(OrderDirection::BUY), 0_dec);
}

TEST_F(DistributedInboundCheckTest, AMessageWithoutAnAgentPayloadIsDropped)
{
    for (const std::string type : {"DISTRIBUTED_CANCEL_ORDERS", "DISTRIBUTED_PLACE_ORDER_LIMIT", "DISTRIBUTED_UNHANDLED"}) {
        ASSERT_NO_THROW(exchange->receiveMessage(instruction(type, limit(OrderDirection::BUY, 1_dec, 100_dec))))
            << type;
        EXPECT_TRUE(simulation->messageQueue().empty()) << type;
    }
    EXPECT_EQ(restingVolume(OrderDirection::BUY), 0_dec);
}

//-------------------------------------------------------------------------
// The agent's account.

TEST_F(DistributedInboundCheckTest, AnOrderFromAnAgentWithoutAnAccountIsRefusedAndOpensNone)
{
    restLimit(kCounterparty, OrderDirection::SELL, 10_dec, 300_dec);

    ASSERT_NO_THROW(send("DISTRIBUTED_PLACE_ORDER_LIMIT", kUnregistered, limit(OrderDirection::BUY, 1_dec, 100_dec)));
    ASSERT_NO_THROW(send("DISTRIBUTED_PLACE_ORDER_MARKET", kUnregistered, market(OrderDirection::BUY, 1_dec)));

    const auto limitRefusal =
        answer<PlaceOrderLimitErrorResponsePayload>("ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT");
    ASSERT_NE(limitRefusal, nullptr);
    EXPECT_EQ(limitRefusal->errorPayload->message, "NONEXISTENT_ACCOUNT");
    const auto marketRefusal =
        answer<PlaceOrderMarketErrorResponsePayload>("ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET");
    ASSERT_NE(marketRefusal, nullptr);
    EXPECT_EQ(marketRefusal->errorPayload->message, "NONEXISTENT_ACCOUNT");
    EXPECT_EQ(answers().size(), 2uz);
    EXPECT_FALSE(exchange->accounts().contains(kUnregistered)) << "an account was opened for the agent";
    EXPECT_EQ(restingVolume(OrderDirection::BUY), 0_dec);
    EXPECT_EQ(restingVolume(OrderDirection::SELL), 10_dec);
}

TEST_F(DistributedInboundCheckTest, AClosePositionsFromAnAgentWithoutAnAccountIsRefused)
{
    ASSERT_NO_THROW(send(
        "DISTRIBUTED_CLOSE_POSITIONS",
        kUnregistered,
        MessagePayload::create<ClosePositionsPayload>(ClosePosition{OrderID{3}}, kBookId)));

    const auto refusal =
        answer<ClosePositionsErrorResponsePayload>("ERROR_RESPONSE_DISTRIBUTED_CLOSE_POSITIONS");
    ASSERT_NE(refusal, nullptr);
    EXPECT_EQ(refusal->errorPayload->message, "NONEXISTENT_ACCOUNT");
    EXPECT_FALSE(exchange->accounts().contains(kUnregistered));
}

TEST_F(DistributedInboundCheckTest, AnAccountThatDoesNotHoldTheBookIsRefused)
{
    constexpr AgentId kNoHoldings{5};
    ASSERT_TRUE(exchange->accounts().registerRemote(kNoHoldings, taosim::accounting::Account::Holdings{}));

    ASSERT_NO_THROW(send("DISTRIBUTED_PLACE_ORDER_LIMIT", kNoHoldings, limit(OrderDirection::BUY, 1_dec, 100_dec)));
    ASSERT_NO_THROW(send(
        "DISTRIBUTED_CLOSE_POSITIONS",
        kNoHoldings,
        MessagePayload::create<ClosePositionsPayload>(ClosePosition{OrderID{3}}, kBookId)));

    const auto refusal =
        answer<PlaceOrderLimitErrorResponsePayload>("ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT");
    ASSERT_NE(refusal, nullptr);
    EXPECT_EQ(refusal->errorPayload->message, "NONEXISTENT_ACCOUNT");
    const auto closeRefusal =
        answer<ClosePositionsErrorResponsePayload>("ERROR_RESPONSE_DISTRIBUTED_CLOSE_POSITIONS");
    ASSERT_NE(closeRefusal, nullptr);
    EXPECT_EQ(closeRefusal->errorPayload->message, "NONEXISTENT_ACCOUNT");
    EXPECT_EQ(restingVolume(OrderDirection::BUY), 0_dec);
}

//-------------------------------------------------------------------------
// The step goes on.

TEST_F(DistributedInboundCheckTest, ARefusedOrDroppedMessageLeavesTheRestOfTheStepToRun)
{
    simulation->queueMessage(instruction(
        "DISTRIBUTED_PLACE_ORDER_LIMIT", kTrader, limit(OrderDirection::BUY, 1_dec, 100_dec, missingBook())));
    simulation->queueMessage(instruction(
        "DISTRIBUTED_PLACE_ORDER_LIMIT",
        kTrader,
        MessagePayload::create<CancelOrdersPayload>(std::vector{taosim::event::Cancellation{OrderID{}}}, kBookId)));
    simulation->queueMessage(instruction(
        "DISTRIBUTED_PLACE_ORDER_MARKET", kUnregistered, market(OrderDirection::BUY, 1_dec)));
    simulation->queueMessage(instruction(
        "DISTRIBUTED_PLACE_ORDER_LIMIT", kTrader, limit(OrderDirection::BUY, 1_dec, 100_dec)));

    ASSERT_NO_THROW(simulation->step());

    std::vector<std::string> delivered;
    for (const auto& msg : simulation->proxy()->messages()) delivered.push_back(msg->type);
    EXPECT_THAT(
        delivered,
        testing::UnorderedElementsAre(
            "ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT",
            "ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET",
            "RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT"));
    EXPECT_EQ(restingVolume(OrderDirection::BUY), 1_dec);
}

//-------------------------------------------------------------------------
