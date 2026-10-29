/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

// The validator's volume cap, enforced when an instruction executes. With each reply batch the validator sends
// the (agent, book) pairs at or over its cap, and the exchange holds the latest set. A placement from a pair in
// that set is refused when it reaches the exchange, which for a delayed instruction is when its delay ends, so an
// instruction accepted under the cap cannot trade once the pair is capped. Cancels, closes and retrieves pass, as
// do the exchange's own orders on a miner's account (margin calls, stop-loss and take-profit closes). Each new set
// replaces the last.

#include "MultiBookExchangeAgent.hpp"
#include "Simulation.hpp"
#include "util.hpp"
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

class VolumeCapAtExecutionTest : public Test
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

    void send(std::string type, AgentId agentId, MessagePayload::Ptr payload)
    {
        exchange->receiveMessage(Message::create(
            simulation->currentTimestamp(),
            simulation->currentTimestamp(),
            "DISTRIBUTED_PROXY_AGENT",
            "EXCHANGE",
            std::move(type),
            MessagePayload::create<DistributedAgentResponsePayload>(agentId, std::move(payload))));
    }

    [[nodiscard]] static MessagePayload::Ptr limit(
        OrderDirection direction, taosim::decimal_t volume, taosim::decimal_t price)
    {
        return MessagePayload::create<PlaceOrderLimitPayload>(direction, volume, price, kBookId);
    }

    [[nodiscard]] static PlaceOrderMarketPayload::Ptr market(OrderDirection direction, taosim::decimal_t volume)
    {
        return MessagePayload::create<PlaceOrderMarketPayload>(direction, volume, kBookId);
    }

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

    void cap(std::vector<std::pair<AgentId, BookId>> pairs) { exchange->setCappedPairs(std::move(pairs)); }

    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange{};
    taosim::book::Book::Ptr book;
};

}  // namespace

//-------------------------------------------------------------------------

TEST_F(VolumeCapAtExecutionTest, ACappedPairsLimitOrderIsRefusedWhenItArrives)
{
    cap({{kTrader, kBookId}});
    send("DISTRIBUTED_PLACE_ORDER_LIMIT", kTrader, limit(OrderDirection::BUY, 1_dec, 100_dec));

    const auto refusal = answer<PlaceOrderLimitErrorResponsePayload>("ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT");
    ASSERT_NE(refusal, nullptr);
    EXPECT_EQ(refusal->errorPayload->message, "VOLUME_CAP_REACHED");
    EXPECT_EQ(answers().size(), 1uz);
    EXPECT_EQ(restingVolume(OrderDirection::BUY), 0_dec);
}

TEST_F(VolumeCapAtExecutionTest, ACappedPairsMarketOrderIsRefusedWhenItArrives)
{
    restLimit(kCounterparty, OrderDirection::SELL, 10_dec, 300_dec);
    cap({{kTrader, kBookId}});
    send("DISTRIBUTED_PLACE_ORDER_MARKET", kTrader, market(OrderDirection::BUY, 1_dec));

    const auto refusal =
        answer<PlaceOrderMarketErrorResponsePayload>("ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET");
    ASSERT_NE(refusal, nullptr);
    EXPECT_EQ(refusal->errorPayload->message, "VOLUME_CAP_REACHED");
    EXPECT_EQ(restingVolume(OrderDirection::SELL), 10_dec) << "the refused order traded";
}

TEST_F(VolumeCapAtExecutionTest, OnlyTheCappedPairIsRefused)
{
    cap({{kTrader, kBookId}});
    send("DISTRIBUTED_PLACE_ORDER_LIMIT", kCounterparty, limit(OrderDirection::BUY, 1_dec, 100_dec));

    EXPECT_EQ(answerCount("RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT"), 1);
    EXPECT_EQ(restingVolume(OrderDirection::BUY), 1_dec);
}

TEST_F(VolumeCapAtExecutionTest, ACappedPairStillCancels)
{
    const auto orderId = restLimit(kTrader, OrderDirection::BUY, 1_dec, 100_dec);
    cap({{kTrader, kBookId}});
    send(
        "DISTRIBUTED_CANCEL_ORDERS",
        kTrader,
        MessagePayload::create<CancelOrdersPayload>(std::vector{taosim::event::Cancellation{orderId}}, kBookId));

    EXPECT_EQ(answerCount("RESPONSE_DISTRIBUTED_CANCEL_ORDERS"), 1);
    EXPECT_EQ(restingVolume(OrderDirection::BUY), 0_dec);
}

// A stop-loss or take-profit close is the exchange acting on a position, not a new instruction.
TEST_F(VolumeCapAtExecutionTest, AStopLossCloseIsNotRefusedForTheCap)
{
    restLimit(kCounterparty, OrderDirection::SELL, 10_dec, 300_dec);
    cap({{kTrader, kBookId}});
    auto close = market(OrderDirection::BUY, 1_dec);
    close->closeReason = 1;
    send("DISTRIBUTED_PLACE_ORDER_MARKET", kTrader, close);

    const auto refusal =
        answer<PlaceOrderMarketErrorResponsePayload>("ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET");
    EXPECT_TRUE(refusal == nullptr || refusal->errorPayload->message != "VOLUME_CAP_REACHED");
}

TEST_F(VolumeCapAtExecutionTest, ANewSetReplacesTheLast)
{
    cap({{kTrader, kBookId}});
    cap({});
    send("DISTRIBUTED_PLACE_ORDER_LIMIT", kTrader, limit(OrderDirection::BUY, 1_dec, 100_dec));

    EXPECT_EQ(answerCount("RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT"), 1);
    EXPECT_EQ(restingVolume(OrderDirection::BUY), 1_dec);
}

//-------------------------------------------------------------------------
