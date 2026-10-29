/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

// The ALGO trader under the exchange's size floor. It sends only market orders and the exchange
// refuses any below minOrderSize, so a parent it cannot send at or above the floor is finished, not
// left EXECUTING: while it is, both the wakeup and the immediate path skip the book, and the agent
// trades nothing there for the rest of the run.
//
// Each test plants a parent in one of the states that can never be sent and delivers the message
// that would keep it going. Driven through the agent's own handlers on a configured Simulation that
// never runs, so what the agent decided is read straight off the message queue.

#include <taosim/agent/ALGOTraderAgent.hpp>

#include "MultiBookExchangeAgent.hpp"
#include "Simulation.hpp"
#include "util.hpp"
#include <taosim/message/ExchangeAgentMessagePayloads.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <range/v3/algorithm/count_if.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

//-------------------------------------------------------------------------

using namespace taosim;
using namespace taosim::literals;
using taosim::agent::ALGOTraderAgent;
using taosim::agent::ALGOTraderStatus;

using namespace testing;

namespace fs = std::filesystem;

//-------------------------------------------------------------------------

namespace
{

const auto kTestDataPath = fs::path{__FILE__}.parent_path().parent_path() / "data";

constexpr BookId kBookId{};

class ALGOTraderFloorTest : public Test
{
protected:
    void SetUp() override
    {
        nodes = taosim::util::parseSimulationFile(kTestDataPath / "ALGOFloor.xml");
        simulation = std::make_unique<Simulation>();
        simulation->setDebug(false);
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        ASSERT_NE(exchange, nullptr);
        // As the exchange holds it: 2.5364 read through a double, a hair below.
        sizeFloor = exchange->config2().minOrderSize;
        ASSERT_NEAR(taosim::util::decimal2double(sizeFloor), 2.5364, 1e-6);
        for (const auto& localAgent : simulation->agents()) {
            if (auto* a = dynamic_cast<ALGOTraderAgent*>(localAgent.get())) algo = a;
        }
        ASSERT_NE(algo, nullptr);
        // A BUY is funded as free quote over the last trade price, which only a trade sets.
        algo->lastPrice().at(kBookId) = DEC(29.57);
        auto& queue = simulation->messageQueue();
        while (!queue.empty()) queue.pop();
    }

    void plant(OrderDirection direction, decimal_t remaining)
    {
        auto& state = algo->state().at(kBookId);
        state.status = ALGOTraderStatus::EXECUTING;
        state.direction = direction;
        state.volumeToBeExecuted = remaining;
    }

    void deliver(const std::string& type, MessagePayload::Ptr payload)
    {
        const auto now = simulation->currentTimestamp();
        algo->receiveMessage(::Message::create(now, now, exchange->name(), algo->name(), type, payload));
    }

    void deliverExecuteTick() { deliver("WAKEUP_ALGOTRADER_EXECUTE", MessagePayload::create<WakeupPayload>(kBookId)); }

    [[nodiscard]] static PlaceOrderMarketPayload::Ptr sent(OrderDirection direction, decimal_t volume)
    {
        return MessagePayload::create<PlaceOrderMarketPayload>(direction, volume, kBookId);
    }

    [[nodiscard]] std::vector<decimal_t> queuedMarketOrders() const
    {
        std::vector<decimal_t> volumes;
        for (const auto& entry : simulation->messageQueue().queue().underlying()) {
            if (entry.pmsg.msg->type != "PLACE_ORDER_MARKET") continue;
            volumes.push_back(
                std::static_pointer_cast<PlaceOrderMarketPayload>(entry.pmsg.msg->payload)->volume);
        }
        return volumes;
    }

    [[nodiscard]] auto queuedExecuteTicks() const
    {
        return ranges::count_if(
            simulation->messageQueue().queue().underlying(),
            [](const auto& entry) { return entry.pmsg.msg->type == "WAKEUP_ALGOTRADER_EXECUTE"; });
    }

    [[nodiscard]] const auto& state() const { return algo->state().at(kBookId); }

    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange{};
    ALGOTraderAgent* algo{};
    decimal_t sizeFloor{};
};

}  // namespace

//-------------------------------------------------------------------------

// A partial fill left 1.5 to go, above the one-unit retirement size and below the floor.
TEST_F(ALGOTraderFloorTest, ARemainderBelowTheFloorIsFinishedNotSent)
{
    plant(OrderDirection::SELL, DEC(1.5));
    deliverExecuteTick();

    EXPECT_THAT(queuedMarketOrders(), IsEmpty())
        << "the agent sent an order below the 2.5364 floor on a 1.5 remainder, which the exchange refuses";
    EXPECT_EQ(state().status, ALGOTraderStatus::ASLEEP);
}

//-------------------------------------------------------------------------

// The immediate path sizes its parent off the touch, and a level whose orders have all filled
// reads as zero until the step clears it.
TEST_F(ALGOTraderFloorTest, AParentOfNothingIsFinishedNotRearmed)
{
    plant(OrderDirection::BUY, 0_dec);
    deliverExecuteTick();

    EXPECT_EQ(queuedExecuteTicks(), 0)
        << "the agent re-armed its execute tick on a parent with nothing in it, which it would "
           "do again on every tick";
    EXPECT_EQ(state().status, ALGOTraderStatus::ASLEEP);
}

//-------------------------------------------------------------------------

TEST_F(ALGOTraderFloorTest, AFillLeavingLessThanTheFloorFinishesTheParent)
{
    plant(OrderDirection::SELL, DEC(4.0));
    deliver(
        "RESPONSE_PLACE_ORDER_MARKET",
        MessagePayload::create<PlaceOrderMarketResponsePayload>(
            OrderID{1}, sent(OrderDirection::SELL, DEC(2.6))));

    EXPECT_EQ(state().status, ALGOTraderStatus::ASLEEP)
        << "1.4 is left, which can never be sent under the floor";
    EXPECT_EQ(queuedExecuteTicks(), 0);
}

//-------------------------------------------------------------------------

TEST_F(ALGOTraderFloorTest, ARefusedOrderIsNotResent)
{
    plant(OrderDirection::SELL, 10_dec);
    deliver(
        "ERROR_RESPONSE_PLACE_ORDER_MARKET",
        MessagePayload::create<PlaceOrderMarketErrorResponsePayload>(
            sent(OrderDirection::SELL, DEC(3.0)),
            MessagePayload::create<ErrorResponsePayload>("refused")));

    EXPECT_THAT(queuedMarketOrders(), IsEmpty())
        << "the agent resent at once with nothing changed, so a refusal that holds repeats "
           "for the rest of the run";
    EXPECT_EQ(state().status, ALGOTraderStatus::ASLEEP);
}

//-------------------------------------------------------------------------

// The fixture's slice draw is about 0.05.
TEST_F(ALGOTraderFloorTest, ASliceBelowTheFloorIsRaisedToIt)
{
    plant(OrderDirection::SELL, 10_dec);
    deliverExecuteTick();

    EXPECT_THAT(queuedMarketOrders(), ElementsAre(sizeFloor));
    EXPECT_EQ(state().status, ALGOTraderStatus::EXECUTING);
}

//-------------------------------------------------------------------------
