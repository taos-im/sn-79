/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

#include "MultiBookExchangeAgent.hpp"
#include "Simulation.hpp"
#include "util.hpp"
#include <taosim/matching/ClearingManager.hpp>
#include <taosim/statshub/StatsHub.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <utility>

//-------------------------------------------------------------------------

using namespace taosim::literals;
using taosim::matching::LimitOrderDesc;

namespace fs = std::filesystem;

namespace
{

const auto kTestDataPath = fs::path{__FILE__}.parent_path() / "data";

void placeLimit(
    MultiBookExchangeAgent* exchange,
    AgentId agentId,
    BookId bookId,
    OrderDirection direction,
    taosim::decimal_t volume,
    taosim::decimal_t price)
{
    const auto payload = MessagePayload::create<PlaceOrderLimitPayload>(
        direction, volume, price, DEC(0.), bookId, Currency::BASE, std::nullopt,
        false, taosim::TimeInForce::GTC, std::nullopt, STPFlag::CO);
    const auto orderResult = exchange->clearingManager().handleOrder(
        LimitOrderDesc{.agentId = agentId, .payload = payload});
    ASSERT_EQ(orderResult.ec, OrderErrorCode::VALID);
    [[maybe_unused]] const auto order = exchange->books()[bookId]->placeLimitOrder(
        OrderClientContext{agentId},
        Timestamp{},
        orderResult.orderSize,
        payload->direction,
        payload->price,
        payload->leverage,
        payload->stpFlag);
}

}  // namespace

//-------------------------------------------------------------------------

class StatsHubTest : public testing::Test
{
protected:
    void SetUp() override
    {
        nodes = taosim::util::parseSimulationFile(kTestDataPath / "MultiAgentFees.xml");
        simulation = std::make_unique<Simulation>();
        simulation->setDebug(false);
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        exchange->accounts().registerLocal("maker");
        exchange->accounts().registerLocal("taker");
        hub = exchange->statsHub().get();
        ASSERT_NE(hub, nullptr);
    }

    static constexpr AgentId kMaker = -1;
    static constexpr AgentId kTaker = -2;
    static constexpr BookId kBookId{};

    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange{};
    taosim::stats::StatsHub* hub{};
};

//-------------------------------------------------------------------------

TEST_F(StatsHubTest, EmptyBookYieldsZeroL1)
{
    const auto snapshot = hub->l1(kBookId);
    EXPECT_EQ(snapshot.bestBidPrice, 0_dec);
    EXPECT_EQ(snapshot.bestBidVolume, 0_dec);
    EXPECT_EQ(snapshot.bidTotalVolume, 0_dec);
    EXPECT_EQ(snapshot.bestAskPrice, 0_dec);
    EXPECT_EQ(snapshot.bestAskVolume, 0_dec);
    EXPECT_EQ(snapshot.askTotalVolume, 0_dec);
}

//-------------------------------------------------------------------------

TEST_F(StatsHubTest, L1ReadsThroughToTheLiveTopOfBook)
{
    placeLimit(exchange, kMaker, kBookId, OrderDirection::BUY, 3_dec, 291_dec);
    placeLimit(exchange, kMaker, kBookId, OrderDirection::BUY, 1_dec, 297_dec);
    placeLimit(exchange, kMaker, kBookId, OrderDirection::SELL, 2_dec, 303_dec);
    placeLimit(exchange, kMaker, kBookId, OrderDirection::SELL, 8_dec, 307_dec);

    const auto snapshot = hub->l1(kBookId);
    EXPECT_EQ(snapshot.bestBidPrice, 297_dec);
    EXPECT_EQ(snapshot.bestBidVolume, 1_dec);
    EXPECT_EQ(snapshot.bidTotalVolume, 4_dec);
    EXPECT_EQ(snapshot.bestAskPrice, 303_dec);
    EXPECT_EQ(snapshot.bestAskVolume, 2_dec);
    EXPECT_EQ(snapshot.askTotalVolume, 10_dec);
}

//-------------------------------------------------------------------------

// A resting order that fills completely stays on its level, at zero, until the step clears filled
// orders. The matcher skips such a level, and so must every agent reading the book through the hub:
// it is not a price anyone can trade at.
TEST_F(StatsHubTest, ALevelWhoseOrdersHaveAllFilledIsNotTheTouch)
{
    placeLimit(exchange, kMaker, kBookId, OrderDirection::BUY, 1_dec, 297_dec);
    placeLimit(exchange, kMaker, kBookId, OrderDirection::SELL, 2_dec, 303_dec);
    placeLimit(exchange, kMaker, kBookId, OrderDirection::SELL, 8_dec, 307_dec);
    placeLimit(exchange, kTaker, kBookId, OrderDirection::BUY, 2_dec, 303_dec);
    ASSERT_FALSE(exchange->books()[kBookId]->sellQueue().front().hasActiveOrders())
        << "the fixture needs the 303 level filled and not yet cleared";

    const auto l1 = hub->l1(kBookId);
    EXPECT_EQ(l1.bestAskPrice, 307_dec);
    EXPECT_EQ(l1.bestAskVolume, 8_dec);
    EXPECT_EQ(l1.bestBidPrice, 297_dec);

    const auto l2 = hub->l2(kBookId, 5);
    ASSERT_FALSE(l2.asks.empty());
    EXPECT_EQ(l2.asks.front().price, 307_dec);
    EXPECT_EQ(l2.asks.front().quantity, 8_dec);
    EXPECT_EQ(l2.asks.size(), 1u);
}

//-------------------------------------------------------------------------
