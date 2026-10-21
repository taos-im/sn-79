/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

// A stop-loss or take-profit close names why it traded and which position order it closes. The tags ride
// on the market order payload, and the trade it makes carries them whichever agent placed it: a local
// agent's close is tagged exactly as a distributed agent's is.

#include "MultiBookExchangeAgent.hpp"
#include "Simulation.hpp"
#include "util.hpp"
#include <taosim/event/L3RecordContainer.hpp>
#include <taosim/matching/ClearingManager.hpp>
#include <taosim/message/MultiBookMessagePayloads.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <variant>

//-------------------------------------------------------------------------

using namespace taosim;
using namespace taosim::matching;
using namespace taosim::literals;

using testing::Test;

namespace fs = std::filesystem;

//-------------------------------------------------------------------------

namespace
{

const auto kTestDataPath = fs::path{__FILE__}.parent_path().parent_path() / "data";

constexpr BookId kBookId{};
constexpr uint8_t kStopLoss = 1;
constexpr OrderID kPositionOrderId = 4242;

class MarketOrderCloseTagTest : public Test
{
protected:
    void SetUp() override
    {
        nodes = taosim::util::parseSimulationFile(kTestDataPath / "MultiAgentFees.xml");
        simulation = std::make_unique<Simulation>();
        simulation->setDebug(false);
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        book = exchange->books()[kBookId];
        exchange->accounts().registerLocal("agent1");
        exchange->accounts().registerLocal("agent2");
        const AgentId maker = exchange->accounts().getAgentId(LocalAgentId{"agent2"});

        const auto ask = MessagePayload::create<PlaceOrderLimitPayload>(
            OrderDirection::SELL, 5_dec, 300_dec, 0_dec, kBookId, Currency::BASE);
        const auto result = exchange->clearingManager().handleOrder(
            LimitOrderDesc{.agentId = maker, .payload = ask});
        ASSERT_EQ(result.ec, OrderErrorCode::VALID);
        [[maybe_unused]] const auto order = book->placeLimitOrder(
            OrderClientContext{maker}, simulation->currentTimestamp(), result.orderSize, OrderDirection::SELL,
            300_dec, 0_dec, STPFlag::CO);
    }

    [[nodiscard]] static PlaceOrderMarketPayload::Ptr stopLossClose()
    {
        auto payload = MessagePayload::create<PlaceOrderMarketPayload>(OrderDirection::BUY, 1_dec, kBookId);
        payload->closeReason = kStopLoss;
        payload->originatingOrderId = kPositionOrderId;
        return payload;
    }

    [[nodiscard]] std::optional<TradeContext> onlyTradeContext() const
    {
        std::optional<TradeContext> found;
        for (const auto& entry : exchange->L3Record().at(kBookId)) {
            if (const auto* trade = std::get_if<taosim::event::TradeEvent>(&entry)) {
                EXPECT_FALSE(found.has_value()) << "the close made more than one trade";
                found = trade->ctx;
            }
        }
        return found;
    }

    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange{};
    taosim::book::Book::Ptr book;
};

}  // namespace

//-------------------------------------------------------------------------

TEST_F(MarketOrderCloseTagTest, ALocalAgentsCloseTradesTagged)
{
    exchange->receiveMessage(Message::create(
        simulation->currentTimestamp(), simulation->currentTimestamp(), "agent1", "EXCHANGE",
        "PLACE_ORDER_MARKET", stopLossClose()));

    const auto ctx = onlyTradeContext();
    ASSERT_TRUE(ctx.has_value()) << "the close did not trade";
    EXPECT_EQ(ctx->aggressingCloseReason, kStopLoss);
    EXPECT_EQ(ctx->aggressingOriginatingOrderId, kPositionOrderId);
}

//-------------------------------------------------------------------------

TEST_F(MarketOrderCloseTagTest, ADistributedAgentsCloseTradesTagged)
{
    exchange->receiveMessage(Message::create(
        simulation->currentTimestamp(), simulation->currentTimestamp(), "DISTRIBUTED_PROXY_AGENT", "EXCHANGE",
        "DISTRIBUTED_PLACE_ORDER_MARKET",
        MessagePayload::create<DistributedAgentResponsePayload>(AgentId{0}, stopLossClose())));

    const auto ctx = onlyTradeContext();
    ASSERT_TRUE(ctx.has_value()) << "the close did not trade";
    EXPECT_EQ(ctx->aggressingCloseReason, kStopLoss);
    EXPECT_EQ(ctx->aggressingOriginatingOrderId, kPositionOrderId);
}

//-------------------------------------------------------------------------
