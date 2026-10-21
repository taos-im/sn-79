/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

// A spot margin call that cannot be filled stays called and is placed again on later steps until the loan
// settles: once a bid it can fill against arrives, the loan is settled and its margin-call entry goes with
// it. Its agent is told once that the call could not be placed, not on every retry.

#include "MultiBookExchangeAgent.hpp"
#include "Simulation.hpp"
#include "util.hpp"
#include <taosim/accounting/margin_utils.hpp>
#include <taosim/agent/DistributedProxyAgent.hpp>
#include <taosim/matching/ClearingManager.hpp>
#include <taosim/message/MultiBookMessagePayloads.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <pugixml.hpp>
#include <range/v3/algorithm/any_of.hpp>

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>

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

class MarginCallRetryTest : public Test
{
protected:
    void SetUp() override
    {
        nodes = taosim::util::parseSimulationFile(kTestDataPath / "MultiAgentFees.xml");
        adjustExchange(nodes.exchange);
        simulation = std::make_unique<Simulation>();
        simulation->setDebug(false);
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        book = exchange->books()[kBookId];
        for (const auto* name : {"agent1", "agent2", "agent3"}) {
            exchange->accounts().registerLocal(name);
        }
        borrower = exchange->accounts().getAgentId(LocalAgentId{"agent1"});
        bidder = exchange->accounts().getAgentId(LocalAgentId{"agent2"});
        seller = exchange->accounts().getAgentId(LocalAgentId{"agent3"});
    }

    virtual void adjustExchange(pugi::xml_node) {}

    void limit(AgentId agentId, OrderDirection direction, decimal_t volume, decimal_t price)
    {
        const auto payload = MessagePayload::create<PlaceOrderLimitPayload>(
            direction, volume, price, 0_dec, kBookId, Currency::BASE);
        const auto result = exchange->clearingManager().handleOrder(
            LimitOrderDesc{.agentId = agentId, .payload = payload});
        ASSERT_EQ(result.ec, OrderErrorCode::VALID);
        [[maybe_unused]] const auto order = book->placeLimitOrder(
            OrderClientContext{agentId}, simulation->currentTimestamp(), result.orderSize, direction, price,
            0_dec, STPFlag::CO);
    }

    [[nodiscard]] OrderID leveragedMarketBuy(AgentId agentId, decimal_t volume, decimal_t leverage)
    {
        const auto payload = MessagePayload::create<PlaceOrderMarketPayload>(
            OrderDirection::BUY, volume, leverage, kBookId, Currency::BASE);
        const auto result = exchange->clearingManager().handleOrder(
            MarketOrderDesc{.agentId = agentId, .payload = payload});
        EXPECT_EQ(result.ec, OrderErrorCode::VALID);
        return book->placeMarketOrder(
            OrderClientContext{agentId}, simulation->currentTimestamp(), result.orderSize,
            OrderDirection::BUY, leverage, STPFlag::CO)->id();
    }

    void fifoMarketSell(AgentId agentId, decimal_t volume)
    {
        const auto payload = MessagePayload::create<PlaceOrderMarketPayload>(
            OrderDirection::SELL, volume, kBookId, Currency::BASE, std::nullopt, STPFlag::CO, SettleType::FIFO);
        const auto result = exchange->clearingManager().handleOrder(
            MarketOrderDesc{.agentId = agentId, .payload = payload});
        ASSERT_EQ(result.ec, OrderErrorCode::VALID);
        [[maybe_unused]] const auto order = book->placeMarketOrder(
            OrderClientContext{agentId}, simulation->currentTimestamp(), result.orderSize, OrderDirection::SELL,
            0_dec, STPFlag::CO, SettleType::FIFO);
    }

    [[nodiscard]] bool hasLoan(OrderID orderId, std::optional<AgentId> agentId = {})
    {
        return exchange->accounts()[agentId.value_or(borrower)][kBookId].getLoan(orderId).has_value();
    }

    // The notices the proxy holds for `agentId`, counted by type; the proxy's buffer is emptied.
    [[nodiscard]] std::map<std::string, int> takeNotices(AgentId agentId)
    {
        std::map<std::string, int> counts;
        for (const auto& msg : simulation->proxy()->messages()) {
            const auto payload = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
            if (payload && payload->agentId == agentId) ++counts[msg->type];
        }
        simulation->proxy()->clearMessages();
        return counts;
    }

    [[nodiscard]] static int errorCount(const std::map<std::string, int>& notices)
    {
        int n = 0;
        for (const auto& [type, count] : notices) {
            if (type.starts_with("ERROR")) n += count;
        }
        return n;
    }

    [[nodiscard]] bool hasMarginCallEntry(OrderID orderId) const
    {
        return ranges::any_of(
            exchange->clearingManager().marginBuys().at(kBookId),
            [orderId](const auto& level) {
                return ranges::any_of(
                    level.second, [orderId](const auto& ctx) { return ctx.orderId == orderId; });
            });
    }

    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange{};
    taosim::book::Book::Ptr book;
    AgentId borrower{}, bidder{}, seller{};
};

// The order floor of the production simulation layout.
class MarginCallOrderFloorTest : public MarginCallRetryTest
{
protected:
    void adjustExchange(pugi::xml_node exchangeNode) override
    {
        exchangeNode.append_attribute("minOrderSize") = "0.25";
    }
};

}  // namespace

//-------------------------------------------------------------------------

TEST_F(MarginCallRetryTest, AMarginCallThatCannotFillIsCalledAgainOnALaterStep)
{
    limit(seller, OrderDirection::SELL, 10_dec, 300_dec);
    const auto loanId = leveragedMarketBuy(borrower, 2_dec, 1_dec);
    ASSERT_TRUE(hasLoan(loanId));
    ASSERT_TRUE(hasMarginCallEntry(loanId));
    // 300 * 1 / (2 * 0.9): both bids below are under it.
    const auto marginCallPrice = exchange->accounts()[borrower][kBookId].getLoan(loanId)->get().marginCallPrice();
    ASSERT_GT(marginCallPrice, 160_dec);

    // The only bid below the margin-call price is the borrower's own, which the margin call does not trade
    // against.
    limit(borrower, OrderDirection::BUY, 1_dec, 150_dec);
    simulation->step();
    ASSERT_TRUE(hasLoan(loanId)) << "the margin call filled against the borrower's own bid";
    EXPECT_TRUE(hasMarginCallEntry(loanId)) << "an unsettled loan lost its margin call";

    limit(bidder, OrderDirection::BUY, 10_dec, 160_dec);
    simulation->step();
    EXPECT_FALSE(hasLoan(loanId)) << "the margin call was not tried again once it could fill";
    EXPECT_FALSE(hasMarginCallEntry(loanId)) << "a settled loan kept its margin-call entry";
}

//-------------------------------------------------------------------------

TEST_F(MarginCallRetryTest, ASettledLoanIsNotCalledAgain)
{
    limit(seller, OrderDirection::SELL, 10_dec, 300_dec);
    const auto loanId = leveragedMarketBuy(borrower, 2_dec, 1_dec);
    limit(bidder, OrderDirection::BUY, 10_dec, 160_dec);

    simulation->step();
    ASSERT_FALSE(hasLoan(loanId));
    EXPECT_FALSE(hasMarginCallEntry(loanId));
    const auto bidLeft = book->bestBuyLevel() ? book->buyQueue().back().volume() : 0_dec;

    simulation->step();
    EXPECT_EQ(book->bestBuyLevel() ? book->buyQueue().back().volume() : 0_dec, bidLeft)
        << "a margin call was placed for a loan already settled";
}

//-------------------------------------------------------------------------

TEST_F(MarginCallRetryTest, ARemoteAgentIsToldOnceThatItsMarginCallCannotBePlaced)
{
    constexpr AgentId remote{0};
    limit(seller, OrderDirection::SELL, 10_dec, 300_dec);
    const auto loanId = leveragedMarketBuy(remote, 2_dec, 1_dec);
    limit(remote, OrderDirection::BUY, 1_dec, 150_dec);

    int refusals = 0;
    for (int i = 0; i < 3; ++i) {
        simulation->step();
        ASSERT_TRUE(hasLoan(loanId, remote)) << "step " << i;
        refusals += errorCount(takeNotices(remote));
    }
    EXPECT_EQ(refusals, 1) << "the agent was told on every retry";

    limit(bidder, OrderDirection::BUY, 10_dec, 160_dec);
    simulation->step();
    auto notices = takeNotices(remote);
    EXPECT_FALSE(hasLoan(loanId, remote));
    EXPECT_EQ(errorCount(notices), 0);
    EXPECT_GE(notices["EVENT_TRADE"], 1) << "the fill was not notified";
}

//-------------------------------------------------------------------------

TEST_F(MarginCallRetryTest, EachCalledLoanIsToldOnItsOwn)
{
    constexpr AgentId remote{0};
    limit(seller, OrderDirection::SELL, 10_dec, 300_dec);
    const auto firstLoan = leveragedMarketBuy(remote, 2_dec, 1_dec);
    const auto secondLoan = leveragedMarketBuy(remote, 2_dec, 1_dec);
    limit(remote, OrderDirection::BUY, 1_dec, 150_dec);

    int refusals = 0;
    for (int i = 0; i < 3; ++i) {
        simulation->step();
        ASSERT_TRUE(hasLoan(firstLoan, remote) && hasLoan(secondLoan, remote)) << "step " << i;
        refusals += errorCount(takeNotices(remote));
    }
    EXPECT_EQ(refusals, 2) << "one refusal per called loan";
}

//-------------------------------------------------------------------------

TEST_F(MarginCallOrderFloorTest, AMarginCallBelowTheOrderFloorStillClosesTheLoan)
{
    limit(seller, OrderDirection::SELL, 10_dec, 300_dec);
    const auto loanId = leveragedMarketBuy(borrower, DEC(0.25), 1_dec);
    // The setup leaves a loan smaller than the minimum order size.
    limit(bidder, OrderDirection::BUY, DEC(0.4), 299_dec);
    fifoMarketSell(borrower, DEC(0.4));
    ASSERT_TRUE(hasLoan(loanId));
    const auto loanLeft = exchange->accounts()[borrower][kBookId].getLoan(loanId)->get().amount();
    ASSERT_GT(loanLeft, 0_dec);
    ASSERT_LT(loanLeft / 160_dec, DEC(0.25)) << "the remaining loan is not below the order floor";

    // The consumed bid stays as a ghost level until the step barrier clears it.
    simulation->clearFilledOrders();
    limit(bidder, OrderDirection::BUY, 1_dec, 160_dec);
    ASSERT_EQ(book->bestBid(), 160_dec);
    simulation->step();

    EXPECT_FALSE(hasLoan(loanId)) << "the margin call was refused on the order floor";
    EXPECT_FALSE(hasMarginCallEntry(loanId));
}

//-------------------------------------------------------------------------
