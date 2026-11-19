/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 *
 * A buy limit pays what its fills cost, with the price band on or off.
 *
 * The placement check sizes a buy's reservation by walking the asks it would take. With the price band
 * active it predicts no sweep and reserves limit x size instead, the most the order can ever pay. The
 * clearing step settles the fill that completes a buy limit out of the order's remaining reservation
 * rather than at price x size. The two amounts agree while the reservation was sized from the fills.
 * Under the band they differ by the price improvement, (limit - fill price) x size summed over the
 * fills the order took, and that difference would move from the buyer to the seller of the last fill.
 *
 * The two fixtures differ in one attribute, maxPriceBand, so a case that holds without the band and
 * fails with it is evidence about the band and nothing else. Fees are zero in both, so a fill moves
 * its notional only.
 */
#include "taosim/decimal/decimal.hpp"
#include "formatting.hpp"

#include "MultiBookExchangeAgent.hpp"
#include "Order.hpp"
#include "Simulation.hpp"
#include "taosim/message/PayloadFactory.hpp"
#include "util.hpp"
#include <taosim/matching/ClearingManager.hpp>

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <pugixml.hpp>

#include <filesystem>
#include <limits>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

//-------------------------------------------------------------------------

using namespace taosim;
using namespace taosim::accounting;
using namespace taosim::book;
using namespace taosim::matching;
using namespace taosim::literals;

using namespace testing;

namespace fs = std::filesystem;

//-------------------------------------------------------------------------

static const auto s_testDataPath = fs::path{__FILE__}.parent_path().parent_path() / "data";

// remoteAgentCount="2" in the fixtures makes 0 and 1 remote; the locals are registered in SetUp.
static constexpr AgentId s_buyer = 0;
static constexpr AgentId s_seller = -1;
static constexpr AgentId s_seeder = -2;
static constexpr BookId s_bookId{};

struct LimitDesc
{
    AgentId agentId;
    OrderDirection direction;
    decimal_t volume;
    decimal_t price;
    decimal_t leverage{};
    taosim::TimeInForce timeInForce{taosim::TimeInForce::GTC};
};

[[nodiscard]] static std::pair<LimitOrder::Ptr, OrderErrorCode> placeLimit(
    MultiBookExchangeAgent* exchange, const LimitDesc& desc)
{
    const auto payload = MessagePayload::create<PlaceOrderLimitPayload>(
        desc.direction,
        desc.volume,
        desc.price,
        desc.leverage,
        s_bookId,
        Currency::BASE,
        std::nullopt,
        false,
        desc.timeInForce,
        std::nullopt,
        STPFlag::CO);
    const auto orderResult = exchange->clearingManager().handleOrder(
        LimitOrderDesc{.agentId = desc.agentId, .payload = payload});
    if (orderResult.ec != OrderErrorCode::VALID) {
        return {nullptr, orderResult.ec};
    }
    return {
        exchange->books()[s_bookId]->placeLimitOrder(
            OrderClientContext{desc.agentId},
            Timestamp{},
            orderResult.orderSize,
            payload->direction,
            payload->price,
            payload->leverage,
            payload->stpFlag),
        orderResult.ec};
}

struct Level
{
    decimal_t price;
    decimal_t volume;
};

// The buy under test and what it meets. `takenCost` is what the fills on arrival are worth and
// `takenVolume` what they deliver; a buy larger than that leaves the difference resting at its limit.
// A leveraged buy trades volume x (1 + leverage), puts up its share of the cost as collateral and
// books the whole cost as its loan. An immediate-or-cancel buy is cut to what it can take.
struct BandedBuyParams
{
    std::string_view config;
    bool banded;
    std::vector<Level> asks;
    decimal_t limit;
    decimal_t volume;
    decimal_t leverage{};
    taosim::TimeInForce timeInForce{taosim::TimeInForce::GTC};
    decimal_t takenVolume;
    decimal_t takenCost;
};

void PrintTo(const BandedBuyParams& params, std::ostream* os)
{
    std::string asks;
    for (const auto& [price, volume] : params.asks) {
        asks += fmt::format("{}{}@{}", asks.empty() ? "" : ", ", volume, price);
    }
    *os << fmt::format(
        "{{.config = {}, .banded = {}, .asks = [{}], .limit = {}, .volume = {}, .leverage = {}, "
        ".timeInForce = {}, .takenVolume = {}, .takenCost = {}}}",
        params.config, params.banded, asks, params.limit, params.volume, params.leverage,
        params.timeInForce == taosim::TimeInForce::IOC ? "IOC" : "GTC",
        params.takenVolume, params.takenCost);
}

//-------------------------------------------------------------------------

struct BandedBuySettlementTest : TestWithParam<BandedBuyParams>
{
    virtual void SetUp() override
    {
        const auto& params = GetParam();

        nodes = taosim::util::parseSimulationFile(s_testDataPath / params.config);
        simulation = std::make_unique<Simulation>();
        simulation->setDebug(false);
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        book = exchange->books()[s_bookId];
        exchange->accounts().registerLocal("seller");
        exchange->accounts().registerLocal("seeder");

        // The band needs a trade for its reference before it bounds anything. One unit at 300.00
        // between the two locals gives a reference of 300.00, so the band's ceiling is 330.00.
        const auto seedAsk = placeLimit(exchange, {
            .agentId = s_seller,
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = DEC(300.0)});
        ASSERT_EQ(seedAsk.second, OrderErrorCode::VALID);
        const auto seedBid = placeLimit(exchange, {
            .agentId = s_seeder,
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = DEC(300.0)});
        ASSERT_EQ(seedBid.second, OrderErrorCode::VALID);
        book->clearFilledOrders();
        ASSERT_TRUE(book->sellQueue().empty());
        ASSERT_TRUE(book->buyQueue().empty());
        // The fixture has to arm the band exactly when the case says it does, or the pair proves
        // nothing.
        ASSERT_EQ(book->bandLimit(true) != std::numeric_limits<decimal_t>::max(), params.banded);

        for (const auto& [price, volume] : params.asks) {
            const auto ask = placeLimit(exchange, {
                .agentId = s_seller,
                .direction = OrderDirection::SELL,
                .volume = volume,
                .price = price});
            ASSERT_EQ(ask.second, OrderErrorCode::VALID);
        }

        buyerQuoteBefore = buyer().quote->getTotal();
        buyerBaseBefore = buyer().base.getTotal();
        sellerQuoteBefore = seller().quote->getTotal();
    }

    [[nodiscard]] const Balances& buyer() const { return exchange->accounts()[s_buyer].at(s_bookId); }
    [[nodiscard]] const Balances& seller() const { return exchange->accounts()[s_seller].at(s_bookId); }
    [[nodiscard]] decimal_t paid() const { return buyerQuoteBefore - buyer().quote->getTotal(); }
    [[nodiscard]] decimal_t bought() const { return buyer().base.getTotal() - buyerBaseBefore; }
    [[nodiscard]] decimal_t received() const { return seller().quote->getTotal() - sellerQuoteBefore; }
    [[nodiscard]] decimal_t totalVolume() const
    {
        return GetParam().volume * taosim::util::dec1p(GetParam().leverage);
    }

    // What the buyer and the seller hold after fills worth `cost`: the seller has the cost, the buyer
    // is out of pocket by its own share of it, and a leveraged buyer owes the cost as its loan.
    void expectSettled(decimal_t cost) const
    {
        const auto& params = GetParam();
        const decimal_t ownShare = cost / taosim::util::dec1p(params.leverage);
        EXPECT_EQ(paid(), ownShare)
            << "the buyer paid " << paid() << " towards fills that cost " << cost
            << "; its limit times its size is " << params.limit * totalVolume();
        EXPECT_EQ(received(), cost)
            << "the seller received " << received() << " for fills worth " << cost;
        EXPECT_EQ(buyer().m_quoteLoan, params.leverage > 0_dec ? cost : 0_dec);
    }

    // Places the buy under test and checks it took what the case says it takes on arrival.
    [[nodiscard]] LimitOrder::Ptr placeBuy()
    {
        const auto& params = GetParam();
        const auto [order, ec] = placeLimit(exchange, {
            .agentId = s_buyer,
            .direction = OrderDirection::BUY,
            .volume = params.volume,
            .price = params.limit,
            .leverage = params.leverage,
            .timeInForce = params.timeInForce});
        EXPECT_EQ(ec, OrderErrorCode::VALID);
        EXPECT_EQ(bought(), params.takenVolume);
        return order;
    }

    // A step ends by dropping the orders that filled during it, which is also where an order that
    // completed while resting gives back what it reserved and did not spend.
    void endStep()
    {
        book->clearFilledOrders();
        EXPECT_EQ(buyer().quote->getReserved(), 0_dec)
            << "an order that is gone must leave nothing reserved";
        EXPECT_EQ(buyer().quote->getFree(), buyer().quote->getTotal());
    }

    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange{};
    Book::Ptr book;
    decimal_t buyerQuoteBefore{};
    decimal_t buyerBaseBefore{};
    decimal_t sellerQuoteBefore{};
};

//-------------------------------------------------------------------------

struct TakingBuySettlementTest : BandedBuySettlementTest {};

TEST_P(TakingBuySettlementTest, PaysWhatItsFillsCost)
{
    const auto& params = GetParam();

    const auto order = placeBuy();
    ASSERT_NE(order, nullptr);
    ASSERT_EQ(order->volume(), 0_dec) << "this case is a buy that completes on arrival";
    endStep();

    expectSettled(params.takenCost);
}

INSTANTIATE_TEST_SUITE_P(
    BandedBuySettlementTests,
    TakingBuySettlementTest,
    Values(
        BandedBuyParams{
            .config = "BandedBuySettlementNoBand.xml",
            .banded = false,
            .asks = {{.price = DEC(300.0), .volume = 3_dec}},
            .limit = DEC(303.0),
            .volume = 3_dec,
            .takenVolume = 3_dec,
            .takenCost = 900_dec
        },
        BandedBuyParams{
            .config = "BandedBuySettlementNoBand.xml",
            .banded = false,
            .asks = {{.price = DEC(300.0), .volume = 2_dec}, {.price = DEC(301.0), .volume = 2_dec}},
            .limit = DEC(305.0),
            .volume = 4_dec,
            .takenVolume = 4_dec,
            .takenCost = 1202_dec
        },
        BandedBuyParams{
            .config = "BandedBuySettlement.xml",
            .banded = true,
            .asks = {{.price = DEC(300.0), .volume = 3_dec}},
            .limit = DEC(303.0),
            .volume = 3_dec,
            .takenVolume = 3_dec,
            .takenCost = 900_dec
        },
        BandedBuyParams{
            .config = "BandedBuySettlement.xml",
            .banded = true,
            .asks = {{.price = DEC(300.0), .volume = 2_dec}, {.price = DEC(301.0), .volume = 2_dec}},
            .limit = DEC(305.0),
            .volume = 4_dec,
            .takenVolume = 4_dec,
            .takenCost = 1202_dec
        },
        // A limit far above the band's ceiling of 330.00 still fills at the ask.
        BandedBuyParams{
            .config = "BandedBuySettlementNoBand.xml",
            .banded = false,
            .asks = {{.price = DEC(300.0), .volume = 3_dec}},
            .limit = DEC(3000.0),
            .volume = 3_dec,
            .takenVolume = 3_dec,
            .takenCost = 900_dec
        },
        BandedBuyParams{
            .config = "BandedBuySettlement.xml",
            .banded = true,
            .asks = {{.price = DEC(300.0), .volume = 3_dec}},
            .limit = DEC(3000.0),
            .volume = 3_dec,
            .takenVolume = 3_dec,
            .takenCost = 900_dec
        },
        BandedBuyParams{
            .config = "BandedBuySettlementNoBand.xml",
            .banded = false,
            .asks = {{.price = DEC(300.0), .volume = 3_dec}},
            .limit = DEC(303.0),
            .volume = 5_dec,
            .timeInForce = taosim::TimeInForce::IOC,
            .takenVolume = 3_dec,
            .takenCost = 900_dec
        },
        BandedBuyParams{
            .config = "BandedBuySettlement.xml",
            .banded = true,
            .asks = {{.price = DEC(300.0), .volume = 3_dec}},
            .limit = DEC(303.0),
            .volume = 5_dec,
            .timeInForce = taosim::TimeInForce::IOC,
            .takenVolume = 3_dec,
            .takenCost = 900_dec
        },
        BandedBuyParams{
            .config = "BandedBuySettlementNoBand.xml",
            .banded = false,
            .asks = {{.price = DEC(300.0), .volume = 2_dec}, {.price = DEC(301.0), .volume = 2_dec}},
            .limit = DEC(305.0),
            .volume = 2_dec,
            .leverage = 1_dec,
            .takenVolume = 4_dec,
            .takenCost = 1202_dec
        },
        BandedBuyParams{
            .config = "BandedBuySettlement.xml",
            .banded = true,
            .asks = {{.price = DEC(300.0), .volume = 2_dec}, {.price = DEC(301.0), .volume = 2_dec}},
            .limit = DEC(305.0),
            .volume = 2_dec,
            .leverage = 1_dec,
            .takenVolume = 4_dec,
            .takenCost = 1202_dec
        }));

//-------------------------------------------------------------------------

struct RestingBuySettlementTest : BandedBuySettlementTest {};

TEST_P(RestingBuySettlementTest, FilledRemainderPaysWhatItsFillsCost)
{
    const auto& params = GetParam();
    const decimal_t restingVolume = totalVolume() - params.takenVolume;
    ASSERT_GT(restingVolume, 0_dec) << "this case is a buy that leaves a remainder resting";

    const auto order = placeBuy();
    ASSERT_NE(order, nullptr);
    ASSERT_EQ(order->volume(), restingVolume);

    // The remainder rests at its limit and a seller comes down to it, so that fill costs limit x size.
    const auto sell = placeLimit(exchange, {
        .agentId = s_seller,
        .direction = OrderDirection::SELL,
        .volume = restingVolume,
        .price = params.limit});
    ASSERT_EQ(sell.second, OrderErrorCode::VALID);
    ASSERT_EQ(bought(), totalVolume());
    endStep();

    expectSettled(params.takenCost + params.limit * restingVolume);
}

TEST_P(RestingBuySettlementTest, CancelledRemainderPaysWhatItsFillsCost)
{
    const auto& params = GetParam();

    const auto order = placeBuy();
    ASSERT_NE(order, nullptr);
    ASSERT_TRUE(book->cancelOrder(order->id()));
    endStep();

    expectSettled(params.takenCost);
}

INSTANTIATE_TEST_SUITE_P(
    BandedBuySettlementTests,
    RestingBuySettlementTest,
    Values(
        BandedBuyParams{
            .config = "BandedBuySettlementNoBand.xml",
            .banded = false,
            .asks = {{.price = DEC(300.0), .volume = 2_dec}},
            .limit = DEC(305.0),
            .volume = 4_dec,
            .takenVolume = 2_dec,
            .takenCost = 600_dec
        },
        BandedBuyParams{
            .config = "BandedBuySettlement.xml",
            .banded = true,
            .asks = {{.price = DEC(300.0), .volume = 2_dec}},
            .limit = DEC(305.0),
            .volume = 4_dec,
            .takenVolume = 2_dec,
            .takenCost = 600_dec
        }));
