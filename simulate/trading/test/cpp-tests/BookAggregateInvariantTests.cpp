/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "MultiBookExchangeAgent.hpp"
#include "Order.hpp"
#include "Simulation.hpp"
#include "formatting.hpp"
#include "util.hpp"
#include <taosim/book/Book.hpp>
#include <taosim/decimal/decimal.hpp>
#include <taosim/matching/ClearingManager.hpp>
#include <taosim/message/PayloadFactory.hpp>

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string_view>
#include <vector>

//-------------------------------------------------------------------------
// Every price level keeps an aggregate volume (TickContainer::volume) and
// every side keeps the sum over its levels (OrderContainer::volume). These are incrementally
// maintained by placement, matching, cancellation and ghost clearing, so this suite drives the
// book through all of those and, after EVERY step, recomputes the truth from the resting orders
// themselves: the aggregate of a level must equal the sum of its orders' totalVolume() (own plus
// borrowed volume, which is what placement adds), and a side's aggregate must equal the sum of its
// levels. Alongside, the structural facts the aggregates rely on: no empty level survives, levels
// are strictly ascending, every order sits on the level of its price, the registered order set is
// exactly the resting set, and the cached best levels are the outermost active ones.

using namespace taosim;
using namespace taosim::accounting;
using namespace taosim::book;
using namespace taosim::matching;
using namespace taosim::literals;

using namespace testing;

namespace fs = std::filesystem;

namespace
{

const auto kTestDataPath = fs::path{__FILE__}.parent_path() / "data";

// volumeDecimals in MultiAgentFees.xml.
constexpr uint32_t kVolumeDecimals = 4;

Timestamp nextTimestamp() noexcept
{
    static Timestamp ts{};
    return ++ts;
}

//-------------------------------------------------------------------------

[[nodiscard]] std::optional<decimal_t> outermostActivePrice(auto&& levelsView)
{
    const auto it = ranges::find_if(
        levelsView, [](const auto& level) { return level.hasActiveOrders(); });
    return it != ranges::end(levelsView) ? std::make_optional(it->price()) : std::nullopt;
}

void expectSideConsistent(const OrderContainer& side, std::string_view name)
{
    decimal_t sumOfLevels = 0_dec;
    std::optional<decimal_t> previousPrice;
    for (const auto& level : side) {
        SCOPED_TRACE(fmt::format("{} level {}", name, level.price()));
        EXPECT_FALSE(level.empty()) << "an orderless level must have been erased";

        decimal_t sumOfOrders = 0_dec;
        bool anyActive = false;
        for (const auto& order : level) {
            EXPECT_EQ(order->price(), level.price());
            sumOfOrders += order->totalVolume();
            anyActive = anyActive || order->volume() > 0_dec;
        }
        EXPECT_EQ(level.volume(), sumOfOrders);
        EXPECT_EQ(level.hasActiveOrders(), anyActive);

        if (previousPrice) {
            EXPECT_LT(*previousPrice, level.price());
        }
        previousPrice = level.price();
        sumOfLevels += level.volume();
    }
    EXPECT_EQ(side.volume(), sumOfLevels) << name << " side aggregate";
}

void expectBookConsistent(const Book& book)
{
    expectSideConsistent(book.buyQueue(), "bid");
    expectSideConsistent(book.sellQueue(), "ask");

    std::set<OrderID> resting;
    for (const auto* side : {&book.buyQueue(), &book.sellQueue()}) {
        for (const auto& level : *side) {
            for (const auto& order : level) {
                EXPECT_TRUE(resting.insert(order->id()).second) << "order resting twice";
            }
        }
    }
    std::set<OrderID> registered;
    for (const auto& [id, order] : book.orderIdMap()) {
        registered.insert(id);
    }
    EXPECT_EQ(resting, registered);

    const auto none = DEC(-1.0);
    auto priceOf = [](auto&& level) { return level.get().price(); };
    const auto bestBid = book.bestBuyLevel().transform(priceOf).value_or(none);
    const auto bestAsk = book.bestSellLevel().transform(priceOf).value_or(none);
    EXPECT_EQ(
        bestBid, outermostActivePrice(book.buyQueue() | ranges::views::reverse).value_or(none));
    EXPECT_EQ(bestAsk, outermostActivePrice(book.sellQueue()).value_or(none));
    if (bestBid != none && bestAsk != none) {
        EXPECT_LT(bestBid, bestAsk);
    }
}

[[nodiscard]] decimal_t sumOfTotals(const TickContainer& level)
{
    return ranges::accumulate(
        level, 0_dec, std::plus{}, [](const auto& order) { return order->totalVolume(); });
}

[[nodiscard]] const TickContainer* levelAt(const OrderContainer& side, decimal_t price)
{
    const auto it =
        ranges::find_if(side, [price](const auto& level) { return level.price() == price; });
    return it != side.end() ? &*it : nullptr;
}

[[nodiscard]] size_t ghostCount(const Book& book)
{
    size_t ghosts{};
    for (const auto* side : {&book.buyQueue(), &book.sellQueue()}) {
        for (const auto& level : *side) {
            ghosts += ranges::count_if(
                level, [](const auto& order) { return order->volume() == 0_dec; });
        }
    }
    return ghosts;
}

//-------------------------------------------------------------------------

class BookAggregateInvariantTest : public Test
{
public:
    static constexpr AgentId agent1 = -1, agent2 = -2, agent3 = -3, agent4 = -4;
    static constexpr std::array<AgentId, 4> kAgents{agent1, agent2, agent3, agent4};
    const BookId bookId{};

    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange{};
    Book::Ptr book;

    // Placement goes through the clearing manager so that reservations exist and matching settles,
    // exactly as in production. A rejected order would otherwise be placed with size 0 and hide
    // from the invariants, so acceptance is checked unless the caller opts out.
    [[nodiscard]] LimitOrder::Ptr tryPlaceLimit(
        AgentId agentId,
        OrderDirection direction,
        decimal_t volume,
        decimal_t price,
        decimal_t leverage = 0_dec,
        STPFlag stpFlag = STPFlag::CO,
        bool postOnly = false)
    {
        const auto payload = MessagePayload::create<PlaceOrderLimitPayload>(
            direction, volume, price, leverage, bookId, Currency::BASE, std::nullopt, postOnly,
            TimeInForce::GTC, std::nullopt, stpFlag);
        const auto result = exchange->clearingManager().handleOrder(
            LimitOrderDesc{.agentId = agentId, .payload = payload});
        if (result.ec != OrderErrorCode::VALID) {
            return nullptr;
        }
        return book->placeLimitOrder(
            OrderClientContext{agentId},
            nextTimestamp(),
            result.orderSize,
            payload->direction,
            payload->price,
            payload->leverage,
            payload->stpFlag);
    }

    // A rejection ends the test: every caller dereferences the order right away.
    LimitOrder::Ptr placeLimit(
        AgentId agentId,
        OrderDirection direction,
        decimal_t volume,
        decimal_t price,
        decimal_t leverage = 0_dec,
        STPFlag stpFlag = STPFlag::CO,
        bool postOnly = false)
    {
        auto order =
            tryPlaceLimit(agentId, direction, volume, price, leverage, stpFlag, postOnly);
        if (!order) {
            throw std::runtime_error{fmt::format(
                "limit {} {}@{} x{} by agent {} rejected",
                direction == OrderDirection::BUY ? "BUY" : "SELL", volume, price, leverage,
                agentId)};
        }
        return order;
    }

    [[nodiscard]] MarketOrder::Ptr tryPlaceMarket(
        AgentId agentId, OrderDirection direction, decimal_t volume, decimal_t leverage = 0_dec)
    {
        const auto payload = MessagePayload::create<PlaceOrderMarketPayload>(
            direction, volume, leverage, bookId, Currency::BASE, std::nullopt, STPFlag::CO);
        const auto result = exchange->clearingManager().handleOrder(
            MarketOrderDesc{.agentId = agentId, .payload = payload});
        if (result.ec != OrderErrorCode::VALID) {
            return nullptr;
        }
        return book->placeMarketOrder(
            OrderClientContext{agentId},
            nextTimestamp(),
            result.orderSize,
            payload->direction,
            payload->leverage,
            payload->stpFlag);
    }

    MarketOrder::Ptr placeMarket(
        AgentId agentId, OrderDirection direction, decimal_t volume, decimal_t leverage = 0_dec)
    {
        auto order = tryPlaceMarket(agentId, direction, volume, leverage);
        if (!order) {
            throw std::runtime_error{fmt::format(
                "market {} {} x{} by agent {} rejected",
                direction == OrderDirection::BUY ? "BUY" : "SELL", volume, leverage, agentId)};
        }
        return order;
    }

    void check() const { expectBookConsistent(*book); }

    [[nodiscard]] const OrderContainer& bids() const { return book->buyQueue(); }
    [[nodiscard]] const OrderContainer& asks() const { return book->sellQueue(); }

protected:
    void SetUp() override
    {
        nodes = taosim::util::parseSimulationFile(kTestDataPath / "MultiAgentFees.xml");
        simulation = std::make_unique<Simulation>();
        simulation->setDebug(false);
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        book = exchange->books()[bookId];

        exchange->accounts().registerLocal("agent1");
        exchange->accounts().registerLocal("agent2");
        exchange->accounts().registerLocal("agent3");
        exchange->accounts().registerLocal("agent4");

        check();
        ASSERT_TRUE(bids().empty());
        ASSERT_TRUE(asks().empty());
    }
};

}  // namespace

//-------------------------------------------------------------------------

TEST_F(BookAggregateInvariantTest, LadderPlacementAggregatesEveryOrder)
{
    // Two orders per level, three levels a side, placed in a shuffled order so that levels are
    // created both at the ends and in the middle of each deque.
    const auto b1 = placeLimit(agent1, OrderDirection::BUY, DEC(1.5), 299_dec);
    check();
    const auto b2 = placeLimit(agent2, OrderDirection::BUY, DEC(2.25), 297_dec);
    check();
    const auto b3 = placeLimit(agent3, OrderDirection::BUY, DEC(0.75), 298_dec);
    check();
    const auto b4 = placeLimit(agent4, OrderDirection::BUY, DEC(1.0), 299_dec);
    check();
    const auto b5 = placeLimit(agent1, OrderDirection::BUY, 1_dec, 297_dec);
    check();
    const auto b6 = placeLimit(agent2, OrderDirection::BUY, DEC(3.0), 298_dec);
    check();

    const auto a1 = placeLimit(agent3, OrderDirection::SELL, DEC(1.0), 302_dec);
    check();
    const auto a2 = placeLimit(agent4, OrderDirection::SELL, DEC(2.0), 301_dec);
    check();
    const auto a3 = placeLimit(agent1, OrderDirection::SELL, DEC(0.25), 303_dec);
    check();
    const auto a4 = placeLimit(agent2, OrderDirection::SELL, DEC(1.75), 302_dec);
    check();
    const auto a5 = placeLimit(agent3, OrderDirection::SELL, DEC(4.0), 301_dec);
    check();
    const auto a6 = placeLimit(agent4, OrderDirection::SELL, DEC(0.5), 303_dec);
    check();

    ASSERT_EQ(bids().size(), 3uz);
    ASSERT_EQ(asks().size(), 3uz);
    EXPECT_EQ(levelAt(bids(), 299_dec)->volume(), b1->volume() + b4->volume());
    EXPECT_EQ(levelAt(bids(), 298_dec)->volume(), b3->volume() + b6->volume());
    EXPECT_EQ(levelAt(bids(), 297_dec)->volume(), b2->volume() + b5->volume());
    EXPECT_EQ(levelAt(asks(), 301_dec)->volume(), a2->volume() + a5->volume());
    EXPECT_EQ(levelAt(asks(), 302_dec)->volume(), a1->volume() + a4->volume());
    EXPECT_EQ(levelAt(asks(), 303_dec)->volume(), a3->volume() + a6->volume());
    EXPECT_EQ(bids().volume(), DEC(9.5));
    EXPECT_EQ(asks().volume(), DEC(9.5));
    EXPECT_EQ(book->bestBid(), 299_dec);
    EXPECT_EQ(book->bestAsk(), 301_dec);
}

//-------------------------------------------------------------------------

TEST_F(BookAggregateInvariantTest, PartialCancelShrinksOrderAndLevel)
{
    const auto b1 = placeLimit(agent1, OrderDirection::BUY, DEC(1.5), 299_dec);
    const auto b2 = placeLimit(agent2, OrderDirection::BUY, DEC(1.0), 299_dec);
    check();

    ASSERT_TRUE(book->cancelOrder(b1->id(), DEC(0.4)));
    check();
    EXPECT_EQ(b1->volume(), DEC(1.1));
    EXPECT_EQ(levelAt(bids(), 299_dec)->volume(), DEC(2.1));
    EXPECT_EQ(bids().volume(), DEC(2.1));

    // Cancelling more than rests is capped at the order's volume and empties it.
    ASSERT_TRUE(book->cancelOrder(b2->id(), DEC(5.0)));
    check();
    EXPECT_EQ(levelAt(bids(), 299_dec)->volume(), DEC(1.1));
    EXPECT_THAT(*levelAt(bids(), 299_dec), SizeIs(1));

    // Unknown ids and repeated cancellations are refused and leave everything untouched.
    EXPECT_FALSE(book->cancelOrder(b2->id()));
    EXPECT_FALSE(book->cancelOrder(OrderID{424242}));
    check();
    EXPECT_EQ(bids().volume(), DEC(1.1));
}

//-------------------------------------------------------------------------

TEST_F(BookAggregateInvariantTest, FullCancelWithSiblingsKeepsTheLevel)
{
    const auto a1 = placeLimit(agent1, OrderDirection::SELL, DEC(1.0), 301_dec);
    const auto a2 = placeLimit(agent2, OrderDirection::SELL, DEC(2.0), 301_dec);
    const auto a3 = placeLimit(agent3, OrderDirection::SELL, DEC(3.0), 301_dec);
    check();

    ASSERT_TRUE(book->cancelOrder(a2->id()));
    check();
    ASSERT_EQ(asks().size(), 1uz);
    EXPECT_THAT(asks().front(), ElementsAre(a1, a3));
    EXPECT_EQ(asks().front().volume(), DEC(4.0));

    ASSERT_TRUE(book->cancelOrder(a1->id()));
    check();
    EXPECT_THAT(asks().front(), ElementsAre(a3));
    EXPECT_EQ(asks().volume(), DEC(3.0));
}

//-------------------------------------------------------------------------

TEST_F(BookAggregateInvariantTest, CancelEmptiesMiddleLevel)
{
    placeLimit(agent1, OrderDirection::BUY, DEC(1.0), 299_dec);
    const auto middle = placeLimit(agent2, OrderDirection::BUY, DEC(2.0), 298_dec);
    placeLimit(agent3, OrderDirection::BUY, DEC(3.0), 297_dec);
    placeLimit(agent1, OrderDirection::SELL, DEC(1.0), 301_dec);
    const auto middleAsk = placeLimit(agent2, OrderDirection::SELL, DEC(2.0), 302_dec);
    placeLimit(agent3, OrderDirection::SELL, DEC(3.0), 303_dec);
    check();
    ASSERT_EQ(bids().size(), 3uz);
    ASSERT_EQ(asks().size(), 3uz);

    ASSERT_TRUE(book->cancelOrder(middle->id()));
    check();
    EXPECT_EQ(bids().size(), 2uz);
    EXPECT_EQ(levelAt(bids(), 298_dec), nullptr);
    EXPECT_EQ(bids().volume(), DEC(4.0));
    EXPECT_EQ(book->bestBid(), 299_dec);

    ASSERT_TRUE(book->cancelOrder(middleAsk->id()));
    check();
    EXPECT_EQ(asks().size(), 2uz);
    EXPECT_EQ(levelAt(asks(), 302_dec), nullptr);
    EXPECT_EQ(asks().volume(), DEC(4.0));
    EXPECT_EQ(book->bestAsk(), 301_dec);
}

//-------------------------------------------------------------------------

TEST_F(BookAggregateInvariantTest, CancelEmptiesBestLevelsAndBestMovesOn)
{
    const auto bestBid = placeLimit(agent1, OrderDirection::BUY, DEC(1.0), 299_dec);
    placeLimit(agent2, OrderDirection::BUY, DEC(2.0), 298_dec);
    const auto bestAsk = placeLimit(agent3, OrderDirection::SELL, DEC(1.0), 301_dec);
    placeLimit(agent4, OrderDirection::SELL, DEC(2.0), 302_dec);
    check();

    // Two steps: a partial cancel first, then the remainder, so the level dies on a partial path.
    ASSERT_TRUE(book->cancelOrder(bestBid->id(), DEC(0.3)));
    check();
    EXPECT_EQ(book->bestBid(), 299_dec);
    ASSERT_TRUE(book->cancelOrder(bestBid->id(), DEC(0.7)));
    check();
    EXPECT_EQ(bids().size(), 1uz);
    EXPECT_EQ(book->bestBid(), 298_dec);
    EXPECT_EQ(bids().volume(), DEC(2.0));

    ASSERT_TRUE(book->cancelOrder(bestAsk->id()));
    check();
    EXPECT_EQ(asks().size(), 1uz);
    EXPECT_EQ(book->bestAsk(), 302_dec);
    EXPECT_EQ(asks().volume(), DEC(2.0));
}

//-------------------------------------------------------------------------

TEST_F(BookAggregateInvariantTest, CancellingEverythingEmptiesBothSides)
{
    std::vector<LimitOrder::Ptr> orders;
    for (const auto [price, volume] : {
             std::pair{297_dec, DEC(1.0)}, {298_dec, DEC(2.0)}, {298_dec, DEC(0.5)},
             {299_dec, DEC(1.25)}}) {
        orders.push_back(placeLimit(agent1, OrderDirection::BUY, volume, price));
    }
    for (const auto [price, volume] : {
             std::pair{301_dec, DEC(1.0)}, {302_dec, DEC(2.0)}, {302_dec, DEC(0.5)},
             {303_dec, DEC(1.25)}}) {
        orders.push_back(placeLimit(agent2, OrderDirection::SELL, volume, price));
    }
    check();
    EXPECT_EQ(bids().volume(), DEC(4.75));
    EXPECT_EQ(asks().volume(), DEC(4.75));

    // Cancel in an order that removes best, worst and middle levels alternately.
    for (const auto index : {3, 0, 7, 4, 1, 5, 2, 6}) {
        ASSERT_TRUE(book->cancelOrder(orders[index]->id()));
        check();
    }
    EXPECT_TRUE(bids().empty());
    EXPECT_TRUE(asks().empty());
    EXPECT_EQ(bids().volume(), 0_dec);
    EXPECT_EQ(asks().volume(), 0_dec);
    EXPECT_TRUE(book->orderIdMap().empty());
    EXPECT_FALSE(book->bestBuyLevel().has_value());
    EXPECT_FALSE(book->bestSellLevel().has_value());

    // The book is fully usable again afterwards.
    placeLimit(agent3, OrderDirection::BUY, DEC(1.0), 299_dec);
    check();
    EXPECT_EQ(bids().volume(), DEC(1.0));
}

//-------------------------------------------------------------------------

TEST_F(BookAggregateInvariantTest, AggressiveLimitConsumesBestLevelLeavingGhost)
{
    const auto resting = placeLimit(agent4, OrderDirection::BUY, DEC(1.0), 299_dec);
    placeLimit(agent4, OrderDirection::BUY, DEC(2.0), 298_dec);
    check();

    // Partial fill: the resting order shrinks, the level with it, the taker never rests.
    const auto taker1 = placeLimit(agent1, OrderDirection::SELL, DEC(0.4), 299_dec);
    check();
    EXPECT_EQ(taker1->volume(), 0_dec);
    EXPECT_EQ(resting->volume(), DEC(0.6));
    EXPECT_EQ(levelAt(bids(), 299_dec)->volume(), DEC(0.6));
    EXPECT_TRUE(asks().empty());
    EXPECT_EQ(bids().volume(), DEC(2.6));

    // Full fill: the resting order stays on its level as a ghost, the level aggregate is zero and
    // the best bid moves on while the ghost level still exists.
    const auto taker2 = placeLimit(agent2, OrderDirection::SELL, DEC(0.6), 299_dec);
    check();
    EXPECT_EQ(taker2->volume(), 0_dec);
    EXPECT_EQ(resting->volume(), 0_dec);
    ASSERT_NE(levelAt(bids(), 299_dec), nullptr);
    EXPECT_THAT(*levelAt(bids(), 299_dec), ElementsAre(resting));
    EXPECT_EQ(levelAt(bids(), 299_dec)->volume(), 0_dec);
    EXPECT_FALSE(levelAt(bids(), 299_dec)->hasActiveOrders());
    EXPECT_EQ(book->bestBid(), 298_dec);
    EXPECT_EQ(bids().volume(), DEC(2.0));
    EXPECT_EQ(ghostCount(*book), 1uz);

    // A ghost cannot be cancelled, and clearing it also drops the now-empty level.
    EXPECT_FALSE(book->cancelOrder(resting->id()));
    check();
    book->clearFilledOrders();
    check();
    EXPECT_EQ(levelAt(bids(), 299_dec), nullptr);
    EXPECT_EQ(ghostCount(*book), 0uz);
    EXPECT_EQ(bids().volume(), DEC(2.0));
}

//-------------------------------------------------------------------------

TEST_F(BookAggregateInvariantTest, AggressiveLimitSweepsLevelsAndRestsRemainder)
{
    placeLimit(agent4, OrderDirection::SELL, DEC(1.0), 301_dec);
    placeLimit(agent3, OrderDirection::SELL, DEC(1.0), 302_dec);
    placeLimit(agent4, OrderDirection::SELL, DEC(1.0), 303_dec);
    check();

    const auto taker = placeLimit(agent1, OrderDirection::BUY, DEC(2.5), 302_dec);
    check();
    EXPECT_EQ(taker->volume(), DEC(0.5));
    ASSERT_NE(levelAt(bids(), 302_dec), nullptr);
    EXPECT_EQ(levelAt(bids(), 302_dec)->volume(), DEC(0.5));
    EXPECT_EQ(bids().volume(), DEC(0.5));
    EXPECT_EQ(levelAt(asks(), 301_dec)->volume(), 0_dec);
    EXPECT_EQ(levelAt(asks(), 302_dec)->volume(), 0_dec);
    EXPECT_EQ(levelAt(asks(), 303_dec)->volume(), DEC(1.0));
    EXPECT_EQ(asks().volume(), DEC(1.0));
    EXPECT_EQ(book->bestAsk(), 303_dec);
    EXPECT_EQ(book->bestBid(), 302_dec);
    EXPECT_EQ(ghostCount(*book), 2uz);

    // The rested remainder can be cancelled like any other order.
    ASSERT_TRUE(book->cancelOrder(taker->id()));
    check();
    EXPECT_TRUE(bids().empty());

    book->clearFilledOrders();
    check();
    ASSERT_EQ(asks().size(), 1uz);
    EXPECT_EQ(asks().front().price(), 303_dec);
    EXPECT_EQ(asks().volume(), DEC(1.0));
}

//-------------------------------------------------------------------------

TEST_F(BookAggregateInvariantTest, MarketOrdersConsumeLevels)
{
    placeLimit(agent4, OrderDirection::BUY, DEC(1.0), 299_dec);
    const auto deeper = placeLimit(agent3, OrderDirection::BUY, DEC(1.0), 298_dec);
    placeLimit(agent4, OrderDirection::BUY, DEC(2.0), 297_dec);
    placeLimit(agent4, OrderDirection::SELL, DEC(1.0), 301_dec);
    placeLimit(agent3, OrderDirection::SELL, DEC(2.0), 302_dec);
    check();

    placeMarket(agent1, OrderDirection::SELL, DEC(1.5));
    check();
    EXPECT_EQ(levelAt(bids(), 299_dec)->volume(), 0_dec);
    EXPECT_EQ(deeper->volume(), DEC(0.5));
    EXPECT_EQ(levelAt(bids(), 298_dec)->volume(), DEC(0.5));
    EXPECT_EQ(bids().volume(), DEC(2.5));
    EXPECT_EQ(book->bestBid(), 298_dec);

    placeMarket(agent2, OrderDirection::BUY, DEC(1.25));
    check();
    EXPECT_EQ(levelAt(asks(), 301_dec)->volume(), 0_dec);
    EXPECT_EQ(levelAt(asks(), 302_dec)->volume(), DEC(1.75));
    EXPECT_EQ(asks().volume(), DEC(1.75));
    EXPECT_EQ(book->bestAsk(), 302_dec);

    // Exhaust a whole side with one market order; ghosts remain until cleared.
    placeMarket(agent1, OrderDirection::SELL, DEC(2.5));
    check();
    EXPECT_EQ(bids().volume(), 0_dec);
    EXPECT_FALSE(bids().hasActiveOrders());
    EXPECT_FALSE(book->bestBuyLevel().has_value());
    EXPECT_EQ(bids().size(), 3uz);

    book->clearFilledOrders();
    check();
    EXPECT_TRUE(bids().empty());
    EXPECT_EQ(asks().size(), 1uz);
    EXPECT_EQ(ghostCount(*book), 0uz);
}

//-------------------------------------------------------------------------

TEST_F(BookAggregateInvariantTest, ClearFilledOrdersKeepsLiveOrdersAndAggregates)
{
    const auto keep1 = placeLimit(agent4, OrderDirection::BUY, DEC(1.0), 299_dec);
    const auto fill = placeLimit(agent3, OrderDirection::BUY, DEC(0.5), 299_dec);
    const auto keep2 = placeLimit(agent4, OrderDirection::BUY, DEC(1.0), 299_dec);
    placeLimit(agent4, OrderDirection::BUY, DEC(1.0), 298_dec);
    check();

    // Price-time priority fills keep1 fully and fill partially; keep2 stays intact.
    placeMarket(agent1, OrderDirection::SELL, DEC(1.2));
    check();
    EXPECT_EQ(keep1->volume(), 0_dec);
    EXPECT_EQ(fill->volume(), DEC(0.3));
    EXPECT_EQ(keep2->volume(), DEC(1.0));
    EXPECT_EQ(levelAt(bids(), 299_dec)->volume(), DEC(1.3));

    book->clearFilledOrders();
    check();
    EXPECT_THAT(*levelAt(bids(), 299_dec), ElementsAre(fill, keep2));
    EXPECT_EQ(levelAt(bids(), 299_dec)->volume(), DEC(1.3));
    EXPECT_EQ(bids().volume(), DEC(2.3));
    EXPECT_EQ(ghostCount(*book), 0uz);

    // Idempotent.
    book->clearFilledOrders();
    check();
    EXPECT_EQ(bids().volume(), DEC(2.3));
}

//-------------------------------------------------------------------------

TEST_F(BookAggregateInvariantTest, LeveragedOrdersAggregateTotalVolume)
{
    const auto plain = placeLimit(agent1, OrderDirection::BUY, DEC(1.0), 299_dec);
    check();
    const auto leveraged = placeLimit(agent2, OrderDirection::BUY, DEC(1.0), 299_dec, DEC(1.0));
    check();
    EXPECT_EQ(leveraged->totalVolume(), DEC(2.0));
    EXPECT_EQ(levelAt(bids(), 299_dec)->volume(), plain->totalVolume() + leveraged->totalVolume());
    EXPECT_EQ(bids().volume(), DEC(3.0));

    // Cancelling part of a leveraged order removes own AND borrowed volume from the level.
    ASSERT_TRUE(book->cancelOrder(leveraged->id(), DEC(0.25)));
    check();
    EXPECT_EQ(leveraged->volume(), DEC(0.75));
    EXPECT_EQ(levelAt(bids(), 299_dec)->volume(), DEC(2.5));

    ASSERT_TRUE(book->cancelOrder(leveraged->id()));
    check();
    EXPECT_EQ(levelAt(bids(), 299_dec)->volume(), DEC(1.0));
    EXPECT_EQ(bids().volume(), DEC(1.0));

    // Leveraged resting orders are consumed by their total volume.
    const auto leveragedAsk = placeLimit(agent3, OrderDirection::SELL, DEC(1.0), 301_dec, DEC(3.0));
    check();
    EXPECT_EQ(levelAt(asks(), 301_dec)->volume(), DEC(4.0));
    placeMarket(agent4, OrderDirection::BUY, DEC(1.0));
    check();
    EXPECT_EQ(leveragedAsk->totalVolume(), DEC(3.0));
    EXPECT_EQ(levelAt(asks(), 301_dec)->volume(), DEC(3.0));
    ASSERT_TRUE(book->cancelOrder(leveragedAsk->id()));
    check();
    EXPECT_TRUE(asks().empty());
    EXPECT_EQ(asks().volume(), 0_dec);
}

//-------------------------------------------------------------------------

TEST_F(BookAggregateInvariantTest, LeveragedFillsWithInexactDivisionStayExact)
{
    // 1 + 0.5 does not divide volumes exactly in decimal, so the resting order's own volume gets
    // rounded on every fill; the level must track the order's actual totalVolume regardless.
    const auto resting = placeLimit(agent4, OrderDirection::BUY, DEC(1.0), 299_dec, DEC(0.5));
    check();
    EXPECT_EQ(levelAt(bids(), 299_dec)->volume(), DEC(1.5));

    for (const auto take : {DEC(0.1), DEC(0.3333), DEC(0.0001), DEC(0.7)}) {
        placeMarket(agent1, OrderDirection::SELL, take);
        check();
    }
    EXPECT_EQ(levelAt(bids(), 299_dec)->volume(), resting->totalVolume());

    placeMarket(agent2, OrderDirection::SELL, DEC(5.0));
    check();
    EXPECT_EQ(resting->volume(), 0_dec);
    EXPECT_EQ(levelAt(bids(), 299_dec)->volume(), 0_dec);
    EXPECT_EQ(bids().volume(), 0_dec);
}

//-------------------------------------------------------------------------

TEST_F(BookAggregateInvariantTest, RestoredRestingVolumeReturnsToTheLevel)
{
    const auto resting = placeLimit(agent4, OrderDirection::BUY, DEC(1.0), 299_dec);
    placeLimit(agent1, OrderDirection::SELL, DEC(0.4), 299_dec);
    check();
    EXPECT_EQ(resting->volume(), DEC(0.6));

    ASSERT_TRUE(book->restoreRestingOrderVolume(resting->id(), DEC(0.25)));
    check();
    EXPECT_EQ(resting->volume(), DEC(0.85));
    EXPECT_EQ(levelAt(bids(), 299_dec)->volume(), DEC(0.85));
    EXPECT_EQ(bids().volume(), DEC(0.85));

    EXPECT_FALSE(book->restoreRestingOrderVolume(resting->id(), 0_dec));
    EXPECT_FALSE(book->restoreRestingOrderVolume(OrderID{424242}, DEC(0.1)));
    check();

    const auto leveraged = placeLimit(agent3, OrderDirection::BUY, DEC(1.0), 298_dec, DEC(1.0));
    check();
    ASSERT_TRUE(book->restoreRestingOrderVolume(leveraged->id(), DEC(0.5)));
    check();
    EXPECT_EQ(leveraged->totalVolume(), DEC(3.0));
    EXPECT_EQ(levelAt(bids(), 298_dec)->volume(), DEC(3.0));
}

//-------------------------------------------------------------------------

TEST_F(BookAggregateInvariantTest, SelfTradePreventionCancelsKeepAggregates)
{
    const auto own = placeLimit(agent4, OrderDirection::BUY, DEC(1.0), 299_dec);
    placeLimit(agent3, OrderDirection::BUY, DEC(1.0), 299_dec);
    check();

    // CO: the resting own order is cancelled, the new one trades with the other agent's order.
    const auto crossing =
        placeLimit(agent4, OrderDirection::SELL, DEC(0.5), 299_dec, 0_dec, STPFlag::CO);
    check();
    EXPECT_EQ(crossing->volume(), 0_dec);
    EXPECT_EQ(book->orderIdMap().count(own->id()), 0uz);
    EXPECT_EQ(levelAt(bids(), 299_dec)->volume(), DEC(0.5));
    EXPECT_EQ(bids().volume(), DEC(0.5));

    // CN: the incoming order is dropped, the resting own order survives untouched.
    const auto own2 = placeLimit(agent1, OrderDirection::SELL, DEC(1.0), 301_dec);
    check();
    const auto dropped =
        placeLimit(agent1, OrderDirection::BUY, DEC(0.5), 301_dec, 0_dec, STPFlag::CN);
    check();
    EXPECT_EQ(dropped->volume(), 0_dec);
    EXPECT_EQ(own2->volume(), DEC(1.0));
    EXPECT_EQ(levelAt(asks(), 301_dec)->volume(), DEC(1.0));
    EXPECT_EQ(book->orderIdMap().count(dropped->id()), 0uz);
}

//-------------------------------------------------------------------------

TEST_F(BookAggregateInvariantTest, GhostOnlySideBehavesAsEmptyForNewOrders)
{
    const auto ask = placeLimit(agent4, OrderDirection::SELL, DEC(1.0), 301_dec);
    placeMarket(agent1, OrderDirection::BUY, DEC(1.0));
    check();
    ASSERT_EQ(ask->volume(), 0_dec);
    ASSERT_EQ(asks().size(), 1uz);
    ASSERT_FALSE(book->bestSellLevel().has_value());

    // Nothing to trade against: a market buy is refused like on an empty side...
    EXPECT_EQ(tryPlaceMarket(agent2, OrderDirection::BUY, DEC(0.5)), nullptr);
    EXPECT_EQ(tryPlaceMarket(agent2, OrderDirection::BUY, DEC(0.5), DEC(1.0)), nullptr);
    check();

    // ...while a post-only bid cannot cross anything and must be accepted and rest.
    const auto postOnly = placeLimit(
        agent2, OrderDirection::BUY, DEC(0.5), 300_dec, 0_dec, STPFlag::CN, true);
    check();
    EXPECT_EQ(postOnly->volume(), DEC(0.5));
    EXPECT_EQ(levelAt(bids(), 300_dec)->volume(), DEC(0.5));
    EXPECT_EQ(book->bestBid(), 300_dec);
}

//-------------------------------------------------------------------------

TEST_F(BookAggregateInvariantTest, RandomizedOperationsKeepAggregates)
{
    std::mt19937 rng{20260827};
    std::uniform_int_distribution<int> percent{0, 99};
    std::uniform_int_distribution<int> priceTicks{-12, 12};    // 297.00 .. 303.00 in 0.25 steps
    std::uniform_int_distribution<int> volumeTicks{1, 300};    // 0.01 .. 3.00
    std::uniform_int_distribution<size_t> agentIndex{0, kAgents.size() - 1};
    std::uniform_int_distribution<int> leveragePick{0, 9};

    auto randomPrice = [&] { return 300_dec + decimal_t{priceTicks(rng)} * DEC(0.25); };
    auto randomVolume = [&] { return decimal_t{volumeTicks(rng)} * DEC(0.01); };
    auto randomLeverage = [&] {
        const auto pick = leveragePick(rng);
        return pick < 6 ? 0_dec : pick < 8 ? DEC(1.0) : pick < 9 ? DEC(0.5) : DEC(2.0);
    };
    auto randomDirection = [&] {
        return percent(rng) < 50 ? OrderDirection::BUY : OrderDirection::SELL;
    };
    auto pickIndex = [&](size_t size) {
        return std::uniform_int_distribution<size_t>{0, size - 1}(rng);
    };
    auto liveOrders = [&] {
        std::vector<LimitOrder::Ptr> live;
        for (const auto& [id, order] : book->orderIdMap()) {
            if (order->volume() > 0_dec) {
                live.push_back(order);
            }
        }
        return live;
    };

    auto directionName = [](OrderDirection direction) {
        return direction == OrderDirection::BUY ? "BUY" : "SELL";
    };

    size_t placed{}, rejected{}, partialCancels{}, fullCancels{}, markets{}, cleared{};
    for (int step = 0; step < 600; ++step) {
        const auto agentId = kAgents[agentIndex(rng)];
        const auto action = percent(rng);
        if (action < 55) {
            const auto direction = randomDirection();
            const auto volume = randomVolume();
            const auto price = randomPrice();
            const auto leverage = randomLeverage();
            SCOPED_TRACE(fmt::format(
                "step {}: agent {} limit {} {}@{} x{}",
                step, agentId, directionName(direction), volume, price, leverage));
            if (tryPlaceLimit(agentId, direction, volume, price, leverage)) {
                ++placed;
            } else {
                ++rejected;
            }
            check();
        } else if (action < 70) {
            const auto live = liveOrders();
            if (!live.empty()) {
                const auto& order = live[pickIndex(live.size())];
                const auto part = util::round(order->volume() * DEC(0.5), kVolumeDecimals);
                SCOPED_TRACE(fmt::format(
                    "step {}: partial cancel {} of order {} ({}@{} x{})",
                    step, part, order->id(), order->volume(), order->price(), order->leverage()));
                if (part > 0_dec && part < order->volume()) {
                    ASSERT_TRUE(book->cancelOrder(order->id(), part));
                    ++partialCancels;
                }
                check();
            }
        } else if (action < 85) {
            const auto live = liveOrders();
            if (!live.empty()) {
                const auto& order = live[pickIndex(live.size())];
                SCOPED_TRACE(fmt::format(
                    "step {}: full cancel of order {} ({}@{} x{})",
                    step, order->id(), order->volume(), order->price(), order->leverage()));
                ASSERT_TRUE(book->cancelOrder(order->id()));
                ++fullCancels;
                check();
            }
        } else if (action < 95) {
            const auto direction = randomDirection();
            const auto volume = randomVolume();
            const auto leverage = randomLeverage();
            SCOPED_TRACE(fmt::format(
                "step {}: agent {} market {} {} x{}",
                step, agentId, directionName(direction), volume, leverage));
            if (tryPlaceMarket(agentId, direction, volume, leverage)) {
                ++markets;
            } else {
                ++rejected;
            }
            check();
        } else {
            SCOPED_TRACE(fmt::format("step {}: clearFilledOrders", step));
            book->clearFilledOrders();
            ++cleared;
            check();
        }
    }

    // The walk must actually have exercised every path.
    EXPECT_GT(placed, 200uz);
    EXPECT_GT(partialCancels, 20uz);
    EXPECT_GT(fullCancels, 30uz);
    EXPECT_GT(markets, 15uz);
    EXPECT_GT(cleared, 10uz);
    EXPECT_LT(rejected, placed / 2) << "too many rejections for the walk to be meaningful";

    book->clearFilledOrders();
    check();
    EXPECT_EQ(ghostCount(*book), 0uz);
}

//-------------------------------------------------------------------------
