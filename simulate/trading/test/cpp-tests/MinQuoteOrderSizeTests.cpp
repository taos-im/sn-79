/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 *
 * The engine must not accept an order the chain will refuse to settle.
 *
 * minOrderSize is a BASE (alpha) floor in every branch of OrderPlacementValidator: a BASE-currency
 * order compares volume directly, and a QUOTE-currency order divides by price first, so it too becomes
 * an alpha comparison. The quote notional was never floored, while the chain constrains precisely that
 * side: SubtensorModule.InitialMinStake = 2000000 rao = 0.002 TAO on add_stake/remove_stake, read from
 * localnet metadata and matching the value the UI already enforces pre-submit (trade.js:1805,
 * nexus-modals.js:839).
 *
 * Live proof that the gap was reachable: order 218 on book 5, 0.0002 alpha @ 0.0001, a notional of
 * 2e-08 TAO, 100000x below the floor. Accepted at placement, rested on the book. Had it been hit,
 * settlement would have been attempted on chain at a size the chain rejects. It had stayed latent only
 * because the UI gate carried the constraint the engine lacked, so orders arriving via the miner SDK or
 * API were ungated.
 *
 * OrderPlacementValidator is shared with the simulation, so the floor is inert unless declared. The
 * pair of fixtures below differ in exactly one attribute, which is what makes the inertness assertions
 * evidence about simulation mode rather than about one config file.
 *
 * The floor is also scoped to REMOTE agents. run/config/exchange_0.xml instantiates 2600+ local
 * background agents (1024 NoiseTrader, 1000 StylizedTrader, 500 Initialization, 100 FuturesTrader,
 * 10 HFT, 1 ALGOTrader) whose orders are synthetic liquidity inside the engine and never settle on
 * chain. Applying a chain-derived minimum to them would thin the book for a constraint they are not
 * subject to, so only agentId >= 0 is floored.
 */

#include "taosim/decimal/decimal.hpp"
#include "taosim/matching/FeePolicy.hpp"
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
#include <memory>
#include <utility>

//-------------------------------------------------------------------------

using namespace taosim;
using namespace taosim::accounting;
using namespace taosim::book;
using namespace taosim::literals;

using namespace testing;

namespace fs = std::filesystem;

//-------------------------------------------------------------------------

static const auto s_testDataPath = fs::path{__FILE__}.parent_path() / "data";

// The chain's floor, and the value the UI enforces. Stated once so a drift breaks one line.
static const decimal_t s_minQuoteOrderSize = DEC(0.002);

//-------------------------------------------------------------------------

template<typename... Args>
requires std::constructible_from<PlaceOrderLimitPayload, Args..., BookId>
[[nodiscard]] static std::pair<LimitOrder::Ptr, OrderErrorCode> placeLimit(
    MultiBookExchangeAgent* exchange, AgentId agentId, BookId bookId, Currency currency, Args&&... args)
{
    const auto payload = MessagePayload::create<PlaceOrderLimitPayload>(
        std::forward<Args>(args)...,
        bookId,
        currency,
        std::nullopt,
        false,
        taosim::TimeInForce::GTC,
        std::nullopt,
        STPFlag::CO);
    const auto orderResult = exchange->clearingManager().handleOrder(
        LimitOrderDesc{.agentId = agentId, .payload = payload});
    // A rejected order must never reach the book, so only place when the validator accepted it.
    LimitOrder::Ptr order;
    if (orderResult.ec == OrderErrorCode::VALID) {
        order = exchange->books()[bookId]->placeLimitOrder(
            OrderClientContext{agentId},
            Timestamp{},
            orderResult.orderSize,
            payload->direction,
            payload->price,
            payload->leverage,
            payload->stpFlag);
    }
    return {order, orderResult.ec};
}

template<typename... Args>
requires std::constructible_from<PlaceOrderMarketPayload, Args..., BookId>
[[nodiscard]] static std::pair<MarketOrder::Ptr, OrderErrorCode> placeMarket(
    MultiBookExchangeAgent* exchange, AgentId agentId, BookId bookId, Currency currency, Args&&... args)
{
    const auto payload = MessagePayload::create<PlaceOrderMarketPayload>(
        std::forward<Args>(args)..., bookId, currency, std::nullopt, STPFlag::CO);
    const auto orderResult = exchange->clearingManager().handleOrder(
        MarketOrderDesc{.agentId = agentId, .payload = payload});
    MarketOrder::Ptr order;
    if (orderResult.ec == OrderErrorCode::VALID) {
        order = exchange->books()[bookId]->placeMarketOrder(
            OrderClientContext{agentId},
            Timestamp{},
            orderResult.orderSize,
            payload->direction,
            payload->leverage,
            payload->stpFlag);
    }
    return {order, orderResult.ec};
}

//-------------------------------------------------------------------------

class QuoteFloorTestBase : public testing::Test
{
public:
    // The floor applies to REMOTE agents only, because only their fills settle on chain. The test
    // configs declare remoteAgentCount="2", so 0 and 1 are remote; negative ids are local.
    const AgentId remoteAgent = 0;
    const AgentId localAgent = -1;
    const AgentId maker = -2;
    const BookId bookId{};

    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange{};
    Book::Ptr book;

    void configureFrom(std::string_view configName)
    {
        nodes = taosim::util::parseSimulationFile(s_testDataPath / configName);
        simulation = std::make_unique<Simulation>();
        simulation->setDebug(false);
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        book = exchange->books()[bookId];
        exchange->accounts().registerLocal("agent1");
        exchange->accounts().registerLocal("agent2");
    }

    // Resting liquidity for the market-order paths to walk. Asserted rather than discarded: a seed that
    // silently failed would leave an empty book, and a market order against an empty book is refused for
    // an unrelated reason, so every market test below would pass without testing anything.
    void seedAt(decimal_t bid, decimal_t ask)
    {
        const auto b = placeLimit(exchange, maker, bookId, Currency::BASE,
            OrderDirection::BUY, 5_dec, bid, DEC(0.));
        ASSERT_EQ(b.second, OrderErrorCode::VALID);
        const auto a = placeLimit(exchange, maker, bookId, Currency::BASE,
            OrderDirection::SELL, 5_dec, ask, DEC(0.));
        ASSERT_EQ(a.second, OrderErrorCode::VALID);
        ASSERT_FALSE(book->buyQueue().empty());
        ASSERT_FALSE(book->sellQueue().empty());
    }

    void seedBook() { seedAt(DEC(299.0), DEC(301.0)); }

    // Liquidity priced near unity. Every rejection case below must clear the BASE floor so that only
    // the quote floor can decide it, otherwise the test passes for the wrong reason and says nothing
    // about the defect. That needs price < minQuoteOrderSize / minOrderSize = 0.002 / 0.0001 = 20
    // quote per base, which the 299/301 book does not satisfy.
    void seedCheapBook() { seedAt(DEC(0.99), DEC(1.0)); }
};

class QuoteFloorSet : public QuoteFloorTestBase
{
protected:
    void SetUp() override { configureFrom("MinQuoteOrderSize.xml"); }
};

class QuoteFloorUnset : public QuoteFloorTestBase
{
protected:
    void SetUp() override { configureFrom("MinQuoteOrderSizeUnset.xml"); }
};

// Base floor lifted clear of the volume grid so it can reject on its own. See the config's comment.
class QuoteFloorBothFloors : public QuoteFloorTestBase
{
protected:
    void SetUp() override { configureFrom("MinQuoteOrderSizeBothFloors.xml"); }
};

//-------------------------------------------------------------------------

TEST_F(QuoteFloorSet, ConfigCarriesTheDeclaredFloor)
{
    EXPECT_EQ(exchange->config2().minQuoteOrderSize, s_minQuoteOrderSize);
    // The base floor must be untouched by the new attribute.
    EXPECT_EQ(exchange->config2().minOrderSize, DEC(0.0001));
}

TEST_F(QuoteFloorUnset, AnUndeclaredFloorIsZeroSoTheCheckCannotFire)
{
    EXPECT_EQ(exchange->config2().minQuoteOrderSize, 0_dec);
    EXPECT_EQ(exchange->config2().minOrderSize, DEC(0.0001));
}

//-------------------------------------------------------------------------

TEST_F(QuoteFloorSet, TheLiveCaseIsRejected)
{
    // Order 218 exactly: 0.0002 alpha @ 0.0001 = 2e-08 TAO.
    const auto [order, ec] = placeLimit(exchange, remoteAgent, bookId, Currency::BASE,
        OrderDirection::BUY, DEC(0.0002), DEC(0.0001), DEC(0.));
    EXPECT_EQ(ec, OrderErrorCode::MINIMUM_ORDER_SIZE_VIOLATION);
    EXPECT_EQ(order, nullptr);
    EXPECT_TRUE(book->buyQueue().empty());
}

TEST_F(QuoteFloorSet, ALocalAgentIsNotFlooredBecauseItsFillsNeverReachTheChain)
{
    // The scoping guard. The same order that TheLiveCaseIsRejected refuses for a remote agent must be
    // accepted for a local one: background agents supply liquidity inside the engine and are not
    // subject to a chain minimum. Without this, enabling the floor would thin the book on every one of
    // the 2600+ local agents in the exchange config.
    const auto [order, ec] = placeLimit(exchange, localAgent, bookId, Currency::BASE,
        OrderDirection::BUY, DEC(0.0002), DEC(0.0001), DEC(0.));
    EXPECT_EQ(ec, OrderErrorCode::VALID);
    EXPECT_NE(order, nullptr);
}

TEST_F(QuoteFloorSet, ALocalAgentsMarketOrderIsNotFlooredEither)
{
    seedCheapBook();
    const auto [order, ec] = placeMarket(exchange, localAgent, bookId, Currency::BASE,
        OrderDirection::BUY, DEC(0.001), DEC(0.));
    EXPECT_EQ(ec, OrderErrorCode::VALID);
    EXPECT_NE(order, nullptr);
}

TEST_F(QuoteFloorUnset, TheLiveCaseIsStillAcceptedWhenNoFloorIsDeclared)
{
    // The regression guard: simulation configs declare no floor, so this order must behave exactly as
    // it did before the check existed. Its base volume 0.0002 clears minOrderSize 0.0001.
    const auto [order, ec] = placeLimit(exchange, remoteAgent, bookId, Currency::BASE,
        OrderDirection::BUY, DEC(0.0002), DEC(0.0001), DEC(0.));
    EXPECT_EQ(ec, OrderErrorCode::VALID);
    EXPECT_NE(order, nullptr);
}

//-------------------------------------------------------------------------

TEST_F(QuoteFloorSet, ANotionalJustBelowTheFloorIsRejected)
{
    // 0.0066 * 0.3 = 0.00198 TAO, one grid step below 0.002.
    const auto [order, ec] = placeLimit(exchange, remoteAgent, bookId, Currency::BASE,
        OrderDirection::BUY, DEC(0.0066), DEC(0.3), DEC(0.));
    EXPECT_EQ(ec, OrderErrorCode::MINIMUM_ORDER_SIZE_VIOLATION);
    EXPECT_EQ(order, nullptr);
}

TEST_F(QuoteFloorSet, ANotionalExactlyAtTheFloorIsAccepted)
{
    // 0.02 * 0.1 = 0.002 TAO exactly. The boundary must be inclusive: the chain rejects BELOW the
    // minimum, so rejecting AT it would refuse an order the chain would settle.
    const auto [order, ec] = placeLimit(exchange, remoteAgent, bookId, Currency::BASE,
        OrderDirection::BUY, DEC(0.02), DEC(0.1), DEC(0.));
    EXPECT_EQ(ec, OrderErrorCode::VALID);
    EXPECT_NE(order, nullptr);
}

TEST_F(QuoteFloorSet, AnOrdinaryOrderIsUnaffected)
{
    const auto [order, ec] = placeLimit(exchange, remoteAgent, bookId, Currency::BASE,
        OrderDirection::BUY, 1_dec, DEC(299.0), DEC(0.));
    EXPECT_EQ(ec, OrderErrorCode::VALID);
    EXPECT_NE(order, nullptr);
}

//-------------------------------------------------------------------------

TEST_F(QuoteFloorSet, AQuoteCurrencyOrderIsFlooredOnItsOwnVolume)
{
    // For a QUOTE-currency order the volume IS the TAO amount. Priced at unity so the implied base
    // 0.0015 clears minOrderSize and only the quote floor can reject this.
    const auto [order, ec] = placeLimit(exchange, remoteAgent, bookId, Currency::QUOTE,
        OrderDirection::BUY, DEC(0.0015), DEC(1.0), DEC(0.));
    EXPECT_EQ(ec, OrderErrorCode::MINIMUM_ORDER_SIZE_VIOLATION);
    EXPECT_EQ(order, nullptr);
}

TEST_F(QuoteFloorSet, AQuoteCurrencyOrderAtTheFloorIsAccepted)
{
    const auto [order, ec] = placeLimit(exchange, remoteAgent, bookId, Currency::QUOTE,
        OrderDirection::BUY, DEC(0.002), DEC(1.0), DEC(0.));
    EXPECT_EQ(ec, OrderErrorCode::VALID);
    EXPECT_NE(order, nullptr);
}

//-------------------------------------------------------------------------

TEST_F(QuoteFloorSet, AMarketBuyBelowTheFloorIsRejected)
{
    seedCheapBook();
    // 0.001 base against the 1.0 ask is 0.001 TAO, below the floor, while the base volume 0.001 is ten
    // times minOrderSize. So only the quote floor can reject this.
    const auto [order, ec] = placeMarket(exchange, remoteAgent, bookId, Currency::BASE,
        OrderDirection::BUY, DEC(0.001), DEC(0.));
    EXPECT_EQ(ec, OrderErrorCode::MINIMUM_ORDER_SIZE_VIOLATION);
    EXPECT_EQ(order, nullptr);
}

TEST_F(QuoteFloorSet, AQuoteCurrencyMarketBuyBelowTheFloorIsRejected)
{
    seedCheapBook();
    // A QUOTE-currency market order states its size in TAO directly. At the 1.0 ask the implied base is
    // 0.0019, well over minOrderSize, so this too is decided by the quote floor alone.
    const auto [order, ec] = placeMarket(exchange, remoteAgent, bookId, Currency::QUOTE,
        OrderDirection::BUY, DEC(0.0019), DEC(0.));
    EXPECT_EQ(ec, OrderErrorCode::MINIMUM_ORDER_SIZE_VIOLATION);
    EXPECT_EQ(order, nullptr);
}

TEST_F(QuoteFloorSet, AMarketSellBelowTheFloorIsRejected)
{
    // The sell path is a separate branch of the validator (line 231), so it needs its own assertion.
    seedCheapBook();
    const auto [order, ec] = placeMarket(exchange, remoteAgent, bookId, Currency::BASE,
        OrderDirection::SELL, DEC(0.001), DEC(0.));
    EXPECT_EQ(ec, OrderErrorCode::MINIMUM_ORDER_SIZE_VIOLATION);
    EXPECT_EQ(order, nullptr);
}

TEST_F(QuoteFloorSet, AnOrdinaryMarketBuyIsUnaffected)
{
    seedBook();
    const auto [order, ec] = placeMarket(exchange, remoteAgent, bookId, Currency::BASE,
        OrderDirection::BUY, 1_dec, DEC(0.));
    EXPECT_EQ(ec, OrderErrorCode::VALID);
    EXPECT_NE(order, nullptr);
}

TEST_F(QuoteFloorUnset, AMarketBuyBelowTheChainFloorIsStillAcceptedWhenUndeclared)
{
    seedCheapBook();
    const auto [order, ec] = placeMarket(exchange, remoteAgent, bookId, Currency::BASE,
        OrderDirection::BUY, DEC(0.001), DEC(0.));
    EXPECT_EQ(ec, OrderErrorCode::VALID);
    EXPECT_NE(order, nullptr);
}

//-------------------------------------------------------------------------

TEST_F(QuoteFloorBothFloors, TheBaseFloorStillRejectsIndependentlyOfTheQuoteFloor)
{
    // 0.2 alpha @ 299.0 is a 59.8 TAO notional, far clear of the quote floor, while 0.2 is under the
    // 0.25 alpha floor. Only the alpha check can reject this, so the new check cannot be mistaken for
    // having replaced the old one.
    const auto [order, ec] = placeLimit(exchange, remoteAgent, bookId, Currency::BASE,
        OrderDirection::BUY, DEC(0.2), DEC(299.0), DEC(0.));
    EXPECT_EQ(ec, OrderErrorCode::MINIMUM_ORDER_SIZE_VIOLATION);
    EXPECT_EQ(order, nullptr);
}

TEST_F(QuoteFloorBothFloors, BothFloorsClearedIsAccepted)
{
    // The complement: over both floors, so neither check fires.
    const auto [order, ec] = placeLimit(exchange, remoteAgent, bookId, Currency::BASE,
        OrderDirection::BUY, DEC(0.3), DEC(299.0), DEC(0.));
    EXPECT_EQ(ec, OrderErrorCode::VALID);
    EXPECT_NE(order, nullptr);
}

TEST_F(QuoteFloorBothFloors, AtProductionFloorsTheAlphaFloorBindsFirstOnACheapBook)
{
    // Why minOrderSize is set to 0.25 rather than left at the grid quantum: on a book pricing alpha near
    // 0.0136 TAO the quote floor alone would demand 0.002/0.0136 = 0.147 alpha, so a miner sizing orders
    // in alpha would meet a limit that moves with price. At 0.25 the alpha floor binds first and the
    // quote floor is a backstop. 0.2 alpha @ 0.0136 fails both; 0.3 fails neither.
    const auto rejected = placeLimit(exchange, remoteAgent, bookId, Currency::BASE,
        OrderDirection::BUY, DEC(0.2), DEC(0.0136), DEC(0.));
    EXPECT_EQ(rejected.second, OrderErrorCode::MINIMUM_ORDER_SIZE_VIOLATION);
    const auto accepted = placeLimit(exchange, remoteAgent, bookId, Currency::BASE,
        OrderDirection::BUY, DEC(0.3), DEC(0.0136), DEC(0.));
    EXPECT_EQ(accepted.second, OrderErrorCode::VALID);
    EXPECT_NE(accepted.first, nullptr);
}

//-------------------------------------------------------------------------
