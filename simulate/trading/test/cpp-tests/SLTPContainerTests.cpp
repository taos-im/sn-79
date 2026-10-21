/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

#include "formatting.hpp"
#include "Order.hpp"
#include <taosim/decimal/decimal.hpp>
#include <taosim/matching/SLTPContainer.hpp>

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

//-------------------------------------------------------------------------

using namespace taosim;
using namespace taosim::matching;
using namespace taosim::literals;

using namespace testing;

//-------------------------------------------------------------------------

namespace
{

//-------------------------------------------------------------------------
// Helpers
//-------------------------------------------------------------------------

struct DispatchRecord
{
    SLTPEntry entry;
    decimal_t observedPrice;
};

struct SLTPSetup
{
    SLTPContainer container;
    std::vector<DispatchRecord> dispatched;

    void setUp(size_t bookCount = 1)
    {
        container.resize(bookCount);
        container.setDispatch(
            [this](const SLTPEntry& entry, decimal_t observedPrice) {
                dispatched.push_back({entry, observedPrice});
            });
    }
};

//-------------------------------------------------------------------------
// onOrderCreated / onOrderTrade — basic trigger insertion
//-------------------------------------------------------------------------

struct TradeInsertionTestParams
{
    OrderDirection originatingSide{OrderDirection::BUY};
    std::optional<decimal_t> stopLoss;
    std::optional<decimal_t> takeProfit;
    decimal_t fillPrice{};
    decimal_t filledVolume{};
    size_t expectedTriggerCount{};
};

void PrintTo(const TradeInsertionTestParams& p, std::ostream* os)
{
    *os << fmt::format(
        "{{side = {}, SL = {}, TP = {}, fillPrice = {}, fillVol = {}, expected = {}}}",
        p.originatingSide == OrderDirection::BUY ? "BUY" : "SELL",
        p.stopLoss ? fmt::format("{}", *p.stopLoss) : "none",
        p.takeProfit ? fmt::format("{}", *p.takeProfit) : "none",
        p.fillPrice,
        p.filledVolume,
        p.expectedTriggerCount);
}

struct TradeInsertionTest : TestWithParam<TradeInsertionTestParams>
{
    void SetUp() override
    {
        setup.setUp();
        params = GetParam();
    }

    SLTPSetup setup;
    TradeInsertionTestParams params;
};

TEST_P(TradeInsertionTest, InsertsCorrectTriggerCount)
{
    constexpr BookId bookId{};
    constexpr OrderID orderId{42};

    setup.container.onOrderCreated({
        .orderId = orderId,
        .bookId = bookId,
        .agentId = -1,
        .originatingSide = params.originatingSide,
        .volume = params.filledVolume,
        .leverage = 0_dec,
        .stopLoss = params.stopLoss,
        .takeProfit = params.takeProfit
    });

    setup.container.onOrderTrade({
        .bookId = bookId,
        .originatingOrderId = orderId,
        .agentId = -1,
        .side = params.originatingSide,
        .fillPrice = params.fillPrice,
        .filledVolume = params.filledVolume
    });

    EXPECT_EQ(setup.container.books()[bookId].triggers.size(), params.expectedTriggerCount);
    // Order fully filled — orderInfo should be purged
    EXPECT_EQ(setup.container.books()[bookId].orderInfo.count(orderId), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    SLTPContainer,
    TradeInsertionTest,
    Values(
        // BUY with SL only — 1 trigger
        TradeInsertionTestParams{
            .originatingSide = OrderDirection::BUY,
            .stopLoss = DEC(-0.05),
            .fillPrice = 100_dec,
            .filledVolume = 5_dec,
            .expectedTriggerCount = 1
        },
        // BUY with TP only — 1 trigger
        TradeInsertionTestParams{
            .originatingSide = OrderDirection::BUY,
            .takeProfit = DEC(0.10),
            .fillPrice = 100_dec,
            .filledVolume = 5_dec,
            .expectedTriggerCount = 1
        },
        // BUY with both SL + TP — 2 triggers
        TradeInsertionTestParams{
            .originatingSide = OrderDirection::BUY,
            .stopLoss = DEC(-0.05),
            .takeProfit = DEC(0.10),
            .fillPrice = 100_dec,
            .filledVolume = 5_dec,
            .expectedTriggerCount = 2
        },
        // SELL with both SL + TP — 2 triggers
        TradeInsertionTestParams{
            .originatingSide = OrderDirection::SELL,
            .stopLoss = DEC(0.05),
            .takeProfit = DEC(-0.10),
            .fillPrice = 100_dec,
            .filledVolume = 5_dec,
            .expectedTriggerCount = 2
        },
        // No SL/TP — onOrderCreated is a no-op, 0 triggers
        TradeInsertionTestParams{
            .fillPrice = 100_dec,
            .filledVolume = 5_dec,
            .expectedTriggerCount = 0
        }));

//-------------------------------------------------------------------------
// onPriceUpdate — trigger firing and cross direction
//-------------------------------------------------------------------------

struct PriceUpdateTestParams
{
    OrderDirection originatingSide{OrderDirection::BUY};
    std::optional<decimal_t> stopLoss;
    std::optional<decimal_t> takeProfit;
    decimal_t fillPrice{};
    decimal_t filledVolume{};
    decimal_t seedPrice{};   // first price update (sets lastPrice, no firing)
    decimal_t crossPrice{};  // second price update (may fire)
    size_t expectedDispatches{};
};

void PrintTo(const PriceUpdateTestParams& p, std::ostream* os)
{
    *os << fmt::format(
        "{{side = {}, SL = {}, TP = {}, fill = {}, seed = {}, cross = {}, dispatches = {}}}",
        p.originatingSide == OrderDirection::BUY ? "BUY" : "SELL",
        p.stopLoss ? fmt::format("{}", *p.stopLoss) : "none",
        p.takeProfit ? fmt::format("{}", *p.takeProfit) : "none",
        p.fillPrice,
        p.seedPrice,
        p.crossPrice,
        p.expectedDispatches);
}

struct PriceUpdateTest : TestWithParam<PriceUpdateTestParams>
{
    void SetUp() override
    {
        setup.setUp();
        params = GetParam();
    }

    SLTPSetup setup;
    PriceUpdateTestParams params;
};

TEST_P(PriceUpdateTest, FiresCorrectly)
{
    constexpr BookId bookId{};
    constexpr OrderID orderId{1};

    setup.container.onOrderCreated({
        .orderId = orderId,
        .bookId = bookId,
        .agentId = -1,
        .originatingSide = params.originatingSide,
        .volume = params.filledVolume,
        .leverage = 0_dec,
        .stopLoss = params.stopLoss,
        .takeProfit = params.takeProfit
    });

    setup.container.onOrderTrade({
        .bookId = bookId,
        .originatingOrderId = orderId,
        .agentId = -1,
        .side = params.originatingSide,
        .fillPrice = params.fillPrice,
        .filledVolume = params.filledVolume
    });

    // Seed the lastPrice (first update never fires)
    setup.container.onPriceUpdate(bookId, params.seedPrice);
    EXPECT_TRUE(setup.dispatched.empty());

    // Cross
    setup.container.onPriceUpdate(bookId, params.crossPrice);
    EXPECT_EQ(setup.dispatched.size(), params.expectedDispatches);
}

INSTANTIATE_TEST_SUITE_P(
    SLTPContainer,
    PriceUpdateTest,
    Values(
        // Long SL: fill@100, SL=0.95 → trigger@95, seed@100, drop to 94 → fires
        PriceUpdateTestParams{
            .originatingSide = OrderDirection::BUY,
            .stopLoss = DEC(-0.05),
            .fillPrice = 100_dec,
            .filledVolume = 5_dec,
            .seedPrice = 100_dec,
            .crossPrice = 94_dec,
            .expectedDispatches = 1
        },
        // Long SL: seed@100, price rises to 106 — does not fire (wrong direction)
        PriceUpdateTestParams{
            .originatingSide = OrderDirection::BUY,
            .stopLoss = DEC(-0.05),
            .fillPrice = 100_dec,
            .filledVolume = 5_dec,
            .seedPrice = 100_dec,
            .crossPrice = 106_dec,
            .expectedDispatches = 0
        },
        // Long TP: fill@100, TP=1.10 → trigger@110, seed@100, rise to 112 → fires
        PriceUpdateTestParams{
            .originatingSide = OrderDirection::BUY,
            .takeProfit = DEC(0.10),
            .fillPrice = 100_dec,
            .filledVolume = 5_dec,
            .seedPrice = 100_dec,
            .crossPrice = 112_dec,
            .expectedDispatches = 1
        },
        // Long TP: seed@100, drop to 90 — does not fire (wrong direction)
        PriceUpdateTestParams{
            .originatingSide = OrderDirection::BUY,
            .takeProfit = DEC(0.10),
            .fillPrice = 100_dec,
            .filledVolume = 5_dec,
            .seedPrice = 100_dec,
            .crossPrice = 90_dec,
            .expectedDispatches = 0
        },
        // Short SL: fill@100, SL=1.05 → trigger@105, seed@100, rise to 106 → fires
        PriceUpdateTestParams{
            .originatingSide = OrderDirection::SELL,
            .stopLoss = DEC(0.05),
            .fillPrice = 100_dec,
            .filledVolume = 5_dec,
            .seedPrice = 100_dec,
            .crossPrice = 106_dec,
            .expectedDispatches = 1
        },
        // Short TP: fill@100, TP=0.90 → trigger@90, seed@100, drop to 89 → fires
        PriceUpdateTestParams{
            .originatingSide = OrderDirection::SELL,
            .takeProfit = DEC(-0.10),
            .fillPrice = 100_dec,
            .filledVolume = 5_dec,
            .seedPrice = 100_dec,
            .crossPrice = 89_dec,
            .expectedDispatches = 1
        },
        // Long SL+TP: seed@100, crash to 80 — only SL fires (TP is above, wrong dir)
        PriceUpdateTestParams{
            .originatingSide = OrderDirection::BUY,
            .stopLoss = DEC(-0.05),
            .takeProfit = DEC(0.10),
            .fillPrice = 100_dec,
            .filledVolume = 5_dec,
            .seedPrice = 100_dec,
            .crossPrice = 80_dec,
            .expectedDispatches = 1
        },
        // Long SL+TP: seed@100, moon to 120 — only TP fires
        PriceUpdateTestParams{
            .originatingSide = OrderDirection::BUY,
            .stopLoss = DEC(-0.05),
            .takeProfit = DEC(0.10),
            .fillPrice = 100_dec,
            .filledVolume = 5_dec,
            .seedPrice = 100_dec,
            .crossPrice = 120_dec,
            .expectedDispatches = 1
        }));

//-------------------------------------------------------------------------
// Dispatch payload correctness
//-------------------------------------------------------------------------

TEST(SLTPContainerTest, DispatchPayloadIsCorrect)
{
    SLTPSetup setup;
    setup.setUp();
    constexpr BookId bookId{};
    constexpr OrderID orderId{7};
    constexpr AgentId agentId{-1};

    setup.container.onOrderCreated({
        .orderId = orderId,
        .bookId = bookId,
        .agentId = agentId,
        .originatingSide = OrderDirection::BUY,
        .volume = 10_dec,
        .leverage = DEC(0.5),
        .currency = Currency::BASE,
        .stopLoss = DEC(-0.05)
    });

    setup.container.onOrderTrade({
        .bookId = bookId,
        .originatingOrderId = orderId,
        .agentId = agentId,
        .side = OrderDirection::BUY,
        .fillPrice = 200_dec,
        .filledVolume = 10_dec
    });

    // Seed + cross
    setup.container.onPriceUpdate(bookId, 200_dec);
    setup.container.onPriceUpdate(bookId, 180_dec);

    ASSERT_EQ(setup.dispatched.size(), 1u);
    const auto& d = setup.dispatched[0];
    EXPECT_EQ(d.entry.originatingOrderId, orderId);
    EXPECT_EQ(d.entry.agentId, agentId);
    EXPECT_EQ(d.entry.bookId, bookId);
    EXPECT_EQ(d.entry.closingSide, OrderDirection::SELL);
    EXPECT_EQ(d.entry.volume, 10_dec);
    EXPECT_EQ(d.entry.leverage, DEC(0.5));
    EXPECT_EQ(d.entry.currency, Currency::BASE);
    EXPECT_EQ(d.entry.triggerPrice, 190_dec);  // 200 * 0.95
    EXPECT_EQ(d.observedPrice, 180_dec);
}

//-------------------------------------------------------------------------
// Partial fills create independent triggers at different prices
//-------------------------------------------------------------------------

TEST(SLTPContainerTest, PartialFillsCreateSeparateTriggers)
{
    SLTPSetup setup;
    setup.setUp();
    constexpr BookId bookId{};
    constexpr OrderID orderId{10};

    setup.container.onOrderCreated({
        .orderId = orderId,
        .bookId = bookId,
        .agentId = -1,
        .originatingSide = OrderDirection::BUY,
        .volume = 8_dec,
        .leverage = 0_dec,
        .stopLoss = DEC(-0.10)
    });

    // Fill 1: 3@100 → SL trigger at 90
    setup.container.onOrderTrade({
        .bookId = bookId,
        .originatingOrderId = orderId,
        .agentId = -1,
        .side = OrderDirection::BUY,
        .fillPrice = 100_dec,
        .filledVolume = 3_dec
    });
    // orderInfo should still exist (5 remaining)
    EXPECT_EQ(setup.container.books()[bookId].orderInfo.count(orderId), 1u);

    // Fill 2: 5@110 → SL trigger at 99
    setup.container.onOrderTrade({
        .bookId = bookId,
        .originatingOrderId = orderId,
        .agentId = -1,
        .side = OrderDirection::BUY,
        .fillPrice = 110_dec,
        .filledVolume = 5_dec
    });
    // orderInfo purged (0 remaining)
    EXPECT_EQ(setup.container.books()[bookId].orderInfo.count(orderId), 0u);

    // Two independent triggers: 90 (vol 3) and 99 (vol 5)
    EXPECT_EQ(setup.container.books()[bookId].triggers.size(), 2u);

    // Seed at 105, drop to 95 — crosses trigger@99 but not trigger@90
    setup.container.onPriceUpdate(bookId, 105_dec);
    setup.container.onPriceUpdate(bookId, 95_dec);
    ASSERT_EQ(setup.dispatched.size(), 1u);
    EXPECT_EQ(setup.dispatched[0].entry.volume, 5_dec);

    // Continue drop to 85 — crosses trigger@90
    setup.container.onPriceUpdate(bookId, 85_dec);
    ASSERT_EQ(setup.dispatched.size(), 2u);
    EXPECT_EQ(setup.dispatched[1].entry.volume, 3_dec);
}

//-------------------------------------------------------------------------
// removeOrder — margin call cleanup
//-------------------------------------------------------------------------

TEST(SLTPContainerTest, RemoveOrderPurgesTriggersAndInfo)
{
    SLTPSetup setup;
    setup.setUp();
    constexpr BookId bookId{};
    constexpr OrderID orderId{20};

    setup.container.onOrderCreated({
        .orderId = orderId,
        .bookId = bookId,
        .agentId = -1,
        .originatingSide = OrderDirection::BUY,
        .volume = 5_dec,
        .leverage = 0_dec,
        .stopLoss = DEC(-0.05),
        .takeProfit = DEC(0.10)
    });

    setup.container.onOrderTrade({
        .bookId = bookId,
        .originatingOrderId = orderId,
        .agentId = -1,
        .side = OrderDirection::BUY,
        .fillPrice = 100_dec,
        .filledVolume = 5_dec
    });

    ASSERT_EQ(setup.container.books()[bookId].triggers.size(), 2u);

    // Simulate margin call
    setup.container.removeOrder(bookId, orderId);

    EXPECT_EQ(setup.container.books()[bookId].triggers.size(), 0u);
    EXPECT_EQ(setup.container.books()[bookId].orderInfo.count(orderId), 0u);

    // Price movement should not fire anything
    setup.container.onPriceUpdate(bookId, 100_dec);
    setup.container.onPriceUpdate(bookId, 80_dec);
    EXPECT_TRUE(setup.dispatched.empty());
}

//-------------------------------------------------------------------------
// removeOrder leaves other orders' triggers intact
//-------------------------------------------------------------------------

TEST(SLTPContainerTest, RemoveOrderLeavesOtherOrders)
{
    SLTPSetup setup;
    setup.setUp();
    constexpr BookId bookId{};
    constexpr OrderID orderA{30};
    constexpr OrderID orderB{31};

    for (auto id : {orderA, orderB}) {
        setup.container.onOrderCreated({
            .orderId = id,
            .bookId = bookId,
            .agentId = -1,
            .originatingSide = OrderDirection::BUY,
            .volume = 1_dec,
            .leverage = 0_dec,
            .stopLoss = DEC(-0.05)
        });
        setup.container.onOrderTrade({
            .bookId = bookId,
            .originatingOrderId = id,
            .agentId = -1,
            .side = OrderDirection::BUY,
            .fillPrice = 100_dec,
            .filledVolume = 1_dec
        });
    }

    ASSERT_EQ(setup.container.books()[bookId].triggers.size(), 2u);

    setup.container.removeOrder(bookId, orderA);

    EXPECT_EQ(setup.container.books()[bookId].triggers.size(), 1u);

    // orderB's trigger should still fire
    setup.container.onPriceUpdate(bookId, 100_dec);
    setup.container.onPriceUpdate(bookId, 90_dec);
    ASSERT_EQ(setup.dispatched.size(), 1u);
    EXPECT_EQ(setup.dispatched[0].entry.originatingOrderId, orderB);
}

//-------------------------------------------------------------------------
// No dispatch set — onPriceUpdate is a no-op
//-------------------------------------------------------------------------

TEST(SLTPContainerTest, NoPriceUpdateWithoutDispatch)
{
    SLTPContainer container;
    container.resize(1);
    // No setDispatch call

    container.onOrderCreated({
        .orderId = 1,
        .bookId = 0,
        .agentId = -1,
        .originatingSide = OrderDirection::BUY,
        .volume = 1_dec,
        .leverage = 0_dec,
        .stopLoss = DEC(-0.05)
    });

    container.onOrderTrade({
        .bookId = 0,
        .originatingOrderId = 1,
        .agentId = -1,
        .side = OrderDirection::BUY,
        .fillPrice = 100_dec,
        .filledVolume = 1_dec
    });

    // Should not crash
    container.onPriceUpdate(0, 100_dec);
    container.onPriceUpdate(0, 80_dec);

    // Trigger still present (never dispatched)
    EXPECT_EQ(container.books()[0].triggers.size(), 1u);
}

//-------------------------------------------------------------------------
// First price update seeds lastPrice without firing
//-------------------------------------------------------------------------

TEST(SLTPContainerTest, FirstPriceUpdateSeedsOnly)
{
    SLTPSetup setup;
    setup.setUp();
    constexpr BookId bookId{};

    setup.container.onOrderCreated({
        .orderId = 1,
        .bookId = bookId,
        .agentId = -1,
        .originatingSide = OrderDirection::BUY,
        .volume = 1_dec,
        .leverage = 0_dec,
        .stopLoss = DEC(-0.05)
    });

    setup.container.onOrderTrade({
        .bookId = bookId,
        .originatingOrderId = 1,
        .agentId = -1,
        .side = OrderDirection::BUY,
        .fillPrice = 100_dec,
        .filledVolume = 1_dec
    });

    // SL trigger at 95. First update at 80 should NOT fire (no previous price).
    setup.container.onPriceUpdate(bookId, 80_dec);
    EXPECT_TRUE(setup.dispatched.empty());

    // Second update at 80 — same price, no transition
    setup.container.onPriceUpdate(bookId, 80_dec);
    EXPECT_TRUE(setup.dispatched.empty());

    // Trigger is still live
    EXPECT_EQ(setup.container.books()[bookId].triggers.size(), 1u);
}

//-------------------------------------------------------------------------
// FIFO coverage — small helpers used by the tests below.
//-------------------------------------------------------------------------

void createFlaggedOrder(
    SLTPSetup& setup,
    OrderID orderId,
    AgentId agentId,
    BookId bookId,
    OrderDirection side,
    decimal_t volume,
    std::optional<decimal_t> sl = {},
    std::optional<decimal_t> tp = {})
{
    setup.container.onOrderCreated({
        .orderId = orderId,
        .bookId = bookId,
        .agentId = agentId,
        .originatingSide = side,
        .volume = volume,
        .leverage = 0_dec,
        .stopLoss = sl,
        .takeProfit = tp
    });
}

void trade(
    SLTPSetup& setup,
    OrderID orderId,
    AgentId agentId,
    BookId bookId,
    OrderDirection side,
    decimal_t fillPrice,
    decimal_t filledVolume)
{
    setup.container.onOrderTrade({
        .bookId = bookId,
        .originatingOrderId = orderId,
        .agentId = agentId,
        .side = side,
        .fillPrice = fillPrice,
        .filledVolume = filledVolume
    });
}

//-------------------------------------------------------------------------
// FIFO: counter-trade fully covers a single slot
//-------------------------------------------------------------------------

TEST(SLTPContainerTest, CounterTradeFullyCoversSlot)
{
    SLTPSetup setup;
    setup.setUp();
    constexpr BookId bookId{};
    constexpr AgentId agentId{-1};

    createFlaggedOrder(setup, 1, agentId, bookId, OrderDirection::BUY, 5_dec, DEC(-0.05), DEC(0.10));
    trade(setup, 1, agentId, bookId, OrderDirection::BUY, 100_dec, 5_dec);

    ASSERT_EQ(setup.container.books()[bookId].triggers.size(), 2u);
    ASSERT_EQ(setup.container.books()[bookId].openSlots[agentId][static_cast<size_t>(OrderDirection::SELL)].size(), 1u);

    // Unflagged counter-SELL of the same volume drains the slot entirely.
    trade(setup, /*anyId=*/99, agentId, bookId, OrderDirection::SELL, 100_dec, 5_dec);

    EXPECT_EQ(setup.container.books()[bookId].triggers.size(), 0u);
    EXPECT_TRUE(setup.container.books()[bookId].openSlots[agentId][static_cast<size_t>(OrderDirection::SELL)].empty());

    // No trigger should fire for either direction now.
    setup.container.onPriceUpdate(bookId, 100_dec);
    setup.container.onPriceUpdate(bookId, 80_dec);
    setup.container.onPriceUpdate(bookId, 120_dec);
    EXPECT_TRUE(setup.dispatched.empty());
}

//-------------------------------------------------------------------------
// FIFO: counter-trade partially covers a slot — volume mirrors down
//-------------------------------------------------------------------------

TEST(SLTPContainerTest, CounterTradePartiallyCoversSlot)
{
    SLTPSetup setup;
    setup.setUp();
    constexpr BookId bookId{};
    constexpr AgentId agentId{-1};

    createFlaggedOrder(setup, 1, agentId, bookId, OrderDirection::BUY, 10_dec, DEC(-0.05), DEC(0.10));
    trade(setup, 1, agentId, bookId, OrderDirection::BUY, 100_dec, 10_dec);

    // Counter-SELL of 4: slot volume 10 → 6; both triggers' volume mirrors.
    trade(setup, /*anyId=*/99, agentId, bookId, OrderDirection::SELL, 100_dec, 4_dec);

    auto& fifo = setup.container.books()[bookId].openSlots[agentId][static_cast<size_t>(OrderDirection::SELL)];
    ASSERT_EQ(fifo.size(), 1u);
    EXPECT_EQ(fifo.front().volume, 6_dec);
    EXPECT_EQ(fifo.front().slIter->second.volume, 6_dec);
    EXPECT_EQ(fifo.front().tpIter->second.volume, 6_dec);

    // Fire SL — dispatched volume reflects the shrunken coverage, not the original.
    setup.container.onPriceUpdate(bookId, 100_dec);
    setup.container.onPriceUpdate(bookId, 90_dec);
    ASSERT_EQ(setup.dispatched.size(), 1u);
    EXPECT_EQ(setup.dispatched[0].entry.volume, 6_dec);
}

//-------------------------------------------------------------------------
// FIFO: counter-trade spans multiple slots — front drained first
//-------------------------------------------------------------------------

TEST(SLTPContainerTest, CounterTradeSpansMultipleSlots)
{
    SLTPSetup setup;
    setup.setUp();
    constexpr BookId bookId{};
    constexpr AgentId agentId{-1};

    // Two flagged BUY fills → two SELL-closing slots (volume 3 and 5).
    createFlaggedOrder(setup, 1, agentId, bookId, OrderDirection::BUY, 3_dec, DEC(-0.05));
    trade(setup, 1, agentId, bookId, OrderDirection::BUY, 100_dec, 3_dec);
    createFlaggedOrder(setup, 2, agentId, bookId, OrderDirection::BUY, 5_dec, DEC(-0.05));
    trade(setup, 2, agentId, bookId, OrderDirection::BUY, 110_dec, 5_dec);

    auto& fifo = setup.container.books()[bookId].openSlots[agentId][static_cast<size_t>(OrderDirection::SELL)];
    ASSERT_EQ(fifo.size(), 2u);

    // Counter-SELL of 4: drains the first slot (3) fully and shrinks the second (5 → 4).
    trade(setup, /*anyId=*/99, agentId, bookId, OrderDirection::SELL, 105_dec, 4_dec);

    ASSERT_EQ(fifo.size(), 1u);
    EXPECT_EQ(fifo.front().volume, 4_dec);
    // The first slot's SL (at 95) is gone; only the second's SL (at ~104.5) remains.
    EXPECT_EQ(setup.container.books()[bookId].triggers.size(), 1u);
    EXPECT_EQ(fifo.front().slIter->second.volume, 4_dec);
}

//-------------------------------------------------------------------------
// FIFO: firing SL erases paired TP but keeps slot alive for the counter
// closing trade to drain.
//-------------------------------------------------------------------------

TEST(SLTPContainerTest, FiringSLEraseTPAndKeepsSlotForCounterTrade)
{
    SLTPSetup setup;
    setup.setUp();
    constexpr BookId bookId{};
    constexpr AgentId agentId{-1};

    createFlaggedOrder(setup, 1, agentId, bookId, OrderDirection::BUY, 5_dec, DEC(-0.05), DEC(0.10));
    trade(setup, 1, agentId, bookId, OrderDirection::BUY, 100_dec, 5_dec);

    auto& perBook = setup.container.books()[bookId];
    auto& fifo = perBook.openSlots[agentId][static_cast<size_t>(OrderDirection::SELL)];
    ASSERT_EQ(fifo.size(), 1u);

    // SL fires: both SL and TP erased from multimap; slot kept with end() iterators.
    setup.container.onPriceUpdate(bookId, 100_dec);
    setup.container.onPriceUpdate(bookId, 90_dec);
    ASSERT_EQ(setup.dispatched.size(), 1u);
    EXPECT_EQ(perBook.triggers.size(), 0u);
    ASSERT_EQ(fifo.size(), 1u);
    EXPECT_EQ(fifo.front().slIter, perBook.triggers.end());
    EXPECT_EQ(fifo.front().tpIter, perBook.triggers.end());
    EXPECT_EQ(fifo.front().volume, 5_dec);

    // The closing market order would now come in as a SELL counter-trade —
    // simulate it; the slot drains, and the drained agent is pruned, so `fifo`
    // must not be read past this point.
    trade(setup, /*anyId=*/99, agentId, bookId, OrderDirection::SELL, 90_dec, 5_dec);
    EXPECT_FALSE(perBook.openSlots.contains(agentId));

    // Subsequent price moves fire nothing.
    setup.container.onPriceUpdate(bookId, 120_dec);
    EXPECT_EQ(setup.dispatched.size(), 1u);
}

//-------------------------------------------------------------------------
// FIFO: per-agent isolation
//-------------------------------------------------------------------------

TEST(SLTPContainerTest, CoverageIsPerAgent)
{
    SLTPSetup setup;
    setup.setUp();
    constexpr BookId bookId{};
    constexpr AgentId agentA{-1};
    constexpr AgentId agentB{-2};

    // Both agents open a long with SL; different fill prices for distinct triggers.
    createFlaggedOrder(setup, 1, agentA, bookId, OrderDirection::BUY, 3_dec, DEC(-0.05));
    trade(setup, 1, agentA, bookId, OrderDirection::BUY, 100_dec, 3_dec);
    createFlaggedOrder(setup, 2, agentB, bookId, OrderDirection::BUY, 3_dec, DEC(-0.05));
    trade(setup, 2, agentB, bookId, OrderDirection::BUY, 200_dec, 3_dec);

    // Agent A sells 3 (closes their own long). Agent B's coverage must be
    // untouched.
    trade(setup, /*anyId=*/99, agentA, bookId, OrderDirection::SELL, 100_dec, 3_dec);

    auto& perBook = setup.container.books()[bookId];
    EXPECT_TRUE(perBook.openSlots[agentA][static_cast<size_t>(OrderDirection::SELL)].empty());
    ASSERT_EQ(perBook.openSlots[agentB][static_cast<size_t>(OrderDirection::SELL)].size(), 1u);
    EXPECT_EQ(perBook.triggers.size(), 1u);  // only B's SL remains
}

//-------------------------------------------------------------------------
// FIFO: same-side second flagged trade appends to the back, not the front
//-------------------------------------------------------------------------

TEST(SLTPContainerTest, NewFlaggedTradeAppendsToFIFOBack)
{
    SLTPSetup setup;
    setup.setUp();
    constexpr BookId bookId{};
    constexpr AgentId agentId{-1};

    createFlaggedOrder(setup, 1, agentId, bookId, OrderDirection::BUY, 3_dec, DEC(-0.05));
    trade(setup, 1, agentId, bookId, OrderDirection::BUY, 100_dec, 3_dec);
    createFlaggedOrder(setup, 2, agentId, bookId, OrderDirection::BUY, 5_dec, DEC(-0.05));
    trade(setup, 2, agentId, bookId, OrderDirection::BUY, 200_dec, 5_dec);

    auto& fifo = setup.container.books()[bookId].openSlots[agentId][static_cast<size_t>(OrderDirection::SELL)];
    ASSERT_EQ(fifo.size(), 2u);
    // Front is the oldest slot (trigger @ 100 * 0.95 = 95).
    EXPECT_EQ(fifo.front().slIter->second.triggerPrice, 95_dec);
    EXPECT_EQ(fifo.front().volume, 3_dec);
    // Back is the newer slot (trigger @ 200 * 0.95 = 190).
    EXPECT_EQ(fifo.back().slIter->second.triggerPrice, 190_dec);
    EXPECT_EQ(fifo.back().volume, 5_dec);
}

//-------------------------------------------------------------------------
// FIFO: unflagged counter-trade still covers prior flagged coverage
//-------------------------------------------------------------------------

TEST(SLTPContainerTest, UnflaggedCounterTradeCoversFlaggedTriggers)
{
    SLTPSetup setup;
    setup.setUp();
    constexpr BookId bookId{};
    constexpr AgentId agentId{-1};

    createFlaggedOrder(setup, 1, agentId, bookId, OrderDirection::BUY, 5_dec, DEC(-0.05));
    trade(setup, 1, agentId, bookId, OrderDirection::BUY, 100_dec, 5_dec);

    // Counter-SELL is entirely UNFLAGGED (no onOrderCreated for this orderId) —
    // it still drains the existing FIFO.
    trade(setup, /*unflagged=*/500, agentId, bookId, OrderDirection::SELL, 100_dec, 5_dec);

    EXPECT_EQ(setup.container.books()[bookId].triggers.size(), 0u);
    EXPECT_TRUE(setup.container.books()[bookId].openSlots[agentId][static_cast<size_t>(OrderDirection::SELL)].empty());
}

//-------------------------------------------------------------------------
// FIFO: covering happens before opening on a single flagged trade that
// reverses the position
//-------------------------------------------------------------------------

TEST(SLTPContainerTest, CoveringPrecedesOpeningOnPositionReversal)
{
    SLTPSetup setup;
    setup.setUp();
    constexpr BookId bookId{};
    constexpr AgentId agentId{-1};

    // Build up a SELL-closing slot (long position of 3 units).
    createFlaggedOrder(setup, 1, agentId, bookId, OrderDirection::BUY, 3_dec, DEC(-0.05));
    trade(setup, 1, agentId, bookId, OrderDirection::BUY, 100_dec, 3_dec);

    // Flagged SELL of 5 that both closes the long (3) and opens a short (2).
    createFlaggedOrder(setup, 2, agentId, bookId, OrderDirection::SELL, 5_dec, DEC(0.05));
    trade(setup, 2, agentId, bookId, OrderDirection::SELL, 110_dec, 5_dec);

    auto& perBook = setup.container.books()[bookId];
    // Old SELL-closing slot fully drained.
    EXPECT_TRUE(perBook.openSlots[agentId][static_cast<size_t>(OrderDirection::SELL)].empty());
    // New BUY-closing slot for the 2-unit short, SL at 110 * 1.05 = 115.5.
    auto& buyFifo = perBook.openSlots[agentId][static_cast<size_t>(OrderDirection::BUY)];
    ASSERT_EQ(buyFifo.size(), 1u);
    EXPECT_EQ(buyFifo.front().volume, 2_dec);
    EXPECT_EQ(buyFifo.front().slIter->second.triggerPrice, DEC(115.5));
    EXPECT_EQ(buyFifo.front().slIter->second.volume, 2_dec);
}

//-------------------------------------------------------------------------
// removeOrder also cleans up FIFO slots (regression)
//-------------------------------------------------------------------------

TEST(SLTPContainerTest, RemoveOrderDropsFIFOSlot)
{
    SLTPSetup setup;
    setup.setUp();
    constexpr BookId bookId{};
    constexpr AgentId agentId{-1};

    createFlaggedOrder(setup, 1, agentId, bookId, OrderDirection::BUY, 5_dec, DEC(-0.05), DEC(0.10));
    trade(setup, 1, agentId, bookId, OrderDirection::BUY, 100_dec, 5_dec);

    auto& openSlots = setup.container.books()[bookId].openSlots;
    ASSERT_EQ(openSlots[agentId][static_cast<size_t>(OrderDirection::SELL)].size(), 1u);

    setup.container.removeOrder(bookId, 1);

    // removeOrder drains the slot list AND prunes the now-empty agent
    // entry from openSlots so the outer map doesn't accumulate
    // [empty, empty] residuals for departed agents.
    EXPECT_FALSE(openSlots.contains(agentId));
    EXPECT_EQ(setup.container.books()[bookId].triggers.size(), 0u);
}

//-------------------------------------------------------------------------
// An unflagged trade that drains an agent's last slot prunes the agent too
//-------------------------------------------------------------------------

TEST(SLTPContainerTest, UnflaggedTradeDrainingTheLastSlotPrunesTheAgent)
{
    SLTPSetup setup;
    setup.setUp();
    constexpr BookId bookId{};
    constexpr AgentId agentId{-1};

    createFlaggedOrder(setup, 1, agentId, bookId, OrderDirection::BUY, 5_dec, DEC(-0.05), DEC(0.10));
    trade(setup, 1, agentId, bookId, OrderDirection::BUY, 100_dec, 5_dec);
    const auto& perBook = setup.container.books()[bookId];
    ASSERT_TRUE(perBook.openSlots.contains(agentId));

    // The counter-trade has no SL/TP of its own, so no order info is kept for it.
    trade(setup, /*unflagged=*/500, agentId, bookId, OrderDirection::SELL, 100_dec, 5_dec);

    EXPECT_EQ(perBook.activeSlotCount, 0u);
    EXPECT_EQ(perBook.triggers.size(), 0u);
    EXPECT_FALSE(perBook.openSlots.contains(agentId)) << "the drained agent's empty entry was left behind";
}

//-------------------------------------------------------------------------
// Multi-book isolation
//-------------------------------------------------------------------------

TEST(SLTPContainerTest, TriggersAreIsolatedPerBook)
{
    SLTPSetup setup;
    setup.setUp(2);
    constexpr BookId book0{0};
    constexpr BookId book1{1};

    setup.container.onOrderCreated({
        .orderId = 1,
        .bookId = book0,
        .agentId = -1,
        .originatingSide = OrderDirection::BUY,
        .volume = 1_dec,
        .leverage = 0_dec,
        .stopLoss = DEC(-0.05)
    });

    setup.container.onOrderTrade({
        .bookId = book0,
        .originatingOrderId = 1,
        .agentId = -1,
        .side = OrderDirection::BUY,
        .fillPrice = 100_dec,
        .filledVolume = 1_dec
    });

    // Price move on book1 should not fire book0's trigger
    setup.container.onPriceUpdate(book1, 100_dec);
    setup.container.onPriceUpdate(book1, 80_dec);
    EXPECT_TRUE(setup.dispatched.empty());

    // Price move on book0 fires
    setup.container.onPriceUpdate(book0, 100_dec);
    setup.container.onPriceUpdate(book0, 80_dec);
    EXPECT_EQ(setup.dispatched.size(), 1u);
}

//-------------------------------------------------------------------------

}  // namespace

//-------------------------------------------------------------------------
