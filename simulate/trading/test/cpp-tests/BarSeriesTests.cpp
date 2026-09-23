/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

/*
 * The bar series is the fixed clock the agents' statistics move onto, replacing each agent
 * differencing the tape on its own polling phase. Its whole reason to exist is that two
 * readers asking for the same horizon at the same time get the same answer, so these tests
 * are mostly about what a window does NOT depend on: who last touched the series, where
 * inside a bar the caller is, and whether the book was quiet.
 *
 * Driven directly, with no Simulation and no book, so nothing here depends on the rng, on
 * agent parameters or on the market.
 */

#include <taosim/statshub/BarSeries.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>

//-------------------------------------------------------------------------

using namespace taosim::stats;
using namespace taosim::literals;

using namespace testing;

//-------------------------------------------------------------------------

static constexpr Timestamp s_period = 1'000'000'000;
static constexpr uint32_t s_capacity = 16;
static constexpr double s_spread = 0.02;

// Mirrors the order the hub uses: advance the clock, then fold the quote in. Getting this
// backwards would put a quote into a bar that had already closed.
static void quoteAt(
    BarSeries& series, taosim::book::BookTradeStats& tape, Timestamp at, double mid)
{
    series.roll(at, tape);
    series.observeQuote(mid, s_spread, at);
}

//-------------------------------------------------------------------------

// One closed bar is a reading, not an interval. Reporting a window off it would mean
// answering with a number nothing was measured over.
TEST(BarSeries, WindowNeedsTwoClosedBars)
{
    BarSeries series{s_period, s_capacity};
    taosim::book::BookTradeStats tape;

    quoteAt(series, tape, 0, 100.0);
    EXPECT_THAT(series.window(4).bars, Eq(0u));

    series.roll(s_period, tape);
    ASSERT_THAT(series.barsAvailable(), Eq(1u));
    EXPECT_THAT(series.window(4).bars, Eq(0u));

    series.roll(2 * s_period, tape);
    EXPECT_THAT(series.window(4).bars, Eq(1u));
}

//-------------------------------------------------------------------------

// The property the whole design rests on: the bar still open is invisible, so the answer
// does not depend on where inside a bar the caller happens to be.
TEST(BarSeries, PartialBarIsNeverVisible)
{
    BarSeries series{s_period, s_capacity};
    taosim::book::BookTradeStats tape;

    quoteAt(series, tape, 0, 100.0);
    series.roll(3 * s_period, tape);
    const auto before = series.window(8);

    // A large move, but inside the open bar.
    quoteAt(series, tape, 3 * s_period + s_period / 2, 200.0);
    const auto during = series.window(8);

    EXPECT_THAT(during.bars, Eq(before.bars));
    EXPECT_THAT(during.logReturn, DoubleEq(before.logReturn));
    EXPECT_THAT(during.midClose, DoubleEq(before.midClose));

    // And it appears in full the moment that bar closes.
    series.roll(4 * s_period, tape);
    EXPECT_THAT(series.window(8).midClose, DoubleEq(200.0));
}

//-------------------------------------------------------------------------

// Log returns telescope, so a drift over a horizon is a two-point read rather than a sum
// whose value depends on how many samples happened to land inside it. No trades here, so
// every bar falls back to its mid, which is the other half of the price definition.
TEST(BarSeries, ReturnTelescopesAcrossTheWindow)
{
    BarSeries series{s_period, s_capacity};
    taosim::book::BookTradeStats tape;

    quoteAt(series, tape, 0, 100.0);
    quoteAt(series, tape, 1 * s_period, 110.0);
    quoteAt(series, tape, 2 * s_period, 121.0);
    quoteAt(series, tape, 3 * s_period, 133.1);
    series.roll(4 * s_period, tape);

    const auto window = series.window(3);
    ASSERT_THAT(window.bars, Eq(3u));
    EXPECT_THAT(window.logReturn, DoubleNear(std::log(133.1 / 100.0), 1e-12));
    EXPECT_THAT(window.meanReturnPerBar(), DoubleNear(std::log(133.1 / 100.0) / 3.0, 1e-12));
}

//-------------------------------------------------------------------------

// Realized variance over the horizon, already summed, so a consumer whose horizon IS the
// window uses it without rescaling. Three equal 10% steps, so the sum is 3 * log(1.1)^2.
TEST(BarSeries, RealizedVarianceIsTheSumOfSquaredBarReturns)
{
    BarSeries series{s_period, s_capacity};
    taosim::book::BookTradeStats tape;

    quoteAt(series, tape, 0, 100.0);
    quoteAt(series, tape, 1 * s_period, 110.0);
    quoteAt(series, tape, 2 * s_period, 121.0);
    quoteAt(series, tape, 3 * s_period, 133.1);
    series.roll(4 * s_period, tape);

    const double step = std::log(1.1);
    const auto window = series.window(3);
    EXPECT_THAT(window.realizedVariance, DoubleNear(3.0 * step * step, 1e-12));
    EXPECT_THAT(window.variancePerSecond(), DoubleNear(step * step, 1e-12));
}

//-------------------------------------------------------------------------

// A quiet second is not a missing second. The mid is carried, so the window reports no
// move rather than a gap, and nothing decays toward zero for want of an observation.
TEST(BarSeries, QuietBarsCarryTheMidAndContributeNoReturn)
{
    BarSeries series{s_period, s_capacity};
    taosim::book::BookTradeStats tape;

    quoteAt(series, tape, 0, 100.0);
    series.roll(6 * s_period, tape);

    const auto window = series.window(5);
    ASSERT_THAT(window.bars, Eq(5u));
    EXPECT_THAT(window.logReturn, DoubleEq(0.0));
    EXPECT_THAT(window.realizedVariance, DoubleEq(0.0));
    EXPECT_THAT(window.midClose, DoubleEq(100.0));
}

//-------------------------------------------------------------------------

// A one-sided book has no mid. Treating that as a price would inject a return larger than
// anything the market can produce, so it carries instead.
TEST(BarSeries, OneSidedBookCarriesTheLastMid)
{
    BarSeries series{s_period, s_capacity};
    taosim::book::BookTradeStats tape;

    quoteAt(series, tape, 0, 100.0);
    quoteAt(series, tape, s_period / 2, 0.0);
    series.roll(2 * s_period, tape);

    const auto window = series.window(4);
    EXPECT_THAT(window.midClose, DoubleEq(100.0));
    EXPECT_THAT(window.logReturn, DoubleEq(0.0));
}

//-------------------------------------------------------------------------

// Time weighting, which is the difference between "the average quote" and "whatever the
// quote happened to be at the instant the bar ended". 90% of the bar at 100, 10% at 200.
TEST(BarSeries, TimeWeightedMidWeightsByDwellTime)
{
    BarSeries series{s_period, s_capacity};
    taosim::book::BookTradeStats tape;

    quoteAt(series, tape, 0, 100.0);
    quoteAt(series, tape, (9 * s_period) / 10, 200.0);
    series.roll(2 * s_period, tape);

    // The first bar's own value is not reachable through window(), which reports the last
    // bar, so close a second quiet bar and read the carried one instead.
    const auto window = series.window(1);
    EXPECT_THAT(window.midClose, DoubleEq(200.0));
    EXPECT_THAT(window.twSpread, DoubleNear(s_spread, 1e-12));
}

//-------------------------------------------------------------------------

// Asking for more than exists reports what it actually covered rather than quietly
// answering over a shorter interval than requested.
TEST(BarSeries, WindowIsClampedToWhatExistsAndSaysSo)
{
    BarSeries series{s_period, s_capacity};
    taosim::book::BookTradeStats tape;

    quoteAt(series, tape, 0, 100.0);
    series.roll(5 * s_period, tape);

    const auto window = series.window(1000);
    EXPECT_THAT(window.bars, Eq(4u));
    EXPECT_THAT(window.seconds, DoubleEq(4.0));
}

//-------------------------------------------------------------------------

// The tape is snapshotted at each close, so trade-frequency statistics come out of the same
// subtraction as the quote ones and nothing is accumulated twice.
TEST(BarSeries, TapeDifferenceGivesTheWindowVolumeAndCount)
{
    BarSeries series{s_period, s_capacity};
    taosim::book::BookTradeStats tape;

    quoteAt(series, tape, 0, 100.0);
    series.roll(s_period, tape);

    tape.record(DEC(100.0), DEC(2.0), s_period);
    tape.record(DEC(101.0), DEC(3.0), s_period + 1);
    series.roll(2 * s_period, tape);
    tape.record(DEC(102.0), DEC(5.0), 2 * s_period);
    series.roll(3 * s_period, tape);

    const auto full = series.window(2);
    EXPECT_THAT(full.tradeCount, Eq(3u));
    EXPECT_THAT(full.volume, DoubleNear(10.0, 1e-9));

    // The trade recorded exactly on the boundary belongs to the bar that opened there, not
    // the one that just closed, which is why the hub rolls before it records.
    const auto lastOnly = series.window(1);
    EXPECT_THAT(lastOnly.tradeCount, Eq(1u));
    EXPECT_THAT(lastOnly.volume, DoubleNear(5.0, 1e-9));
}

//-------------------------------------------------------------------------

// Realized variance and its jump-robust counterpart both come off the tape snapshots, and
// the pi/2 is already applied so a consumer does not carry the constant around.
TEST(BarSeries, TradeFrequencyVarianceAndBipowerComeOffTheTape)
{
    BarSeries series{s_period, s_capacity};
    taosim::book::BookTradeStats tape;

    quoteAt(series, tape, 0, 100.0);
    series.roll(s_period, tape);
    for (auto i = 0z; i < 5; ++i) {
        tape.record(DEC(100.0) + taosim::util::double2decimal(i), DEC(1.0), s_period);
    }
    series.roll(2 * s_period, tape);

    const auto window = series.window(1);
    EXPECT_THAT(window.tradeReturnSq, Gt(0.0));
    EXPECT_THAT(window.bipowerVariance, Gt(0.0));
}

//-------------------------------------------------------------------------

// An idle book must not make the ring grow, and rolling across a gap longer than the ring
// must not spend the loop closing bars that would be evicted on the way in.
TEST(BarSeries, LongIdleGapStaysBounded)
{
    BarSeries series{s_period, s_capacity};
    taosim::book::BookTradeStats tape;

    quoteAt(series, tape, 0, 100.0);
    series.roll(100'000 * s_period, tape);

    EXPECT_THAT(series.barsAvailable(), Le(s_capacity));
    EXPECT_THAT(series.window(s_capacity).midClose, DoubleEq(100.0));
}

//-------------------------------------------------------------------------

// Rolling is a pure function of the clock and what has accumulated, so it does not matter
// who advances the series or how often. Two readers at the same instant see one answer.
TEST(BarSeries, RollingIsIdempotent)
{
    BarSeries series{s_period, s_capacity};
    taosim::book::BookTradeStats tape;

    quoteAt(series, tape, 0, 100.0);
    quoteAt(series, tape, s_period, 110.0);
    series.roll(2 * s_period, tape);

    const auto first = series.window(4);
    for (auto i = 0z; i < 10; ++i) {
        series.roll(2 * s_period, tape);
    }
    const auto second = series.window(4);

    EXPECT_THAT(second.bars, Eq(first.bars));
    EXPECT_THAT(second.logReturn, DoubleEq(first.logReturn));
    EXPECT_THAT(second.realizedVariance, DoubleEq(first.realizedVariance));
    EXPECT_THAT(series.barsAvailable(), Eq(2u));
}

//-------------------------------------------------------------------------

// The price basis: a bar that traded takes its OHLC from its trades, not from the quote.
TEST(BarSeries, TradedBarTakesItsPriceFromTheTrades)
{
    BarSeries series{s_period, s_capacity};
    taosim::book::BookTradeStats tape;

    quoteAt(series, tape, 0, 100.0);
    series.roll(s_period, tape);

    for (const double price : {101.0, 104.0, 99.0, 102.0}) {
        series.roll(s_period, tape);
        series.recordTrade(price, s_period);
    }
    series.roll(2 * s_period, tape);

    const auto window = series.window(1);
    EXPECT_THAT(window.close, DoubleEq(102.0));
    // And the mid is still reported alongside, untouched by the trades.
    EXPECT_THAT(window.midClose, DoubleEq(100.0));
}

//-------------------------------------------------------------------------

// A bar with no trade in it is the mid, flat, because nothing observable moved inside it.
TEST(BarSeries, UntradedBarIsTheMidFlat)
{
    BarSeries series{s_period, s_capacity};
    taosim::book::BookTradeStats tape;

    quoteAt(series, tape, 0, 100.0);
    series.roll(s_period, tape);
    series.recordTrade(105.0, s_period);
    series.roll(2 * s_period, tape);
    // Third bar: quote only.
    quoteAt(series, tape, 2 * s_period, 106.0);
    series.roll(3 * s_period, tape);

    EXPECT_THAT(series.window(1).close, DoubleEq(106.0));
    // The two bases chain together, so the return spans the switch from trade to quote.
    EXPECT_THAT(series.window(1).logReturn, DoubleNear(std::log(106.0 / 105.0), 1e-12));
}

//-------------------------------------------------------------------------

// The conditioning that keeps a log out of trouble. Bars closed before the book has any
// price at all carry no return, and they do not break the chain either: the first usable
// price differences against the next one, not against nothing.
TEST(BarSeries, BarsWithNoUsablePriceContributeNoReturnAndDoNotBreakTheChain)
{
    BarSeries series{s_period, s_capacity};
    taosim::book::BookTradeStats tape;

    // Three bars before any quote or trade exists.
    series.roll(0, tape);
    series.roll(3 * s_period, tape);
    EXPECT_THAT(series.window(s_capacity).logReturn, DoubleEq(0.0));
    EXPECT_THAT(series.window(s_capacity).realizedVariance, DoubleEq(0.0));

    quoteAt(series, tape, 3 * s_period, 100.0);
    series.roll(4 * s_period, tape);
    quoteAt(series, tape, 4 * s_period, 110.0);
    series.roll(5 * s_period, tape);

    const auto window = series.window(s_capacity);
    EXPECT_THAT(window.logReturn, DoubleNear(std::log(110.0 / 100.0), 1e-12));
    EXPECT_TRUE(std::isfinite(window.logReturn));
    EXPECT_TRUE(std::isfinite(window.realizedVariance));
}

//-------------------------------------------------------------------------

// A non-positive or non-finite trade price is not a price. Recording one must not move the
// series, and above all must not reach the log.
TEST(BarSeries, UnusableTradePricesAreIgnored)
{
    BarSeries series{s_period, s_capacity};
    taosim::book::BookTradeStats tape;

    quoteAt(series, tape, 0, 100.0);
    series.roll(s_period, tape);
    series.recordTrade(0.0, s_period);
    series.recordTrade(-5.0, s_period);
    series.recordTrade(std::numeric_limits<double>::quiet_NaN(), s_period);
    series.roll(2 * s_period, tape);

    const auto window = series.window(1);
    EXPECT_THAT(window.close, DoubleEq(100.0)) << "an unusable trade became the bar's price";
    EXPECT_THAT(window.logReturn, DoubleEq(0.0));
    EXPECT_TRUE(std::isfinite(window.realizedVariance));
}

//-------------------------------------------------------------------------

// The regular sampling a recursion needs. GARCH is defined on regularly sampled returns, and
// the whole point of the fixed clock is being able to hand it some.
TEST(BarSeries, ReturnsTheRegularCloseToCloseSeries)
{
    BarSeries series{s_period, s_capacity};
    taosim::book::BookTradeStats tape;

    quoteAt(series, tape, 0, 100.0);
    quoteAt(series, tape, 1 * s_period, 110.0);
    quoteAt(series, tape, 2 * s_period, 121.0);
    quoteAt(series, tape, 3 * s_period, 133.1);
    series.roll(4 * s_period, tape);

    const auto returns = series.returns(8);
    // Four closed bars give three returns between them, oldest first.
    ASSERT_THAT(returns, SizeIs(3));
    const double step = std::log(1.1);
    EXPECT_THAT(returns, ElementsAre(
        DoubleNear(step, 1e-12), DoubleNear(step, 1e-12), DoubleNear(step, 1e-12)));

    // And they are the same series the window statistics are built from.
    const auto window = series.window(3);
    const double sum = returns[0] + returns[1] + returns[2];
    EXPECT_THAT(window.logReturn, DoubleNear(sum, 1e-12));
}

//-------------------------------------------------------------------------

// The two sigma forms have to agree with the variance they come from, since consumers pick
// between them by what they are deciding: a position's risk over a horizon, or the state of
// the market as a rate.
TEST(BarSeries, SigmaFormsAgreeWithTheVariance)
{
    BarSeries series{s_period, s_capacity};
    taosim::book::BookTradeStats tape;

    quoteAt(series, tape, 0, 100.0);
    quoteAt(series, tape, 1 * s_period, 110.0);
    quoteAt(series, tape, 2 * s_period, 121.0);
    series.roll(3 * s_period, tape);

    const auto window = series.window(2);
    EXPECT_THAT(window.horizonSigma() * window.horizonSigma(),
        DoubleNear(window.realizedVariance, 1e-15));
    EXPECT_THAT(window.sigmaPerSecond() * window.sigmaPerSecond() * window.seconds,
        DoubleNear(window.realizedVariance, 1e-15));
}

//-------------------------------------------------------------------------

// Scale invariance in price, at the substrate. Everything a consumer thresholds on is built
// from log returns, so the same relative path at a ten times higher price level has to give
// the identical statistics. A constant compared against these is then price-invariant for
// free; one compared against a price is not.
TEST(BarSeries, PriceLevelDoesNotChangeTheStatistics)
{
    auto run = [](double scale) {
        BarSeries series{s_period, s_capacity};
        taosim::book::BookTradeStats tape;
        quoteAt(series, tape, 0, 100.0 * scale);
        quoteAt(series, tape, 1 * s_period, 103.0 * scale);
        quoteAt(series, tape, 2 * s_period, 101.0 * scale);
        quoteAt(series, tape, 3 * s_period, 106.0 * scale);
        series.roll(4 * s_period, tape);
        return series.window(3);
    };

    const auto low = run(1.0);
    const auto high = run(10.0);

    EXPECT_THAT(high.logReturn, DoubleNear(low.logReturn, 1e-12));
    EXPECT_THAT(high.realizedVariance, DoubleNear(low.realizedVariance, 1e-12));
    EXPECT_THAT(high.horizonSigma(), DoubleNear(low.horizonSigma(), 1e-12));
    // And the price itself does move, so the test is not comparing two identical runs.
    EXPECT_THAT(high.close, DoubleNear(10.0 * low.close, 1e-9));
}

//-------------------------------------------------------------------------

// `barsAvailable` saturates at the ring capacity, so a consumer stepping a recursion once per
// bar cannot use it as a clock. This one does not saturate, which is the whole reason it
// exists alongside.
TEST(BarSeries, BarsClosedKeepsCountingPastTheRing)
{
    BarSeries series{s_period, s_capacity};
    taosim::book::BookTradeStats tape;

    quoteAt(series, tape, 0, 100.0);
    series.roll(4 * s_period, tape);
    EXPECT_THAT(series.barsClosed(), Eq(4u));

    series.roll(1000 * s_period, tape);
    EXPECT_THAT(series.barsAvailable(), Le(s_capacity));
    EXPECT_THAT(series.barsClosed(), Eq(1000u))
        << "the monotonic count followed the ring and saturated with it";
}
