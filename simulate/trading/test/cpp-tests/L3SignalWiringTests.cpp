/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 *
 * Does the L3 signal chain deliver at all?
 *
 * In exchange mode every on-disk L3 log holds its header and nothing else, on both the
 * prediction-time logger and the deferred (reconciled) one, across 1548 files, while the in-memory
 * L3Record demonstrably carries the same events. The current process wired feeds for all 129 books at
 * startup and has not opened a new sink since, and a rotating logger only opens one from inside log(),
 * so log() has never run.
 *
 * Reading the source refuted five candidate causes but could not settle which hop stops the event, and
 * the live answer needs an exchange restart. These tests need neither: they reproduce the exact wiring
 * shapes in isolation, in the test binary, which links separately from the running engine.
 *
 * A pass here is informative, not vacuous: it moves the fault out of the signal machinery and onto the
 * emit site or the instance identity, which is where the live probe should then look.
 */
#include <taosim/matching/ExchangeSignals.hpp>

#include <boost/signals2.hpp>
#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <vector>

//-------------------------------------------------------------------------

namespace bs2 = boost::signals2;
using taosim::matching::ExchangeSignals;

//-------------------------------------------------------------------------

TEST(L3SignalWiringTest, ExchangeSignalsRelaysItsOwnLogSignalsIntoL3)
{
    // The constructor connects orderLog/tradeLog/cancelLog/instructionLog into L3 capturing `this`.
    // If that relay is broken, nothing downstream can ever see an event.
    ExchangeSignals signals;
    int delivered = 0;
    bs2::scoped_connection feed = signals.L3.connect([&](taosim::L3LogEvent) { ++delivered; });

    signals.cancelLog(CancellationWithLogContext{});

    ASSERT_EQ(delivered, 1);
}

//-------------------------------------------------------------------------

TEST(L3SignalWiringTest, FeedSurvivesGrowthOfTheScopedConnectionVector)
{
    // L3Backlog stores its feeds in std::vector<bs2::scoped_connection> and pushes one per book, so the
    // vector reallocates repeatedly during construction. The suspicion was that reallocation destroys
    // moved-from elements and disconnects the live connection with them. Reading Boost refuted it
    // (connection's move ctor resets the source precisely so a moved-from scoped_connection will not
    // disconnect), but the refutation is worth pinning down here, because it is the difference between
    // "the wiring is fine" and a silent unsubscribe that only appears at scale.
    ExchangeSignals signals;
    int delivered = 0;

    std::vector<bs2::scoped_connection> feeds;
    feeds.push_back(signals.L3.connect([&](taosim::L3LogEvent) { ++delivered; }));
    const auto capacityAtFirst = feeds.capacity();

    // Force several reallocations past the capacity the first push_back reserved.
    for (std::size_t i = 0; i < capacityAtFirst * 8 + 16; ++i) {
        feeds.push_back(signals.L3.connect([](taosim::L3LogEvent) {}));
    }
    ASSERT_GT(feeds.capacity(), capacityAtFirst);

    signals.cancelLog(CancellationWithLogContext{});

    ASSERT_EQ(delivered, 1);
}

//-------------------------------------------------------------------------

TEST(L3SignalWiringTest, FullExchangeModeShapeDelivers)
{
    // MultiBookExchangeAgent holds std::map<BookId, std::unique_ptr<ExchangeSignals>> and L3Backlog
    // iterates it, connecting one feed per entry into a per-book store. This reproduces both sides
    // together, including the map growing after the feeds are taken, which would invalidate anything
    // that captured an element by reference rather than through the pointer.
    constexpr int bookCount = 129;  // the live book count
    std::map<int, std::unique_ptr<ExchangeSignals>> signals;
    for (int b = 0; b < bookCount; ++b) {
        signals[b] = std::make_unique<ExchangeSignals>();
    }

    std::vector<int> store(bookCount, 0);
    std::vector<bs2::scoped_connection> feeds;
    for (auto&& [bookId, sig] : signals) {
        feeds.push_back(sig->L3.connect([&store, bookId](taosim::L3LogEvent) { ++store.at(bookId); }));
    }

    // Emit on the first, the last and one in the middle: an off-by-one in the store index would show up
    // as a delivery landing on a book nobody reads.
    signals.at(0)->cancelLog(CancellationWithLogContext{});
    signals.at(64)->cancelLog(CancellationWithLogContext{});
    signals.at(bookCount - 1)->cancelLog(CancellationWithLogContext{});

    EXPECT_EQ(store.at(0), 1);
    EXPECT_EQ(store.at(64), 1);
    EXPECT_EQ(store.at(bookCount - 1), 1);

    int total = 0;
    for (const auto count : store) { total += count; }
    ASSERT_EQ(total, 3);
}

//-------------------------------------------------------------------------

TEST(L3SignalWiringTest, TwoSubscribersOnOneL3SignalBothReceive)
{
    // Live, each book's L3 has two subscribers: the prediction-time L3EventLogger and L3Backlog's
    // deferred feed. Both on-disk families are empty, which is one fact rather than two only if a
    // second subscriber cannot displace the first.
    ExchangeSignals signals;
    int logger = 0, backlog = 0;
    bs2::scoped_connection a = signals.L3.connect([&](taosim::L3LogEvent) { ++logger; });
    bs2::scoped_connection b = signals.L3.connect([&](taosim::L3LogEvent) { ++backlog; });

    signals.cancelLog(CancellationWithLogContext{});

    EXPECT_EQ(logger, 1);
    ASSERT_EQ(backlog, 1);
}
