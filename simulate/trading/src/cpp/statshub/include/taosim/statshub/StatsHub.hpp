/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/book/BookTradeStats.hpp>
#include <taosim/decimal/decimal.hpp>
#include <taosim/statshub/BarSeries.hpp>

#include "common.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace taosim::book { class Book; }

//-------------------------------------------------------------------------
// Shared, engine-fed market statistics for local agents: a read API behind an
// injected pointer instead of RETRIEVE_* round-trips through the message queue.
// The books stay the system of record -- L1 is an exact read-through of the live
// top of book, while running statistics are materialized incrementally off the
// book signals. Everything runs on the single engine thread, exactly like the
// UnsyncSignals feeding it. Running statistics accumulate in double: a statistic
// is not a monetary amount (same convention as BookTradeStats), and hardware FP
// beats software bid128 arithmetic once per-sample moment math dominates the one
// decimal2double conversion at the boundary.

namespace taosim::stats
{

//-------------------------------------------------------------------------

struct L1Snapshot
{
    decimal_t bestAskPrice;
    decimal_t bestAskVolume;
    decimal_t askTotalVolume;
    decimal_t bestBidPrice;
    decimal_t bestBidVolume;
    decimal_t bidTotalVolume;
};

// One price level. Deliberately not the message layer's BookLevel: the hub sits below the
// message layer and a depth read is not a wire type.
struct L2Level
{
    decimal_t price;
    decimal_t quantity;
};

struct L2Snapshot
{
    std::vector<L2Level> bids;
    std::vector<L2Level> asks;
};

class StatsHub
{
public:
    struct Desc
    {
        std::span<const std::shared_ptr<book::Book>> books;
        // Reads the simulation clock. Injected rather than taken as a Simulation pointer:
        // the hub sits below the agent layer and including Simulation here would close a
        // cycle through MultiBookExchangeAgent.
        std::function<Timestamp()> now;
        // Bar width and how many are kept. One second matches the simulation step, so bar
        // boundaries coincide with the state-publish boundary and agents share a clock with
        // what miners see. The depth is the longest horizon anything may ask for.
        Timestamp barPeriod{1'000'000'000};
        uint32_t barCapacity{3600};
    };

    explicit StatsHub(const Desc& desc);
    ~StatsHub() noexcept;

    [[nodiscard]] L1Snapshot l1(BookId bookId) const;
    // Best `depth` levels a side, best first. Both sides can come back empty on a book that
    // has none, so a caller reading front() has to check.
    [[nodiscard]] L2Snapshot l2(BookId bookId, size_t depth) const;

    // The book's monotonic trade accumulator. Kept here rather than on the exchange because
    // it is the same kind of thing as l1(): something an agent reads about a book without
    // asking for it by message. A reader differences two reads of its own, so the hub keeps
    // no per-reader state and any number of agents may read at any cadence.
    [[nodiscard]] const book::BookTradeStats& tradeStats(BookId bookId) const;
    // The whole series, mutable, for checkpoint restore writing into it.
    [[nodiscard]] std::vector<book::BookTradeStats>& tradeStats() noexcept;

    // Fixed-clock statistics. The window covers completed bars only, so what it reports
    // does not depend on where inside the current bar the caller happens to be, and two
    // callers asking for the same horizon at the same time get the same answer.
    [[nodiscard]] BarWindow window(BookId bookId, uint32_t bars) const;
    [[nodiscard]] uint32_t barsAvailable(BookId bookId) const;
    // Monotonic; see BarSeries::barsClosed.
    [[nodiscard]] uint64_t barsClosed(BookId bookId) const;
    // The last `n` close-to-close returns on the fixed clock, oldest first.
    [[nodiscard]] std::vector<double> returns(BookId bookId, uint32_t n) const;
    [[nodiscard]] Timestamp barPeriod() const noexcept;

private:
    std::unique_ptr<struct Impl> m_impl;
};

//-------------------------------------------------------------------------

}  // namespace taosim::stats

//-------------------------------------------------------------------------
