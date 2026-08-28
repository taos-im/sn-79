/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/book/BookTradeStats.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>

//-------------------------------------------------------------------------

namespace taosim::util
{

//-------------------------------------------------------------------------

// Per-agent VARIANCE and TREND estimates differenced out of the book's monotonic trade
// accumulator (BookTradeStats, delivered by RETRIEVE_L1_EXT).
//
// Every agent reads the same public series, which is how it works on a real venue: one
// feed, and participants differ by the HORIZON they view it over, not by each seeing a
// private sample of the underlying prices. So heterogeneity here comes from per-agent
// halflives, and no synthetic per-agent sampling noise is added.
//
// Why differencing rather than each agent keeping its own return history: the book
// already accumulates the realized-variance terms once per trade in O(1), so an agent
// needs only its previous read. That is 4 doubles instead of a ring buffer of returns —
// the StylizedTrader's per-agent buffers cost 64 MB at 1000 instances over 4 books and
// 1.56 GB at ~2000 instances over 48 books, which is why its history is capped at 1000
// samples rather than chosen. Differencing is O(1) per poll and independent of window
// length, and the cost at the book is independent of how many agents read it.
//
// It also fixes the reflexivity problem by construction. A single shared volatility number
// consumed identically by every agent is a synchronised amplifier: volatility up -> all
// widen -> depth thins -> the next trade moves price further -> volatility up. Here the
// measurement window IS the agent's own polling interval and the smoothing is per agent,
// so a population with dispersed cadences does not synchronise even though it reads one
// public series. The book supplies raw accumulators — the role a real market-data feed
// plays — while the estimator stays private to the agent.
//
// Two parameters, deliberately separate:
//   halflifeSeconds — how fast you NOTICE a change
//   (the response gain lives at the CALLER, not here)
// Fusing them is why hand-tuning could not find a good point: a fast estimate with a weak
// response is alert and stable, whereas a slow estimate with a strong response lurches.
// A halflife below a few seconds is not usable by a maker requoting on a ~750 ms timer —
// the estimate would move faster than the quotes it informs — and a halflife approaching
// the volatility-clustering timescale smooths away the stylized fact being reproduced.
class TradeStatsEstimator
{
public:
    void configure(double halflifeSeconds, bool jumpRobust,
                   double slowHalflifeMultiple = 30.0) noexcept
    {
        m_halflifeSeconds = std::max(halflifeSeconds, 1e-9);
        m_jumpRobust = jumpRobust;
        m_slowHalflifeSeconds = m_halflifeSeconds * std::max(slowHalflifeMultiple, 1.0);
    }

    // Fold in a fresh reading. Returns the current estimate (variance per second).
    // The first call only takes a baseline: an interval needs two reads.
    double update(const book::BookTradeStats& stats, Timestamp now) noexcept
    {
        if (!m_haveBaseline) {
            snapshot(stats, now);
            m_haveBaseline = true;
            return m_value;
        }
        const double dt = static_cast<double>(now - m_time) / 1e9;
        const std::uint64_t trades = stats.tradeCount - m_tradeCount;
        // No elapsed time, or no trade in the interval, carries no information — hold the
        // estimate rather than decaying it toward zero on a quiet book, which would read
        // as falling volatility when it is really an absence of observation.
        if (dt <= 0.0 || trades == 0) {
            return m_value;
        }

        double sumSq = m_jumpRobust
            ? (std::numbers::pi / 2.0) * (stats.bipowerSum - m_bipowerSum)
            : stats.logReturnSqSum - m_logReturnSqSum;
        // Bipower lags the squares by one return and both are floating sums, so a tiny
        // negative difference is possible; it is not a signal.
        sumSq = std::max(sumSq, 0.0);
        // Decay the NUMERATOR AND DENOMINATOR separately and take the ratio, rather than
        // smoothing the per-interval ratio sumSq/dt. Averaging ratios reintroduces exactly
        // the sampling-dependence that dividing by elapsed time exists to remove: a long
        // quiet interval contributes a near-zero ratio with full weight and drags the
        // estimate down, so the answer depends on the observer's cadence rather than on the
        // market. Measured on a 0.4 trades/s book, the average-of-ratios form read 9.7e-09
        // against a true 6.7e-08 — low by ~7x. This form is an exponentially weighted
        // ratio-of-sums and degrades to plain realized variance as the halflife grows.
        const double w = std::exp(-std::numbers::ln2 * dt / m_halflifeSeconds);
        m_weightedSq = w * m_weightedSq + sumSq;
        m_weightedDt = w * m_weightedDt + dt;
        m_value = m_weightedDt > 0.0 ? m_weightedSq / m_weightedDt : m_value;

        // Trend, on the same decay. Log-returns telescope, so the sum over any window is
        // just log(p_end/p_start) — a ring buffer of returns was only ever storing two
        // endpoints and a count. Mean return per trade, the analogue of a mean per
        // observation, as a ratio-of-sums for the same reason as the variance.
        m_weightedRet = w * m_weightedRet + (stats.logReturnSum - m_logReturnSum);
        m_weightedCount = w * m_weightedCount + static_cast<double>(trades);
        m_meanLogReturn = m_weightedCount > 0.0 ? m_weightedRet / m_weightedCount : m_meanLogReturn;

        // A second, much slower pass over the same increments. Its only purpose is to give
        // ratio(): the fast estimate RELATIVE to the agent's own recent norm.
        //
        // That ratio is what a consumer should react to, because it is dimensionless. The
        // absolute variance is not comparable to a hand-calibrated sigmaSqr — measured, the
        // shipped 0.01 is ~10^5x any horizon's realized variance — so substituting one for
        // the other would silently rescale every spread and skew. Using the ratio keeps the
        // calibrated value setting the LEVEL and lets the estimator supply only the
        // time-variation, which sidesteps the units question rather than answering it.
        const double ws = std::exp(-std::numbers::ln2 * dt / m_slowHalflifeSeconds);
        m_slowWeightedSq = ws * m_slowWeightedSq + sumSq;
        m_slowWeightedDt = ws * m_slowWeightedDt + dt;
        m_slowValue = m_slowWeightedDt > 0.0 ? m_slowWeightedSq / m_slowWeightedDt : m_slowValue;
        m_primed = true;

        snapshot(stats, now);
        return m_value;
    }

    // Variance per unit time.
    [[nodiscard]] double value() const noexcept { return m_value; }
    // Exponentially weighted mean log-return per trade.
    [[nodiscard]] double meanLogReturn() const noexcept { return m_meanLogReturn; }
    // Slow reference variance, and the fast/slow ratio. ratio() is 1.0 until both are
    // primed, so a consumer scaling by it is inert during warm-up rather than wrong.
    [[nodiscard]] double slowValue() const noexcept { return m_slowValue; }
    [[nodiscard]] double ratio() const noexcept
    {
        return (m_primed && m_slowValue > 0.0) ? m_value / m_slowValue : 1.0;
    }
    [[nodiscard]] bool primed() const noexcept { return m_primed; }

private:
    void snapshot(const book::BookTradeStats& stats, Timestamp now) noexcept
    {
        m_tradeCount = stats.tradeCount;
        m_logReturnSqSum = stats.logReturnSqSum;
        m_logReturnSum = stats.logReturnSum;
        m_bipowerSum = stats.bipowerSum;
        m_time = now;
    }

    double m_halflifeSeconds{60.0};
    bool m_jumpRobust{true};
    // Previous read — the entire per-agent cost.
    std::uint64_t m_tradeCount{};
    double m_logReturnSqSum{};
    double m_logReturnSum{};
    double m_bipowerSum{};
    Timestamp m_time{};
    // Exponentially weighted sums, kept apart so value() is a ratio-of-sums.
    double m_weightedSq{};
    double m_weightedDt{};
    double m_weightedRet{};
    double m_weightedCount{};
    double m_meanLogReturn{};
    double m_slowHalflifeSeconds{1800.0};
    double m_slowWeightedSq{};
    double m_slowWeightedDt{};
    double m_slowValue{};
    double m_value{};
    bool m_primed{false};
    bool m_haveBaseline{false};
};

//-------------------------------------------------------------------------

}  // namespace taosim::util

//-------------------------------------------------------------------------
