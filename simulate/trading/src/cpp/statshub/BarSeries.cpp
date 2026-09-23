/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/statshub/BarSeries.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>

//-------------------------------------------------------------------------

namespace taosim::stats
{

//-------------------------------------------------------------------------

BarSeries::BarSeries(Timestamp period, uint32_t capacity) noexcept
    : m_period{std::max(period, Timestamp{1})}
    , m_bars{std::max(capacity, 2u)}
{}

//-------------------------------------------------------------------------

void BarSeries::start(Timestamp now) noexcept
{
    m_barEnd = (now / m_period + 1) * m_period;
    m_twLast = now;
    m_twElapsed = 0;
    m_started = true;
}

//-------------------------------------------------------------------------

void BarSeries::accrue(Timestamp upTo) noexcept
{
    if (upTo <= m_twLast) return;
    const auto dt = static_cast<double>(upTo - m_twLast);
    m_twMidSum += m_lastMid * dt;
    m_twSpreadSum += m_lastSpread * dt;
    m_twElapsed += upTo - m_twLast;
    m_twLast = upTo;
}

//-------------------------------------------------------------------------

// A price is usable only if it is strictly positive and finite. Everything downstream takes
// a log of a ratio of two of these, so this is the one place that has to be careful: an
// absent quote reads as 0 and an empty book reads as 0, and neither is a price.
[[nodiscard]] static bool usable(double price) noexcept
{
    return price > 0.0 && std::isfinite(price);
}

//-------------------------------------------------------------------------

void BarSeries::closeBar(const book::BookTradeStats& tape)
{
    const double elapsed = static_cast<double>(m_twElapsed);
    // The bar's price comes from its trades. With none, the whole bar is the mid quote,
    // flat, because nothing observable moved inside it.
    const bool traded = m_barHasTrade && usable(m_tradeClose);
    Bar bar{
        .closeTime = m_barEnd,
        .open = traded ? m_tradeOpen : m_lastMid,
        .high = traded ? m_tradeHigh : m_lastMid,
        .low = traded ? m_tradeLow : m_lastMid,
        .close = traded ? m_tradeClose : m_lastMid,
        .hasTrade = traded,
        .midClose = m_lastMid,
        .twMid = elapsed > 0.0 ? m_twMidSum / elapsed : m_lastMid,
        .twSpread = elapsed > 0.0 ? m_twSpreadSum / elapsed : m_lastSpread
    };

    // A return needs a usable price at BOTH ends. Before the book has traded or been
    // two-sided there is none, and treating the absence as a move to or from zero would
    // inject a return larger than anything a market can produce. A bar with no usable price
    // does not break the chain either: the previous close is kept, so the next usable bar
    // differences against it rather than starting over.
    if (m_havePrevClose && usable(m_prevClose) && usable(bar.close)) {
        const double logReturn = std::log(bar.close / m_prevClose);
        if (std::isfinite(logReturn)) {
            m_cumReturn += logReturn;
            m_cumReturnSq += logReturn * logReturn;
        }
    }
    if (usable(bar.close)) {
        m_prevClose = bar.close;
        m_havePrevClose = true;
    }

    bar.cumReturn = m_cumReturn;
    bar.cumReturnSq = m_cumReturnSq;
    bar.tapeAtClose = tape;
    m_bars.push_back(bar);
    ++m_barsClosed;

    m_midOpen = m_midHigh = m_midLow = m_lastMid;
    m_barTouched = false;
    m_barHasTrade = false;
    m_tradeOpen = m_tradeHigh = m_tradeLow = m_tradeClose = 0.0;
    m_twMidSum = m_twSpreadSum = 0.0;
    m_twElapsed = 0;
}

//-------------------------------------------------------------------------

void BarSeries::roll(Timestamp now, const book::BookTradeStats& tape)
{
    if (!m_started) [[unlikely]] {
        start(now);
        return;
    }
    if (now < m_barEnd) return;

    // Bars older than the ring would be evicted the moment they were pushed, so skip
    // straight past them rather than spending the loop on a long idle stretch. Nothing is
    // lost that a reader could have seen.
    const auto pending = (now - m_barEnd) / m_period + 1;
    if (const auto capacity = static_cast<Timestamp>(m_bars.capacity()); pending > capacity) {
        const auto skipped = pending - capacity;
        m_barEnd += skipped * m_period;
        // Counted even though they are not stored. This is a clock: a consumer stepping a
        // recursion once per bar needs to know that time passed, or a long idle stretch
        // leaves its estimate frozen at whatever it was before the gap instead of decaying
        // through it.
        m_barsClosed += skipped;
        m_twLast = std::max(m_twLast, m_barEnd - m_period);
        m_twMidSum = m_twSpreadSum = 0.0;
        m_twElapsed = 0;
    }

    while (m_barEnd <= now) {
        accrue(m_barEnd);
        closeBar(tape);
        m_barEnd += m_period;
    }
}

//-------------------------------------------------------------------------

void BarSeries::recordTrade(double price, Timestamp now)
{
    if (!m_started) [[unlikely]] {
        start(now);
    }
    if (!usable(price)) return;

    if (!m_barHasTrade) {
        m_barHasTrade = true;
        m_tradeOpen = m_tradeHigh = m_tradeLow = price;
    }
    m_tradeHigh = std::max(m_tradeHigh, price);
    m_tradeLow = std::min(m_tradeLow, price);
    m_tradeClose = price;
}

//-------------------------------------------------------------------------

void BarSeries::observeQuote(double mid, double spread, Timestamp now)
{
    if (!m_started) [[unlikely]] {
        start(now);
    }
    accrue(now);

    if (!(mid > 0.0)) return;

    m_lastMid = mid;
    m_lastSpread = spread;
    if (!m_haveMid) {
        m_haveMid = true;
        m_midOpen = m_midHigh = m_midLow = mid;
    }
    if (!m_barTouched) {
        m_barTouched = true;
        m_midOpen = m_midHigh = m_midLow = mid;
    }
    m_midHigh = std::max(m_midHigh, mid);
    m_midLow = std::min(m_midLow, mid);
}

//-------------------------------------------------------------------------

BarWindow BarSeries::window(uint32_t bars) const
{
    BarWindow window{};
    // A window is the difference of two closed bars, so one bar is not yet a window.
    if (m_bars.size() < 2 || bars == 0) return window;

    const auto span = std::min<uint32_t>(bars, static_cast<uint32_t>(m_bars.size()) - 1);
    const auto& last = m_bars.back();
    const auto& first = m_bars[m_bars.size() - 1 - span];

    window.bars = span;
    window.seconds = static_cast<double>(span) * static_cast<double>(m_period) / 1e9;
    window.logReturn = last.cumReturn - first.cumReturn;
    window.realizedVariance = std::max(0.0, last.cumReturnSq - first.cumReturnSq);
    // Both are floating sums differenced over a window, so a tiny negative is arithmetic,
    // not a signal.
    window.tradeReturnSq =
        std::max(0.0, last.tapeAtClose.logReturnSqSum - first.tapeAtClose.logReturnSqSum);
    window.bipowerVariance = (std::numbers::pi / 2.0)
        * std::max(0.0, last.tapeAtClose.bipowerSum - first.tapeAtClose.bipowerSum);
    window.volume = util::decimal2double(
        last.tapeAtClose.volumeSum - first.tapeAtClose.volumeSum);
    window.tradeCount = last.tapeAtClose.tradeCount - first.tapeAtClose.tradeCount;
    window.close = last.close;
    window.midClose = last.midClose;
    window.twSpread = last.twSpread;
    return window;
}

//-------------------------------------------------------------------------

std::vector<double> BarSeries::returns(uint32_t n) const
{
    std::vector<double> res;
    if (m_bars.size() < 2 || n == 0) return res;

    const auto span = std::min<uint32_t>(n, static_cast<uint32_t>(m_bars.size()) - 1);
    res.reserve(span);
    // Differenced out of the cumulative series rather than stored twice.
    for (auto i = m_bars.size() - span; i < m_bars.size(); ++i) {
        res.push_back(m_bars[i].cumReturn - m_bars[i - 1].cumReturn);
    }
    return res;
}

//-------------------------------------------------------------------------

}  // namespace taosim::stats

//-------------------------------------------------------------------------
