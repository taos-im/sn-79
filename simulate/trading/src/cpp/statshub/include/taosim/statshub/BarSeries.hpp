/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/book/BookTradeStats.hpp>

#include <boost/circular_buffer.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

//-------------------------------------------------------------------------

namespace taosim::stats
{

//-------------------------------------------------------------------------

// One closed bar on a fixed clock.
//
// The point of the fixed clock is that two agents asking for the same horizon at the same
// time get the same answer, whenever either of them last looked. Everything a window
// statistic needs is stored CUMULATIVELY as of the close, so a window is the difference of
// two bars: O(1) for any horizon, and with no dependence on the reader's cadence. It is the
// same differencing BookTradeStats already relies on, moved off each reader's private
// polling phase and onto a grid they share.
struct Bar
{
    Timestamp closeTime{};
    // The bar's price, taken from the trades in it. A second with no trade has no trade
    // price, and rather than leaving a hole the whole bar is the mid quote: open, high, low
    // and close all equal, since nothing observable moved.
    double open{}, high{}, low{}, close{};
    bool hasTrade{};
    // The quote close regardless of whether the bar traded, which is what a consumer wanting
    // a price for an untraded book should read.
    double midClose{};
    // Time-weighted over the bar, which is the honest average when the quote sits still for
    // most of it. Sampling at the close would weight a one-microsecond excursion the same as
    // a full second of rest.
    double twMid{}, twSpread{};
    // Since the run began, over the bar price above.
    double cumReturn{}, cumReturnSq{};
    // The tape as of this boundary. Differencing two of these gives the trade-frequency
    // realized variance and bipower over the window, so nothing has to be accumulated twice.
    book::BookTradeStats tapeAtClose{};
};

//-------------------------------------------------------------------------

// A horizon's worth of bars, reduced. `bars` is what was actually covered, which is less
// than asked for during warm-up, so a consumer can tell a short window from a full one
// rather than reading a partial answer as a complete one.
struct BarWindow
{
    uint32_t bars{};
    double seconds{};
    // Telescoped, so this is exactly log(close_end / close_start) over the bars that had a
    // usable price at both ends.
    double logReturn{};
    // Realized variance over the window at bar frequency. Already summed over the horizon:
    // a consumer whose horizon IS this window uses it as is, with no rescaling.
    double realizedVariance{};
    // The same at trade frequency, and its jump-robust counterpart with the pi/2 already
    // applied. The jump part is the difference of the two.
    double tradeReturnSq{};
    double bipowerVariance{};
    double volume{};
    uint64_t tradeCount{};
    double close{};
    double midClose{};
    double twSpread{};

    // Mean log return per bar. With one-second bars this is also per second.
    [[nodiscard]] double meanReturnPerBar() const noexcept
    {
        return bars > 0 ? logReturn / bars : 0.0;
    }

    [[nodiscard]] double variancePerSecond() const noexcept
    {
        return seconds > 0.0 ? realizedVariance / seconds : 0.0;
    }

    // How far the price could plausibly move over THIS horizon, as a fraction. Dimensionless,
    // so a threshold on it is tied neither to a price level nor to a bar width nor to how the
    // variance was measured. This is the quantity to compare against when the thing being
    // decided is itself horizon-shaped, such as how wide a price range an agent will sample.
    [[nodiscard]] double horizonSigma() const noexcept { return std::sqrt(realizedVariance); }

    // The same as a rate, for deciding about the state of the market rather than about a
    // position. Scale it by a horizon in seconds to get back to the above.
    [[nodiscard]] double sigmaPerSecond() const noexcept
    {
        return std::sqrt(variancePerSecond());
    }
};

//-------------------------------------------------------------------------

// Per-book bar accumulator.
//
// Advances lazily: there is no timer and nothing schedules a close. Every entry point rolls
// the clock forward first and only then reads or mutates, so a bar is closed by whoever
// touches the series after its boundary has passed. Rolling is a pure function of `now` and
// what has accumulated, which makes it idempotent and independent of who calls it, and it
// keeps the whole mechanism off the message queue.
class BarSeries
{
public:
    BarSeries(Timestamp period, uint32_t capacity) noexcept;

    // Close out every bar whose boundary has passed. `tape` is snapshotted into each bar
    // closed, so callers that also mutate the tape must roll BEFORE recording into it.
    void roll(Timestamp now, const book::BookTradeStats& tape);

    // Fold in a trade. Whoever calls it must roll first, which is what puts a trade landing
    // on a boundary into the bar that is opening rather than the one that just closed.
    void recordTrade(double price, Timestamp now);

    // Fold in the quote as it stands. `mid` of 0 means the book is one-sided, in which case
    // the last known mid is carried: an absent quote is missing information, not a price of
    // zero, and not a reason to move the series.
    void observeQuote(double mid, double spread, Timestamp now);

    [[nodiscard]] uint32_t barsAvailable() const noexcept
    {
        return static_cast<uint32_t>(m_bars.size());
    }

    // Monotonic count of bars elapsed. `barsAvailable` saturates at the ring capacity, so it
    // cannot serve as a clock; this can. A consumer stepping a recursion once per bar records
    // what it has consumed and asks for the difference. Bars skipped over a gap longer than
    // the ring are counted here even though they are not stored, so the consumer learns that
    // time passed; it can only fold in as many returns as the ring still holds.
    [[nodiscard]] uint64_t barsClosed() const noexcept { return m_barsClosed; }

    // The last `bars` closed bars, clamped to what exists. Never includes the bar still
    // open: that is what makes the answer independent of where inside a bar the caller is.
    [[nodiscard]] BarWindow window(uint32_t bars) const;

    // The last `n` close-to-close returns, oldest first, on the fixed clock. For a consumer
    // running a recursion that assumes REGULARLY sampled returns, GARCH being the one in this
    // tree: feeding it the shared clock is not imposing our clock on the model, it is giving
    // the model the input it always assumed and never had.
    [[nodiscard]] std::vector<double> returns(uint32_t n) const;

    [[nodiscard]] Timestamp period() const noexcept { return m_period; }

private:
    void start(Timestamp now) noexcept;
    void accrue(Timestamp upTo) noexcept;
    void closeBar(const book::BookTradeStats& tape);

    Timestamp m_period;
    Timestamp m_barEnd{};
    Timestamp m_twLast{};
    Timestamp m_twElapsed{};
    bool m_started{};

    double m_lastMid{}, m_lastSpread{};
    bool m_haveMid{};
    double m_midOpen{}, m_midHigh{}, m_midLow{};
    bool m_barTouched{};
    double m_twMidSum{}, m_twSpreadSum{};

    double m_tradeOpen{}, m_tradeHigh{}, m_tradeLow{}, m_tradeClose{};
    bool m_barHasTrade{};

    double m_prevClose{};
    bool m_havePrevClose{};
    double m_cumReturn{}, m_cumReturnSq{};

    uint64_t m_barsClosed{};
    boost::circular_buffer<Bar> m_bars;
};

//-------------------------------------------------------------------------

}  // namespace taosim::stats

//-------------------------------------------------------------------------
