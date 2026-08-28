/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/decimal/decimal.hpp>
#include <Timestamp.hpp>

#include <cmath>
#include <cstdint>

//-------------------------------------------------------------------------

namespace taosim::book
{

//-------------------------------------------------------------------------

// Monotonic per-book trade accumulator, the source for RETRIEVE_L1_EXT.
//
// Every field only ever grows, so a reader obtains interval statistics by
// differencing two reads of its own; the exchange keeps no per-subscriber state
// and any number of agents may poll at any cadence. The L3 record cannot serve
// this purpose since it is cleared every step, well below a typical poll period.
struct BookTradeStats
{
    uint64_t tradeCount{};
    decimal_t volumeSum{};
    decimal_t notionalSum{};
    // Log-return moments across consecutive trades, the realized-variance terms.
    // These are statistics rather than monetary amounts, and double is the correct
    // type for them: a 1bp return squares to ~1e-8, which util::double2decimal
    // truncates away entirely at its default 8 decimal places. They stay double all
    // the way onto the wire; only the exact monetary sums above are decimal.
    double logReturnSum{};
    double logReturnSqSum{};
    // Realized bipower variation, sum |r_i|*|r_{i-1}|. Scaled by pi/2 it estimates the
    // CONTINUOUS part of the variance and is robust to jumps, because a jump enters only
    // through products with its two finite neighbours rather than being squared.
    //
    // This matters for anything that widens quotes on rising volatility: fed plain
    // realized variance, one jump inflates the estimate and the maker withdraws during
    // the crash, amplifying it. With both series available a consumer can respond to
    // diffusive volatility and treat jumps under a separate explicit policy, instead of
    // crash amplification emerging by accident. The jump part is recoverable as
    // logReturnSqSum - (pi/2)*bipowerSum.
    double bipowerSum{};
    double lastAbsLogReturn{};
    decimal_t lastTradePrice{};
    Timestamp lastTradeTime{};

    void record(decimal_t price, decimal_t volume, Timestamp timestamp) noexcept
    {
        const double px = util::decimal2double(price);
        const double prevPx = util::decimal2double(lastTradePrice);
        if (tradeCount > 0 && px > 0.0 && prevPx > 0.0) {
            const double logReturn = std::log(px / prevPx);
            logReturnSum += logReturn;
            logReturnSqSum += logReturn * logReturn;
            // Needs a pair, so this starts contributing one return after the squares do.
            const double absLogReturn = std::abs(logReturn);
            bipowerSum += absLogReturn * lastAbsLogReturn;
            lastAbsLogReturn = absLogReturn;
        }
        ++tradeCount;
        volumeSum += volume;
        notionalSum += price * volume;
        lastTradePrice = price;
        lastTradeTime = timestamp;
    }
};

//-------------------------------------------------------------------------

}  // namespace taosim::book

//-------------------------------------------------------------------------
