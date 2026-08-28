/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>

//-------------------------------------------------------------------------

namespace taosim::util
{

//-------------------------------------------------------------------------

// Realized variance per unit time: sum(r^2) / sum(dt) over paired returns and their
// elapsed intervals.
//
// Two properties matter, and both were absent from the mean-of-squares-minus-
// square-of-mean form this replaces. Dividing by elapsed time rather than by sample
// count makes the estimate independent of how often the observer happened to sample,
// so it measures the market rather than the scheduling. And a sum of squares cannot
// cancel, so the result cannot come out negative on a near-static price series, which
// the previous form did (it inverted the sign of agent demand when it happened).
//
// Returns 0 when no interval is available; callers apply their own floor, since a
// zero-risk estimate implies an unbounded position.
template<typename Returns, typename Intervals>
[[nodiscard]] double realizedVariancePerUnitTime(
    const Returns& returns, const Intervals& intervals) noexcept
{
    const std::size_t n = std::min(std::size(returns), std::size(intervals));
    double sumSq = 0.0;
    double sumDt = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sumSq += returns[i] * returns[i];
        sumDt += intervals[i];
    }
    return sumDt > 0.0 ? sumSq / sumDt : 0.0;
}

//-------------------------------------------------------------------------

}  // namespace taosim::util

//-------------------------------------------------------------------------
