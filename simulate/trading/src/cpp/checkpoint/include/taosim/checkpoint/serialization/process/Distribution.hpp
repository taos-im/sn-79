/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <locale>
#include <sstream>
#include <string>
#include <string_view>

//-------------------------------------------------------------------------

namespace taosim::checkpoint::serialization
{

// A distribution's state as the standard library writes it: its parameters and any value it holds back for the next
// draw (std::normal_distribution draws in pairs, std::poisson_distribution holds a normal of its own), at full
// precision, so a resumed process continues the uninterrupted stream.
template<typename D>
[[nodiscard]] std::string distributionState(const D& d)
{
    std::ostringstream s;
    s.imbue(std::locale::classic());
    s << d;
    return s.str();
}

template<typename D>
void restoreDistribution(D& d, std::string_view state)
{
    std::istringstream s{std::string{state}};
    s.imbue(std::locale::classic());
    D restored;
    if (s >> restored) {
        d = restored;
    }
}

}  // namespace taosim::checkpoint::serialization

//-------------------------------------------------------------------------
