/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

// The uniform-50 endowment draws each agent's cash share of its wealth from [qMin, qMax], unset U[0, 1]. A cash skew
// for a class (the lever named for the third class's drift) is therefore a configuration value: these tests hold the
// range, the unchanged default and the seeded stream that makes a swept pair reproducible.

#include "taosim/accounting/Balances.hpp"
#include <taosim/decimal/decimal.hpp>

#include <gtest/gtest.h>
#include <pugixml.hpp>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

//-------------------------------------------------------------------------

using namespace taosim;
using namespace taosim::accounting;

namespace
{

constexpr RoundParams kRound{.baseDecimals = 4, .quoteDecimals = 8};
constexpr double kPrice = 2.0;
constexpr int kDraws = 1000;

// The cash shares of kDraws endowments drawn with the given extra attributes from one seeded stream.
std::vector<double> cashShares(const std::vector<std::pair<std::string, double>>& extra, unsigned seed = 7)
{
    pugi::xml_document doc;
    auto node = doc.append_child("Balances");
    node.append_attribute("type") = "uniform-50";
    node.append_attribute("wealthMin") = 100.0;
    node.append_attribute("wealthMax") = 200.0;
    node.append_attribute("price") = kPrice;
    node.append_attribute("symbol") = "X";
    for (const auto& [name, value] : extra) node.append_attribute(name.c_str()) = value;

    std::mt19937 rng{seed};
    std::vector<double> shares;
    for (int i = 0; i < kDraws; ++i) {
        const auto bals = Balances::fromXML(node, kRound, &rng);
        const double quote = util::decimal2double(bals.quote->getTotal());
        const double base = util::decimal2double(bals.base.getTotal());
        shares.push_back(quote / (quote + base * kPrice));
    }
    return shares;
}

}  // namespace

//-------------------------------------------------------------------------

TEST(EndowmentShare, EveryCashShareFallsInsideTheConfiguredRange)
{
    const auto shares = cashShares({{"qMin", 0.6}, {"qMax", 0.8}});
    const auto [lo, hi] = std::ranges::minmax(shares);
    EXPECT_GE(lo, 0.6 - 1e-3) << "a share was drawn below qMin";
    EXPECT_LE(hi, 0.8 + 1e-3) << "a share was drawn above qMax";
    EXPECT_LT(lo, 0.62) << "the draw does not reach the bottom of the range";
    EXPECT_GT(hi, 0.78) << "the draw does not reach the top of the range";
}

TEST(EndowmentShare, UnsetTheShareSpansZeroToOneAsShipped)
{
    const auto shares = cashShares({});
    const auto [lo, hi] = std::ranges::minmax(shares);
    EXPECT_LT(lo, 0.05);
    EXPECT_GT(hi, 0.95);
}

TEST(EndowmentShare, TheSameSeedDrawsTheSameEndowments)
{
    EXPECT_EQ(cashShares({{"qMin", 0.6}, {"qMax", 0.8}}, 11), cashShares({{"qMin", 0.6}, {"qMax", 0.8}}, 11));
    EXPECT_NE(cashShares({{"qMin", 0.6}, {"qMax", 0.8}}, 11), cashShares({{"qMin", 0.6}, {"qMax", 0.8}}, 12));
}

//-------------------------------------------------------------------------
