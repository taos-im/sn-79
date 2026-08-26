/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "taosim/util/RootFinding.hpp"

#include <gmock/gmock.h>

#include <cmath>

//-------------------------------------------------------------------------

using namespace testing;
using taosim::util::RootStatus;
using taosim::util::solveScalarBracketed;

namespace
{

// The residual the stylized traders solve for their indifference price: desired
// position minus holdings, where desired position carries a 1/price factor. This is
// the shape that defeated an unbracketed iteration started at a fixed x = 1.0.
double demandResidual(
    double price, double forecast, double variance, double base, double quote,
    double risk, double hara)
{
    return std::log(forecast / price) / (variance * price)
        * (1.0 / risk * (base * price + quote) + hara)
        - base;
}

}  // namespace

//-------------------------------------------------------------------------

TEST(RootFindingTest, FindsRootOfMonotoneResidual)
{
    const auto root = solveScalarBracketed(
        [](double x) { return 2.0 - x; }, 0.5, 10.0);

    EXPECT_THAT(root.converged, IsTrue());
    EXPECT_THAT(root.status, Eq(RootStatus::Converged));
    EXPECT_THAT(root.value, DoubleNear(2.0, 1e-9));
}

TEST(RootFindingTest, ReportsBracketAndEndpointResiduals)
{
    const auto root = solveScalarBracketed(
        [](double x) { return 2.0 - x; }, 0.5, 10.0);

    EXPECT_THAT(root.lo, DoubleEq(0.5));
    EXPECT_THAT(root.hi, DoubleEq(10.0));
    EXPECT_THAT(root.fLo, DoubleNear(1.5, 1e-12));
    EXPECT_THAT(root.fHi, DoubleNear(-8.0, 1e-12));
}

TEST(RootFindingTest, RejectsBracketWithoutSignChange)
{
    // Both ends positive: the root is above the upper bound.
    const auto above = solveScalarBracketed(
        [](double x) { return 20.0 - x; }, 0.5, 10.0);
    EXPECT_THAT(above.converged, IsFalse());
    EXPECT_THAT(above.status, Eq(RootStatus::SameSignPositive));

    // Both ends negative: the root is below the lower bound. The stylized traders
    // rely on this distinction to clamp the minimum price to one tick rather than
    // dropping the decision.
    const auto below = solveScalarBracketed(
        [](double x) { return -1.0 - x; }, 0.5, 10.0);
    EXPECT_THAT(below.converged, IsFalse());
    EXPECT_THAT(below.status, Eq(RootStatus::SameSignNegative));
}

TEST(RootFindingTest, RejectsDegenerateBracket)
{
    auto f = [](double x) { return 2.0 - x; };

    EXPECT_THAT(solveScalarBracketed(f, 5.0, 1.0).status, Eq(RootStatus::BadBracket));
    EXPECT_THAT(solveScalarBracketed(f, 1.0, 1.0).status, Eq(RootStatus::BadBracket));
    EXPECT_THAT(solveScalarBracketed(f, 0.0, 1.0).status, Eq(RootStatus::BadBracket));
}

TEST(RootFindingTest, ReportsNonFiniteBound)
{
    const auto root = solveScalarBracketed(
        [](double x) { return x < 1.0 ? std::nan("") : 2.0 - x; }, 0.5, 10.0);

    EXPECT_THAT(root.converged, IsFalse());
    EXPECT_THAT(root.status, Eq(RootStatus::NonFiniteBound));
}

TEST(RootFindingTest, AcceptsAnEndpointThatIsExactlyTheRoot)
{
    auto f = [](double x) { return 2.0 - x; };

    EXPECT_THAT(solveScalarBracketed(f, 2.0, 10.0).value, DoubleEq(2.0));
    EXPECT_THAT(solveScalarBracketed(f, 0.5, 2.0).value, DoubleEq(2.0));
}

//-------------------------------------------------------------------------

// The regression this replaces: the previous solver started every search at x = 1.0
// regardless of price level, next to the residual's 1/price singularity. Bracketing
// makes the solve independent of the price scale.
TEST(RootFindingTest, DemandResidualSolvesAcrossPriceLevels)
{
    for (const double level : {0.06, 1.0, 300.0, 65000.0}) {
        const double tick = level * 1e-5;
        const double forecast = level * 1.01;
        const double variance = 1e-4;
        const double base = 50.0;
        const double quote = 50.0 * level;

        const auto root = solveScalarBracketed(
            [&](double x) {
                return demandResidual(x, forecast, variance, base, quote, 3e6, 2.0);
            },
            tick, forecast);

        EXPECT_THAT(root.converged, IsTrue()) << "price level " << level;
        EXPECT_THAT(root.value, Gt(tick)) << "price level " << level;
        EXPECT_THAT(root.value, Lt(forecast)) << "price level " << level;
        // The recovered root is a genuine root, not merely a converged iteration.
        EXPECT_THAT(
            demandResidual(root.value, forecast, variance, base, quote, 3e6, 2.0),
            DoubleNear(0.0, 1e-6 * base)) << "price level " << level;
    }
}

TEST(RootFindingTest, DemandResidualOrdersMinimumBelowIndifference)
{
    const double level = 300.0, tick = 0.01, forecast = level * 1.01;
    const double variance = 1e-4, base = 50.0, quote = 50.0 * level;
    const double risk = 3e6, hara = 2.0;

    const auto indifference = solveScalarBracketed(
        [&](double x) { return demandResidual(x, forecast, variance, base, quote, risk, hara); },
        tick, forecast);
    ASSERT_THAT(indifference.converged, IsTrue());

    // Bracketing the minimum above by the indifference price is what makes
    // minimum <= indifference hold by construction rather than by a downstream check.
    const auto minimum = solveScalarBracketed(
        [&](double x) {
            return x * demandResidual(x, forecast, variance, base, quote, risk, hara) - quote;
        },
        tick, indifference.value);

    ASSERT_THAT(minimum.converged, IsTrue());
    EXPECT_THAT(minimum.value, Le(indifference.value));
    EXPECT_THAT(minimum.value, Ge(tick));
}
