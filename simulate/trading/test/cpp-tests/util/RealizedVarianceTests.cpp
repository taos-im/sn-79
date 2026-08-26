/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "taosim/util/RealizedVariance.hpp"

#include <gmock/gmock.h>

#include <cmath>
#include <vector>

//-------------------------------------------------------------------------

using namespace testing;
using taosim::util::realizedVariancePerUnitTime;

namespace
{

// The estimator this replaces: population variance as E[X^2] - E[X]^2. Kept here so
// the regression tests below can demonstrate what the new form fixes.
double meanOfSquaresMinusSquareOfMean(const std::vector<double>& xs)
{
    const auto n = static_cast<double>(xs.size());
    double sum = 0.0, sumSq = 0.0;
    for (double x : xs) { sum += x; sumSq += x * x; }
    return sumSq / n - (sum / n) * (sum / n);
}

}  // namespace

//-------------------------------------------------------------------------

TEST(RealizedVarianceTest, DividesBySumOfIntervals)
{
    // Three returns of 0.1 over 1 s each: sum(r^2)/sum(dt) = 0.03/3 = 0.01.
    const std::vector<double> r{0.1, 0.1, 0.1};
    const std::vector<double> dt{1.0, 1.0, 1.0};

    EXPECT_THAT(realizedVariancePerUnitTime(r, dt), DoubleNear(0.01, 1e-15));
}

TEST(RealizedVarianceTest, IsInvariantToSamplingRate)
{
    // The same diffusion observed at two sampling rates. Over an interval dt a
    // diffusion accumulates variance sigma^2 * dt, so a coarser sample carries a
    // proportionally larger squared return. A per-unit-time estimate must agree.
    const double sigma2 = 4e-6;

    std::vector<double> rFine(64, std::sqrt(sigma2 * 1.0));
    std::vector<double> dtFine(64, 1.0);

    std::vector<double> rCoarse(8, std::sqrt(sigma2 * 8.0));
    std::vector<double> dtCoarse(8, 8.0);

    const double fine = realizedVariancePerUnitTime(rFine, dtFine);
    const double coarse = realizedVariancePerUnitTime(rCoarse, dtCoarse);

    EXPECT_THAT(fine, DoubleNear(sigma2, 1e-18));
    EXPECT_THAT(coarse, DoubleNear(sigma2, 1e-18));
    EXPECT_THAT(coarse, DoubleNear(fine, 1e-18));
}

//-------------------------------------------------------------------------
// Regressions against the previous estimator.

TEST(RealizedVarianceTest, CannotReturnNegativeOnNearStaticSeries)
{
    // A mid-quote that barely moves: near-identical returns around a large offset.
    // E[X^2] - E[X]^2 cancels catastrophically here and can go negative, which
    // inverted the sign of agent demand. A sum of squares cannot.
    std::vector<double> r;
    for (int i = 0; i < 64; ++i) {
        r.push_back(1.0 + static_cast<double>(i % 2) * 1e-13);
    }
    const std::vector<double> dt(r.size(), 1.0);

    EXPECT_THAT(realizedVariancePerUnitTime(r, dt), Ge(0.0));

    // The defect itself, pinned: on this input the previous estimator returns a
    // *negative* variance. Downstream that inverts the sign of agent demand, so the
    // residual whose root sets the indifference price becomes negative at both ends of
    // its bracket and the decision is discarded.
    const double old = meanOfSquaresMinusSquareOfMean(r);
    EXPECT_THAT(old, Lt(0.0)) << "expected the old estimator to go negative, got " << old;
}

TEST(RealizedVarianceTest, IsNonNegativeForAllConstantSeries)
{
    // Sweep offsets across magnitudes where the cancellation bites hardest.
    for (const double offset : {0.0, 1e-6, 1.0, 1e3, 1e6}) {
        const std::vector<double> r(32, offset);
        const std::vector<double> dt(32, 1.0);
        EXPECT_THAT(realizedVariancePerUnitTime(r, dt), Ge(0.0)) << "offset " << offset;
    }
}

//-------------------------------------------------------------------------
// Degenerate inputs.

TEST(RealizedVarianceTest, ReturnsZeroWhenNoIntervalIsAvailable)
{
    EXPECT_THAT(realizedVariancePerUnitTime(std::vector<double>{},
                                            std::vector<double>{}), DoubleEq(0.0));
    EXPECT_THAT(realizedVariancePerUnitTime(std::vector<double>{0.1, 0.2},
                                            std::vector<double>{0.0, 0.0}), DoubleEq(0.0));
}

TEST(RealizedVarianceTest, UsesOnlyThePairedPrefixWhenLengthsDiffer)
{
    // The return and interval buffers are filled in lockstep, but a caller must not
    // read past the shorter of the two.
    const std::vector<double> r{0.1, 0.1, 0.1, 0.1};
    const std::vector<double> dt{1.0, 1.0};

    EXPECT_THAT(realizedVariancePerUnitTime(r, dt), DoubleNear(0.02 / 2.0, 1e-15));
}
