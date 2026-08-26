/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 *
 * A settled on-chain amount must survive the trip into the engine exactly.
 *
 * The L3 log is the engine's record of what happened on chain, so it has to carry the chain's own
 * numbers. Two properties of the generic path made that impossible:
 *
 *   1. rao is 1e-9, so an on-chain amount needs NINE decimal places. double2decimal truncates at
 *      kDefaultDecimalPlaces = 8, which cannot hold one. Measured on real settlements: a settled
 *      0.999496465 alpha was recorded as 0.99949646, and 0.999496453 as 0.99949645.
 *
 *   2. The amount arrives as `rao / 1e9` computed in Python, and that double can sit just BELOW the
 *      exact decimal: 6835900/1e9 is 0.006835899999999999789..., not 0.0068359. TRUNCATION therefore
 *      drops a whole unit in the last place, turning an exactly-settled 0.0068359 into 0.00683589, a
 *      1e-8 error. That is what made trade 5:699's L3 price 0.01366500 where the tape and the
 *      per-agent view both held 0.013665023.
 *
 * chain2decimal fixes both: nine places, and rounding rather than truncation. double2decimal is left
 * exactly as it was, because its truncation is the order-grid behaviour the simulation relies on, and
 * these tests assert that separation directly so a future change cannot quietly couple them.
 */
#include <taosim/decimal/decimal.hpp>

#include <gtest/gtest.h>

//-------------------------------------------------------------------------

namespace
{

// The exact amounts, as the chain reports them: an integer count of rao.
constexpr int64_t kRaoPerUnit = 1'000'000'000;

double asValidatorSendsIt(int64_t rao)
{
    // Exactly what chainlayer_executor.py does: (ir.alpha_amount or 0) / _RAO, in double precision.
    return static_cast<double>(rao) / static_cast<double>(kRaoPerUnit);
}

// An exact decimal built from integers, since decimal_t has no string constructor and the chain's own
// representation is an integer count of rao anyway. Decimal128 divides these exactly.
taosim::decimal_t exactRao(int64_t rao)
{
    return taosim::decimal_t{static_cast<long long>(rao)}
        / taosim::decimal_t{static_cast<long long>(kRaoPerUnit)};
}

taosim::decimal_t ratio(int64_t num, int64_t den)
{
    return taosim::decimal_t{static_cast<long long>(num)}
        / taosim::decimal_t{static_cast<long long>(den)};
}

}  // namespace

//-------------------------------------------------------------------------

// The three amounts measured on real settlements, whose ninth digit the old path discarded.
TEST(ChainAmountPrecision, NinthDecimalDigitSurvives)
{
    const int64_t cases[] = {
        999'496'465,   // trade 5:698 alpha, was recorded 0.99949646
        999'496'453,   // trade 5:700 alpha, was recorded 0.99949645
        500'247'980,   // trade 5:699 alpha, ninth digit zero, was already exact
        6'835'900,     // trade 5:699 tao,   was recorded 0.00683589
    };
    for (const auto rao : cases) {
        const auto got = taosim::util::chain2decimal(asValidatorSendsIt(rao));
        EXPECT_EQ(got, exactRao(rao)) << "rao " << rao << " must arrive exactly";
    }
}

//-------------------------------------------------------------------------

// The failure this replaced: truncating at eight places loses the ninth digit outright.
TEST(ChainAmountPrecision, TheOldPathLosesTheNinthDigit)
{
    const auto viaOldPath = taosim::util::double2decimal(asValidatorSendsIt(999'496'465));
    EXPECT_NE(viaOldPath, exactRao(999'496'465))
        << "if this now passes, double2decimal changed and the two paths have been coupled";
    EXPECT_EQ(viaOldPath, ratio(99'949'646, 100'000'000));
}

//-------------------------------------------------------------------------

// The subtler failure: truncation of a double that sits below its decimal drops a WHOLE last place,
// which is an error 10x larger than the missing digit.
TEST(ChainAmountPrecision, TruncationOfABelowValueLosesAFullUnit)
{
    const auto sent = asValidatorSendsIt(6'835'900);
    ASSERT_LT(taosim::decimal_t{sent}, exactRao(6'835'900))
        << "the premise of this test is that the double sits below the exact decimal";

    const auto viaOldPath = taosim::util::double2decimal(sent);
    EXPECT_EQ(viaOldPath, ratio(683'589, 100'000'000)) << "old path: a 1e-8 loss";

    const auto viaChainPath = taosim::util::chain2decimal(sent);
    EXPECT_EQ(viaChainPath, exactRao(6'835'900)) << "chain path: exact";
}

//-------------------------------------------------------------------------

// The price is a ratio, so its exactness follows from the amounts being exact.
TEST(ChainAmountPrecision, PriceFromExactAmountsMatchesTheTape)
{
    const auto tao = taosim::util::chain2decimal(asValidatorSendsIt(6'835'900));
    const auto alpha = taosim::util::chain2decimal(asValidatorSendsIt(500'247'980));
    const auto price = tao / alpha;

    // The tape and agent_fills both recorded 0.013665023 for trade 5:699. The engine's own quotient
    // must agree with that to well inside the audit's bound, rather than the 0.01366500 it produced
    // from truncated inputs.
    const auto tapePrice = ratio(13'665'023, 1'000'000'000);
    const auto diff = price > tapePrice ? price - tapePrice : tapePrice - price;
    EXPECT_LT(diff, ratio(1, 1'000'000'000)) << "engine quotient vs tape 0.013665023";

    const auto viaOldPath =
        taosim::util::double2decimal(asValidatorSendsIt(6'835'900))
        / taosim::util::double2decimal(asValidatorSendsIt(500'247'980));
    const auto oldDiff = viaOldPath > tapePrice ? viaOldPath - tapePrice : tapePrice - viaOldPath;
    EXPECT_GT(oldDiff, ratio(2, 100'000'000))
        << "the old path's quotient was outside the audit's 2e-8 bound, which is what failed";
}

//-------------------------------------------------------------------------

// Simulation values must be untouched by the new helper's existence: this asserts the ORDER-GRID path
// still truncates, because that behaviour is relied upon and was deliberately not changed.
TEST(ChainAmountPrecision, TheOrderGridPathStillTruncates)
{
    // 0.1234 as a double is 0.12339999999999999857..., and the grid path truncates it.
    EXPECT_EQ(taosim::util::double2decimal(0.1234, 8), ratio(12'339'999, 100'000'000));
    EXPECT_EQ(taosim::util::round(taosim::decimal_t{0.1234}, 4), ratio(1'233, 10'000));
}
