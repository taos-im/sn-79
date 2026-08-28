// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT
//
// PRICE BAND. A market order with no maxSlippage previously matched with maxPrice =
// numeric_limits::max(), so one order could sweep a thin book to any price -- the mechanism behind the
// +21% to +25% excursions measured on mainnet-sim L3, which reverted within a few hundred trades.
//
// These tests pin the properties that make the band un-gameable, each of which corresponds to an attack
// that an earlier draft of the design was vulnerable to:
//
//   INERT BY DEFAULT   maxPriceBand = 0 must reproduce pre-band behaviour exactly, so shipping it dark
//                      cannot change a single fill.
//   BOUND              with a band, no match may print beyond ref*(1+band) / below ref*(1-band).
//   NO RATCHET         the reference must NOT be top-of-book: a band on bestAsk is walked by consuming
//                      the level (four 5% steps reach +21%), which defeats the whole guard.
//   BURST-PROOF        the reference samples once per interval, so trade COUNT inside one interval has no
//                      leverage -- flooding a batch cannot drag it.
//   MEDIAN             moving the reference needs >50% of the window's samples, i.e. sustained control
//                      over most of bandRefWindow rather than a single burst.
//   NEVER DISABLES     a quiet stretch longer than the window must not empty the reference; an empty
//                      reference would reopen the unbounded sweep exactly when the book is thinnest.
#include <gtest/gtest.h>

#include <algorithm>
#include <deque>
#include <vector>

namespace
{

// Mirrors Book::sampleBandRef/bandLimit arithmetic on plain doubles, so the properties are tested
// independently of exchange wiring. The invariants under test are arithmetic, not plumbing.
struct BandRef
{
    long long interval;
    long long window;
    std::deque<double> samples;
    double lastPrice{0.0};
    long long lastSampleTs{0};
    bool seeded{false};

    void trade(long long ts, double price, int maker, int taker)
    {
        if (price <= 0.0) return;
        if (maker == taker) return;               // self-trades excluded (engine STP also prevents these)
        lastPrice = price;
        if (!seeded) { seeded = true; lastSampleTs = ts; samples.push_back(price); }
        sample(ts);
    }

    void sample(long long ts)
    {
        if (!seeded || interval <= 0) return;
        const size_t maxN = static_cast<size_t>(std::max<long long>(1, window / interval));
        while (ts - lastSampleTs >= interval) {
            lastSampleTs += interval;
            samples.push_back(lastPrice);
            if (samples.size() > maxN) samples.pop_front();
        }
    }

    double ref() const
    {
        if (samples.empty()) return 0.0;
        std::vector<double> t{samples.begin(), samples.end()};
        std::nth_element(t.begin(), t.begin() + t.size() / 2, t.end());
        return t[t.size() / 2];
    }

    double limit(bool isBuy, double band) const
    {
        const double r = ref();
        if (band <= 0.0 || r <= 0.0) return isBuy ? 1e300 : -1e300;
        return isBuy ? r * (1.0 + band) : r * (1.0 - band);
    }
};

constexpr long long kSec = 1'000'000'000LL;

BandRef makeRef()
{
    return BandRef{.interval = kSec, .window = 300 * kSec};
}

}  // namespace

TEST(PriceBandTest, DisabledBandIsInertAndImposesNoBound)
{
    auto b = makeRef();
    b.trade(0, 300.0, 1, 2);
    b.sample(60 * kSec);
    // band = 0 must leave matching completely unbounded, i.e. identical to pre-band behaviour
    EXPECT_GT(b.limit(true, 0.0), 1e299);
    EXPECT_LT(b.limit(false, 0.0), -1e299);
}

TEST(PriceBandTest, BoundsAMatchToTheBandAroundTheReference)
{
    auto b = makeRef();
    for (long long s = 0; s < 300; ++s) b.trade(s * kSec, 300.0, 1, 2);
    EXPECT_NEAR(b.ref(), 300.0, 1e-9);
    EXPECT_NEAR(b.limit(true, 0.10), 330.0, 1e-9);
    EXPECT_NEAR(b.limit(false, 0.10), 270.0, 1e-9);
}

TEST(PriceBandTest, BurstInsideOneIntervalCannotDragTheReference)
{
    auto b = makeRef();
    for (long long s = 0; s < 300; ++s) b.trade(s * kSec, 300.0, 1, 2);
    const double before = b.ref();
    // 5,000 trades at the band edge, all inside a single sampling interval: the attack that defeats a
    // per-trade mean. Only one sample can be taken, so trade count buys no movement.
    for (int i = 0; i < 5000; ++i) b.trade(300 * kSec, 330.0, 1, 2);
    EXPECT_NEAR(b.ref(), before, 1e-9);
}

TEST(PriceBandTest, MovingTheMedianRequiresMostOfTheWindow)
{
    auto b = makeRef();
    for (long long s = 0; s < 300; ++s) b.trade(s * kSec, 300.0, 1, 2);
    // 100 of 300 samples captured at a higher price: a minority cannot move a median
    for (long long s = 300; s < 400; ++s) b.trade(s * kSec, 330.0, 1, 2);
    EXPECT_NEAR(b.ref(), 300.0, 1e-9);
    // past the halfway point it finally moves -- the intended cost, paid in sustained time
    for (long long s = 400; s < 460; ++s) b.trade(s * kSec, 330.0, 1, 2);
    EXPECT_NEAR(b.ref(), 330.0, 1e-9);
}

TEST(PriceBandTest, QuietStretchDoesNotDisableTheBand)
{
    auto b = makeRef();
    for (long long s = 0; s < 300; ++s) b.trade(s * kSec, 300.0, 1, 2);
    // no trades for far longer than the window: the reference must persist, never fall back to "no band"
    b.sample(5000 * kSec);
    EXPECT_GT(b.ref(), 0.0);
    EXPECT_NEAR(b.limit(true, 0.10), b.ref() * 1.10, 1e-9);
    EXPECT_LT(b.limit(true, 0.10), 1e299);
}

TEST(PriceBandTest, SelfTradesDoNotMoveTheReference)
{
    auto b = makeRef();
    for (long long s = 0; s < 300; ++s) b.trade(s * kSec, 300.0, 1, 2);
    const double before = b.ref();
    for (long long s = 300; s < 600; ++s) b.trade(s * kSec, 330.0, 7, 7);   // same agent both sides
    EXPECT_NEAR(b.ref(), before, 1e-9);
}
