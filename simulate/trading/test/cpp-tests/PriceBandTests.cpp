// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT
//
// PRICE BAND. A market order with no maxSlippage matches with maxPrice = numeric_limits::max(), so
// without a band a single order can sweep a thin book to any price. The band bounds how far a match
// may reach from the book's slow trailing reference.
//
// These tests pin the properties that make the band hold under pressure:
//
//   INERT BY DEFAULT   maxPriceBand = 0 must reproduce pre-band behaviour exactly, so shipping it dark
//                      cannot change a single fill.
//   BOUND              with a band, no match may print beyond ref*(1+band) / below ref*(1-band).
//   NO RATCHET         the reference must NOT be top-of-book: a band anchored on bestAsk can be walked
//                      by consuming the level, which defeats the guard.
//   BURST-PROOF        the reference samples once per interval, so trade COUNT inside one interval has no
//                      leverage -- flooding a batch cannot drag it.
//   MEDIAN             moving the reference needs >50% of the window's samples, i.e. sustained control
//                      over most of bandRefWindow rather than a single burst.
//   NEVER DISABLES     a quiet stretch longer than the window must not empty the reference; an empty
//                      reference leaves the sweep unbounded exactly when the book is thinnest.
#include <gtest/gtest.h>

#include <algorithm>
#include <deque>
#include <utility>
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

// ── THE WALK BOUND IS NOT THE RESERVATION FIGURE ─────────────────────────────────────────────────
//
// checkIOC decides whether a book can fill an IOC by walking the crossing side and accumulating
// volume, and it bounded that walk with sweepCap. sweepCap answers a different question: what should
// this order RESERVE. For a banded BUY the answer is deliberately 0, because the band can veto the
// cheap fills a sweep prediction would count on, so the order must reserve its full volume at its
// limit price. Used as a price bound that 0 compares against every level price, so the walk broke on
// the FIRST level, collected nothing, and every buy IOC was refused CONTRACT_VIOLATION whatever its
// price or size -- on a simulation-acceptance run, `BUY 1.0@312.35` against a book
// holding 20.8 alpha at 285.5. The sell side was unaffected because sweepCap's sell branch happens to
// return a real price bound; matchCap gives the buy side the same shape.
namespace
{

constexpr double kNoCap = 1e300;
constexpr double kNoFloor = -1e300;

// Mirrors OrderPlacementValidator::sweepCap: a RESERVATION figure.
double sweepCap(bool isBuy, double price, double bandLimit)
{
    if (isBuy) return bandLimit >= kNoCap ? price : 0.0;
    return bandLimit <= kNoFloor ? price : std::max(price, bandLimit);
}

// Mirrors OrderPlacementValidator::matchCap: how far a walk may REACH.
double matchCap(bool isBuy, double price, double bandLimit)
{
    if (isBuy) return bandLimit >= kNoCap ? price : std::min(price, bandLimit);
    return bandLimit <= kNoFloor ? price : std::max(price, bandLimit);
}

// Mirrors checkIOC's accumulation: walk the crossing side until the bound is passed.
double collectedForBuy(const std::vector<std::pair<double, double>>& asks, double bound)
{
    double collected = 0.0;
    for (const auto& [px, qty] : asks) {
        if (bound < px) break;
        collected += qty;
    }
    return collected;
}

}  // namespace

TEST(PriceBandTest, ReservationFigureUsedAsAWalkBoundRefusesEveryBuyIoc)
{
    // The book that was refused: 20.8 alpha at the touch against a 1.0 order.
    const std::vector<std::pair<double, double>> asks{{285.5, 20.8}, {285.6, 14.3}};
    const double orderPx = 312.35;
    const double buyLimit = 283.0 * 1.10;   // trailing reference 283, band 0.10

    // What the defect did: the reservation figure bounds the walk, nothing is collected.
    EXPECT_DOUBLE_EQ(sweepCap(true, orderPx, buyLimit), 0.0);
    EXPECT_DOUBLE_EQ(collectedForBuy(asks, sweepCap(true, orderPx, buyLimit)), 0.0);

    // What the walk bound does: reach as far as the order and the band allow, and fill.
    EXPECT_DOUBLE_EQ(matchCap(true, orderPx, buyLimit), buyLimit);
    EXPECT_GT(collectedForBuy(asks, matchCap(true, orderPx, buyLimit)), 1.0);
}

TEST(PriceBandTest, WalkBoundNeverReachesBeyondTheBandOrTheOrder)
{
    const double buyLimit = 300.0 * 1.10;   // 330

    // An order priced ABOVE the band is capped by the band: it may not print beyond ref*(1+band).
    EXPECT_DOUBLE_EQ(matchCap(true, 400.0, buyLimit), buyLimit);
    // An order priced BELOW the band is capped by itself: the band never widens an order's reach.
    EXPECT_DOUBLE_EQ(matchCap(true, 310.0, buyLimit), 310.0);

    const double sellFloor = 300.0 * 0.90;  // 270
    EXPECT_DOUBLE_EQ(matchCap(false, 200.0, sellFloor), sellFloor);
    EXPECT_DOUBLE_EQ(matchCap(false, 290.0, sellFloor), 290.0);
}

TEST(PriceBandTest, WithTheBandOffTheWalkBoundIsTheOrdersOwnPrice)
{
    // The property every band-off determinism result depends on: with no band, the new bound returns
    // the order's price, exactly as sweepCap did, so no band-off fill can change.
    for (const double px : {0.0001, 1.0, 312.35, 1e6}) {
        EXPECT_DOUBLE_EQ(matchCap(true, px, kNoCap), px);
        EXPECT_DOUBLE_EQ(matchCap(false, px, kNoFloor), px);
        EXPECT_DOUBLE_EQ(matchCap(true, px, kNoCap), sweepCap(true, px, kNoCap));
        EXPECT_DOUBLE_EQ(matchCap(false, px, kNoFloor), sweepCap(false, px, kNoFloor));
    }
}

TEST(PriceBandTest, TheSellSideIsUnchangedByTheWalkBound)
{
    // sweepCap's sell branch was already a price bound, so matchCap must agree with it everywhere --
    // otherwise this fix would silently alter sell behaviour under a band.
    const double sellFloor = 271.5;
    for (const double px : {100.0, 271.5, 300.0, 1e5}) {
        EXPECT_DOUBLE_EQ(matchCap(false, px, sellFloor), sweepCap(false, px, sellFloor));
    }
}
