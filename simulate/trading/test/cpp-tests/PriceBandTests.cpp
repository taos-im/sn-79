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
    // Release rule state, mirroring Book::effectiveBand.
    long long lastTradeTs{0};
    bool refused{false};
    long long firstRefusalTs{0};
    long long releaseAfter{300'000'000'000LL};
    long long releaseStep{30'000'000'000LL};
    double releaseMax{0.5};

    void trade(long long ts, double price, int maker, int taker)
    {
        if (price <= 0.0) return;
        if (maker == taker) return;               // self-trades excluded (engine STP also prevents these)
        lastTradeTs = ts;
        refused = false;
        firstRefusalTs = 0;
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

    void refuse(long long ts)
    {
        if (!refused) { refused = true; firstRefusalTs = ts; }
    }

    // Mirrors Book::effectiveBand: the band in force at ts. The reference never moves here; only the
    // width does, and only while the band has refused a marketable order and nothing has printed.
    double effective(long long ts, double band) const
    {
        if (band <= 0.0 || !seeded || releaseAfter <= 0 || !refused) return band;
        const long long last = lastTradeTs != 0 ? lastTradeTs : firstRefusalTs;
        if (last == 0 || ts <= last) return band;
        const long long silent = ts - last;
        if (silent < releaseAfter) return band;
        const long long step = std::max<long long>(releaseStep, 1);
        const long long widenings = 1 + (silent - releaseAfter) / step;
        return std::min(releaseMax, band * static_cast<double>(1 + widenings));
    }

    double limitAt(long long ts, bool isBuy, double band) const
    {
        const double r = ref();
        const double b = effective(ts, band);
        if (b <= 0.0 || r <= 0.0) return isBuy ? 1e300 : -1e300;
        return isBuy ? r * (1.0 + b) : r * (1.0 - b);
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

// RELEASE. The reference is fed only by trades, so a book whose last print landed a band-width from where
// its agents will trade can never print again: every marketable order is refused and the frozen price is
// re-sampled forever. The release widens the BAND, never moves the REFERENCE: quoting cannot steer it
// (NO RATCHET still holds), a walk during the spell is bounded by the widened band, and the first print
// ends the spell.

TEST(PriceBandTest, ALockedBookWidensTheBandWithoutMovingTheReference)
{
    auto b = makeRef();
    for (long long s = 0; s < 300; ++s) b.trade(s * kSec, 300.0, 1, 2);
    const long long last = 299 * kSec;
    b.refuse(last + kSec);                                               // a sane order was stopped by the band
    EXPECT_DOUBLE_EQ(b.limitAt(last + 299 * kSec, true, 0.10), 330.0);  // not yet bandReleaseAfter
    EXPECT_DOUBLE_EQ(b.limitAt(last + 300 * kSec, true, 0.10), 360.0);  // 0.20 at bandReleaseAfter
    EXPECT_DOUBLE_EQ(b.limitAt(last + 330 * kSec, true, 0.10), 390.0);  // 0.30 one step later
    EXPECT_DOUBLE_EQ(b.limitAt(last + 900 * kSec, true, 0.10), 450.0);  // capped at bandReleaseMax
    EXPECT_DOUBLE_EQ(b.limitAt(last + 900 * kSec, false, 0.10), 150.0);
    EXPECT_DOUBLE_EQ(b.ref(), 300.0);                                    // the reference has not moved
}

TEST(PriceBandTest, TheFirstPrintEndsTheWideningSpell)
{
    auto b = makeRef();
    for (long long s = 0; s < 300; ++s) b.trade(s * kSec, 300.0, 1, 2);
    b.refuse(300 * kSec);
    const long long t = 299 * kSec + 330 * kSec;
    EXPECT_DOUBLE_EQ(b.effective(t, 0.10), 0.30);
    b.trade(t, 350.0, 1, 2);                                             // a print inside the widened band
    EXPECT_DOUBLE_EQ(b.effective(t + kSec, 0.10), 0.10);                // the configured band, at once
    // A print after a long quiet re-samples the window at its price (QuietStretchDoesNotDisableTheBand),
    // so the reference follows the reopening print rather than staying at the frozen one.
    EXPECT_DOUBLE_EQ(b.ref(), 350.0);
}

TEST(PriceBandTest, SilenceWithoutRefusalsDoesNotWiden)
{
    auto b = makeRef();
    for (long long s = 0; s < 300; ++s) b.trade(s * kSec, 300.0, 1, 2);
    EXPECT_DOUBLE_EQ(b.effective(299 * kSec + 3600 * kSec, 0.10), 0.10);  // an hour of quiet, nothing refused
}

TEST(PriceBandTest, SelfTradesDoNotEndTheWideningSpell)
{
    auto b = makeRef();
    for (long long s = 0; s < 300; ++s) b.trade(s * kSec, 300.0, 1, 2);
    b.refuse(300 * kSec);
    const long long t = 299 * kSec + 330 * kSec;
    b.trade(t, 320.0, 7, 7);                                             // a miner printing against itself
    EXPECT_DOUBLE_EQ(b.effective(t + kSec, 0.10), 0.30);                // the spell continues
}

TEST(PriceBandTest, AResumedBookCountsSilenceFromItsLastPrintOrItsFirstRefusal)
{
    // The checkpoint carries the last print with the band. A book restored with it counts silence from that
    // print, so a book locked before the resume releases on schedule after it.
    auto b = makeRef();
    b.seeded = true;
    b.samples.assign(300, 300.0);
    b.lastPrice = 300.0;
    b.lastSampleTs = 5000 * kSec;
    b.lastTradeTs = 4800 * kSec;
    b.refuse(5000 * kSec);
    EXPECT_DOUBLE_EQ(b.effective(5001 * kSec, 0.10), 0.10);   // 201 s since the print
    EXPECT_DOUBLE_EQ(b.effective(5100 * kSec, 0.10), 0.20);   // 300 s since the print
    // A checkpoint written before the field existed restores no last print; silence then counts from the
    // first refusal, never from the sample clock, which moves every time an order is processed.
    auto c = makeRef();
    c.seeded = true;
    c.samples.assign(300, 300.0);
    c.lastPrice = 300.0;
    c.lastSampleTs = 5000 * kSec;
    c.lastTradeTs = 0;
    EXPECT_DOUBLE_EQ(c.effective(9000 * kSec, 0.10), 0.10);   // no refusal yet: nothing to release
    c.refuse(5000 * kSec);
    for (long long s = 5001; s <= 5299; ++s) c.sample(s * kSec);  // orders keep arriving and re-sampling
    EXPECT_DOUBLE_EQ(c.effective(5299 * kSec, 0.10), 0.10);
    EXPECT_DOUBLE_EQ(c.effective(5300 * kSec, 0.10), 0.20);   // 300 s after the first refusal
}

TEST(PriceBandTest, ReleaseDisabledLeavesTheBandFixed)
{
    auto b = makeRef();
    b.releaseAfter = 0;
    for (long long s = 0; s < 300; ++s) b.trade(s * kSec, 300.0, 1, 2);
    b.refuse(300 * kSec);
    EXPECT_DOUBLE_EQ(b.effective(299 * kSec + 3600 * kSec, 0.10), 0.10);
}
