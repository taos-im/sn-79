/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

/*
 * The maker used to replace its quotes on a private timer and schedule a cancel for every
 * order it placed, so the whole population churned continuously whether or not anything had
 * happened. It now keeps one quote a side and changes them for two reasons only: it was hit,
 * or the class's token reached it.
 *
 * What can silently break: the fill reaction stops firing and the maker goes quiet between
 * its turns, or the resting-quote ids drift out of step with the book and old quotes are
 * never cancelled, which accumulates. Both look like a working run from the outside, so the
 * tests here are aimed at exactly those.
 *
 * Driven through a bare Simulation, so no POSIX IPC object is created and the test cannot
 * collide with another taosim on the host.
 */

#include <taosim/agent/HighFrequencyTraderAgent.hpp>

#include "MultiBookExchangeAgent.hpp"
#include "Simulation.hpp"
#include "util.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

//-------------------------------------------------------------------------

using namespace taosim;
using namespace taosim::literals;

using namespace testing;

namespace fs = std::filesystem;

//-------------------------------------------------------------------------

static const auto s_dataPath = fs::path{__FILE__}.parent_path() / "data";

//-------------------------------------------------------------------------

class HighFrequencyQuoteCycleTest : public Test
{
protected:
    static constexpr auto kHft = "HIGH_FREQUENCY_TRADER_AGENT";
    static constexpr BookId kBook = 0;

    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange{};
    std::vector<const agent::HighFrequencyTraderAgent*> makers;
    std::map<AgentId, uint64_t> placements;
    std::map<AgentId, uint64_t> fills;
    Timestamp firstQuoteAt{std::numeric_limits<Timestamp>::max()};
    std::vector<boost::signals2::scoped_connection> feeds;

    void runSimulation(const std::string& fixture = "WakeupChain.xml")
    {
        nodes = taosim::util::parseSimulationFile(s_dataPath / fixture);
        simulation = std::make_unique<Simulation>();
        simulation->setDebug(false);
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        ASSERT_NE(exchange, nullptr);

        std::vector<AgentId> ids;
        for (const auto& localAgent : simulation->agents()) {
            const auto* maker =
                dynamic_cast<const agent::HighFrequencyTraderAgent*>(localAgent.get());
            if (maker == nullptr) continue;
            makers.push_back(maker);
            ids.push_back(exchange->accounts().lookupLocalAgentId(localAgent->name()));
        }
        ASSERT_THAT(makers, Not(IsEmpty()));

        for (auto& book : exchange->books()) {
            feeds.emplace_back(book->signals().orderCreated.connect(
                [this, ids](Order::Ptr, OrderContext ctx) {
                    if (!ranges::contains(ids, ctx.agentId)) return;
                    ++placements[ctx.agentId];
                    firstQuoteAt = std::min(firstQuoteAt, simulation->currentTimestamp());
                }));
            feeds.emplace_back(book->signals().trade.connect(
                [this, ids, &book](Trade::Ptr trade, BookId bookId) {
                    const auto& info = book->orderToClientInfo();
                    const auto it = info.find(trade->restingOrderID());
                    if (it == info.end()) return;
                    if (ranges::contains(ids, it->second.agentId)) ++fills[it->second.agentId];
                }));
        }

        simulation->simulate();
    }

    [[nodiscard]] uint64_t chainTurns(BookId bookId) const
    {
        const auto* chain = exchange->wakeupChains().at(bookId).find(kHft);
        return chain != nullptr ? chain->wakeCount : 0;
    }
};

// The rebalance gate is shut in any realistic configuration, so the tests that need it open
// run against their own fixture. Separated rather than folded in so it is obvious that the
// numbers forcing the gate are not calibration.
class HighFrequencyRebalanceTest : public HighFrequencyQuoteCycleTest
{
};

//-------------------------------------------------------------------------

// The routine churn is on the token now, so the class refreshes at one rate no matter how
// many makers are configured. If the chain never registered or never hopped, the makers
// would be quoting only when hit, which still produces orders and would otherwise pass.
TEST_F(HighFrequencyQuoteCycleTest, RoutineChurnRidesTheChain)
{
    ASSERT_NO_FATAL_FAILURE(runSimulation());

    for (BookId bookId : {kBook, BookId{1}}) {
        const auto* chain = exchange->wakeupChains().at(bookId).find(kHft);
        ASSERT_NE(chain, nullptr) << "book " << bookId << " has no maker chain registered";
        EXPECT_GT(chain->wakeCount, 0u) << "book " << bookId << " chain never hopped";
        EXPECT_EQ(chain->reseedCount, 0u) << "book " << bookId << " healthy chain was repaired";
        EXPECT_EQ(chain->duplicateCount, 0u) << "book " << bookId << " chain forked";
    }
}

//-------------------------------------------------------------------------

// The invariant: one quote a side. A requote cancels and places with independent latencies,
// so two of a side can be in flight together for a moment, but nothing beyond that. An
// id ledger drifting out of step with the book shows up here as unbounded growth, which is
// the failure the old scheduled-cancel design could not have.
TEST_F(HighFrequencyQuoteCycleTest, KeepsAtMostOneQuoteASideResting)
{
    ASSERT_NO_FATAL_FAILURE(runSimulation());

    for (const auto* maker : makers) {
        const auto agentId = exchange->accounts().lookupLocalAgentId(maker->name());
        const auto& account = exchange->accounts()[agentId];
        for (BookId bookId : {kBook, BookId{1}}) {
            // Counting what is actually working, not what is in the set. A fully-filled
            // order stays in the account's active orders and stays findable in the book
            // until the engine reaps it, so the raw set size is not a quote count.
            std::map<OrderDirection, int> perSide;
            for (const auto& order : account.activeOrders().at(bookId)) {
                if (order->totalVolume() <= 0_dec) continue;
                ++perSide[order->direction()];
            }
            for (const auto& [direction, count] : perSide) {
                EXPECT_LE(count, 2)
                    << maker->name() << " book " << bookId << " holds " << count
                    << " resting orders on one side; the ledger has lost track of what it "
                       "has out";
            }
        }
    }
}

//-------------------------------------------------------------------------

// The fill reaction is the thing keeping a maker responsive between its turns, so it is now
// load-bearing rather than a nicety. A stuck coalescing flag would disable it silently: the
// maker would still quote on its turns and still look alive.
TEST_F(HighFrequencyQuoteCycleTest, FillReactionIsArmedAndNotStuck)
{
    ASSERT_NO_FATAL_FAILURE(runSimulation());

    // Asserting the coupling rather than that everything is clear: a fill in the last
    // moments of a run legitimately leaves a reaction in flight, and a test that forbade
    // that would be asserting on where the run happened to stop. What must never happen is
    // the flag standing with nothing pending, which is the state that swallows every later
    // fill without a trace.
    for (const auto* maker : makers) {
        for (BookId bookId : {kBook, BookId{1}}) {
            const bool scheduled = maker->fillWakeScheduled().at(bookId);
            const bool pending =
                maker->pendingBid().at(bookId) || maker->pendingAsk().at(bookId);
            EXPECT_EQ(scheduled, pending)
                << maker->name() << " book " << bookId << ": reaction in flight = "
                << scheduled << " but sides pending = " << pending
                << ". A standing flag with nothing pending means no later fill can ever "
                   "schedule a reaction";
        }
    }
}

//-------------------------------------------------------------------------

// Being hit has to produce requotes between turns. Counting the maker's own reactive
// requotes rather than inferring them from placements: a turn does not always place, since
// the rebalance branch and the affordability checks can both come out empty, so placements
// are not a fixed multiple of turns.
TEST_F(HighFrequencyQuoteCycleTest, BeingHitProducesRequotesBetweenTurns)
{
    ASSERT_NO_FATAL_FAILURE(runSimulation());

    const uint64_t totalFills = ranges::accumulate(fills | views::values, uint64_t{});
    if (totalFills == 0) {
        GTEST_SKIP() << "no maker was hit in this run, so the fill path was never exercised";
    }

    uint64_t reactive{};
    for (const auto* maker : makers) {
        reactive += ranges::accumulate(maker->fillRequotes(), uint64_t{});
    }
    EXPECT_GT(reactive, 0u)
        << totalFills << " maker fills produced no reactive requote at all, so the makers "
           "are quoting only on their turns";
}

//-------------------------------------------------------------------------

// The reservation price is the mid less an inventory skew, and that skew is now converted to
// the current price level along with the spread. Asserting it stays a sane FRACTION of the
// mid rather than a sane absolute distance: a term left on the wrong scale shows up here as a
// skew that is fine at the starting price and absurd once price has moved, which is precisely
// the failure the conversion exists to prevent.
//
// This is a stability guard, not a proof of invariance. The invariance claim wants a
// scale sweep over a whole config, which belongs in one general detector rather than here.
TEST_F(HighFrequencyQuoteCycleTest, ReservationPriceStaysAFractionOfTheMid)
{
    ASSERT_NO_FATAL_FAILURE(runSimulation());

    for (const auto* maker : makers) {
        for (BookId bookId : {kBook, BookId{1}}) {
            const double mid = maker->lastPrice().at(bookId).price;
            const double reservation = maker->pRes().at(bookId);
            if (!(mid > 0.0)) continue;

            EXPECT_GT(reservation, 0.0)
                << maker->name() << " book " << bookId
                << " reservation price went non-positive, which a price-space term on the "
                   "wrong scale does as soon as inventory grows";
            EXPECT_LT(std::abs(reservation - mid) / mid, 1.0)
                << maker->name() << " book " << bookId << ": reservation " << reservation
                << " against mid " << mid << ", a skew of more than the price itself";
        }
    }
}

//-------------------------------------------------------------------------

// The rebalance gate divides inventory by a scale stated in the maker's OWN average quote,
// which is a unit that means something and does not move when somebody changes the funding.
// The fixture asks for five quotes; with a lognormal(2.5, 1) draw the mean quote is about 20,
// so the resolved scale should land near 100 without being exactly it.
TEST_F(HighFrequencyQuoteCycleTest, InventoryScaleIsInUnitsOfTheAgentsOwnQuote)
{
    ASSERT_NO_FATAL_FAILURE(runSimulation());

    const double meanQuote = std::exp(2.5 + 0.5 * 1.0 * 1.0);
    for (const auto* maker : makers) {
        const double scale = maker->psi();
        EXPECT_GT(scale, 0.0) << maker->name() << " has no inventory scale, so the gate "
                                 "divides by zero";
        EXPECT_THAT(scale, AllOf(Gt(5.0 * meanQuote * 0.5), Lt(5.0 * meanQuote * 1.5)))
            << maker->name() << " scale " << scale << " is nowhere near the configured "
               "five average quotes";
    }
}

//-------------------------------------------------------------------------

// The makers do not quote into a market that has not opened. Before the entry delay the book
// is empty and nobody has established a level, so a maker quoting there would be setting the
// price rather than pricing against it. Half the exchange's grace period by default, which
// also leaves the level free to drift before liquidity provision tightens it.
TEST_F(HighFrequencyQuoteCycleTest, MakersWaitForHalfTheGracePeriodBeforeQuoting)
{
    ASSERT_NO_FATAL_FAILURE(runSimulation());

    // The fixture sets a 10 s grace, so the class should be silent for the first 5 s.
    static constexpr Timestamp kEntry = 5'000'000'000;
    ASSERT_LT(firstQuoteAt, std::numeric_limits<Timestamp>::max())
        << "the makers never quoted at all, so this asserts nothing";
    EXPECT_GE(firstQuoteAt, kEntry)
        << "a maker quoted at " << firstQuoteAt << ", inside the entry delay";
    // And they do arrive: silent for the whole run would pass the check above.
    EXPECT_LT(firstQuoteAt, 2 * kEntry)
        << "the makers took far longer than the entry delay to appear";
}


//-------------------------------------------------------------------------

// Shedding inventory used to CONSUME the round: a maker that sent its rebalance market order
// placed no quotes, so it took liquidity and withdrew its own in one action, precisely when
// the book was under pressure. That was survivable while each maker ran its own timer and
// returned within milliseconds. On the class token a maker that steps out stays out until its
// next turn or its next fill, which made the withdrawal the dominant effect and left the
// hot-potato loop damped by everyone leaving rather than by anyone quoting.
//
// The two now happen together. Asserted through the maker's own counters rather than by
// counting orders, because a rebalance and a quote are indistinguishable downstream.
TEST_F(HighFrequencyRebalanceTest, ShedingInventoryDoesNotTakeTheMakerOutOfTheBook)
{
    ASSERT_NO_FATAL_FAILURE(runSimulation("HFTRebalance.xml"));

    uint64_t rebalances{};
    for (const auto* maker : makers) {
        rebalances += ranges::accumulate(maker->rebalances(), uint64_t{});
    }
    ASSERT_GT(rebalances, 0u)
        << "the forcing fixture did not open the rebalance gate, so these tests assert "
           "nothing. Check skipProb/rateProb/inventoryProb against HFTRebalance.xml";

    // Every maker that rebalanced is still quoting at the end of the run. A maker left out of
    // the book by its last rebalance shows up here as a side with nothing live on it while
    // nothing is in flight to replace it.
    for (const auto* maker : makers) {
        for (BookId bookId : {kBook, BookId{1}}) {
            if (maker->rebalances().at(bookId) == 0) continue;
            const auto& bid = maker->restingBid().at(bookId);
            const auto& ask = maker->restingAsk().at(bookId);
            EXPECT_TRUE(bid.live || bid.inFlight || ask.live || ask.inFlight)
                << maker->name() << " book " << bookId << " shed inventory "
                << maker->rebalances().at(bookId)
                << " times and has nothing resting or in flight on either side, so its last "
                   "rebalance took it out of the book";
        }
    }
}

//-------------------------------------------------------------------------

// NO TEST FOR THE OVERSHOOT GUARD, and the reason is worth stating so nobody assumes there is.
//
// A long maker sheds by selling into the bid; if its passive ask were still priced against the
// inventory it is shedding, it would repeat the same offload and land SHORT when both fill.
// Inventory crossing through flat and out the other side is what keeps a hot potato moving
// rather than ending it, so the quote is priced against the inventory the maker EXPECTS to
// hold: `quoteInventory` in placeOrder.
//
// That cannot be asserted here. The only yardstick for "too much inventory" is the maker's own
// scale `psi`, and the fixture that opens the rebalance gate does so partly by shrinking psi to
// about a hundredth of a sane value, so every ratio against it is meaningless by construction.
// Any threshold picked would be measuring the fixture.
//
// What it needs is a per-decision trace: inventory before, rebalance quantity, quote prices,
// inventory after. That is item 3 of the next steps in notes/direct_state_and_wakeup.md, and
// the same gap blocks the sigmaC refit. Until then the guard is reasoned, not verified.
