/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

/*
 * StylizedTrader used to keep its own copy of the top of book, refreshed by a poll running
 * on an interval unrelated to when it decided, and its own variance estimator fed by that
 * same poll. Both are gone: the price is read live from the exchange and the variance and
 * drift come off the shared bar clock.
 *
 * The poll is what made the agent's numbers exist at all, so the tests here are the ones
 * that would fail if removing it had quietly turned the agent into a no-op: it still places
 * orders, and the horizon it reasons over is actually populated.
 *
 * Driven through a bare Simulation, so no POSIX IPC object is created and the test cannot
 * collide with another taosim on the host.
 */

#include <taosim/agent/ALGOTraderAgent.hpp>
#include <taosim/agent/StylizedTraderAgent.hpp>
#include <taosim/statshub/StatsHub.hpp>

#include "MultiBookExchangeAgent.hpp"
#include "Simulation.hpp"
#include "util.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

//-------------------------------------------------------------------------

using namespace taosim;
using namespace testing;

namespace fs = std::filesystem;

//-------------------------------------------------------------------------

static const auto s_dataPath = fs::path{__FILE__}.parent_path() / "data";

//-------------------------------------------------------------------------

class StylizedTraderBarsTest : public Test
{
protected:
    static constexpr BookId kBook = 0;

    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange{};
    std::vector<AgentId> stylizedIds;
    std::vector<AgentId> algoIds;
    uint64_t stylizedOrders{};
    uint64_t algoOrders{};
    std::vector<boost::signals2::scoped_connection> feeds;

    void runSimulation()
    {
        nodes = taosim::util::parseSimulationFile(s_dataPath / "WakeupChain.xml");
        simulation = std::make_unique<Simulation>();
        simulation->setDebug(false);
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        ASSERT_NE(exchange, nullptr);

        // Counting the agent's own orders rather than the book's total: HFT and the other
        // background classes quote on the same book, so a total would pass even with the
        // stylized traders silent.
        //
        // Wired here rather than off the agentsCreated signal, which has already fired by
        // the time configure() returns.
        for (const auto& localAgent : simulation->agents()) {
            const auto id = [&] { return exchange->accounts().lookupLocalAgentId(localAgent->name()); };
            if (dynamic_cast<const agent::StylizedTraderAgent*>(localAgent.get()) != nullptr) {
                stylizedIds.push_back(id());
            }
            else if (dynamic_cast<const agent::ALGOTraderAgent*>(localAgent.get()) != nullptr) {
                algoIds.push_back(id());
            }
        }
        for (auto& book : exchange->books()) {
            feeds.emplace_back(book->signals().orderCreated.connect(
                [this](Order::Ptr, OrderContext ctx) {
                    if (ranges::contains(stylizedIds, ctx.agentId)) ++stylizedOrders;
                    if (ranges::contains(algoIds, ctx.agentId)) ++algoOrders;
                }));
        }

        simulation->simulate();
        ASSERT_THAT(stylizedIds, Not(IsEmpty()));
    }
};

//-------------------------------------------------------------------------

// The one that matters. Every input the agent's decision rests on now comes from somewhere
// else than it did, so if any of those reads returned nothing the agent would still run its
// wakeup chain, still look healthy to the watchdog, and never place anything.
TEST_F(StylizedTraderBarsTest, StillPlacesOrdersWithoutItsPoll)
{
    ASSERT_NO_FATAL_FAILURE(runSimulation());

    EXPECT_GT(stylizedOrders, 0u)
        << "the stylized traders woke but placed nothing, so one of the reads that replaced "
           "the poll is coming back empty";
}

//-------------------------------------------------------------------------

// The horizon has to be populated for the variance to mean anything. A window covering no
// bars would leave the agent on the tick floor in forecast(), which is a floor and not an
// estimate.
TEST_F(StylizedTraderBarsTest, HorizonIsPopulatedAndTheVarianceIsUsable)
{
    ASSERT_NO_FATAL_FAILURE(runSimulation());

    const auto window = exchange->statsHub()->window(kBook, 30);
    EXPECT_GT(window.bars, 0u) << "no closed bars, so nothing was ever measured";
    EXPECT_GT(window.seconds, 0.0);
    EXPECT_GT(window.close, 0.0) << "the bar clock never priced this book at all";
    EXPECT_TRUE(std::isfinite(window.logReturn));
    EXPECT_TRUE(std::isfinite(window.realizedVariance));
    EXPECT_GE(window.realizedVariance, 0.0);
}

//-------------------------------------------------------------------------

// What the fixed clock buys: the answer does not depend on who asked or when they last
// looked. Reading the same horizon twice at the same simulation time is the same window.
TEST_F(StylizedTraderBarsTest, RepeatedReadsAtTheSameInstantAgree)
{
    ASSERT_NO_FATAL_FAILURE(runSimulation());

    const auto* hub = exchange->statsHub().get();
    const auto first = hub->window(kBook, 30);
    const auto second = hub->window(kBook, 30);

    EXPECT_EQ(first.bars, second.bars);
    EXPECT_DOUBLE_EQ(first.logReturn, second.logReturn);
    EXPECT_DOUBLE_EQ(first.realizedVariance, second.realizedVariance);
    EXPECT_DOUBLE_EQ(first.midClose, second.midClose);
}

//-------------------------------------------------------------------------

// ALGOTrader's periodic depth read was a RETRIEVE_L2 round trip and is now taken off the
// hub. Asserting on what that read produced rather than on an order: its execution branch
// sizes a BUY as quoteFree / lastPrice, and lastPrice is only ever set by a trade event, so
// on a book that has not traded yet that path divides by zero. That is older than this
// change and is left alone here rather than papered over with a fixture tuned to dodge it.
//
// Nor on the top-of-book volumes it also stores: the level at the top can legitimately hold
// zero volume until filled orders are cleared, so that reads as absence when it is not.
//
// The consequence to be aware of: handleExecuteTick, the top-of-book read, is not covered.
TEST_F(StylizedTraderBarsTest, AlgoTraderSeesTheBookThroughTheDirectDepthRead)
{
    ASSERT_NO_FATAL_FAILURE(runSimulation());
    ASSERT_THAT(algoIds, SizeIs(1));

    const auto* algo = [this] -> const agent::ALGOTraderAgent* {
        for (const auto& localAgent : simulation->agents()) {
            if (const auto* a = dynamic_cast<const agent::ALGOTraderAgent*>(localAgent.get())) {
                return a;
            }
        }
        return nullptr;
    }();
    ASSERT_NE(algo, nullptr);

    // Repeatedly, and on every book: one push would prove the read works once, but the tick
    // re-arms itself and a chain that stopped after the first would look the same otherwise.
    for (BookId bookId : {kBook, BookId{1}}) {
        EXPECT_THAT(algo->state().at(bookId).volumeStats.bookVolumes().size(), Gt(5u))
            << "book " << bookId << " folded in almost no depth, so either the read comes "
               "back empty or the tick driving it stopped re-arming";
    }
}

//-------------------------------------------------------------------------

// ALGOTrader's GARCH recursion is stepped on the shared clock, not on its own depth tick. The
// tick carries a market-feed latency draw, so the returns it used to see were irregularly
// spaced, and GARCH is defined on regularly sampled ones.
//
// The invariant that matters is the catch-up bookkeeping: bars consumed has to end equal to
// bars closed. Short of that the recursion is either skipping returns, which makes it slower
// than configured, or replaying them, which makes it fictitiously volatile.
TEST_F(StylizedTraderBarsTest, AlgoVolatilityIsSteppedOncePerSharedBar)
{
    ASSERT_NO_FATAL_FAILURE(runSimulation());
    ASSERT_THAT(algoIds, SizeIs(1));

    const auto* algo = [this] -> const agent::ALGOTraderAgent* {
        for (const auto& localAgent : simulation->agents()) {
            if (const auto* a = dynamic_cast<const agent::ALGOTraderAgent*>(localAgent.get())) {
                return a;
            }
        }
        return nullptr;
    }();
    ASSERT_NE(algo, nullptr);

    const auto* hub = exchange->statsHub().get();
    for (BookId bookId : {kBook, BookId{1}}) {
        const auto& volumeStats = algo->state().at(bookId).volumeStats;
        EXPECT_GT(volumeStats.barsConsumed(), 0u)
            << "book " << bookId << " never stepped the recursion at all";
        // Not equality: reading the hub here rolls the clock to the current time, and the
        // agent last stepped on its own tick, so it is legitimately a tick behind. What must
        // hold is that it never runs ahead, which would mean replaying returns, and never
        // falls further behind than one tick, which would mean skipping them.
        const uint64_t closed = hub->barsClosed(bookId);
        EXPECT_LE(volumeStats.barsConsumed(), closed)
            << "book " << bookId << " consumed more bars than have closed, so returns are "
               "being replayed";
        EXPECT_LE(closed - volumeStats.barsConsumed(), 5u)
            << "book " << bookId << " is " << closed - volumeStats.barsConsumed()
            << " bars behind, further than one depth tick, so returns are being skipped";
        EXPECT_TRUE(std::isfinite(volumeStats.estimatedVolatility()));
        EXPECT_GT(volumeStats.estimatedVolatility(), 0.0);
    }
}
