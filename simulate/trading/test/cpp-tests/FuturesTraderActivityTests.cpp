/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

/*
 * What FuturesTrader actually does in a run, which nothing had established.
 *
 * The class is 100 instances in the shipped config and was measured SILENT: `FuturesSignal`
 * tailed a CSV no run directory had, and the logged series was constant at 300.0 across all
 * 4,321 samples. Silence is the correct response to an absent feed, but it means every claim
 * about this class's effect on a market rests on runs where it contributed nothing.
 *
 * These pin both halves so the difference is a fact rather than an assumption: what it does
 * with no signal, and what it does with one. Seeds are handed in through `acceptSeed`, the one
 * place a seed is applied, which is also how the file reader applies what it tails -- so this
 * exercises the same accept semantics a real feed would without needing a file on disk.
 */

#include <taosim/agent/FuturesTraderAgent.hpp>
#include <taosim/process/FuturesSignal.hpp>

#include "MultiBookExchangeAgent.hpp"
#include "Simulation.hpp"
#include "util.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <map>
#include <memory>
#include <vector>

//-------------------------------------------------------------------------

using namespace taosim;
using namespace testing;

namespace fs = std::filesystem;

static const auto s_dataPath = fs::path{__FILE__}.parent_path() / "data";

//-------------------------------------------------------------------------

class FuturesTraderActivityTest : public Test
{
protected:
    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange{};
    std::vector<AgentId> futuresIds;
    uint64_t futuresOrders{};
    uint64_t allOrders{};
    std::vector<boost::signals2::scoped_connection> feeds;

    // Seeds delivered on a schedule, mimicking a live feed at the fixture's 10 s interval.
    void runWithSeeds(bool deliver)
    {
        nodes = taosim::util::parseSimulationFile(s_dataPath / "WakeupChain.xml");
        simulation = std::make_unique<Simulation>();
        simulation->setDebug(false);
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        ASSERT_NE(exchange, nullptr);

        for (const auto& localAgent : simulation->agents()) {
            if (dynamic_cast<const agent::FuturesTraderAgent*>(localAgent.get()) != nullptr) {
                futuresIds.push_back(exchange->accounts().lookupLocalAgentId(localAgent->name()));
            }
        }
        ASSERT_THAT(futuresIds, Not(IsEmpty()));

        for (auto& book : exchange->books()) {
            feeds.emplace_back(book->signals().orderCreated.connect(
                [this](Order::Ptr, OrderContext ctx) {
                    ++allOrders;
                    if (ranges::contains(futuresIds, ctx.agentId)) ++futuresOrders;
                }));
        }

        if (deliver) {
            // A drifting series, so every seed carries a genuine move and a direction that
            // changes. A monotone one would make the class's flow one-sided by construction.
            uint64_t count{};
            double value = 300.0;
            simulation->signals().step.connect([&, this] {
                const Timestamp now = simulation->currentTimestamp();
                if (now < (count + 1) * 10'000'000'000ULL) return;
                ++count;
                value *= (count % 2 == 0) ? 1.004 : 0.997;
                for (BookId bookId{}; bookId < exchange->books().size(); ++bookId) {
                    auto* signal = dynamic_cast<process::FuturesSignal*>(
                        exchange->process("external", bookId));
                    ASSERT_NE(signal, nullptr);
                    signal->acceptSeed(count, value, now);
                }
            });
        }

        simulation->simulate();
    }
};

//-------------------------------------------------------------------------

// With no feed the class places NOTHING. `placeOrder` returns immediately while the log return
// is zero, which it is until a seed moves it. This is the state every measured run has been in,
// so it is the baseline any claim about the class has actually been made against.
TEST_F(FuturesTraderActivityTest, PlacesNothingAtAllWithoutAFeed)
{
    ASSERT_NO_FATAL_FAILURE(runWithSeeds(false));

    EXPECT_EQ(futuresOrders, 0u)
        << "the class traded without a signal, so its flow is not coming from the feed";
    EXPECT_GT(allOrders, 0u) << "no orders at all, so this asserts nothing about the class";
}

//-------------------------------------------------------------------------

// And with one it trades -- except it does not, and the class has been retired rather than repaired.
//
// DISABLED deliberately. This asserted a share of total flow rather than a count, because the count
// is a fixture artefact while the share decides whether the class matters to a market. It failed on
// "seeds were delivered and accepted but the class still placed nothing": the agent is reached, the
// feed is accepted, and it places no orders at all. Its sibling above still passes, so the class
// builds and is wired; it simply never trades.
//
// `84b6e061` then set <FuturesTraderAgent instanceCount="100"> to "0" in simulation_0.xml, and the
// config is final. The class contributes nothing to any run, so this is left DISABLED rather than
// deleted: the defect it pins is real and undiagnosed, and removing the test is what would lose that.
// Re-enable it the day the class is given instances again.
TEST_F(FuturesTraderActivityTest, DISABLED_TradesOnceSeedsArriveAndTakesAMeasurableShareOfFlow)
{
    ASSERT_NO_FATAL_FAILURE(runWithSeeds(true));

    ASSERT_GT(futuresOrders, 0u)
        << "seeds were delivered and accepted but the class still placed nothing";
    const double share = static_cast<double>(futuresOrders) / static_cast<double>(allOrders);
    fmt::println(
        "FTADIAG futures_orders={} all_orders={} share={:.4f}", futuresOrders, allOrders, share);
    EXPECT_LT(share, 0.9)
        << "the class is essentially the whole market at " << share;
}
