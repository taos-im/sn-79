/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

/*
 * Regression test for the HFT own-fill feed.
 *
 * The bug: HighFrequencyTraderAgent handled EVENT_TRADE and
 * RESPONSE_SUBSCRIBE_EVENT_TRADE but never *dispatched* a subscription, and
 * MultiBookExchangeAgent::notifyTradeSubscribers only fans out to registered
 * subscribers. So the agent was never told about its own fills, m_inventory sat at
 * 0 for entire runs, and every term derived from it — the reservation-price skew,
 * the deltaHFT/tauHFT quote cadence, requote-on-fill, and the rebalance branch
 * (the agent's only aggressive channel) — was inert.
 *
 * Why assert on the SUBSCRIPTION rather than on inventory after a trade: the
 * failure was structural, not arithmetic. The signs in handleTrade were always
 * correct. Asserting registration is also fully deterministic — it needs no fill
 * to occur, so there is no dependence on the RNG, on how the book happens to be
 * seeded, or on the sim running long enough for an HFT quote to be hit.
 *
 * This drives a bare Simulation, not a SimulationManager, so no POSIX IPC object
 * is created and the test cannot disturb another taosim running on the same host.
 */

#include <taosim/agent/HighFrequencyTraderAgent.hpp>

#include "MultiBookExchangeAgent.hpp"
#include "Simulation.hpp"
#include "util.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <pugixml.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

//-------------------------------------------------------------------------

using namespace taosim;
using namespace taosim::agent;
using namespace testing;

namespace fs = std::filesystem;

//-------------------------------------------------------------------------

namespace
{

// This file lives in cpp-tests/agent/; the shared fixtures are in cpp-tests/data/.
const auto kTestDataPath = fs::path{__FILE__}.parent_path().parent_path() / "data";

}  // namespace

//-------------------------------------------------------------------------

class HFTOwnTradeFeedTest : public Test
{
protected:
    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange{};

    // Build and run a short simulation.
    void runSimulation()
    {
        nodes = taosim::util::parseSimulationFile(kTestDataPath / "HFTOwnFill.xml");

        ASSERT_TRUE(nodes.simulation.select_node(".//HighFrequencyTraderAgent").node())
            << "HFTOwnFill.xml has no HighFrequencyTraderAgent node";

        simulation = std::make_unique<Simulation>();
        simulation->setDebug(false);
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        ASSERT_NE(exchange, nullptr);

        // Runs start() (which dispatches EVENT_SIMULATION_START, prompting the
        // agents' subscriptions) and then steps to the configured duration.
        simulation->simulate();
    }

    // Names of the configured HFT instances (LocalAgentManager names them
    // "<baseName>_<instanceId>").
    std::vector<std::string> hftNames() const
    {
        std::vector<std::string> names;
        for (const auto& agent : simulation->agents()) {
            if (dynamic_cast<const HighFrequencyTraderAgent*>(agent.get()) != nullptr) {
                names.push_back(agent->name());
            }
        }
        return names;
    }
};

//-------------------------------------------------------------------------

// The subscription is unconditional: own fills are the only source of m_inventory,
// so without it this expectation fails for every instance, which is exactly the
// state every run was in before.
TEST_F(HFTOwnTradeFeedTest, SubscribesToOwnTradeFeed)
{
    ASSERT_NO_FATAL_FAILURE(runSimulation());

    const auto names = hftNames();
    ASSERT_THAT(names, SizeIs(2)) << "expected the 2 configured HFT instances";

    for (const auto& name : names) {
        EXPECT_TRUE(exchange->localOwnTradeSubs().contains(name))
            << name << " is not registered for own-trade events, so it will never "
                       "learn about its own fills and its inventory stays 0";
    }
}

//-------------------------------------------------------------------------

// The full-tape feed must stay empty: an agent on BOTH feeds would net every
// market trade into its inventory, not just its own fills. notifyTradeSubscribers
// guards against double-delivery, but the agent should not be on that feed at all.
TEST_F(HFTOwnTradeFeedTest, DoesNotSubscribeToTheFullTradeTape)
{
    ASSERT_NO_FATAL_FAILURE(runSimulation());

    for (const auto& name : hftNames()) {
        EXPECT_FALSE(exchange->localTradeSubs().contains(name))
            << name << " is on the full-tape feed; its inventory would absorb trades "
                       "it was not a party to";
    }
}

//-------------------------------------------------------------------------
