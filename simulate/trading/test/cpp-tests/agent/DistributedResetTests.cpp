/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

/*
 * Regression tests for the message-queue purge in
 * MultiBookExchangeAgent::handleDistributedAgentReset.
 *
 * The bug (fixed in f5aa286c): the purge filter static_pointer_cast the payload of
 * EVERY queued message to DistributedAgentResponsePayload and read its agentId, so the
 * null guard never fired and unrelated payloads were read type-confused. The first
 * data member of RetrieveL1ExtPayload (bookId) sits at the same offset as the wrapped
 * agentId, so a reset of remote agent k dropped every pending RETRIEVE_L1_EXT for
 * book k. Background agents keep themselves alive through exactly that request ->
 * response -> next-request chain, so a reset of a low-id remote agent silenced them
 * permanently. With dynamic_pointer_cast, only genuinely wrapped messages are purge
 * candidates.
 *
 * Coverage is two-fold: the queue is inspected directly around a reset (what must go,
 * what must stay), and a heartbeat-style background agent is driven through a reset
 * with its receiveMessage calls counted to show it keeps acting.
 */

#include "Agent.hpp"
#include "MultiBookExchangeAgent.hpp"
#include "Simulation.hpp"
#include "util.hpp"
#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/message/MultiBookMessagePayloads.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <range/v3/algorithm/count_if.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

//-------------------------------------------------------------------------

// No blanket using-directive for testing: gtest's testing::Message would make every
// unqualified use of the simulator's ::Message ambiguous.
using testing::Test;
using testing::UnorderedElementsAre;

namespace fs = std::filesystem;

//-------------------------------------------------------------------------

namespace
{

// This file lives in cpp-tests/agent/; the shared fixtures are in cpp-tests/data/.
const auto kTestDataPath = fs::path{__FILE__}.parent_path().parent_path() / "data";

static constexpr BookId kBookId{};
static constexpr Timestamp kHeartbeatDelay{1};
static constexpr auto kHeartbeatAgentName = "HEARTBEAT_TRADER";

//-------------------------------------------------------------------------

// Decorator over Agent: counts receiveMessage calls by type before delegating, so a
// test can assert on an agent's observed activity without entangling the bookkeeping
// with the behavior under test.
class MessageCountingAgent : public Agent
{
public:
    MessageCountingAgent(Simulation* simulation, const std::string& name) noexcept
        : Agent{simulation, name}
    {}

    void receiveMessage(Message::Ptr msg) final
    {
        ++m_receivedCounts[msg->type];
        handleMessage(msg);
    }

    [[nodiscard]] uint32_t receivedCount(const std::string& type) const
    {
        const auto it = m_receivedCounts.find(type);
        return it != m_receivedCounts.end() ? it->second : uint32_t{};
    }

private:
    virtual void handleMessage(Message::Ptr msg) = 0;

    std::map<std::string, uint32_t> m_receivedCounts;
};

//-------------------------------------------------------------------------

// Mimics the heartbeat that keeps HighFrequencyTraderAgent alive: each
// RESPONSE_RETRIEVE_L1_EXT schedules the next RETRIEVE_L1_EXT, so the agent stays
// active only as long as that chain survives in the message queue.
class HeartbeatTraderAgent final : public MessageCountingAgent
{
public:
    HeartbeatTraderAgent(Simulation* simulation, const std::string& name) noexcept
        : MessageCountingAgent{simulation, name}
    {
        m_type = "HeartbeatTraderAgent";
    }

private:
    void handleMessage(Message::Ptr msg) override
    {
        if (msg->type == "EVENT_SIMULATION_START" || msg->type == "RESPONSE_RETRIEVE_L1_EXT") {
            scheduleHeartbeat();
        }
    }

    void scheduleHeartbeat() const
    {
        simulation()->dispatchMessage(
            simulation()->currentTimestamp(),
            kHeartbeatDelay,
            name(),
            "EXCHANGE",
            "RETRIEVE_L1_EXT",
            MessagePayload::create<RetrieveL1ExtPayload>(kBookId));
    }
};

}  // namespace

//-------------------------------------------------------------------------

class DistributedResetTest : public Test
{
protected:
    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange{};

    // MultiAgentFees.xml: one book, remoteAgentCount="2" (remote agents 0 and 1
    // registered at configure), plus the required DistributedProxyAgent.
    void SetUp() override
    {
        nodes = taosim::util::parseSimulationFile(kTestDataPath / "MultiAgentFees.xml");
        simulation = std::make_unique<Simulation>();
        simulation->setDebug(false);
    }

    void configureSimulation()
    {
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        ASSERT_NE(exchange, nullptr);
    }

    // The production shape of a validator-issued reset: the exchange receives a
    // DISTRIBUTED_RESET_AGENT wrapping the target ids (cf. SimulationManager's
    // response ingestion).
    [[nodiscard]] Message::Ptr makeResetMessage(std::vector<AgentId> agentIds) const
    {
        const auto requesterId = agentIds.front();
        return Message::create(
            simulation->currentTimestamp(),
            simulation->currentTimestamp(),
            "DISTRIBUTED_PROXY_AGENT",
            "EXCHANGE",
            "DISTRIBUTED_RESET_AGENT",
            MessagePayload::create<DistributedAgentResponsePayload>(
                requesterId,
                MessagePayload::create<ResetAgentsPayload>(std::move(agentIds))));
    }

    [[nodiscard]] auto queueCountOfType(std::string_view type) const
    {
        return ranges::count_if(
            simulation->messageQueue().queue().underlying(),
            [type](const auto& entry) { return entry.pmsg.msg->type == type; });
    }

    // The wrapped agentIds of the queued messages of the given type, in heap order.
    [[nodiscard]] std::vector<AgentId> queuedWrappedAgentIds(std::string_view type) const
    {
        std::vector<AgentId> ids;
        for (const auto& entry : simulation->messageQueue().queue().underlying()) {
            if (entry.pmsg.msg->type != type) continue;
            if (const auto payload = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(
                    entry.pmsg.msg->payload)) {
                ids.push_back(payload->agentId);
            }
        }
        return ids;
    }

    // A queue mixing every payload shape the reset purge must discriminate between:
    // heartbeat traffic whose type-punned first member collides with low agent ids
    // (bookId 0 in the request, time 0 in the response), proxy notices genuinely
    // wrapped for remote agents 0 and 1, and a plain unwrapped message (whose read
    // under the old cast was out-of-bounds, so its survival is a spared-message
    // check under the fix rather than a deterministic red under the bug).
    void seedMixedQueue() const
    {
        simulation->queueMessage(Message::create(
            Timestamp{}, Timestamp{1}, kHeartbeatAgentName, "EXCHANGE", "RETRIEVE_L1_EXT",
            MessagePayload::create<RetrieveL1ExtPayload>(kBookId)));
        simulation->queueMessage(Message::create(
            Timestamp{}, Timestamp{1}, "EXCHANGE", kHeartbeatAgentName,
            "RESPONSE_RETRIEVE_L1_EXT",
            MessagePayload::create<RetrieveL1ExtResponsePayload>()));
        for (const auto agentId : {AgentId{0}, AgentId{1}}) {
            simulation->queueMessage(Message::create(
                Timestamp{}, Timestamp{1}, "EXCHANGE", "DISTRIBUTED_PROXY_AGENT", "EVENT_TRADE",
                MessagePayload::create<DistributedAgentResponsePayload>(
                    agentId, MessagePayload::create<EmptyPayload>())));
        }
        simulation->queueMessage(Message::create(
            Timestamp{}, Timestamp{1}, kHeartbeatAgentName, kHeartbeatAgentName, "WAKEUP",
            MessagePayload::create<EmptyPayload>()));
    }
};

//-------------------------------------------------------------------------

TEST_F(DistributedResetTest, ResetPurgesOnlyTheTargetAgentsWrappedMessages)
{
    configureSimulation();
    ASSERT_TRUE(simulation->messageQueue().empty());

    seedMixedQueue();
    ASSERT_EQ(simulation->messageQueue().size(), 5uz);

    exchange->receiveMessage(makeResetMessage({AgentId{0}}));

    // Only agent 0's wrapped notice goes; the handler's own reset response arrives.
    EXPECT_EQ(queueCountOfType("EVENT_TRADE"), 1);
    EXPECT_THAT(queuedWrappedAgentIds("EVENT_TRADE"), UnorderedElementsAre(AgentId{1}));
    EXPECT_EQ(queueCountOfType("RETRIEVE_L1_EXT"), 1);
    EXPECT_EQ(queueCountOfType("RESPONSE_RETRIEVE_L1_EXT"), 1);
    EXPECT_EQ(queueCountOfType("WAKEUP"), 1);
    EXPECT_EQ(queueCountOfType("RESPONSE_DISTRIBUTED_RESET_AGENT"), 1);
    EXPECT_EQ(simulation->messageQueue().size(), 5uz);
}

//-------------------------------------------------------------------------

TEST_F(DistributedResetTest, ResetOfAllRemoteAgentsSparesEveryUnwrappedMessage)
{
    configureSimulation();
    seedMixedQueue();

    exchange->receiveMessage(makeResetMessage({AgentId{0}, AgentId{1}}));

    EXPECT_EQ(queueCountOfType("EVENT_TRADE"), 0);
    EXPECT_EQ(queueCountOfType("RETRIEVE_L1_EXT"), 1);
    EXPECT_EQ(queueCountOfType("RESPONSE_RETRIEVE_L1_EXT"), 1);
    EXPECT_EQ(queueCountOfType("WAKEUP"), 1);
    EXPECT_EQ(queueCountOfType("RESPONSE_DISTRIBUTED_RESET_AGENT"), 1);
    EXPECT_EQ(simulation->messageQueue().size(), 4uz);
}

//-------------------------------------------------------------------------

TEST_F(DistributedResetTest, BackgroundHeartbeatAgentKeepsActingAcrossLowIdReset)
{
    // Registration happens inside configure (name routing + account), so the agent
    // must be in place beforehand; the vector is re-sorted there, so keep the raw
    // pointer rather than an index.
    auto owned = std::make_unique<HeartbeatTraderAgent>(simulation.get(), kHeartbeatAgentName);
    const auto* heartbeat = owned.get();
    simulation->agents().push_back(std::move(owned));
    configureSimulation();

    simulation->dispatchMessage(
        simulation->currentTimestamp(), Timestamp{}, "SIMULATION", kHeartbeatAgentName,
        "EVENT_SIMULATION_START");

    // START -> request -> response -> next request left pending, one beat per 2 ticks.
    for (auto i = 0z; i < 3z; ++i) simulation->step();
    const auto responsesBeforeReset = heartbeat->receivedCount("RESPONSE_RETRIEVE_L1_EXT");
    ASSERT_GE(responsesBeforeReset, 1u);
    ASSERT_EQ(queueCountOfType("RETRIEVE_L1_EXT"), 1);

    // Reset remote agent 0 while the book-0 heartbeat request is pending: its bookId
    // type-puns to agentId 0, which is exactly the collision the fix removes.
    exchange->receiveMessage(makeResetMessage({AgentId{0}}));
    EXPECT_EQ(queueCountOfType("RETRIEVE_L1_EXT"), 1);

    for (auto i = 0z; i < 6z; ++i) simulation->step();
    EXPECT_GT(heartbeat->receivedCount("RESPONSE_RETRIEVE_L1_EXT"), responsesBeforeReset);
}

//-------------------------------------------------------------------------
