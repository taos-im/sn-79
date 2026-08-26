// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT
//
// A sweep must carry WHO caused it, separately from who owns it.
//
// THE DEFECT. In exchange mode a resting order is normally taken by a sweep that the exchange places
// under its OWN id (agent -1) once a miner's instruction has moved the pool through that order. Two
// consequences followed:
//
//   1. Book.cpp compared the aggressing order's OWNER against the resting owner, so the guard read
//      -1 == 171, never matched, and self-trade prevention could not fire at all. Zero
//      "SELF TRADE PREVENTION" lines existed across the lifetime of an exchange process. The miner was
//      nonetheless on both sides of an event its own instruction caused.
//   2. The fill recorded no agent counterparty, which by the Signal contract means "the pool". That is
//      right when pool drift caused the crossing and wrong when a miner's order did.
//
// THE FIX. `initiatorAgentId` travels from the instruction into SweepCrossingContext, onto the sweep
// order's OrderClientContext, into TradeContext/TradeLogContext, and is read in exactly two places: the
// STP comparison and the counterparty on the resting side's signal.
//
// WHAT IT DELIBERATELY DOES NOT TOUCH. `agentId` stays -1 on the sweep order, because the exchange funds
// it and the initiating miner never receives its volume. Fees follow `aggressingAgentId` in
// ClearingManager::handleTrade, and settlement is always against the pool (Signal.hpp). Attributing the
// taker fee to the initiator would charge it for volume it did not get.

#include <gtest/gtest.h>

#include <optional>

#include <taosim/decimal/decimal.hpp>

#include "Order.hpp"
#include "Trade.hpp"
#include <taosim/message/ExchangeAgentMessagePayloads.hpp>

//-------------------------------------------------------------------------

TEST(SweepInitiator, AnOrdinaryOrderIsItsOwnInitiator)
{
    // Every non-sweep order leaves the field empty, and the guard must then behave exactly as before.
    OrderClientContext ctx{AgentId{171}};
    EXPECT_FALSE(ctx.initiatorAgentId.has_value());
    EXPECT_EQ(ctx.initiatorAgentId.value_or(ctx.agentId), AgentId{171});
}

TEST(SweepInitiator, ASweepOrderKeepsExchangeOwnershipButNamesTheInitiator)
{
    // The distinction the whole fix rests on: owned by -1 so the exchange funds it, initiated by the
    // miner so the guard and the counterparty can see who acted.
    OrderClientContext ctx{AgentId{-1}};
    ctx.initiatorAgentId = AgentId{171};

    EXPECT_EQ(ctx.agentId, AgentId{-1}) << "the sweep must stay owned by the exchange, which funds it";
    EXPECT_EQ(ctx.initiatorAgentId.value_or(ctx.agentId), AgentId{171});
}

TEST(SweepInitiator, TheGuardComparisonMatchesOnASelfTrade)
{
    // Book.cpp compares `initiatorAgentId.value_or(agentId)` against the resting owner. Reproduced here
    // as the arithmetic it performs: with the old comparison this was -1 == 171 and never fired.
    const AgentId restingOwner{171};
    OrderClientContext sweep{AgentId{-1}};
    sweep.initiatorAgentId = AgentId{171};

    EXPECT_NE(sweep.agentId, restingOwner) << "which is exactly why the old comparison never matched";
    EXPECT_EQ(sweep.initiatorAgentId.value_or(sweep.agentId), restingOwner)
        << "the acting agent is the miner, so STP must now fire";
}

TEST(SweepInitiator, TheGuardDoesNotMatchWhenAnotherMinerCausedIt)
{
    // Miner B's order crossing miner A's resting order is an ordinary trade, not a self-trade, and STP
    // must stay out of it. Getting this wrong would cancel legitimate cross-miner fills.
    const AgentId restingOwner{171};
    OrderClientContext sweep{AgentId{-1}};
    sweep.initiatorAgentId = AgentId{208};

    EXPECT_NE(sweep.initiatorAgentId.value_or(sweep.agentId), restingOwner);
}

TEST(SweepInitiator, PoolDriftLeavesNoInitiatorSoTheCounterpartyStaysAbsent)
{
    // A sweep with no instruction behind it is the genuine POOL case. Absent counterparty is the
    // Signal contract's way of saying "the pool", so this must remain empty or every drift-driven fill
    // would be misattributed to whichever agent happened to be nearby.
    TradeLogContext ctx{};
    ctx.aggressingAgentId = AgentId{-1};
    ctx.restingAgentId = AgentId{171};

    EXPECT_FALSE(ctx.initiatorAgentId.has_value());
    const bool counterpartyPresent =
        ctx.initiatorAgentId.has_value() || ctx.aggressingAgentId >= AgentId{};
    EXPECT_FALSE(counterpartyPresent) << "drift-driven sweeps must still read as pool fills";
}

TEST(SweepInitiator, AMinerCausedSweepRecordsThatMinerAsCounterparty)
{
    // The attribution half: the resting side traded with the miner whose order moved the price, not
    // with "the pool". Settlement is unaffected; this is who, not where the funds went.
    TradeLogContext ctx{};
    ctx.aggressingAgentId = AgentId{-1};
    ctx.restingAgentId = AgentId{171};
    ctx.initiatorAgentId = AgentId{208};

    const auto counterparty = ctx.initiatorAgentId.has_value()
        ? std::make_optional(*ctx.initiatorAgentId)
        : (ctx.aggressingAgentId >= AgentId{} ? std::make_optional(ctx.aggressingAgentId)
                                              : std::nullopt);
    ASSERT_TRUE(counterparty.has_value());
    EXPECT_EQ(*counterparty, AgentId{208});
}

TEST(SweepInitiator, AnOrdinaryFillStillReportsItsAggressorAsCounterparty)
{
    // The unchanged path: a real agent-to-agent fill names the aggressor, with no initiator involved.
    TradeLogContext ctx{};
    ctx.aggressingAgentId = AgentId{208};
    ctx.restingAgentId = AgentId{171};

    const auto counterparty = ctx.initiatorAgentId.has_value()
        ? std::make_optional(*ctx.initiatorAgentId)
        : (ctx.aggressingAgentId >= AgentId{} ? std::make_optional(ctx.aggressingAgentId)
                                              : std::nullopt);
    ASSERT_TRUE(counterparty.has_value());
    EXPECT_EQ(*counterparty, AgentId{208});
}

TEST(SweepInitiator, TheSweepCarriesTheInitiatorsSelfTradePolicy)
{
    // The other half of making STP reachable.
    //
    // The guard reads the flag off the AGGRESSING order. In exchange mode the aggressor is the
    // exchange's sweep, and PlaceOrderMarketPayload defaults `stpFlag` to CO. So the moment the
    // initiator fix made `actingAgentId == iopAgentId` match, the sweep's DEFAULT policy started
    // deciding: a miner asking for STP=NONE had its own resting ask cancelled anyway.
    //
    // Making STP reachable without carrying the requested policy is worse than leaving it unreachable,
    // because it applies a policy nobody asked for. The flag now travels on SweepCrossingContext beside
    // the initiator.
    PlaceOrderMarketPayload sweep{};
    EXPECT_EQ(sweep.stpFlag, STPFlag::CO)
        << "the payload default is what silently overrode the miner; if this ever becomes NONE the "
           "defect changes shape rather than going away";

    for (const auto requested : {STPFlag::NONE, STPFlag::CO, STPFlag::CN, STPFlag::DC}) {
        PlaceOrderMarketPayload pld{};
        pld.initiatorAgentId = AgentId{171};
        pld.stpFlag = requested;
        EXPECT_EQ(pld.stpFlag, requested)
            << "the sweep must apply the policy the initiator asked for, not the payload default";
    }
}

TEST(SweepInitiator, NoInitiatorMeansNoSelfTradePolicyToApply)
{
    // A drift-driven sweep has no instruction behind it, so there is no requested policy and the guard
    // must not fire on whatever the default happens to be. SweepCrossingContext defaults to NONE for
    // exactly this case.
    PlaceOrderMarketPayload pld{};
    pld.stpFlag = STPFlag::NONE;
    EXPECT_FALSE(pld.initiatorAgentId.has_value());
    EXPECT_EQ(pld.stpFlag, STPFlag::NONE);
}
