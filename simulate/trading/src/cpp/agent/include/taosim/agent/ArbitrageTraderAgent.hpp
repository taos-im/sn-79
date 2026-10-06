/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/agent/common.hpp>
#include "Agent.hpp"
#include "Order.hpp"

#include <cstdint>
#include <map>
#include <string>

//-------------------------------------------------------------------------

namespace taosim::agent
{

//-------------------------------------------------------------------------

// Background "house" arbitrageur: a deterrent for INTRA-ROUND wash (where a farmer
// posts a puppet gift and the winner's take in the same batch, so reactive miners
// never see it in a published state). Subscribes to the MINER-ONLY limit-order event,
// so it is never woken by the background-order flow; when a miner rests a limit order
// priced through a lagged fair estimate (an EMA of the book mid sampled on each trade,
// so off-market wash prints barely move it) by more than `edge`, it immediately crosses
// with a take-only IOC at the order's exact price. Take-only (never rests) + miner-only
// subscription => no effect on background market structure. Inert unless present in the
// simulation XML (<ArbitrageTraderAgent .../>).
class ArbitrageTraderAgent : public Agent
{
public:
    ArbitrageTraderAgent() noexcept = default;
    ArbitrageTraderAgent(Simulation* simulation) noexcept;

    virtual void configure(const pugi::xml_node& node) override;
    virtual void receiveMessage(Message::Ptr msg) override;

private:
    void handleSimulationStart();
    void handleMinerLimitOrder(Message::Ptr msg);
    void handleTrade(Message::Ptr msg);

    // Parameters.
    std::string m_exchange;
    double m_edge{8e-4};       // min fractional mispricing vs fair to act on
    double m_cross{2e-4};      // bump take price toward aggressor to guarantee the cross (< edge)
    double m_alpha{0.02};      // fair EMA weight on the latest book-mid sample
    Timestamp m_latency{1};    // order-placement latency; small => wins the intercept race

    // State (rebuildable; no checkpoint persistence required).
    std::map<BookId, double> m_fair;   // per-book lagged fair (EMA of book mids)
    std::uint64_t m_eventsSeen{};   // miner limit orders seen (activity/cost monitor)
    std::uint64_t m_takes{};        // through-fair gifts it attempted to take
    std::uint64_t m_fills{};        // trades this arb was actually a party to
    AgentId m_id{-1};               // this arb's local agent id
};

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------
