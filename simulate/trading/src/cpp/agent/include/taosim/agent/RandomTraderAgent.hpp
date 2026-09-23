/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/agent/common.hpp>
#include "Agent.hpp"
#include "Order.hpp"

//-------------------------------------------------------------------------

namespace taosim::agent
{

//-------------------------------------------------------------------------


class RandomTraderAgent : public Agent
{
public:
    RandomTraderAgent() noexcept = default;
    RandomTraderAgent(Simulation* simulation) noexcept;

    // TODO: Wrap state into a struct and provide a single access point here.
    [[nodiscard]] auto&& topLevel(this auto&& self) noexcept { return self.m_topLevel; }
    [[nodiscard]] auto&& orderFlag(this auto&& self) noexcept { return self.m_orderFlag; }

    virtual void configure(const pugi::xml_node& node) override;
    virtual void receiveMessage(Message::Ptr msg) override;

private:
    void handleSimulationStart();
    void handleSimulationStop();
    void handleTradeSubscriptionResponse();
    void handleWakeup(Message::Ptr msg);
    void scheduleWakeup(BookId bookId, Timestamp delay);

    // First quote goes up after this, and the agent requotes every `m_stepDelay` after that.
    Timestamp m_graceDelay{};
    Timestamp m_stepDelay{};
    void handleLimitOrderPlacementResponse(Message::Ptr msg);
    void handleLimitOrderPlacementErrorResponse(Message::Ptr msg);
    void handleCancelOrdersResponse(Message::Ptr msg);
    void handleCancelOrdersErrorResponse(Message::Ptr msg);
    void handleTrade(Message::Ptr msg);

    void sendOrder(BookId bookId, OrderDirection direction,
        double volume, double price, double leverage);

    // Parameters, injections.
    std::string m_exchange;
    uint32_t m_bookCount;
    Timestamp m_tau;
    double m_quantityMin;
    double m_quantityMax;

    // State.
    std::vector<TopLevel> m_topLevel;
    std::vector<bool> m_orderFlag;
};

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------
