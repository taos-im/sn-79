/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/message/MultiBookMessagePayloads.hpp>
#include "Agent.hpp"
#include <common.hpp>

#include <string>
#include <vector>

//-------------------------------------------------------------------------

namespace taosim::agent
{

//-------------------------------------------------------------------------

// Terminal order rejection surfaced to the exchange-service response so the
// validator can mark the originating external order REJECTED instead of
// leaving it acknowledged-but-dead. clientOrderId echoes the numeric token the
// validator stamps on external orders, giving exact order attribution even
// when an agent has several in-flight orders on one (book, side).
struct OrderReject
{
    AgentId agentId;
    BookId bookId;
    OrderDirection direction;
    std::string reason;
    std::optional<ClientOrderID> clientOrderId;
    // The miner needs to identify WHICH of its orders was refused, and clientOrderId is optional. Both
    // are in scope at every recording site (errPld->requestPayload), and without them a rejection notice
    // would have to claim a volume of zero, which is worse than not reporting one.
    taosim::decimal_t volume{};
    std::optional<taosim::decimal_t> price;  // absent for a market order, which has no limit
};

class DistributedProxyAgent : public Agent
{
public:
    DistributedProxyAgent(Simulation* simulation);

    [[nodiscard]] std::span<Message::Ptr> messages() noexcept { return m_messages; }
    [[nodiscard]] bool exchangeServiceMode() const noexcept { return m_exchangeServiceMode; }
    [[nodiscard]] auto& tradeSignal() noexcept { return m_tradeSignal; }
    [[nodiscard]] std::span<const OrderReject> orderRejects() const noexcept
    {
        return m_orderRejects;
    }

    void clearMessages() noexcept { m_messages.clear(); };
    void clearOrderRejects() noexcept { m_orderRejects.clear(); }
    void addOrderReject(OrderReject reject) { m_orderRejects.push_back(std::move(reject)); }

    virtual void receiveMessage(Message::Ptr msg) override;
    virtual void configure(const pugi::xml_node& node) override;

private:
    void handleMessageForExchangeService(Message::Ptr msg);

    std::vector<Message::Ptr> m_messages;
    bool m_exchangeServiceMode{};
    UnsyncSignal<void(EventTradePayload::Ptr)> m_tradeSignal;
    std::vector<OrderReject> m_orderRejects;
};

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------
