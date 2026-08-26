/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/agent/DistributedProxyAgent.hpp>

#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include "Simulation.hpp"
#include "json_util.hpp"
#include "util.hpp"


//-------------------------------------------------------------------------

namespace taosim::agent
{

//-------------------------------------------------------------------------

DistributedProxyAgent::DistributedProxyAgent(Simulation* simulation)
    : Agent{simulation, "DISTRIBUTED_PROXY_AGENT"}
{}

//-------------------------------------------------------------------------

void DistributedProxyAgent::receiveMessage(Message::Ptr msg)
{
    static const std::set<std::string> ignoredMessageTypes{
        "MULTIBOOK_STATE_PUBLISH",
        "EVENT_SIMULATION_START"
    };

    if (ignoredMessageTypes.contains(msg->type)) {
        return;
    }

    if (m_exchangeServiceMode) {
        handleMessageForExchangeService(msg);
    } else {
        m_messages.push_back(msg);
    }
}

//-------------------------------------------------------------------------

void DistributedProxyAgent::configure(const pugi::xml_node& node)
{
    Agent::configure(node);

    m_exchangeServiceMode = node.attribute("exchangeServiceMode").as_bool();
}

//-------------------------------------------------------------------------

void DistributedProxyAgent::handleMessageForExchangeService(Message::Ptr msg)
{
    // Which responses actually reach this proxy. Without it, "the miner received no notice" cannot be
    // told apart from "the exchange generated no response" or "the response was never delivered here",
    // and those have entirely different fixes.
    simulation()->logDebug("proxy(exchange-service) received {}", msg->type);

    // Agent responses are held in m_messages for the response packer, which serializes them with the
    // simulation's notice serializer and clears the buffer -- same buffer and same serializer as
    // simulation mode, so a notice type the simulation forwards is forwarded here too without being
    // named again.
    //
    // The two exceptions below are the responses this proxy already reports through another channel.

    // Terminal placement failures are reported as rejects instead, which the validator matches to its
    // external-order registry to mark the originating order REJECTED and turns into the miner's refusal
    // notice. Forwarding the response as a notice as well would refuse the same order twice.
    if (msg->type.starts_with("ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER")) {
        const auto pld = std::static_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
        const bool isLimit = msg->type.find("LIMIT") != std::string::npos;
        if (isLimit) {
            const auto errPld =
                std::static_pointer_cast<PlaceOrderLimitErrorResponsePayload>(pld->payload);
            m_orderRejects.push_back({
                .agentId = pld->agentId,
                .bookId = errPld->requestPayload->bookId,
                .direction = errPld->requestPayload->direction,
                .reason = errPld->errorPayload->message,
                .clientOrderId = errPld->requestPayload->clientOrderId,
                .volume = errPld->requestPayload->volume,
                .price = errPld->requestPayload->price});
        } else {
            const auto errPld =
                std::static_pointer_cast<PlaceOrderMarketErrorResponsePayload>(pld->payload);
            m_orderRejects.push_back({
                .agentId = pld->agentId,
                .bookId = errPld->requestPayload->bookId,
                .direction = errPld->requestPayload->direction,
                .reason = errPld->errorPayload->message,
                .clientOrderId = errPld->requestPayload->clientOrderId,
                .volume = errPld->requestPayload->volume,
                .price = std::nullopt});
        }
        return;
    }

    // EVENT_TRADE is the one response NOT forwarded as a notice. An exchange fill is not final until it
    // settles on chain and settlement can roll back, so a fill is announced from the reconciled result
    // rather than from the match; forwarding this too would report every fill twice.
    if (msg->type != "EVENT_TRADE") {
        m_messages.push_back(msg);
        return;
    }

    const auto pld = std::static_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
    const auto subPld = std::static_pointer_cast<EventTradePayload>(pld->payload);

    if (subPld->isResting) {
        fmt::println("TRADE NOTIF {}", json::jsonSerializable2str(subPld));
        m_tradeSignal(subPld);
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------
