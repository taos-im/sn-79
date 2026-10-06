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
    // EVENT_SIMULATION_START BELONGS IN THE STATE UPDATE, like every other event.
    //
    // Ignoring it here was the first of two places a lifecycle notice was dropped on its way to an
    // agent; the second was checkGlobalDuplicate in both state serializers, which discarded every
    // notice whose payload is not an agent response. Between them, FinanceAgentBase.update's
    // `case "EVENT_SIMULATION_START" | "ESS": self.onStart(event)` could never fire, and neither
    // could its end-of-simulation counterpart -- the handlers existed, were dispatched correctly,
    // and were never reached.
    //
    // The engine already serialises both (NoticePack packs StartSimulationPayload with its logDir)
    // and already fans both to every agent (ValidatorRequest's EmptyPayload/StartSimulationPayload
    // branch). Only the two drops above stood in the way.
    //
    // Safe against a resumed run re-announcing itself: Simulation::start() is the sole dispatcher of
    // both events and both its callers guard on state() == INACTIVE (Simulation.cpp:119,
    // MultiAssetSimulationOrchestrator.cpp:719), so a resume never re-sends them. Safe against the
    // grace period too: publishState() returns before any serializer runs while warmingUp(), and the
    // buffer is only cleared inside the serializer, so a notice queued at step 0 survives to the
    // first real publish.
    static const std::set<std::string> ignoredMessageTypes{
        "MULTIBOOK_STATE_PUBLISH"
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
        const bool isLimit = msg->type.contains("LIMIT");
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
        m_tradeSignal(subPld);
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------
