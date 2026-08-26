/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/message/MultiBookMessagePayloads.hpp>
#include <taosim/serialization/msgpack/common.hpp>

//-------------------------------------------------------------------------

namespace taosim::message::serialization
{

//-------------------------------------------------------------------------

void packMessagePayload(auto& o, MessagePayload::Ptr payload)
{
    if (auto pld = std::dynamic_pointer_cast<StartSimulationPayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<PlaceOrderMarketPayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<PlaceOrderMarketResponsePayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<PlaceOrderMarketErrorResponsePayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<PlaceOrderLimitPayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<PlaceOrderLimitResponsePayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<PlaceOrderLimitErrorResponsePayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<RetrieveOrdersPayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<RetrieveOrdersResponsePayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<CancelOrdersPayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<CancelOrdersResponsePayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<CancelOrdersErrorResponsePayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<ClosePositionsPayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<ClosePositionsResponsePayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<ClosePositionsErrorResponsePayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<RetrieveL2Payload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<RetrieveL2ResponsePayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<RetrieveL1Payload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<RetrieveL1ResponsePayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<RetrieveL1ExtPayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<RetrieveL1ExtResponsePayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<SubscribeEventTradeByOrderPayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<EventOrderMarketPayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<EventOrderLimitPayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<EventTradePayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<ResetAgentsPayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<ResetAgentsResponsePayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<ResetAgentsErrorResponsePayload>(payload)) {
        o.pack(*pld);
    }
    else if (auto pld = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(payload)) {
        o.pack_map(2);

        o.pack("agentId");
        o.pack(pld->agentId);

        o.pack("payload");
        packMessagePayload(o, pld->payload);
    }
    else {
        o.pack_nil();
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::message::serialization

//-------------------------------------------------------------------------
