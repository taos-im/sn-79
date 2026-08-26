/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/message/PayloadFactory.hpp>

#include <taosim/accounting/Balance.hpp>
#include "Cancellation.hpp"
#include "ClosePosition.hpp"
#include <taosim/message/MultiBookMessagePayloads.hpp>
#include "Order.hpp"
#include <taosim/serialization/msgpack/utils.hpp>

#include <fmt/core.h>

#include <vector>

//-------------------------------------------------------------------------

MessagePayload::Ptr PayloadFactory::createFromJsonMessage(const rapidjson::Value& json)
{
    static constexpr auto ctx = std::source_location::current().function_name();

    const auto& payloadJson = json["payload"];
    std::string_view type = json["type"].GetString();

    if (type == "PLACE_ORDER_MARKET") {
        return PlaceOrderMarketPayload::fromJson(payloadJson);
    }
    else if (type == "RESPONSE_PLACE_ORDER_MARKET") {
        return PlaceOrderMarketResponsePayload::fromJson(payloadJson);
    }
    else if (type == "ERROR_RESPONSE_PLACE_ORDER_MARKET") {
        return PlaceOrderMarketErrorResponsePayload::fromJson(payloadJson);
    }
    else if (type == "PLACE_ORDER_LIMIT") {
        return PlaceOrderLimitPayload::fromJson(payloadJson);
    }
    else if (type == "RESPONSE_PLACE_ORDER_LIMIT") {
        return PlaceOrderLimitResponsePayload::fromJson(payloadJson);
    }
    else if (type == "ERROR_RESPONSE_PLACE_ORDER_LIMIT") {
        return PlaceOrderLimitErrorResponsePayload::fromJson(payloadJson);
    }
    else if (type == "RETRIEVE_ORDERS") {
        return RetrieveOrdersPayload::fromJson(payloadJson);
    }
    else if (type == "CANCEL_ORDERS") {
        return CancelOrdersPayload::fromJson(payloadJson);
    }
    else if (type == "CLOSE_POSITIONS") {
        return ClosePositionsPayload::fromJson(payloadJson);
    }
    else if (type == "RESPONSE_CANCEL_ORDERS") {
        return CancelOrdersResponsePayload::fromJson(payloadJson);
    }
    else if (type == "ERROR_RESPONSE_CANCEL_ORDERS") {
        return CancelOrdersErrorResponsePayload::fromJson(payloadJson);
    }
    else if (type == "RETRIEVE_L1" || type == "WAKEUP") {
        return RetrieveL1Payload::fromJson(payloadJson);
    }
    else if (type == "RESPONSE_RETRIEVE_L1") {
        return RetrieveL1ResponsePayload::fromJson(payloadJson);
    }
    else if (type == "RETRIEVE_L1_EXT") {
        return RetrieveL1ExtPayload::fromJson(payloadJson);
    }
    else if (type == "RESPONSE_RETRIEVE_L1_EXT") {
        return RetrieveL1ExtResponsePayload::fromJson(payloadJson);
    }
    else if (type == "RETRIEVE_BOOK") {
        return RetrieveL2Payload::fromJson(payloadJson);
    }
    else if (type == "SUBSCRIBE_EVENT_ORDER_MARKET") {
        return MessagePayload::create<EmptyPayload>();
    }
    else if (type == "SUBSCRIBE_EVENT_ORDER_LIMIT") {
        return MessagePayload::create<EmptyPayload>();
    }
    else if (type == "SUBSCRIBE_EVENT_TRADE") {
        return MessagePayload::create<EmptyPayload>();
    }
    else if (type == "SUBSCRIBE_EVENT_ORDER_TRADE") {
        return SubscribeEventTradeByOrderPayload::fromJson(payloadJson);
    }
    else if (type == "SUBSCRIBE_EVENT_TRADE_OWN") {
        return MessagePayload::create<EmptyPayload>();
    }
    else if (type == "RESET_AGENT") {
        return ResetAgentsPayload::fromJson(payloadJson);
    }
    else if (type == "RESPONSE_RESET_AGENT") {
        return ResetAgentsResponsePayload::fromJson(payloadJson);
    }
    else if (type == "ERROR_RESPONSE_RESET_AGENT") {
        return ResetAgentsErrorResponsePayload::fromJson(payloadJson);
    }
    else if (type == "EVENT_SIMULATION_START" || type == "EVENT_SIMULATION_END") {
        return MessagePayload::create<EmptyPayload>();
    }

    throw std::runtime_error{fmt::format("{}: Unrecognized message type '{}'", ctx, type)};
}

//-------------------------------------------------------------------------

MessagePayload::Ptr PayloadFactory::createFromMessagePack(const msgpack::object& o, std::string_view type)
{
    using namespace std::literals::string_view_literals;

    static constexpr auto ctx = std::source_location::current().function_name();

    if (auto prefix = "DISTRIBUTED_"sv; type.starts_with(prefix)) {
        try {
            return MessagePayload::create<DistributedAgentResponsePayload>(
                *taosim::serialization::msgpackFindMap<AgentId>(o, "agentId"),
                createFromMessagePack(
                    *taosim::serialization::msgpackFindMapObj(o, "payload"),
                    type.substr(prefix.size())
                )
            );
        }
        catch (const std::exception& e) {
            throw std::runtime_error{fmt::format(
                "{}: Error creating payload of type '{}': {}", ctx, type, e.what()
            )};
        }
    }

    auto makePayload = [&]<std::derived_from<MessagePayload> T> {
        try {
            auto pld = std::make_shared<T>();
            o.convert(*pld);
            return pld;
        }
        catch (const std::exception& e) {
            throw std::runtime_error{fmt::format(
                "{}: Error creating payload of type '{}': {}", ctx, type, e.what())};
        }
    };

    if (type == "PLACE_ORDER_MARKET") {
        return makePayload.operator()<PlaceOrderMarketPayload>();
    }
    else if (type == "RESPONSE_PLACE_ORDER_MARKET") {
        return makePayload.operator()<PlaceOrderMarketResponsePayload>();
    }
    else if (type == "ERROR_RESPONSE_PLACE_ORDER_MARKET") {
        return makePayload.operator()<PlaceOrderMarketErrorResponsePayload>();
    }
    else if (type == "PLACE_ORDER_LIMIT") {
        return makePayload.operator()<PlaceOrderLimitPayload>();
    }
    else if (type == "RESPONSE_PLACE_ORDER_LIMIT") {
        return makePayload.operator()<PlaceOrderLimitResponsePayload>();
    }
    else if (type == "ERROR_RESPONSE_PLACE_ORDER_LIMIT") {
        return makePayload.operator()<PlaceOrderLimitErrorResponsePayload>();
    }
    else if (type == "RETRIEVE_ORDERS") {
        return makePayload.operator()<RetrieveOrdersPayload>();
    }
    else if (type == "CANCEL_ORDERS") {
        return makePayload.operator()<CancelOrdersPayload>();
    }
    else if (type == "CLOSE_POSITIONS") {
        return makePayload.operator()<ClosePositionsPayload>();
    }
    else if (type == "RESPONSE_CANCEL_ORDERS") {
        return makePayload.operator()<CancelOrdersResponsePayload>();
    }
    else if (type == "ERROR_RESPONSE_CANCEL_ORDERS") {
        return makePayload.operator()<CancelOrdersErrorResponsePayload>();
    }
    else if (type == "RETRIEVE_L1" || type == "WAKEUP") {
        return makePayload.operator()<RetrieveL1Payload>();
    }
    else if (type == "RESPONSE_RETRIEVE_L1") {
        return makePayload.operator()<RetrieveL1ResponsePayload>();
    }
    else if (type == "RETRIEVE_L1_EXT") {
        return makePayload.operator()<RetrieveL1ExtPayload>();
    }
    else if (type == "RESPONSE_RETRIEVE_L1_EXT") {
        return makePayload.operator()<RetrieveL1ExtResponsePayload>();
    }
    else if (type == "RETRIEVE_L2") {
        return makePayload.operator()<RetrieveL2Payload>();
    }
    else if (type == "RESPONSE_RETRIEVE_L2") {
        return makePayload.operator()<RetrieveL2ResponsePayload>();
    }
    else if (type == "RETRIEVE_BOOK") {
        return makePayload.operator()<RetrieveL2Payload>();
    }
    else if (type == "SUBSCRIBE_EVENT_ORDER_MARKET") {
        return MessagePayload::create<EmptyPayload>();
    }
    else if (type == "SUBSCRIBE_EVENT_ORDER_LIMIT") {
        return MessagePayload::create<EmptyPayload>();
    }
    else if (type == "SUBSCRIBE_EVENT_TRADE") {
        return MessagePayload::create<EmptyPayload>();
    }
    else if (type == "SUBSCRIBE_EVENT_ORDER_TRADE") {
        return makePayload.operator()<SubscribeEventTradeByOrderPayload>();
    }
    else if (type == "SUBSCRIBE_EVENT_TRADE_OWN") {
        return MessagePayload::create<EmptyPayload>();
    }
    else if (type == "RESET_AGENT") {
        return makePayload.operator()<ResetAgentsPayload>();
    }
    else if (type == "RESPONSE_RESET_AGENT") {
        return makePayload.operator()<ResetAgentsResponsePayload>();
    }
    else if (type == "ERROR_RESPONSE_RESET_AGENT") {
        return makePayload.operator()<ResetAgentsErrorResponsePayload>();
    }
    else if (type == "EVENT_SIMULATION_START" || type == "EVENT_SIMULATION_END") {
        return MessagePayload::create<EmptyPayload>();
    }
    else {
        return MessagePayload::create<EmptyPayload>();
    }

    throw std::runtime_error{fmt::format("{}: Unrecognized message type '{}'", ctx, type)};
}

//-------------------------------------------------------------------------