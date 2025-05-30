/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "ExchangeAgentMessagePayloads.hpp"

#include "json_util.hpp"
#include "util.hpp"

//-------------------------------------------------------------------------

void StartSimulationPayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("logDir", rapidjson::Value{logDir.c_str(), allocator}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void StartSimulationPayload::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    jsonSerialize(json, key);
}

//-------------------------------------------------------------------------

StartSimulationPayload::Ptr StartSimulationPayload::fromJson(const rapidjson::Value& json)
{
    return MessagePayload::create<StartSimulationPayload>(json["logDir"].GetString());
}

//-------------------------------------------------------------------------

void PlaceOrderMarketPayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("direction", rapidjson::Value{std::to_underlying(direction)}, allocator);
        json.AddMember("volume", rapidjson::Value{taosim::util::decimal2double(volume)}, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
        taosim::json::setOptionalMember(json, "clientOrderId", clientOrderId);
        json.AddMember(
            "stpFlag",
            rapidjson::Value{magic_enum::enum_name(stpFlag).data(), allocator},
            allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void PlaceOrderMarketPayload::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("direction", rapidjson::Value{std::to_underlying(direction)}, allocator);
        json.AddMember("volume", rapidjson::Value{taosim::util::packDecimal(volume)}, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
        taosim::json::setOptionalMember(json, "clientOrderId", clientOrderId);
        json.AddMember(
            "stpFlag",
            rapidjson::Value{magic_enum::enum_name(stpFlag).data(), allocator},
            allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

PlaceOrderMarketPayload::Ptr PlaceOrderMarketPayload::fromJson(const rapidjson::Value& json)
{
    return MessagePayload::create<PlaceOrderMarketPayload>(
        OrderDirection{json["direction"].GetUint()},
        taosim::json::getDecimal(json["volume"]),
        json["bookId"].GetUint(),
        !json["clientOrderId"].IsNull()
            ? std::make_optional(json["clientOrderId"].GetUint())
            : std::nullopt,
        json.HasMember("stpFlag")
            ? magic_enum::enum_cast<STPFlag>(json["stpFlag"].GetUint())
                .value_or(STPFlag::CO)
            : STPFlag::CO
        );
}

//-------------------------------------------------------------------------

void PlaceOrderMarketResponsePayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("orderId", rapidjson::Value{}.SetUint(id), allocator);
        requestPayload->jsonSerialize(json, "requestPayload");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void PlaceOrderMarketResponsePayload::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("orderId", rapidjson::Value{}.SetUint(id), allocator);
        requestPayload->checkpointSerialize(json, "requestPayload");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

PlaceOrderMarketResponsePayload::Ptr PlaceOrderMarketResponsePayload::fromJson(
    const rapidjson::Value& json)
{
    return MessagePayload::create<PlaceOrderMarketResponsePayload>(
        json["orderId"].GetUint(),
        PlaceOrderMarketPayload::fromJson(json["requestPayload"]));
}

//-------------------------------------------------------------------------

void PlaceOrderMarketErrorResponsePayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        requestPayload->jsonSerialize(json, "requestPayload");
        errorPayload->jsonSerialize(json, "errorPayload");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void PlaceOrderMarketErrorResponsePayload::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        requestPayload->checkpointSerialize(json, "requestPayload");
        errorPayload->checkpointSerialize(json, "errorPayload");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

PlaceOrderMarketErrorResponsePayload::Ptr PlaceOrderMarketErrorResponsePayload::fromJson(
    const rapidjson::Value& json)
{
    return MessagePayload::create<PlaceOrderMarketErrorResponsePayload>(
        PlaceOrderMarketPayload::fromJson(json["requestPayload"]),
        ErrorResponsePayload::fromJson(json["errorPayload"]));
}

//-------------------------------------------------------------------------

void PlaceOrderLimitPayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [&](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember(
            "direction", rapidjson::Value{std::to_underlying(direction)}, allocator);
        json.AddMember("volume", rapidjson::Value{taosim::util::decimal2double(volume)}, allocator);
        json.AddMember("price", rapidjson::Value{taosim::util::decimal2double(price)}, allocator);
        json.AddMember("leverage", rapidjson::Value{taosim::util::decimal2double(leverage)}, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
        taosim::json::setOptionalMember(json, "clientOrderId", clientOrderId);
        json.AddMember(
            "flag",
            rapidjson::Value{magic_enum::enum_name(flag).data(), allocator},
            allocator);
        json.AddMember(
            "stpFlag",
            rapidjson::Value{magic_enum::enum_name(stpFlag).data(), allocator},
            allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void PlaceOrderLimitPayload::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [&](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("direction", rapidjson::Value{std::to_underlying(direction)}, allocator);
        json.AddMember("volume", rapidjson::Value{taosim::util::packDecimal(volume)}, allocator);
        json.AddMember("price", rapidjson::Value{taosim::util::packDecimal(price)}, allocator);
        json.AddMember("leverage", rapidjson::Value{taosim::util::packDecimal(leverage)}, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
        taosim::json::setOptionalMember(json, "clientOrderId", clientOrderId);
        json.AddMember(
            "flag",
            rapidjson::Value{magic_enum::enum_name(flag).data(), allocator},
            allocator);
        json.AddMember(
            "stpFlag",
            rapidjson::Value{magic_enum::enum_name(stpFlag).data(), allocator},
            allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

PlaceOrderLimitPayload::Ptr PlaceOrderLimitPayload::fromJson(const rapidjson::Value& json)
{
    return MessagePayload::create<PlaceOrderLimitPayload>(
        OrderDirection{json["direction"].GetUint()},
        taosim::json::getDecimal(json["volume"]),
        taosim::json::getDecimal(json["price"]),
        json.HasMember("leverage")
            ? taosim::json::getDecimal(json["leverage"])
            : 0_dec,
        json["bookId"].GetUint(),
        !json["clientOrderId"].IsNull()
            ? std::make_optional(json["clientOrderId"].GetUint())
            : std::nullopt,
        json.HasMember("flag")
            ? magic_enum::enum_cast<LimitOrderFlag>(json["flag"].GetUint())
                .value_or(LimitOrderFlag::NONE)
            : LimitOrderFlag::NONE,
        json.HasMember("stpFlag")
            ? magic_enum::enum_cast<STPFlag>(json["stpFlag"].GetUint())
                .value_or(STPFlag::CO)
            : STPFlag::CO
        );
}

//-------------------------------------------------------------------------

namespace taosim
{

bool violatesPostOnly(Book::Ptr book, PlaceOrderLimitPayload::Ptr limitOrderPayload) noexcept
{
    switch (limitOrderPayload->direction) {
        case OrderDirection::BUY:
            return !book->sellQueue().empty()
                && limitOrderPayload->price >= book->sellQueue().front().price();
        case OrderDirection::SELL:
            return !book->buyQueue().empty()
                && limitOrderPayload->price <= book->buyQueue().back().price();
        default:
            std::unreachable();
    }
}

bool violatesImmediateOrCancel(
    Book::Ptr book, PlaceOrderLimitPayload::Ptr limitOrderPayload) noexcept
{
    return !violatesPostOnly(book, limitOrderPayload);
}

}  // namespace taosim

//-------------------------------------------------------------------------

void PlaceOrderLimitResponsePayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("orderId", rapidjson::Value{id}, allocator);
        requestPayload->jsonSerialize(json, "requestPayload");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void PlaceOrderLimitResponsePayload::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("orderId", rapidjson::Value{id}, allocator);
        requestPayload->checkpointSerialize(json, "requestPayload");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

PlaceOrderLimitResponsePayload::Ptr PlaceOrderLimitResponsePayload::fromJson(
    const rapidjson::Value& json)
{
    return MessagePayload::create<PlaceOrderLimitResponsePayload>(
        json["orderId"].GetUint(),
        PlaceOrderLimitPayload::fromJson(json["requestPayload"]));
}

//-------------------------------------------------------------------------

void PlaceOrderLimitErrorResponsePayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        requestPayload->jsonSerialize(json, "requestPayload");
        errorPayload->jsonSerialize(json, "errorPayload");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void PlaceOrderLimitErrorResponsePayload::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        requestPayload->checkpointSerialize(json, "requestPayload");
        errorPayload->checkpointSerialize(json, "errorPayload");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

PlaceOrderLimitErrorResponsePayload::Ptr PlaceOrderLimitErrorResponsePayload::fromJson(
    const rapidjson::Value& json)
{
    return MessagePayload::create<PlaceOrderLimitErrorResponsePayload>(
        PlaceOrderLimitPayload::fromJson(json["requestPayload"]),
        ErrorResponsePayload::fromJson(json["errorPayload"]));
}

//-------------------------------------------------------------------------

void RetrieveOrdersPayload::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        rapidjson::Value orderIdsJson{rapidjson::kArrayType};
        for (OrderID orderId : ids) {
            orderIdsJson.PushBack(orderId, allocator);
        }
        json.AddMember("orderIds", orderIdsJson, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void RetrieveOrdersPayload::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    jsonSerialize(json, key);
}

//-------------------------------------------------------------------------

RetrieveOrdersPayload::Ptr RetrieveOrdersPayload::fromJson(const rapidjson::Value& json)
{
    std::vector<OrderID> orderIds;
    for (const auto& orderId : json["orderIds"].GetArray()) {
        orderIds.push_back(orderId.GetUint());
    }
    return MessagePayload::create<RetrieveOrdersPayload>(
        std::move(orderIds), json["bookId"].GetUint());
}

//-------------------------------------------------------------------------

void RetrieveOrdersResponsePayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        rapidjson::Value ordersJson{rapidjson::kArrayType};
        for (const auto& order : orders) {
            rapidjson::Document orderJson{rapidjson::kObjectType, &allocator};
            order.jsonSerialize(orderJson);
            ordersJson.PushBack(orderJson, allocator);
        }
        json.AddMember("orders", ordersJson, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void RetrieveOrdersResponsePayload::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        rapidjson::Value ordersJson{rapidjson::kArrayType};
        for (const auto& order : orders) {
            rapidjson::Document orderJson{rapidjson::kObjectType, &allocator};
            order.checkpointSerialize(orderJson);
            ordersJson.PushBack(orderJson, allocator);
        }
        json.AddMember("orders", ordersJson, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

RetrieveOrdersResponsePayload::Ptr RetrieveOrdersResponsePayload::fromJson(
    const rapidjson::Value& json)
{
    std::vector<LimitOrder> orders;
    for (const auto& order : json["orders"].GetArray()) {
        orders.emplace_back(
            order["orderId"].GetUint(),
            order["timestamp"].GetUint64(),
            taosim::json::getDecimal(order["volume"]),
            OrderDirection{order["direction"].GetUint()},
            taosim::json::getDecimal(order["price"]));
    }
    return MessagePayload::create<RetrieveOrdersResponsePayload>(
        std::move(orders),
        json["bookId"].GetUint());
}

//-------------------------------------------------------------------------

void CancelOrdersPayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        rapidjson::Value cancellationsJson{rapidjson::kArrayType};
        for (const Cancellation& cancellation : cancellations) {
            rapidjson::Document cancellationJson{rapidjson::kObjectType, &allocator};
            cancellation.jsonSerialize(cancellationJson);
            cancellationsJson.PushBack(cancellationJson, allocator);
        }
        json.AddMember("cancellations", cancellationsJson, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void CancelOrdersPayload::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        rapidjson::Value cancellationsJson{rapidjson::kArrayType};
        for (const Cancellation& cancellation : cancellations) {
            rapidjson::Document cancellationJson{rapidjson::kObjectType, &allocator};
            cancellation.checkpointSerialize(cancellationJson);
            cancellationsJson.PushBack(cancellationJson, allocator);
        }
        json.AddMember("cancellations", cancellationsJson, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

CancelOrdersPayload::Ptr CancelOrdersPayload::fromJson(const rapidjson::Value& json)
{
    return MessagePayload::create<CancelOrdersPayload>(
        [&json] {
            std::vector<Cancellation> cancellations;
            for (const auto& cancellationJson : json["cancellations"].GetArray()) {
                cancellations.emplace_back(
                    cancellationJson["orderId"].GetUint(),
                    !cancellationJson["volume"].IsNull()
                        ? std::make_optional(taosim::json::getDecimal(json["volume"]))
                        : std::nullopt);
            }
            return cancellations;
        }(),
        json["bookId"].GetUint());
}

//-------------------------------------------------------------------------

void CancelOrdersResponsePayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        rapidjson::Value orderIdsJson{rapidjson::kArrayType};
        for (OrderID orderId : orderIds) {
            orderIdsJson.PushBack(orderId, allocator);
        }
        json.AddMember("orderIds", orderIdsJson, allocator);
        requestPayload->jsonSerialize(json, "requestPayload");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void CancelOrdersResponsePayload::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        rapidjson::Value orderIdsJson{rapidjson::kArrayType};
        for (OrderID orderId : orderIds) {
            orderIdsJson.PushBack(orderId, allocator);
        }
        json.AddMember("orderIds", orderIdsJson, allocator);
        requestPayload->checkpointSerialize(json, "requestPayload");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

CancelOrdersResponsePayload::Ptr CancelOrdersResponsePayload::fromJson(
    const rapidjson::Value& json)
{
    std::vector<OrderID> orderIds;
    for (const auto& orderId : json["orderIds"].GetArray()) {
        orderIds.push_back(orderId.GetUint());
    }
    return MessagePayload::create<CancelOrdersResponsePayload>(
        std::move(orderIds),
        CancelOrdersPayload::fromJson(json["requestPayload"]));
}

//-------------------------------------------------------------------------

void CancelOrdersErrorResponsePayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        rapidjson::Value orderIdsJson{rapidjson::kArrayType};
        for (OrderID orderId : orderIds) {
            orderIdsJson.PushBack(orderId, allocator);
        }
        json.AddMember("orderIds", orderIdsJson, allocator);
        requestPayload->jsonSerialize(json, "requestPayload");
        errorPayload->jsonSerialize(json, "errorPayload");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void CancelOrdersErrorResponsePayload::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        rapidjson::Value orderIdsJson{rapidjson::kArrayType};
        for (OrderID orderId : orderIds) {
            orderIdsJson.PushBack(orderId, allocator);
        }
        json.AddMember("orderIds", orderIdsJson, allocator);
        requestPayload->checkpointSerialize(json, "requestPayload");
        errorPayload->checkpointSerialize(json, "errorPayload");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

CancelOrdersErrorResponsePayload::Ptr CancelOrdersErrorResponsePayload::fromJson(
    const rapidjson::Value& json)
{
    std::vector<OrderID> orderIds;
    for (const auto& orderId : json["orderIds"].GetArray()) {
        orderIds.push_back(orderId.GetUint());
    }
    return MessagePayload::create<CancelOrdersErrorResponsePayload>(
        std::move(orderIds),
        CancelOrdersPayload::fromJson(json["requestPayload"]),
        ErrorResponsePayload::fromJson(json["errorPayload"]));
}

//-------------------------------------------------------------------------

void RetrieveBookPayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("depth", rapidjson::Value{depth}, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void RetrieveBookPayload::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    jsonSerialize(json, key);
}

//-------------------------------------------------------------------------

RetrieveBookPayload::Ptr RetrieveBookPayload::fromJson(const rapidjson::Value& json)
{
    return MessagePayload::create<RetrieveBookPayload>(
        json["depth"].GetUint(), json["bookId"].GetUint());
}

//-------------------------------------------------------------------------

void RetrieveBookResponsePayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("time", rapidjson::Value{time}, allocator);
        rapidjson::Value tickContainersJson{rapidjson::kArrayType};
        for (const TickContainer::ContainerType& tickContainer : tickContainers) {
            rapidjson::Document tickContainerJson{rapidjson::kArrayType, &allocator};
            for (const TickContainer::value_type& order : tickContainer) {
                rapidjson::Document orderJson{rapidjson::kObjectType, &allocator};
                order->jsonSerialize(orderJson);
                tickContainerJson.PushBack(orderJson, allocator);
            }
            tickContainersJson.PushBack(tickContainerJson, allocator);
        }
        json.AddMember("tickContainers", tickContainersJson, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void RetrieveBookResponsePayload::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("time", rapidjson::Value{time}, allocator);
        rapidjson::Value tickContainersJson{rapidjson::kArrayType};
        for (const TickContainer::ContainerType& tickContainer : tickContainers) {
            rapidjson::Document tickContainerJson{rapidjson::kArrayType, &allocator};
            for (const TickContainer::value_type& order : tickContainer) {
                rapidjson::Document orderJson{rapidjson::kObjectType, &allocator};
                order->jsonSerialize(orderJson);
                tickContainerJson.PushBack(orderJson, allocator);
            }
            tickContainersJson.PushBack(tickContainerJson, allocator);
        }
        json.AddMember("tickContainers", tickContainersJson, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

RetrieveBookResponsePayload::Ptr RetrieveBookResponsePayload::fromJson(
    const rapidjson::Value& json)
{
    return MessagePayload::create<RetrieveBookResponsePayload>(
        json["time"].GetUint64(),
        [&] {
            std::vector<TickContainer::ContainerType> tickContainers;
            for (const rapidjson::Value& tickContainerJson : json["tickContainers"].GetArray()) {
                tickContainers.push_back([&] {
                    TickContainer::ContainerType tickContainer;
                    for (const rapidjson::Value& orderJson : tickContainerJson.GetArray()) {
                        // Just provide conservative enough rounding details for now...
                        tickContainer.push_back(LimitOrder::fromJson(orderJson, 12, 12));
                    }
                    return tickContainer;
                }());
            }
            return tickContainers;
        }());
}

//-------------------------------------------------------------------------

void RetrieveL1Payload::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void RetrieveL1Payload::checkpointSerialize(rapidjson::Document& json, const std::string& key) const
{
    jsonSerialize(json, key);
}

//-------------------------------------------------------------------------

RetrieveL1Payload::Ptr RetrieveL1Payload::fromJson(const rapidjson::Value& json)
{
    return MessagePayload::create<RetrieveL1Payload>(json["bookId"].GetUint());
}

//-------------------------------------------------------------------------

void RetrieveL1ResponsePayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("timestamp", rapidjson::Value{time}, allocator);
        json.AddMember(
            "bestAskPrice", rapidjson::Value{taosim::util::decimal2double(bestAskPrice)}, allocator);
        json.AddMember(
            "bestAskVolume", rapidjson::Value{taosim::util::decimal2double(bestAskVolume)}, allocator);
        json.AddMember(
            "askTotalVolume", rapidjson::Value{taosim::util::decimal2double(askTotalVolume)}, allocator);
        json.AddMember(
            "bestBidPrice", rapidjson::Value{taosim::util::decimal2double(bestBidPrice)}, allocator);
        json.AddMember(
            "bestBidVolume", rapidjson::Value{taosim::util::decimal2double(bestBidVolume)}, allocator);
        json.AddMember(
            "bidTotalVolume", rapidjson::Value{taosim::util::decimal2double(bidTotalVolume)}, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void RetrieveL1ResponsePayload::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("timestamp", rapidjson::Value{time}, allocator);
        json.AddMember(
            "bestAskPrice", rapidjson::Value{taosim::util::packDecimal(bestAskPrice)}, allocator);
        json.AddMember(
            "bestAskVolume", rapidjson::Value{taosim::util::packDecimal(bestAskVolume)}, allocator);
        json.AddMember(
            "askTotalVolume", rapidjson::Value{taosim::util::packDecimal(askTotalVolume)}, allocator);
        json.AddMember(
            "bestBidPrice", rapidjson::Value{taosim::util::packDecimal(bestBidPrice)}, allocator);
        json.AddMember(
            "bestBidVolume", rapidjson::Value{taosim::util::packDecimal(bestBidVolume)}, allocator);
        json.AddMember(
            "bidTotalVolume", rapidjson::Value{taosim::util::packDecimal(bidTotalVolume)}, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

RetrieveL1ResponsePayload::Ptr RetrieveL1ResponsePayload::fromJson(const rapidjson::Value& json)
{
    return MessagePayload::create<RetrieveL1ResponsePayload>(
        json["timestamp"].GetUint64(),
        taosim::json::getDecimal(json["bestAskPrice"]),
        taosim::json::getDecimal(json["bestAskVolume"]),
        taosim::json::getDecimal(json["askTotalVolume"]),
        taosim::json::getDecimal(json["bestBidPrice"]),
        taosim::json::getDecimal(json["bestBidVolume"]),
        taosim::json::getDecimal(json["bidTotalVolume"]),
        json["bookId"].GetUint());
}

//-------------------------------------------------------------------------

void SubscribeEventTradeByOrderPayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("orderId", rapidjson::Value{id}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void SubscribeEventTradeByOrderPayload::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    jsonSerialize(json, key);
}

//-------------------------------------------------------------------------

SubscribeEventTradeByOrderPayload::Ptr SubscribeEventTradeByOrderPayload::fromJson(
    const rapidjson::Value& json)
{
    return MessagePayload::create<SubscribeEventTradeByOrderPayload>(json["orderId"].GetUint());
}

//-------------------------------------------------------------------------

void EventOrderMarketPayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        order.jsonSerialize(json, "order");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void EventOrderMarketPayload::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        order.checkpointSerialize(json, "order");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

EventOrderMarketPayload::Ptr EventOrderMarketPayload::fromJson(const rapidjson::Value& json)
{
    return MessagePayload::create<EventOrderMarketPayload>(
        MarketOrder{
            json["orderId"].GetUint(),
            json["timestamp"].GetUint64(),
            taosim::json::getDecimal(json["volume"]),
            OrderDirection{json["direction"].GetUint()}});
}

//-------------------------------------------------------------------------

void EventOrderLimitPayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        order.jsonSerialize(json, "order");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void EventOrderLimitPayload::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        order.checkpointSerialize(json, "order");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

EventOrderLimitPayload::Ptr EventOrderLimitPayload::fromJson(const rapidjson::Value& json)
{
    return MessagePayload::create<EventOrderLimitPayload>(
        LimitOrder{
            json["orderId"].GetUint(),
            json["timestamp"].GetUint64(),
            taosim::json::getDecimal(json["volume"]),
            OrderDirection{json["direction"].GetUint()},
            taosim::json::getDecimal(json["price"])});
}

//-------------------------------------------------------------------------

void EventTradePayload::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        trade.jsonSerialize(json, "trade");
        context.jsonSerialize(json, "context");
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
        taosim::json::setOptionalMember(json, "clientOrderId", clientOrderId);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void EventTradePayload::checkpointSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        trade.checkpointSerialize(json, "trade");
        context.checkpointSerialize(json, "context");
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
        taosim::json::setOptionalMember(json, "clientOrderId", clientOrderId);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

EventTradePayload::Ptr EventTradePayload::fromJson(const rapidjson::Value& json)
{
    return MessagePayload::create<EventTradePayload>(
        Trade{
            json["tradeId"].GetUint(),
            json["timestamp"].GetUint64(),
            OrderDirection{json["direction"].GetUint()},
            json["aggressingOrderId"].GetUint(),
            json["restingOrderId"].GetUint(),
            taosim::json::getDecimal(json["volume"]),
            taosim::json::getDecimal(json["price"])},
        TradeLogContext(
            json["aggressingAgentId"].GetUint(),
            json["restingAgentId"].GetUint(),
            json["bookId"].GetUint(),
            taosim::exchange::Fees{
                .maker = taosim::json::getDecimal(json["fees"]["maker"]),
                .taker = taosim::json::getDecimal(json["fees"]["taker"])}
        ),
        json["bookId"].GetUint(),
        !json["clientOrderId"].IsNull()
            ? std::make_optional(json["clientOrderId"].GetUint())
            : std::nullopt);
}

//-------------------------------------------------------------------------

void ResetAgentsPayload::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        rapidjson::Value agentIdsJson{rapidjson::kArrayType};
        for (AgentId agentId : agentIds) {
            agentIdsJson.PushBack(agentId, allocator);
        }
        json.AddMember("agentIds", agentIdsJson, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void ResetAgentsPayload::checkpointSerialize(rapidjson::Document& json, const std::string& key) const
{
    jsonSerialize(json, key);
}

//-------------------------------------------------------------------------

ResetAgentsPayload::Ptr ResetAgentsPayload::fromJson(const rapidjson::Value& json)
{
    std::vector<AgentId> agentIds;
    for (const auto& agentId : json["agentIds"].GetArray()) {
        agentIds.push_back(agentId.GetInt());
    }
    return MessagePayload::create<ResetAgentsPayload>(std::move(agentIds));
}

//-------------------------------------------------------------------------

void ResetAgentsResponsePayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        rapidjson::Value agentIdsJson{rapidjson::kArrayType};
        for (AgentId agentId : agentIds) {
            agentIdsJson.PushBack(agentId, allocator);
        }
        json.AddMember("agentIds", agentIdsJson, allocator);
        requestPayload->jsonSerialize(json, "requestPayload");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void ResetAgentsResponsePayload::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        rapidjson::Value agentIdsJson{rapidjson::kArrayType};
        for (AgentId agentId : agentIds) {
            agentIdsJson.PushBack(agentId, allocator);
        }
        json.AddMember("agentIds", agentIdsJson, allocator);
        requestPayload->checkpointSerialize(json, "requestPayload");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

ResetAgentsResponsePayload::Ptr ResetAgentsResponsePayload::fromJson(const rapidjson::Value& json)
{
    std::vector<AgentId> agentIds;
    for (const auto& agentId : json["agentIds"].GetArray()) {
        agentIds.push_back(agentId.GetInt());
    }
    return MessagePayload::create<ResetAgentsResponsePayload>(
        std::move(agentIds),
        ResetAgentsPayload::fromJson(json["requestPayload"]));
}

//-------------------------------------------------------------------------

void ResetAgentsErrorResponsePayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        rapidjson::Value agentIdsJson{rapidjson::kArrayType};
        for (AgentId agentId : agentIds) {
            agentIdsJson.PushBack(agentId, allocator);
        }
        json.AddMember("agentIds", agentIdsJson, allocator);
        requestPayload->jsonSerialize(json, "requestPayload");
        errorPayload->jsonSerialize(json, "errorPayload");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void ResetAgentsErrorResponsePayload::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        rapidjson::Value agentIdsJson{rapidjson::kArrayType};
        for (AgentId agentId : agentIds) {
            agentIdsJson.PushBack(agentId, allocator);
        }
        json.AddMember("agentIds", agentIdsJson, allocator);
        requestPayload->checkpointSerialize(json, "requestPayload");
        errorPayload->checkpointSerialize(json, "errorPayload");
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

ResetAgentsErrorResponsePayload::Ptr ResetAgentsErrorResponsePayload::fromJson(
    const rapidjson::Value& json)
{
    return MessagePayload::create<ResetAgentsErrorResponsePayload>(
        [&json] {
            std::vector<AgentId> agentIds;
            for (const auto& agentId : json["agentIds"].GetArray()) {
                agentIds.push_back(agentId.GetInt());
            }
            return agentIds;
        }(),
        ResetAgentsPayload::fromJson(json["requestPayload"]),
        ErrorResponsePayload::fromJson(json["errorPayload"]));
}

//-------------------------------------------------------------------------

void WakeupForCancellationPayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("orderToCancelId", rapidjson::Value{orderToCancelId}, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void WakeupForCancellationPayload::checkpointSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    jsonSerialize(json, key);
}

//-------------------------------------------------------------------------