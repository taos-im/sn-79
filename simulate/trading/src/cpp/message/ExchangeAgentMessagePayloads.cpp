/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/message/ExchangeAgentMessagePayloads.hpp>

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

StartSimulationPayload::Ptr StartSimulationPayload::fromJson(const rapidjson::Value& json)
{
    return MessagePayload::create<StartSimulationPayload>(json["logDir"].GetString());
}

//-------------------------------------------------------------------------

void PlaceOrderMarketPayload::L3Serialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("d", rapidjson::Value{std::to_underlying(direction)}, allocator);
        json.AddMember("v", rapidjson::Value{taosim::util::decimal2double(volume)}, allocator);
        json.AddMember("b", rapidjson::Value{bookId}, allocator);
        json.AddMember("n", rapidjson::Value{std::to_underlying(currency)}, allocator);
        taosim::json::setOptionalMember(json, "ci", clientOrderId);
        json.AddMember(
            "s", rapidjson::Value{magic_enum::enum_name(stpFlag).data(), allocator}, allocator);
        json.AddMember("l", rapidjson::Value{taosim::util::decimal2double(leverage)}, allocator);
        std::visit(
            [&](auto&& flag) {
                using T = std::remove_cvref_t<decltype(flag)>;
                if constexpr (std::same_as<T, SettleType>) {
                    json.AddMember(
                        "f", rapidjson::Value{magic_enum::enum_name(flag).data(), allocator}, allocator);
                } else if constexpr (std::same_as<T, OrderID>) {
                    json.AddMember("f", rapidjson::Value{flag}, allocator);
                } else {
                    static_assert(false, "Non-exhaustive visitor");
                }
            }, settleFlag);
        taosim::json::setOptionalMember(json, "sl", stopLoss);
        taosim::json::setOptionalMember(json, "tp", takeProfit);
        taosim::json::setOptionalMember(json, "ph", placeholder);
    };
    return taosim::json::serializeHelper(json, key, serialize);
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
        json.AddMember("currency", rapidjson::Value{std::to_underlying(currency)}, allocator);
        taosim::json::setOptionalMember(json, "clientOrderId", clientOrderId);
        json.AddMember(
            "stpFlag",
            rapidjson::Value{magic_enum::enum_name(stpFlag).data(), allocator},
            allocator);
        json.AddMember("leverage", rapidjson::Value{taosim::util::decimal2double(leverage)}, allocator);
        std::visit([&](auto&& flag) {
            using T = std::remove_cvref_t<decltype(flag)>;
            if constexpr (std::same_as<T, SettleType>) {
                json.AddMember(
                    "settleFlag",
                    rapidjson::Value{magic_enum::enum_name(flag).data(), allocator},
                    allocator);
            } else if constexpr (std::same_as<T, OrderID>) {
                json.AddMember("settleFlag", rapidjson::Value{flag}, allocator);
            } else {
                static_assert(false, "Non-exhaustive visitor");
            }
        }, settleFlag);
        json.AddMember(
            "maxSlippage", rapidjson::Value{taosim::util::decimal2double(maxSlippage)}, allocator);
        json.AddMember("delegate", rapidjson::Value{delegate.c_str(), allocator}, allocator);
        taosim::json::setOptionalMember(json, "stopLoss", stopLoss);
        taosim::json::setOptionalMember(json, "takeProfit", takeProfit);
        taosim::json::setOptionalMember(json, "placeholder", placeholder);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

PlaceOrderMarketPayload::Ptr PlaceOrderMarketPayload::fromJson(const rapidjson::Value& json)
{
    auto getOptDec = [&](const char* k) -> std::optional<taosim::decimal_t> {
        if (!json.HasMember(k) || json[k].IsNull()) {
            return std::nullopt;
        }
        return std::make_optional(taosim::json::getDecimal(json[k]));
    };

    return MessagePayload::create<PlaceOrderMarketPayload>(
        OrderDirection{json["direction"].GetUint()},
        taosim::json::getDecimal(json["volume"]),
        json.HasMember("leverage")
            ? taosim::json::getDecimal(json["leverage"])
            : 0_dec,
        json["bookId"].GetUint(),
        Currency{json["currency"].GetUint()},
        !json["clientOrderId"].IsNull()
            ? std::make_optional(json["clientOrderId"].GetUint())
            : std::nullopt,
        json.HasMember("stpFlag")
            ? magic_enum::enum_cast<STPFlag>(json["stpFlag"].GetUint())
                .value_or(STPFlag::CO)
            : STPFlag::CO,
        json.HasMember("settleFlag")
            ? (json["settleFlag"].IsInt() && magic_enum::enum_cast<SettleType>(json["settleFlag"].GetInt()).has_value()
                ? SettleFlag(magic_enum::enum_cast<SettleType>(json["settleFlag"].GetInt()).value())
                : SettleFlag(static_cast<OrderID>(json["settleFlag"].GetUint())))
            : SettleFlag(SettleType::FIFO),
        getOptDec("stopLoss"),
        getOptDec("takeProfit"),
        getOptDec("placeholder")
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

PlaceOrderMarketErrorResponsePayload::Ptr PlaceOrderMarketErrorResponsePayload::fromJson(
    const rapidjson::Value& json)
{
    return MessagePayload::create<PlaceOrderMarketErrorResponsePayload>(
        PlaceOrderMarketPayload::fromJson(json["requestPayload"]),
        ErrorResponsePayload::fromJson(json["errorPayload"]));
}

//-------------------------------------------------------------------------

void PlaceOrderLimitPayload::L3Serialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [&](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("d", rapidjson::Value{std::to_underlying(direction)}, allocator);
        json.AddMember("v", rapidjson::Value{taosim::util::decimal2double(volume)}, allocator);
        json.AddMember("p", rapidjson::Value{taosim::util::decimal2double(price)}, allocator);
        json.AddMember("l", rapidjson::Value{taosim::util::decimal2double(leverage)}, allocator);
        json.AddMember("b", rapidjson::Value{bookId}, allocator);
        json.AddMember("n", rapidjson::Value{std::to_underlying(currency)}, allocator);
        taosim::json::setOptionalMember(json, "ci", clientOrderId);
        json.AddMember("y", rapidjson::Value{postOnly}, allocator);
        json.AddMember(
            "r", rapidjson::Value{magic_enum::enum_name(timeInForce).data(), allocator}, allocator);
        taosim::json::setOptionalMember(json, "x", expiryPeriod);
        json.AddMember(
            "s", rapidjson::Value{magic_enum::enum_name(stpFlag).data(), allocator}, allocator);
        std::visit(
            [&](auto&& flag) {
                using T = std::remove_cvref_t<decltype(flag)>;
                if constexpr (std::same_as<T, SettleType>) {
                    json.AddMember(
                        "f", rapidjson::Value{magic_enum::enum_name(flag).data(), allocator}, allocator);
                } else if constexpr (std::same_as<T, OrderID>) {
                    json.AddMember("f", rapidjson::Value{flag}, allocator);
                } else {
                    static_assert(false, "Non-exhaustive visitor");
                }
            },
            settleFlag);
        taosim::json::setOptionalMember(json, "sl", stopLoss);
        taosim::json::setOptionalMember(json, "tp", takeProfit);
        taosim::json::setOptionalMember(json, "ph", placeholder);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void PlaceOrderLimitPayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [&](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("direction", rapidjson::Value{std::to_underlying(direction)}, allocator);
        json.AddMember("volume", rapidjson::Value{taosim::util::decimal2double(volume)}, allocator);
        json.AddMember("price", rapidjson::Value{taosim::util::decimal2double(price)}, allocator);
        json.AddMember("leverage", rapidjson::Value{taosim::util::decimal2double(leverage)}, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
        json.AddMember("currency", rapidjson::Value{std::to_underlying(currency)}, allocator);
        taosim::json::setOptionalMember(json, "clientOrderId", clientOrderId);
        json.AddMember("postOnly", rapidjson::Value{postOnly}, allocator);
        json.AddMember("allowPartial", rapidjson::Value{allowPartial}, allocator);
        json.AddMember(
            "timeInForce",
            rapidjson::Value{magic_enum::enum_name(timeInForce).data(), allocator},
            allocator);
        taosim::json::setOptionalMember(json, "expiryPeriod", expiryPeriod);
        json.AddMember(
            "stpFlag",
            rapidjson::Value{magic_enum::enum_name(stpFlag).data(), allocator},
            allocator);
        std::visit([&](auto&& flag) {
            using T = std::remove_cvref_t<decltype(flag)>;
            if constexpr (std::is_same_v<T, SettleType>) {
                json.AddMember(
                    "settleFlag",
                    rapidjson::Value{magic_enum::enum_name(flag).data(), allocator},
                    allocator);
            } else if constexpr (std::is_same_v<T, OrderID>) {
                json.AddMember("settleFlag", rapidjson::Value{flag}, allocator);
            }
        }, settleFlag);
        json.AddMember("delegate", rapidjson::Value{delegate.c_str(), allocator}, allocator);
        taosim::json::setOptionalMember(json, "stopLoss", stopLoss);
        taosim::json::setOptionalMember(json, "takeProfit", takeProfit);
        taosim::json::setOptionalMember(json, "placeholder", placeholder);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

PlaceOrderLimitPayload::Ptr PlaceOrderLimitPayload::fromJson(const rapidjson::Value& json)
{
    auto getOptDec = [&](const char* k) -> std::optional<taosim::decimal_t> {
        if (!json.HasMember(k) || json[k].IsNull()) {
            return std::nullopt;
        }
        return std::make_optional(taosim::json::getDecimal(json[k]));
    };

    auto payload = MessagePayload::create<PlaceOrderLimitPayload>(
        OrderDirection{json["direction"].GetUint()},
        taosim::json::getDecimal(json["volume"]),
        taosim::json::getDecimal(json["price"]),
        json.HasMember("leverage")
            ? taosim::json::getDecimal(json["leverage"])
            : 0_dec,
        json["bookId"].GetUint(),
        Currency{json["currency"].GetUint()},
        !json["clientOrderId"].IsNull()
            ? std::make_optional(json["clientOrderId"].GetUint())
            : std::nullopt,
        json.HasMember("postOnly") ? json["postOnly"].GetBool() : false,
        json.HasMember("timeInForce")
            ? magic_enum::enum_cast<taosim::TimeInForce>(json["timeInForce"].GetUint())
                .value_or(taosim::TimeInForce::GTC)
            : taosim::TimeInForce::GTC,
        json.HasMember("expiryPeriod")
            ? !json["expiryPeriod"].IsNull()
                ? std::make_optional(json["expiryPeriod"].GetUint64())
                : std::nullopt
            : std::nullopt,
        json.HasMember("stpFlag")
            ? magic_enum::enum_cast<STPFlag>(json["stpFlag"].GetUint())
                .value_or(STPFlag::CO)
            : STPFlag::CO,
        json.HasMember("settleFlag")
            ? (json["settleFlag"].IsInt() && magic_enum::enum_cast<SettleType>(json["settleFlag"].GetInt()).has_value()
                ? SettleFlag(magic_enum::enum_cast<SettleType>(json["settleFlag"].GetInt()).value())
                : SettleFlag(static_cast<OrderID>(json["settleFlag"].GetUint())))
            : SettleFlag(SettleType::FIFO),
        getOptDec("stopLoss"),
        getOptDec("takeProfit"),
        getOptDec("placeholder")
        );
    payload->allowPartial = json.HasMember("allowPartial") && !json["allowPartial"].IsNull()
        ? json["allowPartial"].GetBool()
        : true;
    return payload;
}

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

RetrieveOrdersResponsePayload::Ptr RetrieveOrdersResponsePayload::fromJson(
    const rapidjson::Value& json)
{
    std::vector<LimitOrder> orders;
    for (const auto& order : json["orders"].GetArray()) {
        orders.emplace_back(
            order["orderId"].GetUint(),
            order["timestamp"].GetUint64(),
            // Currency{order["type"].GetUint()},
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
        for (const taosim::event::Cancellation& cancellation : cancellations) {
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

CancelOrdersPayload::Ptr CancelOrdersPayload::fromJson(const rapidjson::Value& json)
{
    return MessagePayload::create<CancelOrdersPayload>(
        [&json] {
            std::vector<taosim::event::Cancellation> cancellations;
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

void ClosePositionsPayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        rapidjson::Value closePositionsJson{rapidjson::kArrayType};
        for (const ClosePosition& closePosition : closePositions) {
            rapidjson::Document closePositionJson{rapidjson::kObjectType, &allocator};
            closePosition.jsonSerialize(closePositionJson);
            closePositionsJson.PushBack(closePositionJson, allocator);
        }
        json.AddMember("closePositions", closePositionsJson, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

ClosePositionsPayload::Ptr ClosePositionsPayload::fromJson(const rapidjson::Value& json)
{
    return MessagePayload::create<ClosePositionsPayload>(
        [&json] {
            std::vector<ClosePosition> ClosePositions;
            for (const auto& closePositionJson : json["closePositions"].GetArray()) {
                ClosePositions.emplace_back(
                    closePositionJson["orderId"].GetUint(),
                    !closePositionJson["volume"].IsNull()
                        ? std::make_optional(taosim::json::getDecimal(json["volume"]))
                        : std::nullopt);
            }
            return ClosePositions;
        }(),
        json["bookId"].GetUint());
}

//-------------------------------------------------------------------------

void ClosePositionsResponsePayload::jsonSerialize(
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

ClosePositionsResponsePayload::Ptr ClosePositionsResponsePayload::fromJson(
    const rapidjson::Value& json)
{
    std::vector<OrderID> orderIds;
    for (const auto& orderId : json["orderIds"].GetArray()) {
        orderIds.push_back(orderId.GetUint());
    }
    return MessagePayload::create<ClosePositionsResponsePayload>(
        std::move(orderIds),
        ClosePositionsPayload::fromJson(json["requestPayload"]));
}

//-------------------------------------------------------------------------

void ClosePositionsErrorResponsePayload::jsonSerialize(
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

ClosePositionsErrorResponsePayload::Ptr ClosePositionsErrorResponsePayload::fromJson(
    const rapidjson::Value& json)
{
    std::vector<OrderID> orderIds;
    for (const auto& orderId : json["orderIds"].GetArray()) {
        orderIds.push_back(orderId.GetUint());
    }
    return MessagePayload::create<ClosePositionsErrorResponsePayload>(
        std::move(orderIds),
        ClosePositionsPayload::fromJson(json["requestPayload"]),
        ErrorResponsePayload::fromJson(json["errorPayload"]));
}

//-------------------------------------------------------------------------

void RetrieveL2Payload::jsonSerialize(
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

RetrieveL2Payload::Ptr RetrieveL2Payload::fromJson(const rapidjson::Value& json)
{
    return MessagePayload::create<RetrieveL2Payload>(
        json["depth"].GetUint(), json["bookId"].GetUint());
}

//-------------------------------------------------------------------------

void RetrieveL2ResponsePayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("time", rapidjson::Value{time}, allocator);
        rapidjson::Value bidsJson{rapidjson::kArrayType};
        for (const auto& level : bids) {
            rapidjson::Document levelJson{rapidjson::kObjectType, &allocator};
            levelJson.AddMember(
                "price",
                rapidjson::Value{taosim::util::decimal2double(level.price)},
                allocator);
            levelJson.AddMember(
                "quantity",
                rapidjson::Value{taosim::util::decimal2double(level.quantity)},
                allocator);
            bidsJson.PushBack(levelJson, allocator);
        }
        json.AddMember("bids", bidsJson, allocator);
        rapidjson::Value asksJson{rapidjson::kArrayType};
        for (const auto& level : asks) {
            rapidjson::Document levelJson{rapidjson::kObjectType, &allocator};
            levelJson.AddMember(
                "price",
                rapidjson::Value{taosim::util::decimal2double(level.price)},
                allocator);
            levelJson.AddMember(
                "quantity",
                rapidjson::Value{taosim::util::decimal2double(level.quantity)},
                allocator);
            asksJson.PushBack(levelJson, allocator);
        }
        json.AddMember("asks", asksJson, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

RetrieveL2ResponsePayload::Ptr RetrieveL2ResponsePayload::fromJson(
    const rapidjson::Value& json)
{
    return MessagePayload::create<RetrieveL2ResponsePayload>(
        json["time"].GetUint64(),
        [&] {
            std::vector<BookLevel> bids;
            for (const auto& levelJson : json["bids"].GetArray()) {
                bids.push_back({
                    .price = taosim::json::getDecimal(levelJson["price"]),
                    .quantity = taosim::json::getDecimal(levelJson["quantity"])
                });
            }
            return bids;
        }(),
        [&] {
            std::vector<BookLevel> asks;
            for (const auto& levelJson : json["asks"].GetArray()) {
                asks.push_back({
                    .price = taosim::json::getDecimal(levelJson["price"]),
                    .quantity = taosim::json::getDecimal(levelJson["quantity"])
                });
            }
            return asks;
        }(),
        json["bookId"].GetUint());
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

void RetrieveL1ExtPayload::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

RetrieveL1ExtPayload::Ptr RetrieveL1ExtPayload::fromJson(const rapidjson::Value& json)
{
    return MessagePayload::create<RetrieveL1ExtPayload>(json["bookId"].GetUint());
}

//-------------------------------------------------------------------------

void RetrieveL1ExtResponsePayload::jsonSerialize(
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
        json.AddMember("tradeCount", rapidjson::Value{tradeStats.tradeCount}, allocator);
        json.AddMember(
            "volumeSum", rapidjson::Value{taosim::util::decimal2double(tradeStats.volumeSum)}, allocator);
        json.AddMember(
            "notionalSum", rapidjson::Value{taosim::util::decimal2double(tradeStats.notionalSum)}, allocator);
        json.AddMember("logReturnSum", rapidjson::Value{tradeStats.logReturnSum}, allocator);
        json.AddMember("logReturnSqSum", rapidjson::Value{tradeStats.logReturnSqSum}, allocator);
        json.AddMember(
            "lastTradePrice",
            rapidjson::Value{taosim::util::decimal2double(tradeStats.lastTradePrice)},
            allocator);
        json.AddMember("lastTradeTime", rapidjson::Value{tradeStats.lastTradeTime}, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

RetrieveL1ExtResponsePayload::Ptr RetrieveL1ExtResponsePayload::fromJson(
    const rapidjson::Value& json)
{
    return MessagePayload::create<RetrieveL1ExtResponsePayload>(
        json["timestamp"].GetUint64(),
        taosim::json::getDecimal(json["bestAskPrice"]),
        taosim::json::getDecimal(json["bestAskVolume"]),
        taosim::json::getDecimal(json["askTotalVolume"]),
        taosim::json::getDecimal(json["bestBidPrice"]),
        taosim::json::getDecimal(json["bestBidVolume"]),
        taosim::json::getDecimal(json["bidTotalVolume"]),
        taosim::book::BookTradeStats{
            .tradeCount = json["tradeCount"].GetUint64(),
            .volumeSum = taosim::json::getDecimal(json["volumeSum"]),
            .notionalSum = taosim::json::getDecimal(json["notionalSum"]),
            .logReturnSum = json["logReturnSum"].GetDouble(),
            .logReturnSqSum = json["logReturnSqSum"].GetDouble(),
            .lastTradePrice = taosim::json::getDecimal(json["lastTradePrice"]),
            .lastTradeTime = json["lastTradeTime"].GetUint64()},
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
        auto& allocator = json.GetAllocator();
        order.jsonSerialize(json, "order");
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
        json.AddMember("agentId", rapidjson::Value{agentId}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

EventOrderMarketPayload::Ptr EventOrderMarketPayload::fromJson(const rapidjson::Value& json)
{
    auto payload = MessagePayload::create<EventOrderMarketPayload>(
        MarketOrder{
            json["orderId"].GetUint(),
            json["timestamp"].GetUint64(),
            // Currency{json["currency"].GetUint()},
            taosim::json::getDecimal(json["volume"]),
            OrderDirection{json["direction"].GetUint()}});
    if (json.HasMember("bookId")) payload->bookId = json["bookId"].GetUint();
    if (json.HasMember("agentId")) payload->agentId = json["agentId"].GetInt();
    return payload;
}

//-------------------------------------------------------------------------

void EventOrderLimitPayload::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        order.jsonSerialize(json, "order");
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
        json.AddMember("agentId", rapidjson::Value{agentId}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

EventOrderLimitPayload::Ptr EventOrderLimitPayload::fromJson(const rapidjson::Value& json)
{
    auto payload = MessagePayload::create<EventOrderLimitPayload>(
        LimitOrder{
            json["orderId"].GetUint(),
            json["timestamp"].GetUint64(),
            // Currency{json["currency"].GetUint()},
            taosim::json::getDecimal(json["volume"]),
            OrderDirection{json["direction"].GetUint()},
            taosim::json::getDecimal(json["price"])});
    if (json.HasMember("bookId")) payload->bookId = json["bookId"].GetUint();
    if (json.HasMember("agentId")) payload->agentId = json["agentId"].GetInt();
    return payload;
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
        if (!delegate.empty()) {
            json.AddMember("delegate", rapidjson::Value{delegate.c_str(), allocator}, allocator);
        }
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
            taosim::matching::Fees{
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
