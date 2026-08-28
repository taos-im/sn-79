/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/book/Book.hpp>
#include <taosim/book/BookTradeStats.hpp>
#include <taosim/book/serialization/BookTradeStats.hpp>
#include <taosim/decimal/serialization/decimal.hpp>
#include "Cancellation.hpp"
#include "ClosePosition.hpp"
#include <taosim/message/MessagePayload.hpp>
#include "Order.hpp"
#include "Trade.hpp"
#include "common.hpp"
#include "Flags.hpp"

#include <msgpack.hpp>

#include <optional>
#include <vector>

using STPFlag = taosim::STPFlag;
using SettleFlag = taosim::SettleFlag;
using SettleType = taosim::SettleType;

//-------------------------------------------------------------------------

struct StartSimulationPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<StartSimulationPayload>;

    std::string logDir;

    StartSimulationPayload() noexcept = default;

    StartSimulationPayload(const std::string& logDir) noexcept
        : logDir{logDir}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(logDir);
};

//-------------------------------------------------------------------------

struct PlaceOrderMarketPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<PlaceOrderMarketPayload>;

    OrderDirection direction;
    taosim::decimal_t volume;
    taosim::decimal_t leverage;
    BookId bookId;
    Currency currency{Currency::BASE};
    std::optional<ClientOrderID> clientOrderId;
    STPFlag stpFlag{STPFlag::CO};
    SettleFlag settleFlag{SettleType::FIFO};
    // Agent-settable, same meaning as on the limit payload: when false, a swap that cannot be filled
    // in full is refused rather than filled short. It matters here only because a SELL draws on ONE
    // named delegate, so the executable size is the delegate's stake and not the summed free balance
    // the agent sized against. Defaults true, which is the historical behaviour.
    bool allowPartial{true};
    taosim::decimal_t maxSlippage;
    std::string delegate;
    std::optional<taosim::decimal_t> stopLoss;
    std::optional<taosim::decimal_t> takeProfit;
    std::optional<taosim::decimal_t> placeholder;
    bool skipMinSizeCheck{};
    uint8_t closeReason{};        // 0=none, 1=SL, 2=TP — set by SL/TP dispatch only
    OrderID originatingOrderId{}; // LOB ID of the position order that triggered SL/TP
    // Set by sweepCrossing only, and deliberately NOT serialised: the sweep message is created and
    // consumed inside this process, so this never has to survive a wire hop. See OrderClientContext.
    std::optional<AgentId> initiatorAgentId;

    PlaceOrderMarketPayload() noexcept = default;

    PlaceOrderMarketPayload(
        OrderDirection direction,
        taosim::decimal_t volume,
        BookId bookId,
        Currency currency = Currency::BASE,
        std::optional<ClientOrderID> clientOrderId = {},
        STPFlag stpFlag = STPFlag::CO,
        SettleFlag settleFlag = SettleType::FIFO,
        std::optional<taosim::decimal_t> stopLoss = {},
        std::optional<taosim::decimal_t> takeProfit = {},
        std::optional<taosim::decimal_t> placeholder = {}) noexcept
        : direction{direction},
          volume{volume},
          leverage{0_dec},
          bookId{bookId},
          currency{currency},
          clientOrderId{clientOrderId},
          stpFlag{stpFlag},
          settleFlag{settleFlag},
          stopLoss{stopLoss},
          takeProfit{takeProfit},
          placeholder{placeholder}
    {}

    PlaceOrderMarketPayload(
        OrderDirection direction,
        taosim::decimal_t volume,
        taosim::decimal_t leverage,
        BookId bookId,
        Currency currency = Currency::BASE,
        std::optional<ClientOrderID> clientOrderId = {},
        STPFlag stpFlag = STPFlag::CO,
        SettleFlag settleFlag = SettleType::FIFO,
        std::optional<taosim::decimal_t> stopLoss = {},
        std::optional<taosim::decimal_t> takeProfit = {},
        std::optional<taosim::decimal_t> placeholder = {}) noexcept
        : direction{direction},
          volume{volume},
          leverage{leverage},
          bookId{bookId},
          currency{currency},
          clientOrderId{clientOrderId},
          stpFlag{stpFlag},
          settleFlag{settleFlag},
          stopLoss{stopLoss},
          takeProfit{takeProfit},
          placeholder{placeholder}
    {}

    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(
        direction,
        volume,
        leverage,
        bookId,
        currency,
        clientOrderId,
        MSGPACK_NVP("stp", stpFlag),
        settleFlag,
        allowPartial,
        MSGPACK_NVP("max_slippage", maxSlippage),
        delegate,
        MSGPACK_NVP("stopLoss", stopLoss),
        MSGPACK_NVP("takeProfit", takeProfit),
        MSGPACK_NVP("placeholder", placeholder));
};

//-------------------------------------------------------------------------

struct PlaceOrderMarketResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<PlaceOrderMarketResponsePayload>;

    OrderID id;
    PlaceOrderMarketPayload::Ptr requestPayload;

    PlaceOrderMarketResponsePayload() noexcept = default;

    PlaceOrderMarketResponsePayload(
        OrderID id, PlaceOrderMarketPayload::Ptr requestPayload) noexcept
        : id{id}, requestPayload{requestPayload}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(MSGPACK_NVP("orderId", id), requestPayload);
};

//-------------------------------------------------------------------------

struct PlaceOrderMarketErrorResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<PlaceOrderMarketErrorResponsePayload>;

    PlaceOrderMarketPayload::Ptr requestPayload;
    ErrorResponsePayload::Ptr errorPayload;

    PlaceOrderMarketErrorResponsePayload() noexcept = default;

    PlaceOrderMarketErrorResponsePayload(
        PlaceOrderMarketPayload::Ptr requestPayload,
        ErrorResponsePayload::Ptr errorPayload) noexcept
        : requestPayload{requestPayload}, errorPayload{std::move(errorPayload)}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(requestPayload, errorPayload);
};

//-------------------------------------------------------------------------

struct PlaceOrderLimitPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<PlaceOrderLimitPayload>;

    OrderDirection direction;
    taosim::decimal_t volume;
    taosim::decimal_t price;
    taosim::decimal_t leverage;
    BookId bookId;
    Currency currency{Currency::BASE};
    std::optional<ClientOrderID> clientOrderId;
    bool postOnly{};
    // Agent-settable. When false, a marketable limit too large to fill entirely
    // within its limit rests whole (all-or-nothing) instead of filling the
    // marketable portion and resting the remainder. Defaults true.
    bool allowPartial{true};
    taosim::TimeInForce timeInForce{taosim::TimeInForce::GTC};
    std::optional<Timestamp> expiryPeriod;
    STPFlag stpFlag{STPFlag::CO};
    SettleFlag settleFlag{SettleType::FIFO};
    std::string delegate;
    std::optional<uint64_t> interfaceOrderId;
    std::optional<taosim::decimal_t> stopLoss;
    std::optional<taosim::decimal_t> takeProfit;
    std::optional<taosim::decimal_t> placeholder;

    PlaceOrderLimitPayload() noexcept = default;

    PlaceOrderLimitPayload(
        OrderDirection direction,
        taosim::decimal_t volume,
        taosim::decimal_t price,
        BookId bookId,
        Currency currency = Currency::BASE,
        std::optional<ClientOrderID> clientOrderId = {},
        bool postOnly = false,
        taosim::TimeInForce timeInForce = taosim::TimeInForce::GTC,
        std::optional<Timestamp> expiryPeriod = {},
        STPFlag stpFlag = STPFlag::CO,
        SettleFlag settleFlag = SettleType::FIFO,
        std::optional<taosim::decimal_t> stopLoss = {},
        std::optional<taosim::decimal_t> takeProfit = {},
        std::optional<taosim::decimal_t> placeholder = {}) noexcept
        : direction{direction},
          volume{volume},
          price{price},
          bookId{bookId},
          currency{currency},
          clientOrderId{clientOrderId},
          postOnly{postOnly},
          timeInForce{timeInForce},
          expiryPeriod{expiryPeriod},
          stpFlag{stpFlag},
          settleFlag{settleFlag},
          stopLoss{stopLoss},
          takeProfit{takeProfit},
          placeholder{placeholder}
    {}

    PlaceOrderLimitPayload(
        OrderDirection direction,
        taosim::decimal_t volume,
        taosim::decimal_t price,
        taosim::decimal_t leverage,
        BookId bookId,
        Currency currency = Currency::BASE,
        std::optional<ClientOrderID> clientOrderId = {},
        bool postOnly = false,
        taosim::TimeInForce timeInForce = taosim::TimeInForce::GTC,
        std::optional<Timestamp> expiryPeriod = {},
        STPFlag stpFlag = STPFlag::CO,
        SettleFlag settleFlag = SettleType::FIFO,
        std::optional<taosim::decimal_t> stopLoss = {},
        std::optional<taosim::decimal_t> takeProfit = {},
        std::optional<taosim::decimal_t> placeholder = {}) noexcept
        : direction{direction},
          volume{volume},
          price{price},
          leverage{leverage},
          bookId{bookId},
          currency{currency},
          clientOrderId{clientOrderId},
          postOnly{postOnly},
          timeInForce{timeInForce},
          expiryPeriod{expiryPeriod},
          stpFlag{stpFlag},
          settleFlag{settleFlag},
          stopLoss{stopLoss},
          takeProfit{takeProfit},
          placeholder{placeholder}
    {}

    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(
        direction,
        volume,
        price,
        leverage,
        bookId,
        currency,
        clientOrderId,
        postOnly,
        allowPartial,
        timeInForce,
        expiryPeriod,
        MSGPACK_NVP("stp", stpFlag),
        settleFlag,
        delegate,
        MSGPACK_NVP("interfaceOrderId", interfaceOrderId),
        MSGPACK_NVP("stopLoss", stopLoss),
        MSGPACK_NVP("takeProfit", takeProfit),
        MSGPACK_NVP("placeholder", placeholder));
};

//-------------------------------------------------------------------------

struct PlaceOrderLimitResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<PlaceOrderLimitResponsePayload>;

    OrderID id;
    PlaceOrderLimitPayload::Ptr requestPayload;

    PlaceOrderLimitResponsePayload() noexcept = default;

    PlaceOrderLimitResponsePayload(OrderID id, PlaceOrderLimitPayload::Ptr requestPayload) noexcept
        : id{id}, requestPayload{requestPayload}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(MSGPACK_NVP("orderId", id), requestPayload);
};

//-------------------------------------------------------------------------

struct PlaceOrderLimitErrorResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<PlaceOrderLimitErrorResponsePayload>;

    PlaceOrderLimitPayload::Ptr requestPayload;
    ErrorResponsePayload::Ptr errorPayload;

    PlaceOrderLimitErrorResponsePayload() noexcept = default;

    PlaceOrderLimitErrorResponsePayload(
        PlaceOrderLimitPayload::Ptr requestPayload, ErrorResponsePayload::Ptr errorPayload) noexcept
        : requestPayload{requestPayload}, errorPayload{errorPayload}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(requestPayload, errorPayload);
};

//-------------------------------------------------------------------------

struct RetrieveOrdersPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<RetrieveOrdersPayload>;

    std::vector<OrderID> ids;
    BookId bookId;

    RetrieveOrdersPayload() noexcept = default;

    RetrieveOrdersPayload(std::vector<OrderID> ids, BookId bookId) noexcept
        : ids{std::move(ids)}, bookId{bookId}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(MSGPACK_NVP("orderIds", ids), bookId);
};

//-------------------------------------------------------------------------

struct RetrieveOrdersResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<RetrieveOrdersResponsePayload>;

    std::vector<LimitOrder> orders;
    BookId bookId;

    RetrieveOrdersResponsePayload() noexcept = default;

    RetrieveOrdersResponsePayload(std::vector<LimitOrder> orders, BookId bookId) noexcept
        : orders{std::move(orders)}, bookId{bookId}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(orders, bookId);
};

//-------------------------------------------------------------------------

struct CancelOrdersPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<CancelOrdersPayload>;

    std::vector<taosim::event::Cancellation> cancellations;
    BookId bookId;

    CancelOrdersPayload() noexcept = default;

    CancelOrdersPayload(
        std::vector<taosim::event::Cancellation> cancellations, BookId bookId) noexcept
        : cancellations{std::move(cancellations)}, bookId{bookId}
    {}

    CancelOrdersPayload(taosim::event::Cancellation cancellation, BookId bookId) noexcept
        : cancellations{cancellation}, bookId{bookId}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(cancellations, bookId);
};

//-------------------------------------------------------------------------

struct CancelOrdersResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<CancelOrdersResponsePayload>;

    std::vector<OrderID> orderIds;
    CancelOrdersPayload::Ptr requestPayload;

    CancelOrdersResponsePayload() noexcept = default;

    CancelOrdersResponsePayload(
        std::vector<OrderID> orderIds,
        CancelOrdersPayload::Ptr requestPayload) noexcept
        : orderIds{std::move(orderIds)}, requestPayload{requestPayload}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(orderIds, requestPayload);
};

//-------------------------------------------------------------------------

struct CancelOrdersErrorResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<CancelOrdersErrorResponsePayload>;

    std::vector<OrderID> orderIds;
    CancelOrdersPayload::Ptr requestPayload;
    ErrorResponsePayload::Ptr errorPayload;

    CancelOrdersErrorResponsePayload() noexcept = default;

    CancelOrdersErrorResponsePayload(
        std::vector<OrderID> orderIds,
        CancelOrdersPayload::Ptr requestPayload,
        ErrorResponsePayload::Ptr errorPayload) noexcept
        : orderIds{std::move(orderIds)}, requestPayload{requestPayload}, errorPayload{errorPayload}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(orderIds, requestPayload, errorPayload);
};

//-------------------------------------------------------------------------

struct ClosePositionsPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<ClosePositionsPayload>;

    std::vector<ClosePosition> closePositions;
    BookId bookId;

    ClosePositionsPayload() = default;

    ClosePositionsPayload(std::vector<ClosePosition> closePositions, BookId bookId) noexcept
        : closePositions{std::move(closePositions)}, bookId{bookId}
    {}

    ClosePositionsPayload(ClosePosition closePositions, BookId bookId) noexcept
        : closePositions{closePositions}, bookId{bookId}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(MSGPACK_NVP("closes", closePositions), bookId);
};

//-------------------------------------------------------------------------

struct ClosePositionsResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<ClosePositionsResponsePayload>;

    std::vector<OrderID> orderIds;
    ClosePositionsPayload::Ptr requestPayload;

    ClosePositionsResponsePayload() noexcept = default;

    ClosePositionsResponsePayload(
        std::vector<OrderID> orderIds,
        ClosePositionsPayload::Ptr requestPayload) noexcept
        : orderIds{std::move(orderIds)}, requestPayload{requestPayload}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(orderIds, requestPayload);
};

//-------------------------------------------------------------------------

struct ClosePositionsErrorResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<ClosePositionsErrorResponsePayload>;

    std::vector<OrderID> orderIds;
    ClosePositionsPayload::Ptr requestPayload;
    ErrorResponsePayload::Ptr errorPayload;

    ClosePositionsErrorResponsePayload() noexcept = default;

    ClosePositionsErrorResponsePayload(
        std::vector<OrderID> orderIds,
        ClosePositionsPayload::Ptr requestPayload,
        ErrorResponsePayload::Ptr errorPayload) noexcept
        : orderIds{std::move(orderIds)}, requestPayload{requestPayload}, errorPayload{errorPayload}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(orderIds, requestPayload, errorPayload);
};

//-------------------------------------------------------------------------

struct RetrieveL2Payload : public MessagePayload
{
    using Ptr = std::shared_ptr<RetrieveL2Payload>;

    size_t depth;
    BookId bookId;

    RetrieveL2Payload() noexcept = default;

    RetrieveL2Payload(size_t depth, BookId bookId) noexcept
        : depth{depth}, bookId{bookId}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(depth, bookId);
};

//-------------------------------------------------------------------------

struct BookLevel
{
    taosim::decimal_t price;
    taosim::decimal_t quantity;

    MSGPACK_DEFINE_MAP(price, quantity);
};

struct RetrieveL2ResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<RetrieveL2ResponsePayload>;

    Timestamp time;
    std::vector<BookLevel> bids;
    std::vector<BookLevel> asks;
    BookId bookId;

    RetrieveL2ResponsePayload() noexcept = default;

    RetrieveL2ResponsePayload(
        Timestamp time,
        std::vector<BookLevel> bids,
        std::vector<BookLevel> asks,
        BookId bookId) noexcept
        : time{time}, bids{std::move(bids)}, asks{std::move(asks)}, bookId{bookId}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(time, bids, asks, bookId);
};

//-------------------------------------------------------------------------

struct RetrieveL1Payload : public MessagePayload
{
    using Ptr = std::shared_ptr<RetrieveL1Payload>;

    BookId bookId;

    RetrieveL1Payload() = default;

    RetrieveL1Payload(BookId bookId) noexcept : bookId{bookId} {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(bookId);
};

//-------------------------------------------------------------------------

struct RetrieveL1ResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<RetrieveL1ResponsePayload>;

    Timestamp time{};
    taosim::decimal_t bestAskPrice{};
    taosim::decimal_t bestAskVolume{};
    taosim::decimal_t askTotalVolume{};
    taosim::decimal_t bestBidPrice{};
    taosim::decimal_t bestBidVolume{};
    taosim::decimal_t bidTotalVolume{};
    BookId bookId;

    RetrieveL1ResponsePayload() noexcept = default;

    RetrieveL1ResponsePayload(Timestamp time, BookId bookId) noexcept
        : time{time}, bookId{bookId}
    {}

    RetrieveL1ResponsePayload(
        Timestamp time,
        taosim::decimal_t bestAskPrice,
        taosim::decimal_t bestAskVolume,
        taosim::decimal_t askTotalVolume,
        taosim::decimal_t bestBidPrice,
        taosim::decimal_t bestBidVolume,
        taosim::decimal_t bidTotalVolume,
        BookId bookId) noexcept
        : time{time},
          bestAskPrice{bestAskPrice},
          bestAskVolume{bestAskVolume},
          askTotalVolume{askTotalVolume},
          bestBidPrice{bestBidPrice},
          bestBidVolume{bestBidVolume},
          bidTotalVolume{bidTotalVolume},
          bookId{bookId}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(
        MSGPACK_NVP("timestamp", time),
        bestAskPrice,
        bestAskVolume,
        askTotalVolume,
        bestBidPrice,
        bestBidVolume,
        bidTotalVolume,
        bookId);
};

//-------------------------------------------------------------------------

struct RetrieveL1ExtPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<RetrieveL1ExtPayload>;

    BookId bookId;

    RetrieveL1ExtPayload() = default;

    RetrieveL1ExtPayload(BookId bookId) noexcept : bookId{bookId} {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(bookId);
};

//-------------------------------------------------------------------------

// L1 plus the book's cumulative trade statistics. A separate message from
// RETRIEVE_L1 rather than an extension of it, so existing pollers and any remote
// client keep their wire format untouched.
struct RetrieveL1ExtResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<RetrieveL1ExtResponsePayload>;

    Timestamp time{};
    taosim::decimal_t bestAskPrice{};
    taosim::decimal_t bestAskVolume{};
    taosim::decimal_t askTotalVolume{};
    taosim::decimal_t bestBidPrice{};
    taosim::decimal_t bestBidVolume{};
    taosim::decimal_t bidTotalVolume{};
    // Monotonic since the run began. Difference two reads for interval statistics;
    // no normalization is applied here, so the reader owns the choice of whether
    // realized variance is expressed per trade, per unit time, or annualized.
    taosim::book::BookTradeStats tradeStats{};
    BookId bookId{};

    RetrieveL1ExtResponsePayload() noexcept = default;

    RetrieveL1ExtResponsePayload(Timestamp time, BookId bookId) noexcept
        : time{time}, bookId{bookId}
    {}

    RetrieveL1ExtResponsePayload(
        Timestamp time,
        taosim::decimal_t bestAskPrice,
        taosim::decimal_t bestAskVolume,
        taosim::decimal_t askTotalVolume,
        taosim::decimal_t bestBidPrice,
        taosim::decimal_t bestBidVolume,
        taosim::decimal_t bidTotalVolume,
        taosim::book::BookTradeStats tradeStats,
        BookId bookId) noexcept
        : time{time},
          bestAskPrice{bestAskPrice},
          bestAskVolume{bestAskVolume},
          askTotalVolume{askTotalVolume},
          bestBidPrice{bestBidPrice},
          bestBidVolume{bestBidVolume},
          bidTotalVolume{bidTotalVolume},
          tradeStats{tradeStats},
          bookId{bookId}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(
        MSGPACK_NVP("timestamp", time),
        bestAskPrice,
        bestAskVolume,
        askTotalVolume,
        bestBidPrice,
        bestBidVolume,
        bidTotalVolume,
        tradeStats,
        bookId);
};

//-------------------------------------------------------------------------

struct SubscribeEventTradeByOrderPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<SubscribeEventTradeByOrderPayload>;

    OrderID id;

    SubscribeEventTradeByOrderPayload() noexcept = default;

    SubscribeEventTradeByOrderPayload(OrderID id) noexcept : id{id} {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(MSGPACK_NVP("orderId", id));
};

//-------------------------------------------------------------------------

struct EventOrderMarketPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<EventOrderMarketPayload>;

    MarketOrder order;
    // Origin of the order. Defaulted for backward compatibility; populated at dispatch
    // so subscribers can tell which book and which agent (incl. remote/miner) placed it.
    BookId bookId{};
    AgentId agentId{};

    EventOrderMarketPayload() noexcept = default;

    EventOrderMarketPayload(const MarketOrder& order) noexcept : order{order} {}

    EventOrderMarketPayload(const MarketOrder& order, BookId bookId, AgentId agentId) noexcept
        : order{order}, bookId{bookId}, agentId{agentId} {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(order, bookId, agentId);
};

//-------------------------------------------------------------------------

struct EventOrderLimitPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<EventOrderLimitPayload>;

    LimitOrder order;
    // Origin of the order. Defaulted for backward compatibility; populated at dispatch
    // so subscribers can tell which book and which agent (incl. remote/miner) placed it.
    BookId bookId{};
    AgentId agentId{};

    EventOrderLimitPayload() noexcept = default;

    EventOrderLimitPayload(const LimitOrder& order) noexcept : order{order} {}

    EventOrderLimitPayload(const LimitOrder& order, BookId bookId, AgentId agentId) noexcept
        : order{order}, bookId{bookId}, agentId{agentId} {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(order, bookId, agentId);
};

//-------------------------------------------------------------------------

struct EventTradePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<EventTradePayload>;

    Trade trade;
    TradeLogContext context;
    BookId bookId;
    std::optional<ClientOrderID> clientOrderId;
    std::string delegate;
    Currency currency{Currency::QUOTE};
    bool isResting{};

    EventTradePayload() noexcept = default;

    EventTradePayload(
        const Trade& trade,
        const TradeLogContext& context,
        BookId bookId,
        std::optional<ClientOrderID> clientOrderId = {},
        std::string delegate = {},
        Currency currency = Currency::QUOTE,
        bool isResting = {}) noexcept
        : trade{trade},
          context{context},
          bookId{bookId},
          clientOrderId{clientOrderId},
          delegate{std::move(delegate)},
          currency{currency},
          isResting{isResting}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(trade, context, bookId, clientOrderId, delegate, currency);
};

//-------------------------------------------------------------------------

struct ResetAgentsPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<ResetAgentsPayload>;

    std::vector<AgentId> agentIds;

    ResetAgentsPayload() noexcept = default;

    ResetAgentsPayload(std::vector<AgentId> agentIds) noexcept
        : agentIds{std::move(agentIds)}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(agentIds);
};

//-------------------------------------------------------------------------

struct ResetAgentsResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<ResetAgentsResponsePayload>;

    std::vector<AgentId> agentIds;
    ResetAgentsPayload::Ptr requestPayload;

    ResetAgentsResponsePayload() noexcept = default;

    ResetAgentsResponsePayload(
        std::vector<AgentId> agentIds,ResetAgentsPayload::Ptr requestPayload) noexcept
        : agentIds{std::move(agentIds)}, requestPayload{requestPayload}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(agentIds, requestPayload);
};

//-------------------------------------------------------------------------

struct ResetAgentsErrorResponsePayload : public MessagePayload
{
    using Ptr = std::shared_ptr<ResetAgentsErrorResponsePayload>;

    std::vector<AgentId> agentIds;
    ResetAgentsPayload::Ptr requestPayload;
    ErrorResponsePayload::Ptr errorPayload;

    ResetAgentsErrorResponsePayload() noexcept = default;

    ResetAgentsErrorResponsePayload(
        std::vector<AgentId> agentIds,
        ResetAgentsPayload::Ptr requestPayload,
        ErrorResponsePayload::Ptr errorPayload) noexcept
        : agentIds{std::move(agentIds)}, requestPayload{requestPayload}, errorPayload{errorPayload}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(agentIds, requestPayload, errorPayload);
};

//-------------------------------------------------------------------------
