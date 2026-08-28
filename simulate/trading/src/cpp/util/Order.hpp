/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/decimal/serialization/decimal.hpp>
#include "CheckpointSerializable.hpp"
#include "JsonSerializable.hpp"
#include "common.hpp"
#include "Flags.hpp"

#include <msgpack.hpp>

//-------------------------------------------------------------------------

using ClientOrderID = std::decay_t<OrderID>;
using taosim::STPFlag;
using taosim::SettleFlag;
using taosim::SettleType;

enum class OrderDirection : uint32_t
{
    BUY,
    SELL
};

MSGPACK_ADD_ENUM(OrderDirection);

enum class Currency : uint32_t
{
    BASE,
    QUOTE    
};

MSGPACK_ADD_ENUM(Currency);

[[nodiscard]] constexpr std::string_view OrderDirection2StrView(OrderDirection dir) noexcept
{
    return magic_enum::enum_name(dir);
}

template<>
struct fmt::formatter<OrderDirection>
{
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(OrderDirection dir, FormatContext& ctx) const
    {
        return fmt::format_to(ctx.out(), "{}", OrderDirection2StrView(dir));
    }
};

enum class OrderErrorCode : uint32_t
{
    VALID,
    NONEXISTENT_ACCOUNT,
    INSUFFICIENT_BASE,
    INSUFFICIENT_QUOTE,
    EMPTY_BOOK,
    PRICE_INCREMENT_VIOLATED,
    VOLUME_INCREMENT_VIOLATED,
    EXCEEDING_LOAN,
    CONTRACT_VIOLATION,
    INVALID_LEVERAGE,
    INVALID_VOLUME,
    INVALID_PRICE,
    EXCEEDING_MAX_ORDERS,
    DUAL_POSITION,
    MINIMUM_ORDER_SIZE_VIOLATION
};

[[nodiscard]] constexpr std::string_view OrderErrorCode2StrView(OrderErrorCode ec) noexcept
{
    return magic_enum::enum_name(ec);
}

template<>
struct fmt::formatter<OrderErrorCode>
{
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(OrderErrorCode ec, FormatContext& ctx) const
    {
        return fmt::format_to(ctx.out(), "{}", OrderErrorCode2StrView(ec));
    }
};

//-------------------------------------------------------------------------

struct BasicOrder : public JsonSerializable
{
    BasicOrder() noexcept = default;

    BasicOrder(
        OrderID id,
        Timestamp timestamp,
        taosim::decimal_t volume,
        taosim::decimal_t leverage = 0_dec) noexcept;

    virtual ~BasicOrder() noexcept = default;

    [[nodiscard]] OrderID id() const noexcept { return m_id; }
    [[nodiscard]] Timestamp timestamp() const noexcept { return m_timestamp; }
    [[nodiscard]] taosim::decimal_t volume() const noexcept { return m_volume; }
    [[nodiscard]] taosim::decimal_t totalVolume() const noexcept { return m_volume * taosim::util::dec1p(m_leverage); }
    [[nodiscard]] taosim::decimal_t leverage() const noexcept { return m_leverage; }
    
    void removeVolume(taosim::decimal_t decrease);
    void removeLeveragedVolume(taosim::decimal_t decrease);
    void setVolume(taosim::decimal_t newVolume);
    void setLeverage(taosim::decimal_t newLeverage);

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    OrderID m_id;
    Timestamp m_timestamp;
    taosim::decimal_t m_volume;
    taosim::decimal_t m_leverage{};

    MSGPACK_DEFINE_MAP(
        MSGPACK_NVP("orderId", m_id),
        MSGPACK_NVP("timestamp", m_timestamp),
        MSGPACK_NVP("volume", m_volume),
        MSGPACK_NVP("leverage", m_leverage));
};

//-------------------------------------------------------------------------

struct Order : public BasicOrder
{
    using Ptr = std::shared_ptr<Order>;

    Order() noexcept = default;

    Order(
        OrderID orderId,
        Timestamp timestamp,
        taosim::decimal_t volume,
        OrderDirection direction,
        taosim::decimal_t leverage = 0_dec,
        STPFlag stpFlag = STPFlag::CO,
        SettleFlag settleFlag = SettleType::FIFO,
        Currency currency = Currency::BASE,
        std::optional<taosim::decimal_t> stopLoss = std::nullopt,
        std::optional<taosim::decimal_t> takeProfit = std::nullopt,
        std::optional<taosim::decimal_t> placeholder = std::nullopt) noexcept;

    [[nodiscard]] OrderDirection direction() const noexcept { return m_direction; }
    [[nodiscard]] STPFlag stpFlag() const noexcept { return m_stpFlag; }
    [[nodiscard]] SettleFlag settleFlag() const noexcept { return m_settleFlag; }
    [[nodiscard]] Currency currency() const noexcept { return m_currency; }
    [[nodiscard]] std::optional<taosim::decimal_t> stopLoss() const noexcept { return m_stopLoss; }
    [[nodiscard]] std::optional<taosim::decimal_t> takeProfit() const noexcept { return m_takeProfit; }
    [[nodiscard]] std::optional<taosim::decimal_t> placeholder() const noexcept { return m_placeholder; }
    [[nodiscard]] bool hasSLTP() const noexcept { return m_stopLoss || m_takeProfit; }

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    OrderDirection m_direction;
    STPFlag m_stpFlag{STPFlag::CO};
    SettleFlag m_settleFlag{SettleType::FIFO};
    Currency m_currency{Currency::BASE};
    std::optional<taosim::decimal_t> m_stopLoss;
    std::optional<taosim::decimal_t> m_takeProfit;
    std::optional<taosim::decimal_t> m_placeholder;

    MSGPACK_DEFINE_MAP(
        MSGPACK_NVP("orderId", m_id),
        MSGPACK_NVP("timestamp", m_timestamp),
        MSGPACK_NVP("volume", m_volume),
        MSGPACK_NVP("leverage", m_leverage),
        MSGPACK_NVP("direction", m_direction),
        MSGPACK_NVP("stpFlag", m_stpFlag),
        MSGPACK_NVP("settleFlag", m_settleFlag),
        MSGPACK_NVP("currency", m_currency),
        MSGPACK_NVP("stopLoss", m_stopLoss),
        MSGPACK_NVP("takeProfit", m_takeProfit),
        MSGPACK_NVP("placeholder", m_placeholder));
};

//-------------------------------------------------------------------------

struct MarketOrder : public Order
{
    using Ptr = std::shared_ptr<MarketOrder>;

    taosim::decimal_t m_maxSlippage{0_dec};  // 0 = no price limit

    MarketOrder() noexcept = default;

    MarketOrder(
        OrderID orderId,
        Timestamp timestamp,
        taosim::decimal_t volume,
        OrderDirection direction,
        taosim::decimal_t leverage = 0_dec,
        STPFlag stpFlag = STPFlag::CO,
        SettleFlag settleFlag = SettleType::FIFO,
        Currency currency = Currency::BASE,
        taosim::decimal_t maxSlippage = 0_dec,
        std::optional<taosim::decimal_t> stopLoss = std::nullopt,
        std::optional<taosim::decimal_t> takeProfit = std::nullopt,
        std::optional<taosim::decimal_t> placeholder = std::nullopt) noexcept;

    [[nodiscard]] taosim::decimal_t maxSlippage() const noexcept { return m_maxSlippage; }

    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(
        MSGPACK_NVP("orderId", m_id),
        MSGPACK_NVP("timestamp", m_timestamp),
        MSGPACK_NVP("volume", m_volume),
        MSGPACK_NVP("leverage", m_leverage),
        MSGPACK_NVP("direction", m_direction),
        MSGPACK_NVP("stpFlag", m_stpFlag),
        MSGPACK_NVP("settleFlag", m_settleFlag),
        MSGPACK_NVP("currency", m_currency),
        MSGPACK_NVP("stopLoss", m_stopLoss),
        MSGPACK_NVP("takeProfit", m_takeProfit),
        MSGPACK_NVP("placeholder", m_placeholder));
};

//-------------------------------------------------------------------------

struct LimitOrder : public Order
{
    using Ptr = std::shared_ptr<LimitOrder>;

    LimitOrder() noexcept = default;

    LimitOrder(
        OrderID orderId,
        Timestamp timestamp,
        taosim::decimal_t volume,
        OrderDirection direction,
        taosim::decimal_t price,
        taosim::decimal_t leverage = 0_dec,
        STPFlag stpFlag = STPFlag::CO,
        SettleFlag settleFlag = SettleType::FIFO,
        bool postOnly = false,
        taosim::TimeInForce timeInForce = taosim::TimeInForce::GTC,
        std::optional<Timestamp> expiryPeriod = std::nullopt,
        Currency currency = Currency::BASE,
        std::optional<taosim::decimal_t> stopLoss = std::nullopt,
        std::optional<taosim::decimal_t> takeProfit = std::nullopt,
        std::optional<taosim::decimal_t> placeholder = std::nullopt) noexcept;

    [[nodiscard]] taosim::decimal_t price() const noexcept { return m_price; };
    [[nodiscard]] bool postOnly() const noexcept { return m_postOnly; }
    [[nodiscard]] taosim::TimeInForce timeInForce() const noexcept { return m_timeInForce; }
    [[nodiscard]] std::optional<Timestamp> expiryPeriod() const noexcept { return m_expiryPeriod; }

    void setPrice(taosim::decimal_t newPrice);

    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json, int priceDecimals, int volumeDecimals);

    taosim::decimal_t m_price;
    bool m_postOnly{};
    taosim::TimeInForce m_timeInForce{taosim::TimeInForce::GTC};
    std::optional<Timestamp> m_expiryPeriod;

    MSGPACK_DEFINE_MAP(
        MSGPACK_NVP("orderId", m_id),
        MSGPACK_NVP("timestamp", m_timestamp),
        MSGPACK_NVP("volume", m_volume),
        MSGPACK_NVP("leverage", m_leverage),
        MSGPACK_NVP("direction", m_direction),
        MSGPACK_NVP("stpFlag", m_stpFlag),
        MSGPACK_NVP("settleFlag", m_settleFlag),
        MSGPACK_NVP("currency", m_currency),
        MSGPACK_NVP("price", m_price),
        MSGPACK_NVP("postOnly", m_postOnly),
        MSGPACK_NVP("timeInForce", m_timeInForce),
        MSGPACK_NVP("expiryPeriod", m_expiryPeriod),
        MSGPACK_NVP("stopLoss", m_stopLoss),
        MSGPACK_NVP("takeProfit", m_takeProfit),
        MSGPACK_NVP("placeholder", m_placeholder));
};

//-------------------------------------------------------------------------

struct OrderClientContext
{
    AgentId agentId;
    std::optional<ClientOrderID> clientOrderId;
    std::string delegate;
    Currency currency{Currency::QUOTE};
    // SL/TP close metadata — populated only for exchange-triggered market closes.
    uint8_t closeReason{0};       // 0=none, 1=SL, 2=TP
    OrderID originatingOrderId{0}; // LOB ID of the position order that spawned the SL/TP
    // WHO CAUSED this order to exist, when that differs from who owns it.
    //
    // Set only on sweep orders, which the exchange places under its own id (agent -1) to reconcile the
    // book with the pool after a miner's instruction moved the price. `agentId` stays -1 because the
    // exchange funds the sweep and the initiating miner never receives its volume, so fees and
    // settlement must not follow this field. It exists so the two places that need to know WHO acted,
    // rather than who paid, can ask: the self-trade guard, and the counterparty recorded on the fill.
    // Empty means the order is its own initiator, which is every ordinary order.
    std::optional<AgentId> initiatorAgentId;

    // WHICH INSTRUCTION created this order, so "was already resting" can be asked exactly.
    //
    // Two orders belong to one instruction precisely when the exchange minted them while dispatching
    // that instruction: a marketable order and the sweep raised to fill it. That pair must be exempt
    // from self-trade prevention, because cancelling the resting side destroys the order the sweep
    // exists to fill. Everything else, including two orders a miner sends in the SAME batch, is a
    // separate instruction and must remain subject to STP.
    //
    // Empty means "not minted under instruction dispatch", which covers every simulation-mode order.
    // The comparison therefore FAILS CLOSED: an empty seq never compares equal, so the guard falls
    // back to order-id ordering and STP protects rather than stands down. A plain integer defaulting
    // to 0 would do the opposite -- every unstamped order would look like one instruction and STP
    // would silently never fire anywhere.
    std::optional<uint64_t> instrSeq;

    OrderClientContext() noexcept = default;

    OrderClientContext(
        AgentId agentId,
        std::optional<ClientOrderID> clientOrderId = {},
        std::string delegate = {},
        Currency currency = Currency::QUOTE) noexcept
        : agentId{agentId},
          clientOrderId{clientOrderId},
          delegate{std::move(delegate)},
          currency{currency}
    {}

    [[nodiscard]] static OrderClientContext fromJson(const rapidjson::Value& json);

    // EVERY FIELD, not the original four. This map omitted closeReason, originatingOrderId,
    // initiatorAgentId and instrSeq, so any context crossing msgpack would silently lose them.
    //
    // It is unreachable today -- nothing packs an OrderClientContext: no Order map includes the context,
    // SLTPContainer holds three but declares no MSGPACK map at all, and there are no direct pack sites.
    // That is exactly why it is fixed NOW: no data exists in the old encoding, so there is no migration
    // question, and it becomes a wire-compatibility problem the moment anything packs a context, which is
    // what checkpointing the SL/TP trigger registry would do.
    //
    // The two optionals do not fail the same way, which is why the omission mattered. An empty instrSeq
    // fails CLOSED by design: it never compares equal, so the guard falls back to order-id ordering and
    // STP protects. An empty initiatorAgentId means "the order is its own initiator", so a sweep that
    // lost the field is indistinguishable from an ordinary order and both the self-trade guard and the
    // counterparty recorded on the fill change behaviour silently.
    MSGPACK_DEFINE_MAP(
        agentId, clientOrderId, delegate, currency,
        closeReason, originatingOrderId, initiatorAgentId, instrSeq);
};

//-------------------------------------------------------------------------

struct OrderContext : public JsonSerializable
{
    AgentId agentId;
    BookId bookId;
    std::optional<ClientOrderID> clientOrderId;

    OrderContext() = default;

    OrderContext(
        AgentId agentId, BookId bookId, std::optional<ClientOrderID> clientOrderId = {}) noexcept
        : agentId{agentId}, bookId{bookId}, clientOrderId{clientOrderId}
    {}

    void jsonSerialize(rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static OrderContext fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(agentId, bookId, clientOrderId);
};

//-------------------------------------------------------------------------

struct OrderLogContext : public JsonSerializable
{
    using Ptr = std::shared_ptr<OrderLogContext>;

    AgentId agentId;
    BookId bookId;

    OrderLogContext() noexcept = default;

    OrderLogContext(AgentId agentId, BookId bookId) noexcept
        : agentId{agentId}, bookId{bookId}
    {}

    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    void jsonSerialize(rapidjson::Document& json, const std::string& key = {}) const override;

    MSGPACK_DEFINE_MAP(agentId, bookId);
};

//-------------------------------------------------------------------------

struct OrderWithLogContext : public JsonSerializable
{
    using Ptr = std::shared_ptr<OrderWithLogContext>;

    Order::Ptr order;
    OrderLogContext::Ptr logContext;

    OrderWithLogContext(Order::Ptr order, OrderLogContext::Ptr logContext) noexcept
        : order{order}, logContext{logContext}
    {}

    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    void jsonSerialize(rapidjson::Document& json, const std::string& key = {}) const override;

    MSGPACK_DEFINE_MAP(order, logContext);
};

//-------------------------------------------------------------------------
