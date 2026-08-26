/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/matching/Fees.hpp>
#include "JsonSerializable.hpp"
#include "Order.hpp"
#include "Timestamp.hpp"
#include "common.hpp"
#include <taosim/mp/mp.hpp>
#include "util.hpp"

#include <memory>
#include <optional>

#include <msgpack.hpp>

//-------------------------------------------------------------------------

// 64-bit deliberately. The exchange mints from ONE monotonic counter that is never reset
// and resumes from a persisted high-water mark across restarts, so the width is a hard
// ceiling on how many trades can ever be identified: at uint32_t that ceiling is
// 4,294,967,295, past which the counter wraps and reissues an id an earlier trade already
// used. Two unrelated trades sharing an id is precisely the cross-surface collision the
// canonical <netuid>:<seq> identity exists to prevent.
//
// It also removes a width mismatch that was already live: the settlement structs declare
// tradeId as optional<uint64_t>, so a large id decoded into a fill but threw std::bad_cast
// as a staged-correction map KEY, discarding that whole block of the reconciliation.
//
// No migration is needed. msgpack encodes integers by VALUE, not by declared type, so the
// checkpoint's packed counter (checkpoint/serialization/book/Book.hpp) and the correction
// map keys are byte-identical for any value below 2^32, in both directions; the high-water
// file is plain text.
using TradeID = uint64_t;

// THE single place a trade id is assigned. Both fills that the books matched
// (Book::logTrade) and fills settled straight off the pool reserves
// (MultiBookExchangeAgent::mintPoolTradeId) call through here, so there is one counter, one
// increment, and one thing to reason about when asking whether an id can repeat.
//
// It is assigned when the fill is CREATED, not when it settles, because the id is the
// correlation key: it travels out with the settlement request so the validator can stage a correction
// against it, key the on-chain settlement to it, and hand the same id back on the result.
// Deferring assignment to settlement would leave that whole chain with nothing to key by.
[[nodiscard]] inline std::optional<TradeID> assignTradeId(
    const std::shared_ptr<TradeID>& counter) noexcept
{
    // No shared counter means the books hold private sequences, where an id would collide
    // across books; callers must treat that as "no id" rather than invent one.
    if (!counter) return std::nullopt;
    return (*counter)++;
}

//-------------------------------------------------------------------------

struct Trade : public JsonSerializable
{
    using Ptr = std::shared_ptr<Trade>;

    Trade() noexcept = default;

    Trade(
        TradeID id,
        Timestamp timestamp,
        OrderDirection direction,
        OrderID aggressingOrderID,
        OrderID restingOrderID,
        taosim::decimal_t volume,
        taosim::decimal_t price) noexcept;

    [[nodiscard]] TradeID id() const noexcept { return m_id; }
    [[nodiscard]] OrderDirection direction() const noexcept { return m_direction; }
    [[nodiscard]] Timestamp timestamp() const noexcept { return m_timestamp; }
    [[nodiscard]] OrderID aggressingOrderID() const noexcept { return m_aggressingOrderID; }
    [[nodiscard]] OrderID restingOrderID() const noexcept { return m_restingOrderID; }
    [[nodiscard]] taosim::decimal_t volume() const noexcept { return m_volume; }
    [[nodiscard]] taosim::decimal_t price() const noexcept { return m_price; }

    void setTimestamp(Timestamp timestamp) noexcept { m_timestamp = timestamp; }

    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    template<typename... Args>
    requires std::constructible_from<Trade, Args...> && taosim::mp::IsPointer<typename Trade::Ptr>
    [[nodiscard]] static Ptr create(Args&&... args) noexcept
    {
        return Trade::Ptr{new Trade(std::forward<Args>(args)...)};
    }

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    TradeID m_id;
    Timestamp m_timestamp;
    OrderDirection m_direction;
    OrderID m_aggressingOrderID;
    OrderID m_restingOrderID;
    taosim::decimal_t m_volume;
    taosim::decimal_t m_price;

    MSGPACK_DEFINE_MAP(
        MSGPACK_NVP("tradeId", m_id),
        MSGPACK_NVP("direction", m_direction),
        MSGPACK_NVP("timestamp", m_timestamp),
        MSGPACK_NVP("aggressingOrderId", m_aggressingOrderID),
        MSGPACK_NVP("restingOrderId", m_restingOrderID),
        MSGPACK_NVP("volume", m_volume),
        MSGPACK_NVP("price", m_price));
};

//-------------------------------------------------------------------------

struct TradeContext : public JsonSerializable
{
    BookId bookId;
    AgentId aggressingAgentId;
    AgentId restingAgentId;
    taosim::matching::Fees fees;
    // SL/TP close metadata — 0/0 for regular orders.
    uint8_t aggressingCloseReason{0};   // 0=none, 1=SL, 2=TP
    OrderID aggressingOriginatingOrderId{0};
    // Who caused the aggressing order to exist, when that is not its owner. Set only for sweep orders,
    // which the exchange places under its own id after a miner's instruction moved the pool. Used for
    // attribution and the self-trade guard; fees and settlement follow aggressingAgentId, not this.
    std::optional<AgentId> initiatorAgentId;

    TradeContext() = default;

    TradeContext(
        BookId bookId,
        AgentId aggressingAgentId,
        AgentId restingAgentId,
        taosim::matching::Fees fees) noexcept
        : bookId{bookId},
          aggressingAgentId{aggressingAgentId},
          restingAgentId{restingAgentId},
          fees{fees}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static TradeContext fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(bookId, aggressingAgentId, restingAgentId, fees);
};

//-------------------------------------------------------------------------

struct TradeLogContext : public JsonSerializable
{
    using Ptr = std::shared_ptr<TradeLogContext>;

    AgentId aggressingAgentId;
    AgentId restingAgentId;
    BookId bookId;
    taosim::matching::Fees fees;
    // SL/TP close metadata — 0/0 for regular orders.
    uint8_t aggressingCloseReason{0};   // 0=none, 1=SL, 2=TP
    OrderID aggressingOriginatingOrderId{0};
    // Who caused the aggressing order, when that is not its owner. Set only on sweep orders, which the
    // exchange places under its own id after a miner's instruction moved the pool. Read when recording
    // the counterparty on the resting side's fill; fees and settlement follow aggressingAgentId.
    std::optional<AgentId> initiatorAgentId;

    TradeLogContext() noexcept = default;

    TradeLogContext(
        AgentId aggressingAgentId,
        AgentId restingAgentId,
        BookId bookId,
        taosim::matching::Fees fees) noexcept
        : aggressingAgentId{aggressingAgentId},
          restingAgentId{restingAgentId},
          bookId{bookId},
          fees{fees}
    {}

    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    template<typename... Args>
    requires std::constructible_from<TradeLogContext, Args...>
        && taosim::mp::IsPointer<typename TradeLogContext::Ptr>
    [[nodiscard]] static Ptr create(Args&&... args) noexcept
    {
        return TradeLogContext::Ptr{new TradeLogContext(std::forward<Args>(args)...)};
    }

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(bookId, aggressingAgentId, restingAgentId, fees);
};

//-------------------------------------------------------------------------

struct TradeWithLogContext : public JsonSerializable
{
    using Ptr = std::shared_ptr<TradeWithLogContext>;

    Trade::Ptr trade;
    TradeLogContext::Ptr logContext;

    TradeWithLogContext() noexcept = default;

    TradeWithLogContext(Trade::Ptr trade, TradeLogContext::Ptr logContext) noexcept
        : trade{trade}, logContext{logContext}
    {}

    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    template<typename... Args>
    requires std::constructible_from<TradeWithLogContext, Args...>
        && taosim::mp::IsPointer<typename Trade::Ptr>
    [[nodiscard]] static Ptr create(Args&&... args) noexcept
    {
        return TradeWithLogContext::Ptr{new TradeWithLogContext(std::forward<Args>(args)...)};
    }

    [[nodiscard]] static Ptr fromJson(const rapidjson::Value& json);

    MSGPACK_DEFINE_MAP(trade, logContext);
};

//-------------------------------------------------------------------------
