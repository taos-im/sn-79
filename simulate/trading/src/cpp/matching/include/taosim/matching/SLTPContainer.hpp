/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "Order.hpp"
#include "common.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <array>
#include <functional>
#include <list>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>
#include <utility>
#include <vector>

//-------------------------------------------------------------------------

namespace taosim::matching
{

//-------------------------------------------------------------------------
// The direction of price movement that fires an SL/TP trigger.

enum class SLTPCross : uint8_t
{
    AT_OR_ABOVE,  // fires when the latest price crosses up through triggerPrice
    AT_OR_BELOW   // fires when the latest price crosses down through triggerPrice
};

//-------------------------------------------------------------------------
// Captured-at-creation context for an SL/TP-bearing order. Kept in a
// per-book side store so that subsequent fills can derive trigger entries
// even if the originating order itself (e.g. an immediately-traded market
// order, or a fully-filled limit order) is no longer reachable through
// the book. Decremented as fills come in and erased when the remaining
// volume reaches zero.

struct SLTPOrderInfo
{
    AgentId agentId{};
    BookId bookId{};
    OrderClientContext clientCtx;
    OrderDirection originatingSide{OrderDirection::BUY};
    decimal_t leverage;
    Currency currency{Currency::BASE};
    decimal_t remainingVolume;
    std::optional<decimal_t> stopLoss;
    std::optional<decimal_t> takeProfit;
    std::optional<decimal_t> placeholder;
};

//-------------------------------------------------------------------------
// One pending stop-loss / take-profit trigger. Held in an ordered
// container keyed by the trigger price until the latest market price
// crosses it.

struct SLTPEntry
{
    OrderID originatingOrderId{};
    OrderClientContext clientCtx;
    AgentId agentId{};
    BookId bookId{};
    OrderDirection closingSide{OrderDirection::BUY};
    decimal_t volume;
    decimal_t leverage;
    Currency currency{Currency::BASE};
    decimal_t triggerPrice;
    SLTPCross cross{SLTPCross::AT_OR_BELOW};
    // Extra per-trigger context, retained because orderInfo is purged once
    // the originating order is fully filled.
    decimal_t fillPrice;
    decimal_t placeholder;
    decimal_t baseSl;
    decimal_t baseTp;
    bool isSL{false};
    // Set once a price band has cut this close and its owner was told, so an armed close fired again is
    // not told again. Not checkpointed.
    bool bandTold{false};
};

//-------------------------------------------------------------------------
// One coverage slot in an agent's per-book FIFO. Each slot represents the
// still-pending coverage from a fill that opened (or grew) a position.
// Its volume shrinks as counter-trades cover it; when volume hits zero
// both underlying triggers are erased from the multimap. slIter / tpIter
// are `triggers.end()` when absent (e.g. the order only had SL, or the
// trigger already fired — see onPriceUpdate, which leaves the slot alive
// with end() iterators so the closing market order's counter-trade still
// finds the slot to drain).

struct Slot
{
    std::multimap<decimal_t, SLTPEntry>::iterator slIter;
    std::multimap<decimal_t, SLTPEntry>::iterator tpIter;
    decimal_t volume;
};

//-------------------------------------------------------------------------
// Per-book ordered triggers, the side store of in-flight order info, and
// the most recent price observed for the book. We keep every trigger —
// both above- and below-direction crossings — in one multimap keyed by
// trigger price; the per-entry `cross` field tells them apart. By
// tracking lastPrice we only ever need to look at the contiguous range
// of entries with keys in [min(prev, new), max(prev, new)] on each price
// update — two log-time member lookups (multimap::lower_bound /
// upper_bound). ranges::lower_bound on bidirectional iterators
// silently degrades to a linear scan, so the member calls are
// deliberate.
//
// openSlots is the per-agent, per-closingSide FIFO of live coverage
// slots. Index 0 = BUY-closing (covers shorts), 1 = SELL-closing (covers
// longs). A trade on side S shrinks the agent's FIFO at openSlots[agent][S];
// the remainder (if any) opens new coverage in openSlots[agent][opposite(S)].

struct SLTPPerBook
{
    std::multimap<decimal_t, SLTPEntry> triggers;
    std::unordered_map<OrderID, SLTPOrderInfo> orderInfo;
    std::optional<decimal_t> lastPrice;
    std::unordered_map<AgentId, std::array<std::list<Slot>, 2>> openSlots;
    // Running total across all (agent, side) FIFOs in this book. Lets the
    // hot paths short-circuit when the book has no SL/TP activity without
    // having to enumerate `openSlots`.
    size_t activeSlotCount{};
    // Closes a price band cut, holding what they did not close. Not checkpointed: a restart drops them
    // as it drops other in-flight state.
    std::vector<SLTPEntry> armed;
};

//-------------------------------------------------------------------------
// Owns the SL/TP trigger inventory for an exchange agent, listens to a
// per-book price feed, and emits closing market orders via a pluggable
// dispatch callback. The container itself stays agnostic to how the
// feed and the dispatch are wired up — callers configure both via
// `priceFeed(bookId)` and `setDispatch(...)`.

class SLTPContainer
{
public:
    using DispatchFn = std::function<void(const SLTPEntry&, decimal_t /*observedPrice*/)>;

    struct OnOrderCreatedContext
    {
        OrderID orderId{};
        BookId bookId{};
        AgentId agentId{};
        OrderClientContext clientCtx;
        OrderDirection originatingSide{OrderDirection::BUY};
        decimal_t volume;
        decimal_t leverage;
        Currency currency{Currency::BASE};
        std::optional<decimal_t> stopLoss;
        std::optional<decimal_t> takeProfit;
        // Optional per-order parameter carried through to the trigger.
        std::optional<decimal_t> placeholder;
    };

    struct OnOrderTradeContext
    {
        BookId bookId{};
        OrderID originatingOrderId{};
        // The agent's side of this particular fill and the agent itself.
        // Both are needed even for unflagged orders so the covering step
        // can walk the agent's FIFO of live triggers and shrink them as
        // the counter-trade consumes prior coverage.
        AgentId agentId{};
        OrderDirection side{OrderDirection::BUY};
        decimal_t fillPrice;
        decimal_t filledVolume;
    };

    void resize(size_t bookCount);

    [[nodiscard]] size_t bookCount() const noexcept { return m_books.size(); }

    // Register an order carrying SL/TP information at creation time.
    // Both resting (limit) and aggressing (market or marketable limit)
    // orders should pass through here so subsequent fills can derive
    // trigger entries even if the originating order is gone by then.
    // No-op when neither stopLoss nor takeProfit is set.
    void onOrderCreated(const OnOrderCreatedContext& ctx);

    // Insert SL/TP triggers for a single fill of a previously-registered
    // order. Trigger prices are computed as ctx.fillPrice * proportion;
    // the side of the closing market order is opposite the originating
    // side. The fill is debited from the order's remaining volume; once
    // zero, the side-store entry is purged.
    void onOrderTrade(const OnOrderTradeContext& ctx);

    // Remove all triggers and the order-info entry for an order whose
    // position has been forcibly closed (e.g. margin call).
    void removeOrder(BookId bookId, OrderID orderId);

    // Debit the order's `orderInfo.remainingVolume` by the canceled
    // amount and purge the side-store entry if the residual reaches
    // zero. Triggers and slots from prior fills are intentionally left
    // alone — those represent positions that already exist and must
    // still close on a price crossing. No-op for orderIds the container
    // doesn't track (unflagged orders); safe to call unconditionally
    // from the agent's cancel signal hook.
    void onOrderCanceled(BookId bookId, OrderID orderId, decimal_t volumeCanceled);

    // Drain all entries that the price transition `lastPrice -> price`
    // crossed and dispatch a closing market order for each.
    void onPriceUpdate(BookId bookId, decimal_t price);

    // AN SL/TP CLOSE A PRICE BAND CUT STAYS ARMED. `entry` carries what is left of the close; it fires
    // again from fireArmed, as a margin call is placed again each step while the price stays past it.
    void rearm(SLTPEntry entry);

    // Dispatch every armed close on the book whose trigger `price` is still past, each at most once per call: a close
    // the band cuts again re-arms for the next call. A close whose price has come back inside stays armed.
    void fireArmed(BookId bookId, decimal_t price);

    // Replace the dispatch callback used to spawn closing market orders.
    // Default-constructed containers have no dispatch; calling
    // onPriceUpdate before setting one is a no-op.
    void setDispatch(DispatchFn fn) noexcept { m_dispatch = std::move(fn); }

    [[nodiscard]] bool hasDispatch() const noexcept { return static_cast<bool>(m_dispatch); }

    // Mutable accessor for the per-book scoped_connection holding the
    // active price feed; assigning a new connection here automatically
    // tears down the previous one.
    [[nodiscard]] auto&& priceFeed(this auto&& self, BookId bookId) noexcept
    {
        return self.m_priceFeeds.at(bookId);
    }

    [[nodiscard]] auto&& books(this auto&& self) noexcept { return self.m_books; }

private:
    std::vector<SLTPPerBook> m_books;
    std::vector<bs2::scoped_connection> m_priceFeeds;
    DispatchFn m_dispatch;
};

//-------------------------------------------------------------------------

}  // namespace taosim::matching

//-------------------------------------------------------------------------

template<>
struct fmt::formatter<taosim::matching::SLTPEntry>
{
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const taosim::matching::SLTPEntry& e, FormatContext& ctx) const
    {
        return fmt::format_to(ctx.out(),
            "SLTPEntry{{order#{}, agent#{}, book#{}, leg={}, closingSide={}, "
            "triggerPrice={}, vol={}, cross={}}}",
            e.originatingOrderId, e.agentId, e.bookId,
            e.isSL ? "SL" : "TP",
            e.closingSide,
            e.triggerPrice, e.volume,
            e.cross == taosim::matching::SLTPCross::AT_OR_ABOVE ? "AT_OR_ABOVE" : "AT_OR_BELOW");
    }
};

//-------------------------------------------------------------------------

template<>
struct fmt::formatter<taosim::matching::Slot>
{
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const taosim::matching::Slot& s, FormatContext& ctx) const
    {
        auto fmtIter = [](const auto& it, const auto& triggers) {
            if (it == triggers.end()) return std::string{"-"};
            return fmt::format("@{}", it->first);
        };
        return fmt::format_to(ctx.out(),
            "Slot{{vol={}, sl={}, tp={}}}",
            s.volume,
            s.slIter == s.tpIter || s.slIter == decltype(s.slIter){}
                ? std::string{"-"} : fmt::format("@{}", s.slIter->first),
            s.tpIter == decltype(s.tpIter){}
                ? std::string{"-"} : fmt::format("@{}", s.tpIter->first));
    }
};

//-------------------------------------------------------------------------

template<>
struct fmt::formatter<taosim::matching::SLTPPerBook>
{
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const taosim::matching::SLTPPerBook& b, FormatContext& ctx) const
    {
        auto out = fmt::format_to(ctx.out(),
            "PerBook{{lastPrice={}, activeSlots={}, triggers=[",
            b.lastPrice ? fmt::format("{}", *b.lastPrice) : std::string{"none"},
            b.activeSlotCount);
        bool first = true;
        for (const auto& [price, entry] : b.triggers) {
            out = fmt::format_to(out, "{}{}", first ? "" : ", ", entry);
            first = false;
        }
        out = fmt::format_to(out, "], orderInfo=[");
        first = true;
        for (const auto& [oid, info] : b.orderInfo) {
            out = fmt::format_to(out,
                "{}order#{}(agent#{}, side={}, vol={}, sl={}, tp={})",
                first ? "" : ", ",
                oid, info.agentId, info.originatingSide, info.remainingVolume,
                info.stopLoss ? fmt::format("{}", *info.stopLoss) : std::string{"-"},
                info.takeProfit ? fmt::format("{}", *info.takeProfit) : std::string{"-"});
            first = false;
        }
        out = fmt::format_to(out, "], openSlots=[");
        first = true;
        for (const auto& [agentId, sides] : b.openSlots) {
            for (size_t side = 0; side < sides.size(); ++side) {
                if (sides[side].empty()) continue;
                out = fmt::format_to(out,
                    "{}agent#{}/{}=[{}]",
                    first ? "" : ", ",
                    agentId,
                    side == 0 ? "BUY-close" : "SELL-close",
                    fmt::join(sides[side], ", "));
                first = false;
            }
        }
        return fmt::format_to(out, "]}}");
    }
};

//-------------------------------------------------------------------------

template<>
struct fmt::formatter<taosim::matching::SLTPContainer>
{
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const taosim::matching::SLTPContainer& c, FormatContext& ctx) const
    {
        auto out = fmt::format_to(ctx.out(), "SLTPContainer{{");
        for (size_t i = 0; i < c.books().size(); ++i) {
            out = fmt::format_to(out, "{}book#{}: {}", i == 0 ? "" : ", ", i, c.books()[i]);
        }
        return fmt::format_to(out, "}}");
    }
};

//-------------------------------------------------------------------------
