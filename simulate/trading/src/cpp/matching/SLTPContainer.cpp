/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/matching/SLTPContainer.hpp>

#include "util.hpp"
#include <taosim/util/SLTPDebug.hpp>

#include <range/v3/algorithm/find_if.hpp>

#include <algorithm>
#include <utility>

//-------------------------------------------------------------------------

namespace taosim::matching
{

//-------------------------------------------------------------------------
// Drop the outer `openSlots` entry for an agent whose per-side FIFOs
// are both empty. Otherwise the map slowly accumulates one residual
// `[empty, empty]` pair per agent that ever opened a slot, which the
// hot-path `find()` then has to walk past on every trade. Cheap to call
// at the end of any FIFO mutation.

namespace
{

void pruneAgentIfEmpty(SLTPPerBook& perBook, AgentId agentId)
{
    auto it = perBook.openSlots.find(agentId);
    if (it == perBook.openSlots.end()) return;
    if (it->second[0].empty() && it->second[1].empty()) {
        perBook.openSlots.erase(it);
    }
}

}  // namespace

//-------------------------------------------------------------------------

void SLTPContainer::resize(size_t bookCount)
{
    m_books.resize(bookCount);
    m_priceFeeds.resize(bookCount);
}

//-------------------------------------------------------------------------

void SLTPContainer::onOrderCreated(const OnOrderCreatedContext& ctx)
{
    if (!ctx.stopLoss && !ctx.takeProfit) return;
    if (ctx.bookId >= m_books.size()) return;

    m_books[ctx.bookId].orderInfo.insert_or_assign(
        ctx.orderId,
        SLTPOrderInfo{
            .agentId = ctx.agentId,
            .bookId = ctx.bookId,
            .clientCtx = ctx.clientCtx,
            .originatingSide = ctx.originatingSide,
            .leverage = ctx.leverage,
            .currency = ctx.currency,
            .remainingVolume = ctx.volume,
            .stopLoss = ctx.stopLoss,
            .takeProfit = ctx.takeProfit,
            .placeholder = ctx.placeholder
        });
    util::SLTPDebugger::log(
        "onOrderCreated: order#{} agent#{} book={} side={} vol={} sl={} tp={}",
        ctx.orderId, ctx.agentId, ctx.bookId, std::to_underlying(ctx.originatingSide),
        ctx.volume,
        ctx.stopLoss ? fmt::format("{}", *ctx.stopLoss) : std::string{"-"},
        ctx.takeProfit ? fmt::format("{}", *ctx.takeProfit) : std::string{"-"});
}

//-------------------------------------------------------------------------

void SLTPContainer::onOrderTrade(const OnOrderTradeContext& ctx)
{
    if (ctx.bookId >= m_books.size()) return;
    auto& perBook = m_books[ctx.bookId];

    // Fast path: books that have never carried any SL/TP activity (no
    // orderInfo and no active slots) skip the whole dance. This runs on
    // every trade side, so it has to be cheap.
    if (perBook.orderInfo.empty() && perBook.activeSlotCount == 0) return;

    // Covering step: a trade on ctx.side reduces any open coverage whose
    // closingSide equals ctx.side (those triggers would close by trading
    // in the same direction this trade just did). Walk the FIFO from the
    // front, shrinking and erasing slots in arrival order until the trade
    // volume is consumed — oldest coverage goes first. We use find()
    // rather than operator[] so unflagged agents don't get default-inserted
    // into openSlots — that would bloat the map over long runs.
    util::SLTPDebugger::log(
        "onOrderTrade: order#{} agent#{} book={} side={} fillPrice={} filledVol={}",
        ctx.originatingOrderId, ctx.agentId, ctx.bookId,
        std::to_underlying(ctx.side), ctx.fillPrice, ctx.filledVolume);

    decimal_t remaining = ctx.filledVolume;
    size_t slotsDrained{};
    bool partialCover{};
    if (auto fifoIt = perBook.openSlots.find(ctx.agentId);
        fifoIt != perBook.openSlots.end()) {
        auto& coveringFifo = fifoIt->second[std::to_underlying(ctx.side)];
        while (!coveringFifo.empty() && remaining > 0_dec) {
            auto& slot = coveringFifo.front();
            if (slot.volume <= remaining) {
                remaining -= slot.volume;
                if (slot.slIter != perBook.triggers.end()) {
                    perBook.triggers.erase(slot.slIter);
                }
                if (slot.tpIter != perBook.triggers.end()) {
                    perBook.triggers.erase(slot.tpIter);
                }
                coveringFifo.pop_front();
                --perBook.activeSlotCount;
                ++slotsDrained;
            } else {
                slot.volume -= remaining;
                if (slot.slIter != perBook.triggers.end()) {
                    slot.slIter->second.volume = slot.volume;
                }
                if (slot.tpIter != perBook.triggers.end()) {
                    slot.tpIter->second.volume = slot.volume;
                }
                remaining = 0_dec;
                partialCover = true;
            }
        }
    }
    if (slotsDrained > 0 || partialCover) {
        util::SLTPDebugger::log(
            "  covering: drained {} slot(s){} (residual={})",
            slotsDrained, partialCover ? " + 1 partial" : "", remaining);
    }

    // Opening step: only the residual that survived the covering pass
    // becomes new coverage. On a position reversal (e.g. flagged SELL 5
    // against an existing long of 3), covering drains 3 and opens a
    // 2-unit slot on the opposite side; with no prior position the full
    // fill survives and a slot for the whole filled volume is opened.
    auto infoIt = perBook.orderInfo.find(ctx.originatingOrderId);
    if (infoIt == perBook.orderInfo.end()) {
        pruneAgentIfEmpty(perBook, ctx.agentId);
        return;
    }
    auto& info = infoIt->second;

    if (remaining > 0_dec && (info.stopLoss || info.takeProfit)) {
        const auto closingSide = ctx.side == OrderDirection::BUY
            ? OrderDirection::SELL
            : OrderDirection::BUY;

        auto& openingFifo =
            perBook.openSlots[ctx.agentId][std::to_underlying(closingSide)];
        openingFifo.push_back(Slot{
            .slIter = perBook.triggers.end(),
            .tpIter = perBook.triggers.end(),
            .volume = remaining
        });
        ++perBook.activeSlotCount;
        auto& slot = openingFifo.back();

        // Cross direction is derived from where the final trigger lands
        // relative to the fill, so callers don't need to care about BUY
        // vs SELL when supplying proportions: long SL with baseSl = -0.05
        // ends up below the fill (AT_OR_BELOW), short SL with
        // baseSl = +0.05 ends up above (AT_OR_ABOVE), and so on.
        const auto baseSl = info.stopLoss.value_or(0_dec);
        const auto baseTp = info.takeProfit.value_or(0_dec);
        const auto placeholder = info.placeholder.value_or(0_dec);
        auto insertTrigger = [&](decimal_t triggerPrice, bool isSL) {
            return perBook.triggers.emplace(
                triggerPrice,
                SLTPEntry{
                    .originatingOrderId = ctx.originatingOrderId,
                    .clientCtx = info.clientCtx,
                    .agentId = info.agentId,
                    .bookId = info.bookId,
                    .closingSide = closingSide,
                    .volume = remaining,
                    .leverage = info.leverage,
                    .currency = info.currency,
                    .triggerPrice = triggerPrice,
                    .cross = triggerPrice >= ctx.fillPrice
                        ? SLTPCross::AT_OR_ABOVE
                        : SLTPCross::AT_OR_BELOW,
                    .fillPrice = ctx.fillPrice,
                    .placeholder = placeholder,
                    .baseSl = baseSl,
                    .baseTp = baseTp,
                    .isSL = isSL
                });
        };

        if (info.stopLoss) {
            slot.slIter = insertTrigger(ctx.fillPrice * util::dec1p(baseSl), true);
        }
        if (info.takeProfit) {
            slot.tpIter = insertTrigger(ctx.fillPrice * util::dec1p(baseTp), false);
        }
        util::SLTPDebugger::log(
            "  opened slot: agent#{} closingSide={} vol={} slTrigger={} tpTrigger={}",
            ctx.agentId, std::to_underlying(closingSide), remaining,
            slot.slIter != perBook.triggers.end()
                ? fmt::format("{}", slot.slIter->first) : std::string{"-"},
            slot.tpIter != perBook.triggers.end()
                ? fmt::format("{}", slot.tpIter->first) : std::string{"-"});
    }

    info.remainingVolume -= ctx.filledVolume;
    if (info.remainingVolume <= 0_dec) {
        perBook.orderInfo.erase(infoIt);
    }
    pruneAgentIfEmpty(perBook, ctx.agentId);
}

//-------------------------------------------------------------------------

void SLTPContainer::onOrderCanceled(BookId bookId, OrderID orderId, decimal_t volumeCanceled)
{
    if (bookId >= m_books.size()) return;
    auto& perBook = m_books[bookId];

    auto it = perBook.orderInfo.find(orderId);
    if (it == perBook.orderInfo.end()) return;

    it->second.remainingVolume -= volumeCanceled;
    const bool fullyCanceled = it->second.remainingVolume <= 0_dec;
    if (fullyCanceled) {
        perBook.orderInfo.erase(it);
    }
    util::SLTPDebugger::log(
        "onOrderCanceled: order#{} book={} canceledVol={} ({})",
        orderId, bookId, volumeCanceled,
        fullyCanceled ? "purged from orderInfo" : "remaining template kept");
}

//-------------------------------------------------------------------------

void SLTPContainer::removeOrder(BookId bookId, OrderID orderId)
{
    if (bookId >= m_books.size()) return;
    auto& perBook = m_books[bookId];

    const bool hadInfo = perBook.orderInfo.contains(orderId);
    perBook.orderInfo.erase(orderId);
    size_t triggersErased{};
    std::optional<AgentId> touchedAgent;

    for (auto it = perBook.triggers.begin(); it != perBook.triggers.end(); ) {
        if (it->second.originatingOrderId == orderId) {
            ++triggersErased;
            touchedAgent = it->second.agentId;
            // Null the handle in the owning slot; drop the slot from its
            // FIFO once both handles are gone so a forced close leaves no
            // stale coverage behind.
            auto& slotList =
                perBook.openSlots[it->second.agentId][std::to_underlying(it->second.closingSide)];
            const auto slotIt = ranges::find_if(slotList, [it](const auto& slot) {
                return slot.slIter == it || slot.tpIter == it;
            });
            if (slotIt != slotList.end()) {
                if (slotIt->slIter == it) slotIt->slIter = perBook.triggers.end();
                if (slotIt->tpIter == it) slotIt->tpIter = perBook.triggers.end();
                if (slotIt->slIter == perBook.triggers.end()
                    && slotIt->tpIter == perBook.triggers.end()) {
                    slotList.erase(slotIt);
                    --perBook.activeSlotCount;
                }
            }
            it = perBook.triggers.erase(it);
        } else {
            ++it;
        }
    }
    if (touchedAgent) pruneAgentIfEmpty(perBook, *touchedAgent);
    if (hadInfo || triggersErased > 0) {
        util::SLTPDebugger::log(
            "removeOrder: order#{} book={} (info={}, triggers erased={})",
            orderId, bookId, hadInfo ? "yes" : "no", triggersErased);
    }
}

//-------------------------------------------------------------------------

void SLTPContainer::onPriceUpdate(BookId bookId, decimal_t price)
{
    if (bookId >= m_books.size()) return;
    if (!m_dispatch) return;

    auto& perBook = m_books[bookId];

    // Without a previous reference price we have no transition to
    // evaluate; record the price and bail. This avoids spurious
    // triggering on the first observation.
    if (!perBook.lastPrice) {
        perBook.lastPrice = price;
        return;
    }

    const auto prev = *perBook.lastPrice;
    perBook.lastPrice = price;
    if (prev == price) return;

    const bool ascending = price > prev;
    const auto lo = std::min(prev, price);
    const auto hi = std::max(prev, price);

    // The closed range [lo, hi] captures every crossed threshold;
    // lower_bound gets us to the first in O(log N), and we stop when the
    // key leaves the range. We deliberately don't cache upper_bound(hi) —
    // firing a trigger can erase its paired trigger anywhere in the map,
    // which would invalidate a cached `last` if it happened to point at
    // that paired node. Keying on `it->first <= hi` is robust.
    for (auto it = perBook.triggers.lower_bound(lo);
         it != perBook.triggers.end() && it->first <= hi; ) {
        const auto& entry = it->second;
        const bool fires = ascending
            ? entry.cross == SLTPCross::AT_OR_ABOVE
            : entry.cross == SLTPCross::AT_OR_BELOW;
        if (fires) {
            // The slot owning this trigger keeps its volume intact so the
            // closing market order's counter-trade will find it in the
            // FIFO and drain it via the covering step; only the trigger
            // iterators are nulled out, and the paired trigger (SL when
            // TP fires, or vice versa) is erased too — a single closure
            // retires both legs of the slot.
            auto& slotList =
                perBook.openSlots[entry.agentId][std::to_underlying(entry.closingSide)];
            const auto slotIt = ranges::find_if(slotList, [it](const auto& slot) {
                return slot.slIter == it || slot.tpIter == it;
            });
            if (slotIt != slotList.end()) {
                const auto paired = (slotIt->slIter == it) ? slotIt->tpIter : slotIt->slIter;
                if (paired != perBook.triggers.end()) {
                    perBook.triggers.erase(paired);
                }
                slotIt->slIter = perBook.triggers.end();
                slotIt->tpIter = perBook.triggers.end();
            }
            util::SLTPDebugger::log(
                "onPriceUpdate: {} fires for order#{} agent#{} book={} triggerPrice={} observedPrice={} ({})",
                entry.isSL ? "SL" : "TP",
                entry.originatingOrderId, entry.agentId, bookId,
                it->first, price, ascending ? "ascending" : "descending");
            m_dispatch(entry, price);
            it = perBook.triggers.erase(it);
        } else {
            ++it;
        }
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::matching

//-------------------------------------------------------------------------
