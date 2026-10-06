/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/matching/SLTPContainer.hpp>
#include <taosim/decimal/serialization/decimal.hpp>
#include <taosim/serialization/msgpack/common.hpp>

#include <unordered_set>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

//-------------------------------------------------------------------------
// SLTPEntry

template<>
struct convert<taosim::matching::SLTPEntry>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::matching::SLTPEntry& v) const
    {
        if (o.type != msgpack::type::MAP) throw taosim::serialization::MsgPackError{};

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();
            if      (key == "originatingOrderId") v.originatingOrderId = val.as<OrderID>();
            else if (key == "clientCtx")          val.convert(v.clientCtx);
            else if (key == "agentId")            v.agentId = val.as<AgentId>();
            else if (key == "bookId")             v.bookId = val.as<BookId>();
            else if (key == "closingSide")        v.closingSide = val.as<OrderDirection>();
            else if (key == "volume")             v.volume = val.as<taosim::decimal_t>();
            else if (key == "leverage")           v.leverage = val.as<taosim::decimal_t>();
            else if (key == "currency")           v.currency = val.as<Currency>();
            else if (key == "triggerPrice")       v.triggerPrice = val.as<taosim::decimal_t>();
            else if (key == "cross")
                v.cross = static_cast<taosim::matching::SLTPCross>(val.as<uint8_t>());
            else if (key == "fillPrice")          v.fillPrice = val.as<taosim::decimal_t>();
            else if (key == "placeholder")        v.placeholder = val.as<taosim::decimal_t>();
            else if (key == "baseSl")             v.baseSl = val.as<taosim::decimal_t>();
            else if (key == "baseTp")             v.baseTp = val.as<taosim::decimal_t>();
            else if (key == "isSL")               v.isSL = val.as<bool>();
        }
        return o;
    }
};

template<>
struct pack<taosim::matching::SLTPEntry>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::matching::SLTPEntry& v) const
    {
        o.pack_map(15);
        o.pack("originatingOrderId"); o.pack(v.originatingOrderId);
        o.pack("clientCtx");          o.pack(v.clientCtx);
        o.pack("agentId");            o.pack(v.agentId);
        o.pack("bookId");             o.pack(v.bookId);
        o.pack("closingSide");        o.pack(v.closingSide);
        o.pack("volume");             o.pack(v.volume);
        o.pack("leverage");           o.pack(v.leverage);
        o.pack("currency");           o.pack(v.currency);
        o.pack("triggerPrice");       o.pack(v.triggerPrice);
        o.pack("cross");              o.pack(static_cast<uint8_t>(v.cross));
        o.pack("fillPrice");          o.pack(v.fillPrice);
        o.pack("placeholder");        o.pack(v.placeholder);
        o.pack("baseSl");             o.pack(v.baseSl);
        o.pack("baseTp");             o.pack(v.baseTp);
        o.pack("isSL");               o.pack(v.isSL);
        return o;
    }
};

//-------------------------------------------------------------------------
// SLTPOrderInfo

template<>
struct convert<taosim::matching::SLTPOrderInfo>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::matching::SLTPOrderInfo& v) const
    {
        if (o.type != msgpack::type::MAP) throw taosim::serialization::MsgPackError{};

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();
            if      (key == "agentId")         v.agentId = val.as<AgentId>();
            else if (key == "bookId")          v.bookId = val.as<BookId>();
            else if (key == "clientCtx")       val.convert(v.clientCtx);
            else if (key == "originatingSide") v.originatingSide = val.as<OrderDirection>();
            else if (key == "leverage")        v.leverage = val.as<taosim::decimal_t>();
            else if (key == "currency")        v.currency = val.as<Currency>();
            else if (key == "remainingVolume") v.remainingVolume = val.as<taosim::decimal_t>();
            else if (key == "stopLoss")        val.convert(v.stopLoss);
            else if (key == "takeProfit")      val.convert(v.takeProfit);
            else if (key == "placeholder")     val.convert(v.placeholder);
        }
        return o;
    }
};

template<>
struct pack<taosim::matching::SLTPOrderInfo>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::matching::SLTPOrderInfo& v) const
    {
        o.pack_map(10);
        o.pack("agentId");         o.pack(v.agentId);
        o.pack("bookId");          o.pack(v.bookId);
        o.pack("clientCtx");       o.pack(v.clientCtx);
        o.pack("originatingSide"); o.pack(v.originatingSide);
        o.pack("leverage");        o.pack(v.leverage);
        o.pack("currency");        o.pack(v.currency);
        o.pack("remainingVolume"); o.pack(v.remainingVolume);
        o.pack("stopLoss");        o.pack(v.stopLoss);
        o.pack("takeProfit");      o.pack(v.takeProfit);
        o.pack("placeholder");     o.pack(v.placeholder);
        return o;
    }
};

//-------------------------------------------------------------------------
// SLTPPerBook — Slot iterators are round-tripped as (price, orderId) pairs.
// openSlots is deferred until triggers is populated so iterators can be
// reconstructed via equal_range lookups into the restored multimap.

template<>
struct convert<taosim::matching::SLTPPerBook>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::matching::SLTPPerBook& v) const
    {
        if (o.type != msgpack::type::MAP) throw taosim::serialization::MsgPackError{};

        const msgpack::object* openSlotsObj = nullptr;

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "triggers") {
                if (val.type != msgpack::type::ARRAY) throw taosim::serialization::MsgPackError{};
                for (const auto& pair : val.via.array) {
                    if (pair.type != msgpack::type::ARRAY || pair.via.array.size < 2)
                        throw taosim::serialization::MsgPackError{};
                    auto price = pair.via.array.ptr[0].as<taosim::decimal_t>();
                    auto entry = pair.via.array.ptr[1].as<taosim::matching::SLTPEntry>();
                    v.triggers.emplace(price, std::move(entry));
                }
            }
            else if (key == "orderInfo") {
                if (val.type != msgpack::type::ARRAY) throw taosim::serialization::MsgPackError{};
                for (const auto& pair : val.via.array) {
                    if (pair.type != msgpack::type::ARRAY || pair.via.array.size < 2)
                        throw taosim::serialization::MsgPackError{};
                    auto orderId = pair.via.array.ptr[0].as<OrderID>();
                    auto info    = pair.via.array.ptr[1].as<taosim::matching::SLTPOrderInfo>();
                    v.orderInfo.emplace(orderId, std::move(info));
                }
            }
            else if (key == "lastPrice") {
                if (val.type != msgpack::type::NIL)
                    v.lastPrice = val.as<taosim::decimal_t>();
            }
            else if (key == "activeSlotCount") {
                v.activeSlotCount = val.as<size_t>();
            }
            else if (key == "openSlots") {
                openSlotsObj = &val;
            }
        }

        // Restore openSlots after triggers are populated so iterators can
        // be reconstructed via equal_range. A single trigger node is claimed
        // by at most one slot: an order filled multiple times at the same
        // price produces several identical-keyed nodes and an equal number of
        // slots, and (price, orderId, isSL) alone can't tell them apart.
        // Tracking claimed nodes by address keeps the slot-to-node mapping
        // one-to-one, so the FIFO drain in onOrderTrade never erases a node
        // twice through two slots holding the same iterator.
        if (openSlotsObj && openSlotsObj->type == msgpack::type::ARRAY) {
            std::unordered_set<const taosim::matching::SLTPEntry*> claimed;
            for (const auto& agentEntry : openSlotsObj->via.array) {
                if (agentEntry.type != msgpack::type::ARRAY || agentEntry.via.array.size < 2)
                    continue;
                const auto  agentId  = agentEntry.via.array.ptr[0].as<AgentId>();
                const auto& sidesArr = agentEntry.via.array.ptr[1];
                if (sidesArr.type != msgpack::type::ARRAY || sidesArr.via.array.size < 2)
                    continue;

                auto& sides = v.openSlots[agentId];

                for (size_t side = 0; side < 2; ++side) {
                    const auto& slotsArr = sidesArr.via.array.ptr[side];
                    if (slotsArr.type != msgpack::type::ARRAY) continue;

                    for (const auto& sObj : slotsArr.via.array) {
                        if (sObj.type != msgpack::type::MAP) continue;

                        bool slAtEnd = true, tpAtEnd = true;
                        taosim::decimal_t slPrice{}, tpPrice{};
                        OrderID slOrderId{}, tpOrderId{};
                        taosim::decimal_t volume{};

                        for (const auto& [sk, sv] : sObj.via.map) {
                            auto skey = sk.as<std::string_view>();
                            if      (skey == "slAtEnd")   slAtEnd   = sv.as<bool>();
                            else if (skey == "slPrice")   slPrice   = sv.as<taosim::decimal_t>();
                            else if (skey == "slOrderId") slOrderId = sv.as<OrderID>();
                            else if (skey == "tpAtEnd")   tpAtEnd   = sv.as<bool>();
                            else if (skey == "tpPrice")   tpPrice   = sv.as<taosim::decimal_t>();
                            else if (skey == "tpOrderId") tpOrderId = sv.as<OrderID>();
                            else if (skey == "volume")    volume    = sv.as<taosim::decimal_t>();
                        }

                        taosim::matching::Slot slot;
                        slot.volume = volume;

                        slot.slIter = v.triggers.end();
                        if (!slAtEnd) {
                            auto [lo, hi] = v.triggers.equal_range(slPrice);
                            for (auto it = lo; it != hi; ++it) {
                                if (it->second.originatingOrderId == slOrderId && it->second.isSL
                                    && !claimed.contains(&it->second)) {
                                    slot.slIter = it;
                                    claimed.insert(&it->second);
                                    break;
                                }
                            }
                        }

                        slot.tpIter = v.triggers.end();
                        if (!tpAtEnd) {
                            auto [lo, hi] = v.triggers.equal_range(tpPrice);
                            for (auto it = lo; it != hi; ++it) {
                                if (it->second.originatingOrderId == tpOrderId && !it->second.isSL
                                    && !claimed.contains(&it->second)) {
                                    slot.tpIter = it;
                                    claimed.insert(&it->second);
                                    break;
                                }
                            }
                        }

                        sides[side].push_back(std::move(slot));
                    }
                }
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::matching::SLTPPerBook>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::matching::SLTPPerBook& v) const
    {
        o.pack_map(5);

        o.pack("triggers");
        o.pack_array(v.triggers.size());
        for (const auto& [price, entry] : v.triggers) {
            o.pack_array(2);
            o.pack(price);
            o.pack(entry);
        }

        o.pack("orderInfo");
        o.pack_array(v.orderInfo.size());
        for (const auto& [orderId, info] : v.orderInfo) {
            o.pack_array(2);
            o.pack(orderId);
            o.pack(info);
        }

        o.pack("lastPrice");
        if (v.lastPrice.has_value()) o.pack(*v.lastPrice);
        else                         o.pack_nil();

        o.pack("activeSlotCount");
        o.pack(v.activeSlotCount);

        // openSlots: array of [agentId, [side0-slots, side1-slots]]
        // Each slot: map with slAtEnd/slPrice/slOrderId/tpAtEnd/tpPrice/tpOrderId/volume.
        o.pack("openSlots");
        o.pack_array(v.openSlots.size());
        for (const auto& [agentId, sides] : v.openSlots) {
            o.pack_array(2);
            o.pack(agentId);
            o.pack_array(2);
            for (const auto& slotList : sides) {
                o.pack_array(slotList.size());
                for (const auto& slot : slotList) {
                    const bool slAtEnd = (slot.slIter == v.triggers.end());
                    const bool tpAtEnd = (slot.tpIter == v.triggers.end());
                    o.pack_map(7);
                    o.pack("slAtEnd");   o.pack(slAtEnd);
                    o.pack("slPrice");   o.pack(slAtEnd ? taosim::decimal_t{0} : slot.slIter->first);
                    o.pack("slOrderId");
                    o.pack(slAtEnd ? OrderID{} : slot.slIter->second.originatingOrderId);
                    o.pack("tpAtEnd");   o.pack(tpAtEnd);
                    o.pack("tpPrice");   o.pack(tpAtEnd ? taosim::decimal_t{0} : slot.tpIter->first);
                    o.pack("tpOrderId");
                    o.pack(tpAtEnd ? OrderID{} : slot.tpIter->second.originatingOrderId);
                    o.pack("volume");    o.pack(slot.volume);
                }
            }
        }

        return o;
    }
};

//-------------------------------------------------------------------------
// SLTPContainer — only m_books (the per-book trigger state) is persistent;
// m_priceFeeds and m_dispatch are runtime wiring rewired each batch.

template<>
struct convert<taosim::matching::SLTPContainer>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::matching::SLTPContainer& v) const
    {
        if (o.type != msgpack::type::ARRAY) throw taosim::serialization::MsgPackError{};
        const auto& arr = o.via.array;
        if (arr.size != v.bookCount()) throw taosim::serialization::MsgPackError{};
        for (size_t i = 0; i < arr.size; ++i)
            arr.ptr[i].convert(v.books()[i]);
        return o;
    }
};

template<>
struct pack<taosim::matching::SLTPContainer>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::matching::SLTPContainer& v) const
    {
        o.pack_array(v.bookCount());
        for (const auto& book : v.books())
            o.pack(book);
        return o;
    }
};

//-------------------------------------------------------------------------

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
