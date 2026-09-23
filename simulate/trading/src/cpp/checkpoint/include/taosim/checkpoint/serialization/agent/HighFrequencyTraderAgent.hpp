/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/agent/HighFrequencyTraderAgent.hpp>
#include <taosim/checkpoint/serialization/agent/common.hpp>
#include <taosim/serialization/msgpack/boost/circular_buffer.hpp>
#include <taosim/serialization/msgpack/common.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

//-------------------------------------------------------------------------

template<>
struct convert<taosim::agent::HighFrequencyTraderAgent>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::agent::HighFrequencyTraderAgent& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "topLevel") {
                val.convert(v.topLevel());
            }
            else if (key == "inventory") {
                val.convert(v.inventory());
            }
            else if (key == "baseFree") {
                val.convert(v.baseFree());
            }
            else if (key == "quoteFree") {
                val.convert(v.quoteFree());
            }
            else if (key == "restingBid") {
                val.convert(v.restingBid());
            }
            else if (key == "restingAsk") {
                val.convert(v.restingAsk());
            }
            else if (key == "pendingBid") {
                val.convert(v.pendingBid());
            }
            else if (key == "pendingAsk") {
                val.convert(v.pendingAsk());
            }
            else if (key == "fillWakeScheduled") {
                val.convert(v.fillWakeScheduled());
            }
            else if (key == "lastPrice") {
                val.convert(v.lastPrice());
            }
            else if (key == "id") {
                val.convert(v.id());
            }
            else if (key == "pRes") {
                // 2026-08 audit fix D2 migration: pRes was a scalar in pre-batch
                // checkpoints; broadcast it into the (configure()-sized) per-book
                // vector instead of throwing msgpack::type_error on restore.
                if (val.type == msgpack::type::FLOAT64 || val.type == msgpack::type::FLOAT32) {
                    const double scalar = val.as<double>();
                    for (auto& p : v.pRes()) p = scalar;
                } else {
                    val.convert(v.pRes());
                }
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::agent::HighFrequencyTraderAgent>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::agent::HighFrequencyTraderAgent& v) const
    {
        // "deltaHFT" went with the self timer it fed: the maker's routine churn is on a token now, so
// nothing reads an inventory-adaptive interval any more.
//
// "orderFlag" and "tauHFT" were dropped when the maker stopped scheduling a cancel
        // per order: the first was never read by anything, and the second was the quote
        // lifetime, which a quote no longer has. The resting-quote ids take their place, and
        // they DO have to survive a restore: without them the agent cannot cancel what it
        // left in the book and would simply quote on top of it.
        o.pack_map(12);

        o.pack("topLevel");
        o.pack(v.topLevel());

        o.pack("inventory");
        o.pack(v.inventory());

        o.pack("baseFree");
        o.pack(v.baseFree());

        o.pack("quoteFree");
        o.pack(v.quoteFree());

        o.pack("restingBid");
        o.pack(v.restingBid());

        o.pack("restingAsk");
        o.pack(v.restingAsk());

        o.pack("pendingBid");
        o.pack(v.pendingBid());

        o.pack("pendingAsk");
        o.pack(v.pendingAsk());

        o.pack("fillWakeScheduled");
        o.pack(v.fillWakeScheduled());

        o.pack("lastPrice");
        o.pack(v.lastPrice());

        o.pack("id");
        o.pack(v.id());

        o.pack("pRes");
        o.pack(v.pRes());

        return o;
    }
};

//-------------------------------------------------------------------------

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------