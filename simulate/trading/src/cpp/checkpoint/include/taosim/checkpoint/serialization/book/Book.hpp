/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/book/Book.hpp>
#include <taosim/book/serialization/OrderContainer.hpp>
#include <taosim/book/serialization/TickContainer.hpp>

#include <range/v3/action/reverse.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::book::Book>
{
    const msgpack::object& operator()(const msgpack::object& o, taosim::book::Book& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "buyQueue") {
                val.convert(v.buyQueue());
            }
            else if (key == "sellQueue") {
                val.convert(v.sellQueue());
            }
            else if (key == "orderIdCounter") {
                val.convert(v.orderIdCounter());
            }
            else if (key == "tradeIdCounter") {
                val.convert(v.tradeIdCounter());
            }
            else if (key == "orderToClientInfo") {
                val.convert(v.orderToClientInfo());
            }
            // THE BAND TRAVELS WITH THE BOOK. Absent here, a resumed book is unbounded until its
            // first print, and that print then becomes the reference. Keys are read individually so
            // a checkpoint written before this change still loads: the missing keys simply leave the
            // defaults, which is the old behaviour rather than a parse failure.
            else if (key == "bandSamples") {
                val.convert(v.bandSamples());
            }
            else if (key == "bandLastPrice") {
                val.convert(v.bandLastPrice());
            }
            else if (key == "bandRefCached") {
                val.convert(v.bandRefCached());
            }
            else if (key == "bandLastSampleTs") {
                val.convert(v.bandLastSampleTs());
            }
            else if (key == "bandSeeded") {
                val.convert(v.bandSeeded());
            }
            // The release rule counts silence from the last print; without it a book locked across a resume
            // would count from nothing and never widen.
            else if (key == "bandLastTradeTs") {
                val.convert(v.bandLastTradeTs());
            }
            else if (key == "bandRefusedSinceTrade") {
                val.convert(v.bandRefusedSinceTrade());
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::book::Book>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o, const taosim::book::Book& v) const
    {
        o.pack_map(12);

        o.pack("buyQueue");
        o.pack(v.buyQueue());

        o.pack("sellQueue");
        o.pack(v.sellQueue());

        o.pack("orderIdCounter");
        o.pack(*v.orderIdCounter());

        o.pack("tradeIdCounter");
        o.pack(*v.tradeIdCounter());

        o.pack("orderToClientInfo");
        o.pack(v.orderToClientInfo());

        o.pack("bandSamples");
        o.pack(v.bandSamples());

        o.pack("bandLastPrice");
        o.pack(v.bandLastPrice());

        o.pack("bandRefCached");
        o.pack(v.bandRefCached());

        o.pack("bandLastSampleTs");
        o.pack(v.bandLastSampleTs());

        o.pack("bandSeeded");
        o.pack(v.bandSeeded());
        o.pack("bandLastTradeTs");
        o.pack(v.bandLastTradeTs());
        o.pack("bandRefusedSinceTrade");
        o.pack(v.bandRefusedSinceTrade());

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
