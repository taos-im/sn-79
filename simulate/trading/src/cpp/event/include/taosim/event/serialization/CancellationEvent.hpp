/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/event/CancellationEvent.hpp>
#include <taosim/serialization/msgpack/common.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::event::CancellationEvent>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::event::CancellationEvent& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "cancellation") {
                v.cancellation = val.as<taosim::event::Cancellation>();
            }
            else if (key == "timestamp") {
                v.timestamp = val.as<Timestamp>();
            }
            else if (key == "price") {
                v.price = val.as<taosim::decimal_t>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::event::CancellationEvent>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::event::CancellationEvent& v) const
    {
        if constexpr (std::same_as<Stream, taosim::serialization::HumanReadableStream>) {
            o.pack_map(7);
    
            o.pack("y");
            o.pack("c");

            o.pack("i");
            o.pack(v.cancellation.id);

            o.pack("t");
            o.pack(v.timestamp);

            o.pack("p");
            o.pack(v.price);

            o.pack("q");
            o.pack(v.cancellation.volume);
            o.pack("s");
            o.pack(std::to_underlying(v.direction));
            o.pack("l");
            o.pack(v.leverage);
        }
        else if constexpr (std::same_as<Stream, taosim::serialization::BinaryStream>) {
            o.pack_map(4);

            o.pack("event");
            o.pack("cancel");

            o.pack("cancellation");
            o.pack(v.cancellation);

            o.pack("timestamp");
            o.pack(v.timestamp);

            o.pack("price");
            o.pack(v.price);
        }
        else {
            static_assert(false, "Unrecognized Stream type");
        }

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
