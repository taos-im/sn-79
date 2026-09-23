/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/book/OrderContainer.hpp>
#include <taosim/book/serialization/TickContainer.hpp>
#include <taosim/serialization/msgpack/common.hpp>

#include <range/v3/numeric/accumulate.hpp>

#include <functional>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::book::OrderContainer>
{
    msgpack::object const& operator()(
        const msgpack::object& o, taosim::book::OrderContainer& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "levels") {
                if (val.type != msgpack::type::ARRAY) {
                    throw taosim::serialization::MsgPackError{};
                }
                for (const auto& val2 : val.via.array) {
                    taosim::book::TickContainer tc;
                    tc.orderContainer() = &v;
                    val2.convert(tc);
                    v.push_back(std::move(tc));
                }
            }
            else if (key == "volume") {
                v.volume() = val.as<taosim::decimal_t>();
            }
        }

        // Derived from the restored levels for the same reason a level derives its own aggregate.
        v.volume() = ranges::accumulate(
            v, taosim::decimal_t{}, std::plus{}, [](const auto& level) { return level.volume(); });

        return o;
    }
};

template<>
struct pack<taosim::book::OrderContainer>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::book::OrderContainer& v) const
    {
        o.pack_map(2);

        o.pack("levels");
        o.pack_array(v.size());
        for (const auto& level : v) {
            o.pack(level);
        }

        o.pack("volume");
        o.pack(v.volume());

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------