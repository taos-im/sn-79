/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/book/TickContainer.hpp>
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
struct convert<taosim::book::TickContainer>
{
    msgpack::object const& operator()(
        const msgpack::object& o,taosim::book::TickContainer& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "price") {
                v.price() = val.as<taosim::decimal_t>();
            }
            else if (key == "volume") {
                v.volume() = val.as<taosim::decimal_t>();
            }
            else if (key == "orders") {
                if (val.type != msgpack::type::ARRAY) {
                    throw taosim::serialization::MsgPackError{};
                }
                std::vector<std::shared_ptr<LimitOrder>> orders;
                for (const auto& val2 : val.via.array) {
                    auto order = std::make_shared<LimitOrder>();
                    val2.convert(*order);
                    orders.push_back(order);
                }
                for (auto&& order : orders | views::reverse) {
                    v.push_front(std::move(order));
                }
            }
        }

        // The aggregate is derived rather than trusted: it must equal the sum of the restored
        // orders regardless of what the writer recorded.
        v.volume() = ranges::accumulate(
            v, taosim::decimal_t{}, std::plus{},
            [](const auto& order) { return order->totalVolume(); });

        return o;
    }
};

template<>
struct pack<taosim::book::TickContainer>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::book::TickContainer& v) const
    {
        o.pack_map(3);

        o.pack("price");
        o.pack(v.price());
    
        o.pack("volume");
        o.pack(v.volume());

        o.pack("orders");
        o.pack_array(v.size());
        for (const auto order : v) {
            o.pack(order);
        }

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
