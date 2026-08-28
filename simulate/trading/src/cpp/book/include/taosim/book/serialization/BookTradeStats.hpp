/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/book/BookTradeStats.hpp>
#include <taosim/decimal/serialization/decimal.hpp>
#include <taosim/serialization/msgpack/common.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::book::BookTradeStats>
{
    msgpack::object const& operator()(
        msgpack::object const& o, taosim::book::BookTradeStats& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "tradeCount") {
                v.tradeCount = val.as<uint64_t>();
            }
            else if (key == "volumeSum") {
                v.volumeSum = val.as<taosim::decimal_t>();
            }
            else if (key == "notionalSum") {
                v.notionalSum = val.as<taosim::decimal_t>();
            }
            else if (key == "logReturnSum") {
                v.logReturnSum = val.as<double>();
            }
            else if (key == "logReturnSqSum") {
                v.logReturnSqSum = val.as<double>();
            }
            else if (key == "lastTradePrice") {
                v.lastTradePrice = val.as<taosim::decimal_t>();
            }
            else if (key == "lastTradeTime") {
                v.lastTradeTime = val.as<Timestamp>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::book::BookTradeStats>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::book::BookTradeStats& v) const
    {
        o.pack_map(7);

        o.pack("tradeCount");
        o.pack(v.tradeCount);

        o.pack("volumeSum");
        o.pack(v.volumeSum);

        o.pack("notionalSum");
        o.pack(v.notionalSum);

        o.pack("logReturnSum");
        o.pack(v.logReturnSum);

        o.pack("logReturnSqSum");
        o.pack(v.logReturnSqSum);

        o.pack("lastTradePrice");
        o.pack(v.lastTradePrice);

        o.pack("lastTradeTime");
        o.pack(v.lastTradeTime);

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
