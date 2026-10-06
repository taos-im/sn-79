/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/decimal/decimal.hpp>
#include <taosim/serialization/msgpack/common.hpp>

#include <cstring>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

//-------------------------------------------------------------------------

template<>
struct convert<taosim::decimal_t>
{
    const msgpack::object& operator()(const msgpack::object& o, taosim::decimal_t& v) const
    {
        if (o.type == msgpack::type::FLOAT64) {
            v = taosim::util::double2decimal(o.as<double>());
        }
        else if (o.type == msgpack::type::BIN) {
            v = taosim::util::unpackDecimal(o.as<taosim::PackedDecimal>());
        }
        else if (o.type == msgpack::type::POSITIVE_INTEGER) {
            v = taosim::decimal_t{o.as<uint64_t>()};
        }
        else if (o.type == msgpack::type::NEGATIVE_INTEGER) {
            v = taosim::decimal_t{o.as<int64_t>()};
        }
        else {
            throw taosim::serialization::MsgPackError{};
        }
        return o;
    }
};

template<>
struct pack<taosim::decimal_t>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::decimal_t& v) const
    {
        if constexpr (std::same_as<Stream, taosim::serialization::HumanReadableStream>) {
            o.pack(taosim::util::decimal2double(v));
        }
        else if constexpr (std::same_as<Stream, taosim::serialization::BinaryStream>) {
            o.pack(taosim::util::packDecimal(v));
        }
        else {
            static_assert(false, "Unrecognized Stream type");
        }
        return o;
    }
};

//-------------------------------------------------------------------------

template<>
struct convert<taosim::PackedDecimal>
{
    const msgpack::object& operator()(const msgpack::object& o, taosim::PackedDecimal& v) const
    {
        if (o.type != msgpack::type::BIN) {
            throw taosim::serialization::MsgPackError{};
        }
        if (o.via.bin.size != sizeof(v.data)) {
            throw taosim::serialization::MsgPackError{};
        }
        std::memcpy(v.data, o.via.bin.ptr, sizeof(v.data));
        return o;
    }
};

template<>
struct pack<taosim::PackedDecimal>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::PackedDecimal& v) const
    {
        o.pack_bin(sizeof(v.data));
        o.pack_bin_body(reinterpret_cast<const char*>(v.data), sizeof(v.data));
        return o;
    }
};

//-------------------------------------------------------------------------

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
