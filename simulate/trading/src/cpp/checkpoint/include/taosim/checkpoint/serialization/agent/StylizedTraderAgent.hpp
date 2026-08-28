/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/agent/StylizedTraderAgent.hpp>
#include <taosim/checkpoint/serialization/agent/common.hpp>
#include <taosim/serialization/msgpack/boost/circular_buffer.hpp>
#include <taosim/serialization/msgpack/common.hpp>

//-------------------------------------------------------------------------

MSGPACK_ADD_ENUM(taosim::agent::StylizedTraderAgent::RegimeState);

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

//-------------------------------------------------------------------------

template<>
struct convert<taosim::agent::StylizedTraderAgent>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::agent::StylizedTraderAgent& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "tauF") {
                using T = std::remove_cvref_t<decltype(v.tauF())>;
                v.tauF() = val.as<T>();
            }
            else if (key == "orderFlag") {
                using T = std::remove_cvref_t<decltype(v.orderFlag())>;
                v.orderFlag() = val.as<T>();
            }
            else if (key == "regimeChangeProb") {
                using T = std::remove_cvref_t<decltype(v.regimeChangeProb())>;
                v.regimeChangeProb() = val.as<T>();
            }
            else if (key == "regimeState") {
                using T = std::remove_cvref_t<decltype(v.regimeState())>;
                v.regimeState() = val.as<T>();
            }
            else if (key == "topLevel") {
                using T = std::remove_cvref_t<decltype(v.topLevel())>;
                v.topLevel() = val.as<T>();
            }
            else if (key == "lastMid") {
                // Was "priceHist", a full ring buffer of which only the last element was
                // ever read. Checkpoints written before this change carry the old key and
                // will not restore this field; the agent reseeds it on the first L1.
                using T = std::remove_cvref_t<decltype(v.lastMid())>;
                v.lastMid() = val.as<T>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::agent::StylizedTraderAgent>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::agent::StylizedTraderAgent& v) const
    {
        // MUST match the number of key/value pairs packed below.
        o.pack_map(6);

        o.pack("tauF");
        o.pack(v.tauF());

        o.pack("orderFlag");
        o.pack(v.orderFlag());

        o.pack("regimeChangeProb");
        o.pack(v.regimeChangeProb());

        o.pack("regimeState");
        o.pack(v.regimeState());

        o.pack("topLevel");
        o.pack(v.topLevel());

        o.pack("lastMid");
        o.pack(v.lastMid());

        return o;
    }
};

//-------------------------------------------------------------------------

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------