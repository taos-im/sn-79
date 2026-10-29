/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/process/JumpDiffusion.hpp>
#include <taosim/checkpoint/serialization/process/Distribution.hpp>
#include <taosim/checkpoint/serialization/process/RNG.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::process::JumpDiffusion>
{
    const msgpack::object& operator()(const msgpack::object& o, taosim::process::JumpDiffusion& v) const
    {    
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "rng") {
                v.state().rng = val.as<taosim::process::RNG>();
            }
            else if (key == "dJ") {
                v.state().dJ = val.as<double>();
            }
            else if (key == "t") {
                v.state().t = val.as<double>();
            }
            else if (key == "W") {
                v.state().W = val.as<double>();
            }
            else if (key == "value") {
                val.convert(v.state().value);
            }
            // Absent from a checkpoint written before they were kept: the distributions then start afresh, as they did.
            else if (key == "gaussian") {
                taosim::checkpoint::serialization::restoreDistribution(v.gaussian(), val.as<std::string_view>());
            }
            else if (key == "jump") {
                taosim::checkpoint::serialization::restoreDistribution(v.jump(), val.as<std::string_view>());
            }
            else if (key == "poisson") {
                taosim::checkpoint::serialization::restoreDistribution(v.poisson(), val.as<std::string_view>());
            }
        }
        
        return o;
    }
};

template<>
struct pack<taosim::process::JumpDiffusion>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::process::JumpDiffusion& v) const
    {
        o.pack_map(8);

        o.pack("rng");
        o.pack(v.state().rng);

        o.pack("dJ");
        o.pack(v.state().dJ);

        o.pack("t");
        o.pack(v.state().t);

        o.pack("W");
        o.pack(v.state().W);

        o.pack("value");
        o.pack(v.state().value);

        o.pack("gaussian");
        o.pack(taosim::checkpoint::serialization::distributionState(v.gaussian()));

        o.pack("jump");
        o.pack(taosim::checkpoint::serialization::distributionState(v.jump()));

        o.pack("poisson");
        o.pack(taosim::checkpoint::serialization::distributionState(v.poisson()));

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------