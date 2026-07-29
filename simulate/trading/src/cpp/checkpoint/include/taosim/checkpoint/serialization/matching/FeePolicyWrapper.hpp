/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/matching/FeePolicyWrapper.hpp>
#include <taosim/checkpoint/serialization/matching/helpers.hpp>
#include <taosim/serialization/msgpack/common.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::matching::FeePolicyWrapper>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::matching::FeePolicyWrapper& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        auto convertAgentBaseNameFeePolicies = [&](const msgpack::object& o) {
            if (o.type != msgpack::type::MAP) {
                throw taosim::serialization::MsgPackError{};
            }
            for (const auto& [k, val] : o.via.map) {
                auto key = k.as<std::string>();

                // find(), not .at(): a checkpoint fee-policy key absent from the current
                // config (policies changed between runs) is skipped, not treated as fatal
                // (which would abort the whole restore and reset the exchange).
                auto it = v.agentBaseNameFeePolicies().find(key);
                if (it == v.agentBaseNameFeePolicies().end()) {
                    continue;
                }
                auto feePolicy = it->second.get();

                if (auto fp = dynamic_cast<taosim::matching::DynamicFeePolicy*>(feePolicy)) {
                    val.convert(*fp);
                }
                else if (auto fp = dynamic_cast<taosim::matching::TieredFeePolicy*>(feePolicy)) {
                    val.convert(*fp);
                }
            }
        };

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "defaultFeePolicy") {
                auto fp = v.feePolicy().get();
                taosim::checkpoint::serialization::unpackFeePolicy(val, *fp);
            }
            else if (key == "agentBaseNameFeePolicies") {
                convertAgentBaseNameFeePolicies(val);
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::matching::FeePolicyWrapper>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::matching::FeePolicyWrapper& v) const
    {
        using taosim::checkpoint::serialization::packFeePolicy;

        o.pack_map(2);

        o.pack("defaultFeePolicy");
        packFeePolicy(o, *v.feePolicy());

        o.pack("agentBaseNameFeePolicies");
        o.pack_map(v.agentBaseNameFeePolicies().size());
        for (auto&& [baseName, feePolicy] : v.agentBaseNameFeePolicies()) {
            o.pack(baseName);
            packFeePolicy(o, *feePolicy);
        }

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
