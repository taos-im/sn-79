/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/accounting/AccountRegistry.hpp>
#include <taosim/checkpoint/serialization/accounting/Account.hpp>
#include <taosim/serialization/msgpack/boost/bimap.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::accounting::AccountRegistry>
{
    const msgpack::object& operator()(
        const msgpack::object& o, taosim::accounting::AccountRegistry& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "localIdCounter") {
                v.localIdCounter() = val.as<AgentId>();
            }
            else if (key == "remoteIdCounter") {
                v.remoteIdCounter() = val.as<AgentId>();
            }
            else if (key == "accounts") {
                auto acctsFromCkpt = val.as<taosim::accounting::AccountRegistry::Accounts>();
                for (auto&& [agentId, acct] : acctsFromCkpt) {
                    // insert_or_assign, not .at(): the checkpoint may hold accounts for
                    // agentIds absent from the freshly-constructed registry (deregistered
                    // agents, runtime-added external/miner UIDs). .at() would throw on the
                    // first such account and abort the whole restore, resetting the exchange.
                    v.accounts().insert_or_assign(agentId, std::move(acct));
                }
            }
            else if (key == "agentIdToBaseName") {
                v.agentIdToBaseName() =
                    val.as<taosim::accounting::AccountRegistry::AgentIdToBaseNameMap>();
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::accounting::AccountRegistry>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o, const taosim::accounting::AccountRegistry& v) const
    {
        o.pack_map(4);

        o.pack("localIdCounter");
        o.pack(v.localIdCounter());

        o.pack("remoteIdCounter");
        o.pack(v.remoteIdCounter());

        o.pack("accounts");
        o.pack(v.accounts());

        o.pack("agentIdToBaseName");
        o.pack(v.agentIdToBaseName());

        return o;
    }
};
    
}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------