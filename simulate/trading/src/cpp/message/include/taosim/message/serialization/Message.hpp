/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/message/Message.hpp>
#include <taosim/message/MessagePayload.hpp>
#include <taosim/message/PayloadFactory.hpp>
#include <taosim/message/serialization/DistributedAgentResponsePayload.hpp>
#include <taosim/message/serialization/helpers.hpp>
#include <taosim/serialization/msgpack/common.hpp>
#include <taosim/serialization/msgpack/utils.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<Message>
{
    const msgpack::object& operator()(const msgpack::object& o, Message& v)
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        const auto type = taosim::serialization::msgpackFindMap<std::string_view>(o, "type");
        if (!type) {
            throw taosim::serialization::MsgPackError{};
        }
        v.type = *type;

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "occurrence") {
                v.occurrence = val.as<Timestamp>();
            }
            else if (key == "arrival") {
                v.arrival = val.as<Timestamp>();
            }
            else if (key == "source") {
                v.source = val.as<std::string>();
            }
            else if (key == "target") {
                v.targets = val.as<std::vector<std::string>>();
            }
            else if (key == "payload") {
                v.payload = PayloadFactory::createFromMessagePack(
                    val, *type, PayloadFactory::UnknownType::EMPTY_PAYLOAD);
            }
        }

        return o;
    }
};

template<>
struct pack<Message>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o, const Message& v) const
    {
        o.pack_map(6);

        o.pack("occurrence");
        o.pack(v.occurrence);
    
        o.pack("arrival");
        o.pack(v.arrival);

        o.pack("source");
        o.pack(v.source);

        o.pack("target");
        o.pack(v.targets);

        o.pack("type");
        o.pack(v.type);

        o.pack("payload");
        taosim::message::serialization::packMessagePayload(o, v.payload);

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------