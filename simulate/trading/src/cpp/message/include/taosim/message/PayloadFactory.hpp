/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/message/MessagePayload.hpp>

#include <msgpack.hpp>

#include <cstdint>
#include <string_view>

//-------------------------------------------------------------------------

class PayloadFactory
{
public:
    // What a msgpack decode does with a type the factory has no payload for. Inbound messages are
    // refused, as the JSON factory refuses them. A checkpointed message queue also holds the engine's
    // own events and responses, which have no payload here; those restore with an empty payload.
    enum class UnknownType : uint8_t
    {
        REJECT,
        EMPTY_PAYLOAD
    };

    [[nodiscard]] static MessagePayload::Ptr createFromJsonMessage(const rapidjson::Value& json);
    [[nodiscard]] static MessagePayload::Ptr createFromMessagePack(
        const msgpack::object& o, std::string_view type, UnknownType unknownType = UnknownType::REJECT);

private:
    PayloadFactory() = default;
};

//-------------------------------------------------------------------------
