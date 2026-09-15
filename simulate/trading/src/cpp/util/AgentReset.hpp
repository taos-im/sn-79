/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "JsonSerializable.hpp"
#include "Timestamp.hpp"
#include "common.hpp"
#include "json_util.hpp"

#include <msgpack.hpp>

#include <cstdint>
#include <memory>

//-------------------------------------------------------------------------

// An agent reset as it appears in the per-book L3 log. The reset cancels the agent's resting orders
// through the book, so L3 already holds one ordinary cancellation per order; this record marks the reset
// that caused them, once per book, whether or not the agent had orders on that book. It is the only L3
// trace of a reset for an agent with nothing resting, and the only record that says the cancellations
// beside it were not the agent's own.
struct AgentResetLogContext : public JsonSerializable
{
    using Ptr = std::shared_ptr<AgentResetLogContext>;

    AgentId agentId{};
    BookId bookId{};
    Timestamp timestamp{};
    uint32_t cancelled{};

    AgentResetLogContext() noexcept = default;

    AgentResetLogContext(AgentId agentId, BookId bookId, Timestamp timestamp, uint32_t cancelled) noexcept
        : agentId{agentId}, bookId{bookId}, timestamp{timestamp}, cancelled{cancelled}
    {}

    // {"r":{"a":<agent>,"n":<orders cancelled on this book>},"g":{"a":<agent>,"b":<book>,"j":<timestamp>}}
    // The "g" group has the shape of every other L3 record's, so the logger's canonical book-id rewrite
    // applies to it and a reader that keys on "g" needs no new case.
    void L3Serialize(rapidjson::Document& json, const std::string& key = {}) const;

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    MSGPACK_DEFINE_MAP(agentId, bookId, timestamp, cancelled);
};

//-------------------------------------------------------------------------
