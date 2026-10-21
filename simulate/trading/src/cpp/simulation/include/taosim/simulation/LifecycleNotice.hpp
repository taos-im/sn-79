// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT

#pragma once

#include <Simulation.hpp>
#include <taosim/message/Message.hpp>
#include <taosim/message/MessagePayload.hpp>
#include <taosim/net/net.hpp>

#include <rapidjson/document.h>

#include <cstdint>
#include <filesystem>
#include <span>
#include <string_view>

//-------------------------------------------------------------------------

namespace taosim::simulation
{

//-------------------------------------------------------------------------
// The validator-facing lifecycle notices, in the exact shape
// SimulationManager::publishStartInfo / publishEndInfo send: one HTTP GET to the
// general-message endpoint carrying {"messages": [...]}, each message
// {timestamp, delay: 0, source: "SIMULATION", target: "*", type, payload}. The
// validator's /account route walks the batch and hands EVENT_SIMULATION_START
// (payload.logDir) to on_start and EVENT_SIMULATION_END to on_end. Factored out so
// the multi-asset orchestrator can send one START per realization in one request.

[[nodiscard]] Message::Ptr makeStartNotice(uint64_t start, const std::filesystem::path& logDir);

[[nodiscard]] Message::Ptr makeEndNotice(uint64_t start);

[[nodiscard]] rapidjson::Document makeNoticeBatch(std::span<const Message::Ptr> messages);

//-------------------------------------------------------------------------
// EVENT_SIMULATION_FAILED: a multi-asset run stopped on a failed realization. It
// names the realization, its background, the simulation time of the failure and the
// error, clipped to kFailureNoticeErrorLength bytes on a character boundary. The
// validator pages on it and takes it for no run end.

inline constexpr size_t kFailureNoticeErrorLength = 200;

struct SimulationFailedPayload : public MessagePayload
{
    using Ptr = std::shared_ptr<SimulationFailedPayload>;

    uint32_t realization{};
    std::string background;
    uint64_t time{};
    std::string error;

    SimulationFailedPayload(uint32_t realization, std::string background, uint64_t time, std::string error) noexcept
        : realization{realization}, background{std::move(background)}, time{time}, error{std::move(error)}
    {}

    void jsonSerialize(rapidjson::Document& json, const std::string& key = {}) const override;
};

[[nodiscard]] Message::Ptr makeFailureNotice(
    uint32_t realization, std::string_view background, uint64_t time, std::string_view error);

//-------------------------------------------------------------------------
// Blocking send on a private io_context, retrying as asyncSendOverNetwork does
// until the validator answers. `logger` receives the transport's debug lines.
// With `failFast`, the first transport failure gives up instead.

void postGeneralMessage(
    const Simulation& logger,
    const net::NetworkingInfo& netInfo,
    std::string_view endpoint,
    const rapidjson::Document& body,
    bool failFast = false);

//-------------------------------------------------------------------------

}  // namespace taosim::simulation

//-------------------------------------------------------------------------
