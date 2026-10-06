// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT

#pragma once

#include <Simulation.hpp>
#include <taosim/message/Message.hpp>
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
// Blocking send on a private io_context, retrying as asyncSendOverNetwork does
// until the validator answers. `logger` receives the transport's debug lines.

void postGeneralMessage(
    const Simulation& logger,
    const net::NetworkingInfo& netInfo,
    std::string_view endpoint,
    const rapidjson::Document& body);

//-------------------------------------------------------------------------

}  // namespace taosim::simulation

//-------------------------------------------------------------------------
