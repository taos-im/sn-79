/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <pugixml.hpp>

#include <taosim/net/net.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

//-------------------------------------------------------------------------

namespace taosim::simulation::multiasset
{

//-------------------------------------------------------------------------
// Root element identifying a multi-asset run; lets `-f` route between the
// single-Simulation manager and the orchestrator without a separate flag.

inline constexpr std::string_view kRootElement{"MultiAssetSimulation"};

//-------------------------------------------------------------------------
// One background = one realization of a portfolio: a standalone <Simulation>
// config of bookCount assets with per-book quote balances, referenced by (relative) path.
// `instanceCount` requests that many independently-seeded realizations of it.

struct BackgroundDesc
{
    std::string name;
    std::filesystem::path path;
    uint32_t instanceCount{1};
};

//-------------------------------------------------------------------------
// Minimal top-level descriptor of a multi-asset run, parsed from the
// <MultiAssetSimulation> wrapper. Aggregate-initializable.

struct MultiAssetConfig
{
    std::vector<BackgroundDesc> backgrounds;  // one per listed <Background>
    uint32_t requestedWorkers{};              // thread count; 0 => derive from hardware
    std::optional<uint64_t> masterSeed;       // root of every per-realization rngSeed;
                                              // absent => backgrounds keep their own
    uint64_t start{};                         // shared global time grid, stamped onto every
    uint64_t step{};                          // realization; step & duration are required (> 0)
    uint64_t duration{};
    bool useMessagePack{};                    // publish state to the validator each step (IPC)
    uint64_t gracePeriod{};                   // warmup: suppress publishing/checkpointing below this
    uint64_t ckptIntervalInSteps{};           // checkpoint every N global steps; 0 => disabled
    uint64_t ckptNumLastFilesToKeep{};        // retain only the most recent N checkpoint dirs
    bool ckptMeasureWallClockTime{};          // log per-save wall-clock time
    uint32_t ckptNumWorkers{1};               // dedicated pool size for parallel block writes
    bool sltpDebug{};                         // the SL/TP tracer; a background's own flag is ignored
    bool traceTime{};                         // print the sim clock (TIME) each global step
    bool measureStepWallClockTime{};          // print the per-step PROCESSED wall-clock breakdown
    // Where EVENT_SIMULATION_START/END go: HTTP to host:port + generalMsgEndpoint, as
    // SimulationManager sends them. Empty host or port => offline, nothing is sent.
    net::NetworkingInfo netInfo;
    std::string generalMsgEndpoint{"/"};
    std::filesystem::path baseDir;            // where run artifacts are preserved
    std::filesystem::path sourcePath;         // the multi-asset config itself

    // Parse the wrapper node; `configDir` resolves the relative background paths.
    [[nodiscard]] static MultiAssetConfig fromXML(
        const pugi::xml_node& node, const std::filesystem::path& configDir);
};

//-------------------------------------------------------------------------

}  // namespace taosim::simulation::multiasset

//-------------------------------------------------------------------------
