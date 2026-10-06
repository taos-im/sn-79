/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/checkpoint/CheckpointToken.hpp>

#include <pugixml.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

//-------------------------------------------------------------------------

namespace taosim::simulation { class SimulationOrchestrator; }

//-------------------------------------------------------------------------

namespace taosim::simulation::multiasset
{

//-------------------------------------------------------------------------
// Cheap routing probe: true iff the file's root element is the multi-asset
// wrapper. Parses only as far as the root element.

[[nodiscard]] bool isMultiAssetConfig(const std::filesystem::path& configPath);

//-------------------------------------------------------------------------
// Build the orchestrator matching `configPath`'s root element: the multi-asset
// wrapper yields a MultiAssetSimulationOrchestrator, anything else a
// SimulationManager. Lets callers drive any run mode through run() alone.

[[nodiscard]] std::unique_ptr<simulation::SimulationOrchestrator> makeSimulationOrchestrator(
    const std::filesystem::path& configPath, const std::filesystem::path& baseDir);

//-------------------------------------------------------------------------
// Restore-mode counterpart: route by the persisted run-dir config.xml's root element
// to the matching orchestrator's fromCheckpoint.

[[nodiscard]] std::unique_ptr<simulation::SimulationOrchestrator> makeOrchestratorFromCheckpoint(
    const checkpoint::CheckpointToken& ckptToken);

//-------------------------------------------------------------------------
// Inputs to the worker/cohort layout decision. Aggregate-initializable.

struct ResourceAllocationDesc
{
    uint32_t simulationCount{};       // N: total Simulations (sum of per-background instanceCount)
    uint32_t requestedWorkers{};      // thread count; 0 => derive from hardwareConcurrency
    uint32_t hardwareConcurrency{};
    uint32_t checkpointWorkers{};     // threads reserved for checkpoint I/O
};

//-------------------------------------------------------------------------
// The resolved layout: how many worker threads, and how the N Simulations are
// spread over them as cohorts. The barrier is sized to workerCount, decoupling
// the Simulation count from the core count.

struct ResourceAllocation
{
    uint32_t workerCount{};                // threads, each draining a cohort
    uint32_t simulationCount{};            // == ResourceAllocationDesc::simulationCount
    uint32_t threadPoolSize{};             // workerCount + checkpointWorkers
    std::vector<uint32_t> cohortSizes;     // Simulations assigned per worker
};

//-------------------------------------------------------------------------
// Pure: choose the worker count (honoring the request and the thread budget)
// and balance the Simulations across workers. No side effects.

[[nodiscard]] ResourceAllocation computeResourceAllocation(const ResourceAllocationDesc& desc);

//-------------------------------------------------------------------------
// Derive a deterministic, distinct seed for one realization (instance) of a
// background, so realizations are reproducible yet independent.

[[nodiscard]] uint64_t realizationSeed(
    uint64_t masterSeed, uint32_t backgroundIdx, uint32_t instanceIdx);

//-------------------------------------------------------------------------
// Adjust a background <Simulation> config into one concrete realization: stamp
// the per-realization id (quote balances stay per book, as in a single market; the
// shared-wallet stamp was removed on 30 September 2026). `seed`, when given, is
// stamped as `rngSeed` (the attribute Simulation reads),
// overriding any the background declares; when absent the background's own
// rngSeed (or device seeding) stands. Returns an owned copy; the source node is
// left untouched.

[[nodiscard]] pugi::xml_document adjustedBackgroundDoc(
    const pugi::xml_node& backgroundNode, std::optional<uint64_t> seed, std::string_view id);

//-------------------------------------------------------------------------

}  // namespace taosim::simulation::multiasset

//-------------------------------------------------------------------------
