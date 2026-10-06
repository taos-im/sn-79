/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/checkpoint/CheckpointSource.hpp>
#include <taosim/checkpoint/CheckpointToken.hpp>
#include <taosim/simulation/multiasset/MultiAssetConfig.hpp>
#include <taosim/simulation/multiasset/helpers.hpp>
#include <taosim/simulation/SimulationOrchestrator.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>

//-------------------------------------------------------------------------

namespace taosim::simulation::multiasset
{

//-------------------------------------------------------------------------
// Drives the multi-asset run mode: many realizations of an X-asset portfolio,
// scheduled as cohorts of Simulations over a fixed set of worker threads (see
// ResourceAllocation). Built from a <MultiAssetSimulation> config that names
// the per-asset background configs by relative path.
//
// Relationship to simulation::SimulationManager: the run machinery (worker
// thread pool, per-step barrier, state publishing, checkpointing) is reused by
// COMPOSITION rather than inheritance — SimulationManager's factory-built,
// privately-held internals don't lend themselves to subclassing, and the
// cohort generalization stays in one place. The owned run state lives behind
// the PIMPL. Both kinds satisfy the shared SimulationOrchestrator interface.

class MultiAssetSimulationOrchestrator final
    : public simulation::SimulationOrchestrator
    , public checkpoint::CheckpointSource
{
public:
    ~MultiAssetSimulationOrchestrator() override;

    void run() override;

    [[nodiscard]] const MultiAssetConfig& config() const noexcept;
    [[nodiscard]] const ResourceAllocation& allocation() const noexcept;

    // CheckpointSource: each realization is a checkpoint block; the writes run on a
    // dedicated pool (the run pool is parked on the per-step barrier when a checkpoint
    // is taken); the common state is the grid timestamp plus each realization's log
    // file sizes (every realization has its own log directory).
    [[nodiscard]] bool warmingUp() const override;
    [[nodiscard]] std::span<const std::unique_ptr<Simulation>> checkpointBlocks() const override;
    [[nodiscard]] boost::asio::thread_pool& checkpointExecutor() const override;
    void writeCommonCheckpoint(const std::filesystem::path& commonFile) const override;

    // `resumeTimestamp`, when set, stamps each realization's 'current' grid time so a
    // restored run continues from the checkpoint instant rather than the start.
    [[nodiscard]] static std::unique_ptr<MultiAssetSimulationOrchestrator> fromConfig(
        const std::filesystem::path& configPath,
        const std::filesystem::path& baseDir,
        std::optional<uint64_t> resumeTimestamp = {});
    [[nodiscard]] static std::unique_ptr<MultiAssetSimulationOrchestrator> fromCheckpoint(
        const checkpoint::CheckpointToken& ckptToken);

private:
    MultiAssetSimulationOrchestrator();

    std::unique_ptr<struct Impl> m_impl;
};

//-------------------------------------------------------------------------

}  // namespace taosim::simulation::multiasset

//-------------------------------------------------------------------------
