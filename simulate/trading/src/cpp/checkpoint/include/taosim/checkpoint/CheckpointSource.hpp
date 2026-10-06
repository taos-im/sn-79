/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <filesystem>
#include <memory>
#include <span>

//-------------------------------------------------------------------------

namespace boost::asio { class thread_pool; }

class Simulation;

//-------------------------------------------------------------------------

namespace taosim::checkpoint
{

//-------------------------------------------------------------------------
// The run-mode-specific surface a CheckpointManager needs to persist a checkpoint.
// SimulationManager (single config) and the multi-asset orchestrator each implement
// it; checkpointing runs once every N steps, so the virtual indirection is free.

struct CheckpointSource
{
    virtual ~CheckpointSource() = default;

    // Skip checkpointing while the run is still warming up.
    [[nodiscard]] virtual bool warmingUp() const = 0;

    // The per-block simulations to persist (one checkpoint file each); the first also
    // dates the checkpoint. A "block" is a single-config block or a multi-asset
    // realization — both are just Simulations.
    [[nodiscard]] virtual std::span<const std::unique_ptr<Simulation>>
        checkpointBlocks() const = 0;

    // Execution context for the parallel per-block writes. Implementations must size it
    // with spare capacity beyond their stepping threads: the writes are posted while the
    // steppers are parked on the run barrier, so a pool sized to the steppers deadlocks.
    [[nodiscard]] virtual boost::asio::thread_pool& checkpointExecutor() const = 0;

    // Persist the run-mode-specific "common" state (e.g. timestamp + log file sizes).
    virtual void writeCommonCheckpoint(const std::filesystem::path& commonFile) const = 0;

protected:
    CheckpointSource() noexcept = default;
};

//-------------------------------------------------------------------------

}  // namespace taosim::checkpoint

//-------------------------------------------------------------------------
