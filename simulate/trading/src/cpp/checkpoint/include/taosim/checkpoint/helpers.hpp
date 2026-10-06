/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/checkpoint/CheckpointToken.hpp>
#include <taosim/serialization/msgpack/common.hpp>

#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <vector>

//-------------------------------------------------------------------------

class MultiBookExchangeAgent;
class Simulation;

namespace taosim::simulation
{

class SimulationManager;

}  // namespace taosim::simulation

//-------------------------------------------------------------------------

namespace taosim::checkpoint
{

// Restore the per-book L3 event counters from a checkpoint's "signals" section.
//
// ONE implementation, because two was the defect. The exchange-service apply path used
// `ev.convert(exch->signals())`, msgpack's default convert for
// std::map<BookId, std::unique_ptr<ExchangeSignals>>, which CREATES new objects and destroys the
// originals, severing every logger and event-backlog feed connected at configure time.
//
// This removes the bug class rather than the bug: the section is converted into plain integers and
// assigned to the existing objects, so no code path deserialises an ExchangeSignals at all. The on-disk
// format is unchanged, because pack<ExchangeSignals> already writes only the counter, making the section
// a map<BookId, positive integer> on disk either way.
//
// NEVER convert the signals map wholesale. There is no custom convert to stop you; msgpack's default
// will happily replace every object, and the only symptom is log files that contain their header
// and nothing else -- a failure with no error and no missing file, so nothing reports it.
void restoreSignalCounters(MultiBookExchangeAgent* exchange, const msgpack::object& section);

[[nodiscard]] CheckpointToken postProcessToken(const CheckpointToken& token);

[[nodiscard]] std::filesystem::path runDirFromToken(const CheckpointToken& token);

using PathFactory = std::function<std::filesystem::path(const std::filesystem::path&)>;

[[nodiscard]] std::filesystem::path runDirLatest(const std::filesystem::path& baseDir);

inline static const std::map<CheckpointToken, PathFactory> s_tokenToRunDirFactory{
    {"latest", &runDirLatest}
};

[[nodiscard]] std::filesystem::path ckptDirLatest(const std::filesystem::path& runDir);

inline static const std::map<CheckpointToken, PathFactory> s_tokenToCkptDirFactory{
    {"latest", &ckptDirLatest}
};

[[nodiscard]] std::filesystem::path ckptDirFromToken(const CheckpointToken& token);

[[nodiscard]] std::vector<std::filesystem::path> ckptDirsSortedByWriteTime(
    const std::filesystem::path& path);

// Atomically publish `data` to `path`: write to a sibling temp file, then rename it
// over the target, so a crash mid-write leaves either the previous file or the
// complete new one, never a truncated one. Throws CheckpointError on any failure.
void atomicWrite(const std::filesystem::path& path, std::span<const char> data);

class CheckpointManager;

void setupUsingCkptData(
    taosim::simulation::SimulationManager* simuMngr,
    const msgpack::object& commonObj,
    std::span<msgpack::object_handle> blockObjHandles);

void setupUsingCkptDataMultiAsset(
    std::span<const std::unique_ptr<Simulation>> simulations,
    CheckpointManager* checkpointManager,
    const msgpack::object& commonObj,
    std::span<msgpack::object_handle> blockObjHandles);

// Re-link the shared-quote topology a msgpack restore severs: one quote Balance per
// account and one order-id + trade-id counter across the exchange's books (each is
// deserialized per book independently). No-op when quote is not shared.
void reestablishSharedTopology(MultiBookExchangeAgent& exchange);

// Common + numbered block files of a checkpoint directory, as msgpack object handles
// (which own their buffers — keep the result alive while converting from it).
struct LoadedCheckpoint
{
    msgpack::object_handle common;
    std::vector<msgpack::object_handle> blocks;
};

[[nodiscard]] LoadedCheckpoint loadCheckpointObjects(const std::filesystem::path& ckptDir);

}  // namespace taosim::checkpoint

//-------------------------------------------------------------------------
