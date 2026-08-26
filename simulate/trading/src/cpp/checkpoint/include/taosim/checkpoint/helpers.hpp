/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/checkpoint/CheckpointToken.hpp>
#include <taosim/serialization/msgpack/common.hpp>

#include <filesystem>
#include <functional>
#include <span>
#include <vector>

//-------------------------------------------------------------------------

class MultiBookExchangeAgent;

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
// originals, severing every logger and event-backlog feed connected at configure time. That exact bug was
// found and fixed on 2026-05-12 (971786f9, "preserve L3EventLogger connections") in the OTHER apply path
// and survived here for three months, because nobody knew there were two.
//
// This removes the bug class rather than the bug: the section is converted into plain integers and
// assigned to the existing objects, so no code path deserialises an ExchangeSignals at all. The on-disk
// format is unchanged, because pack<ExchangeSignals> already writes only the counter, making the section
// a map<BookId, positive integer> on disk either way.
//
// NEVER convert the signals map wholesale. There is no custom convert to stop you; msgpack's default
// will happily replace every object and the only symptom is log files that contain their header and
// nothing else, which is how this went unnoticed from May to August.
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

void setupUsingCkptData(
    taosim::simulation::SimulationManager* simuMngr,
    const msgpack::object& commonObj,
    std::span<msgpack::object_handle> blockObjHandles);

}  // namespace taosim::checkpoint

//-------------------------------------------------------------------------
