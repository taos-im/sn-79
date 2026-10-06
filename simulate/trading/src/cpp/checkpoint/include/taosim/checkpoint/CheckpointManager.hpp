/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/checkpoint/CheckpointSource.hpp>

#include <filesystem>
#include <memory>
#include <regex>
#include <string_view>

//-------------------------------------------------------------------------

namespace taosim::checkpoint
{

//-------------------------------------------------------------------------

struct CheckpointingDesc
{
    CheckpointSource* source;
    std::filesystem::path runDir;
    size_t intervalInSteps{};
    ptrdiff_t numLastFilesToKeep{};
    bool measureWallClockTime{};
};

//-------------------------------------------------------------------------
// Owns the checkpoint store (directory layout, retention, interval gating, timing)
// and drives a CheckpointSource to serialize the run-mode-specific state. Decoupled
// from any concrete run mode via that interface. The owning runner connects
// saveCheckpoint() to its per-step signal.

class CheckpointManager
{
public:
    explicit CheckpointManager(const CheckpointingDesc& desc);
    ~CheckpointManager();

    [[nodiscard]] ssize_t& stepCounter() noexcept;

    void saveCheckpoint();

    // Write a checkpoint NOW, ignoring the step interval. For an INTENDED stop only.
    //
    // saveCheckpoint() above is interval-gated, and the interval is in STEPS, so its wall-clock spacing
    // depends entirely on step rate: at a low step count per interval it can be many minutes. A resume
    // therefore rewinds by up to one interval, while every consumer downstream -- the trade record, the
    // data service, any ledger keyed by (book, trade id) -- has kept the interim minutes. The engine then
    // re-mints ids those consumers already hold, giving two economically distinct trades one identity.
    //
    // A checkpoint taken AT the stop removes the rewind, so a resume is exactly current and no id is
    // re-minted. That is what makes `taosim -c latest` usable across a deliberate restart.
    void saveCheckpointOnShutdown();

    static constexpr std::string_view s_storeDirName{"ckpt"};
    static constexpr std::string_view s_dirExtension{".ckptd"};
    static constexpr std::string_view s_fileExtension{".ckpt"};
    static inline const std::regex s_relevantLogFilePattern{R"(((L[23]|fees).*\.log)|(.*\.csv))"};

private:
    void saveCheckpointImpl();
    void saveCheckpointMeasured();
    void cleanup();

    std::unique_ptr<struct Impl> m_impl;
};

//-------------------------------------------------------------------------

}  // namespace taosim::checkpoint

//-------------------------------------------------------------------------
