/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <filesystem>
#include <regex>

//-------------------------------------------------------------------------

namespace taosim::simulation
{

class SimulationManager;

}  // namespace taosim::simulation

//-------------------------------------------------------------------------

namespace taosim::checkpoint
{

//-------------------------------------------------------------------------

struct CheckpointingDesc
{
    simulation::SimulationManager* simuMngr;
    std::filesystem::path runDir;
    size_t intervalInSteps{};
    ssize_t numLastFilesToKeep{};
    bool measureWallClockTime{};
};

//-------------------------------------------------------------------------

class CheckpointManager
{
public:
    explicit CheckpointManager(const CheckpointingDesc& desc);

    [[nodiscard]] auto&& stepCounter(this auto&& self) noexcept { return self.m_stepCounter; }

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

    simulation::SimulationManager* m_simuMngr;
    std::filesystem::path m_dir;
    size_t m_intervalInSteps;
    ssize_t m_numLastFilesToKeep;
    bool m_measureWallClockTime;
    ssize_t m_stepCounter{-1};
    std::filesystem::path m_latestCkptDir;

    friend class simulation::SimulationManager;
};

//-------------------------------------------------------------------------

}  // namespace taosim::checkpoint

//-------------------------------------------------------------------------
