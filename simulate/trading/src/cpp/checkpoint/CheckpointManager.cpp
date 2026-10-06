/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/checkpoint/CheckpointManager.hpp>

#include <Simulation.hpp>
#include <taosim/checkpoint/CheckpointError.hpp>
#include <taosim/checkpoint/helpers.hpp>
#include <taosim/checkpoint/serialization/helpers.hpp>
#include <taosim/message/serialization/MessageQueue.hpp>
#include <taosim/serialization/msgpack/common.hpp>

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <fmt/chrono.h>
#include <msgpack.hpp>

#include <atomic>
#include <chrono>
#include <latch>

//-------------------------------------------------------------------------

namespace taosim::checkpoint
{

//-------------------------------------------------------------------------
// Hidden run state. The dir layout, retention bound, interval and the (local)
// save-timing all live here; the run-mode state comes through m_source.

struct Impl
{
    CheckpointSource* source;
    std::filesystem::path dir;
    size_t intervalInSteps;
    ssize_t numLastFilesToKeep;
    bool measureWallClockTime;
    ssize_t stepCounter{-1};
    std::filesystem::path latestCkptDir;
};

//-------------------------------------------------------------------------

CheckpointManager::CheckpointManager(const CheckpointingDesc& desc)
    : m_impl{std::make_unique<Impl>(Impl{
        .source = desc.source,
        .intervalInSteps = desc.intervalInSteps,
        .numLastFilesToKeep = desc.numLastFilesToKeep,
        .measureWallClockTime = desc.measureWallClockTime})}
{
    if (desc.runDir.empty()) {
        throw CheckpointError{"'runDir' must be non-empty"};
    }
    if (m_impl->intervalInSteps == 0) {
        throw CheckpointError{"'intervalInSteps' must be non-zero"};
    }

    m_impl->dir = desc.runDir / s_storeDirName;
}

//-------------------------------------------------------------------------
// Out-of-line so Impl is complete at the point of destruction (PIMPL).

CheckpointManager::~CheckpointManager() = default;

//-------------------------------------------------------------------------

ssize_t& CheckpointManager::stepCounter() noexcept
{
    return m_impl->stepCounter;
}

//-------------------------------------------------------------------------

void CheckpointManager::saveCheckpoint()
{
    if (m_impl->source->warmingUp()) return;
    if (++m_impl->stepCounter == 0 || m_impl->stepCounter % m_impl->intervalInSteps != 0) return;

    fmt::println("Saving checkpoint...");

    try {
        if (m_impl->measureWallClockTime) {
            saveCheckpointMeasured();
        } else {
            saveCheckpointImpl();
        }
        fmt::println("Checkpoint saved successfully.");
    }
    catch (const std::exception& e) {
        fmt::println("Error saving checkpoint at {}", e.what());
    }
}

//-------------------------------------------------------------------------

void CheckpointManager::saveCheckpointOnShutdown()
{
    // Warm-up has no meaningful state to resume into, and the interval save skips it for the same
    // reason. The step counter is deliberately NOT touched: this is out-of-band, and advancing it
    // would shift the phase of every later interval save on a resumed run.
    if (m_impl->source->warmingUp()) return;

    fmt::println("Saving checkpoint on shutdown...");

    try {
        // Never the measured variant: its wall-clock timing is a benchmark of the periodic path and
        // this one happens once, while a supervisor is already counting down to SIGKILL.
        saveCheckpointImpl();
        fmt::println("Shutdown checkpoint saved successfully.");
    }
    catch (const std::exception& e) {
        // Never rethrow: a failed checkpoint must not turn a clean stop into a crash. The next
        // resume falls back to the last periodic checkpoint, which is the behaviour without this.
        fmt::println("Error saving shutdown checkpoint: {}", e.what());
    }
}

//-------------------------------------------------------------------------

void CheckpointManager::saveCheckpointImpl()
{
    const auto blocks = m_impl->source->checkpointBlocks();
    const auto simuTime = blocks.front()->currentTimestamp();

    m_impl->latestCkptDir = m_impl->dir / fmt::format("{}{}", simuTime, s_dirExtension);
    std::filesystem::create_directories(m_impl->latestCkptDir);

    // Common (run-mode specific) state.
    m_impl->source->writeCommonCheckpoint(
        m_impl->latestCkptDir / fmt::format("common{}", s_fileExtension));

    // Blocks (generic), written in parallel on the source's executor.
    std::atomic_flag blockWriteFailed;
    std::latch latch{std::ssize(blocks)};
    for (auto&& simulation : blocks) {
        boost::asio::post(
            m_impl->source->checkpointExecutor(),
            [&] {
                // atomicWrite throws on failure; catch here so a block-write
                // error is logged rather than escaping the thread-pool task
                // (which would std::terminate), and — critically — so the latch
                // is ALWAYS counted down. Skipping count_down() would leave
                // latch.wait() below blocked forever and hang the simulation.
                try {
                    taosim::serialization::BinaryStream stream;
                    msgpack::packer packer{stream};
                    taosim::checkpoint::serialization::packBlock(packer, *simulation);

                    const auto blockCkptFile = m_impl->latestCkptDir
                        / fmt::format("{}{}", simulation->blockIdx(), s_fileExtension);
                    atomicWrite(blockCkptFile, {stream.data(), stream.size()});
                }
                catch (const std::exception& e) {
                    fmt::println("Error saving checkpoint block: {}", e.what());
                    blockWriteFailed.test_and_set();
                }

                latch.count_down();
            });
    }
    latch.wait();

    if (blockWriteFailed.test()) {
        // A checkpoint missing a block file cannot be restored from; drop it and
        // keep the older intact checkpoints instead of cleaning them up.
        std::error_code ec;
        fs::remove_all(m_impl->latestCkptDir, ec);
        throw CheckpointError{ec
            ? fmt::format(
                "incomplete checkpoint '{}' could not be removed ({}) — restore may pick it",
                m_impl->latestCkptDir.string(), ec.message())
            : fmt::format(
                "incomplete checkpoint '{}' discarded", m_impl->latestCkptDir.string())};
    }

    cleanup();
}

//-------------------------------------------------------------------------

void CheckpointManager::saveCheckpointMeasured()
{
    using namespace std::chrono;

    const auto t0 = high_resolution_clock::now();
    saveCheckpointImpl();
    const auto t1 = high_resolution_clock::now();

    fmt::println("Took {:.4f}s", duration<double>(t1 - t0).count());
}

//-------------------------------------------------------------------------

void CheckpointManager::cleanup()
{
    const auto dirs = ckptDirsSortedByWriteTime(m_impl->dir);

    auto dirsToRemoveView = dirs
        | ranges::views::filter([&](auto&& f) {
            static const std::regex pattern{fmt::format("^\\d+\\{}$", s_dirExtension)};
            const auto name = f.filename();
            return std::regex_match(name.string(), pattern)
                && name != m_impl->latestCkptDir.filename();
        })
        | ranges::views::take(std::max(0z, std::ssize(dirs) - m_impl->numLastFilesToKeep));

    for (auto&& dir : dirsToRemoveView) {
        fs::remove_all(dir);
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::checkpoint

//-------------------------------------------------------------------------
