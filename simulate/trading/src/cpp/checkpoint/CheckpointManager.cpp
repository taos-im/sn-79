/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/checkpoint/CheckpointManager.hpp>

#include <taosim/checkpoint/CheckpointError.hpp>
#include <taosim/checkpoint/SimulatorState.hpp>
#include <taosim/checkpoint/serialization/SimulatorState.hpp>
#include <taosim/checkpoint/helpers.hpp>

#include <fmt/chrono.h>

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <latch>
#include <stdexcept>

//-------------------------------------------------------------------------

namespace
{

// Write `size` bytes from `data` to `path` (create/truncate), close. Raw POSIX
// with explicit error checks rather than std::ofstream so a failed write (e.g.
// disk full) THROWS instead of silently setting failbit — atomicWrite relies on
// this to avoid renaming a truncated temp file over a good checkpoint. No fsync:
// we want atomicity (via rename below), not power-loss durability, and per-file
// fsync cost multi-second stalls per checkpoint (esp. on WSL2).
void writeFile(const std::filesystem::path& path, const char* data, std::size_t size)
{
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error{fmt::format(
            "open({}) failed: {}", path.string(), std::strerror(errno))};
    }
    std::size_t written = 0;
    while (written < size) {
        const auto n = ::write(fd, data + written, size - written);
        if (n < 0) {
            const auto err = errno;
            ::close(fd);
            throw std::runtime_error{fmt::format(
                "write({}) failed: {}", path.string(), std::strerror(err))};
        }
        written += static_cast<std::size_t>(n);
    }
    ::close(fd);
}

// Atomically publish `size` bytes to `path`: write to a sibling temp file, then
// rename it over the target. POSIX rename is atomic on the same filesystem, so a
// process crash / restart mid-write leaves either the previous checkpoint or the
// complete new one, never a truncated file. Throws on any failure (and removes
// the temp) so the caller aborts rather than leaving a partial file at the real
// path. Not fsync-backed: survives process crashes, not power loss (see A/B/C
// tradeoff — atomicity was the goal, fsync's stall cost was not worth it).
void atomicWrite(const std::filesystem::path& path, const char* data, std::size_t size)
{
    auto tmp = path;
    tmp += ".tmp";
    writeFile(tmp, data, size);
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::error_code rm_ec;
        std::filesystem::remove(tmp, rm_ec);
        throw std::runtime_error{fmt::format(
            "rename({} -> {}) failed: {}", tmp.string(), path.string(), ec.message())};
    }
}

}  // namespace

//-------------------------------------------------------------------------

namespace taosim::checkpoint
{

//-------------------------------------------------------------------------

CheckpointManager::CheckpointManager(const CheckpointingDesc& desc)
    : m_simuMngr{desc.simuMngr},
      m_intervalInSteps{desc.intervalInSteps},
      m_numLastFilesToKeep{desc.numLastFilesToKeep},
      m_measureWallClockTime{desc.measureWallClockTime}
{
    if (desc.runDir.empty()) {
        throw CheckpointError{"'runDir' must be non-empty"};
    }
    if (m_intervalInSteps == 0) {
        throw CheckpointError{"'intervalInSteps' must be non-zero"};
    }

    m_dir = desc.runDir / s_storeDirName;

    m_simuMngr->stepSignal().connect([this] { saveCheckpoint(); });
}

//-------------------------------------------------------------------------

void CheckpointManager::saveCheckpoint()
{
    if (m_simuMngr->warmingUp()) return;
    if (++m_stepCounter == 0 || m_stepCounter % m_intervalInSteps != 0) return;

    fmt::println("Saving checkpoint...");

    try {
        if (m_measureWallClockTime) {
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
    if (m_simuMngr->warmingUp()) return;

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
    const auto simuTime = m_simuMngr->simulations().front()->currentTimestamp();
    
    m_latestCkptDir = m_dir / fmt::format("{}{}", simuTime, s_dirExtension);
    std::filesystem::create_directories(m_latestCkptDir);

    // Common.
    taosim::serialization::BinaryStream stream;
    msgpack::packer packer{stream};
    taosim::checkpoint::serialization::packCommon(packer, m_simuMngr);

    const auto commonCkptFile = m_latestCkptDir / fmt::format("common{}", s_fileExtension);
    atomicWrite(commonCkptFile, stream.data(), stream.size());

    // Blocks.
    std::latch latch{m_simuMngr->blockInfo().count};
    for (auto&& simulation : m_simuMngr->simulations()) {
        boost::asio::post(
            *m_simuMngr->threadPool(),
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

                    const auto blockCkptFile =
                        m_latestCkptDir / fmt::format("{}{}", simulation->blockIdx(), s_fileExtension);
                    atomicWrite(blockCkptFile, stream.data(), stream.size());
                }
                catch (const std::exception& e) {
                    fmt::println("Error saving checkpoint block: {}", e.what());
                }

                latch.count_down();
            });
    }
    latch.wait();

    cleanup();
}

//-------------------------------------------------------------------------

void CheckpointManager::saveCheckpointMeasured()
{
    using namespace std::chrono;

    auto& measurements = m_simuMngr->measurements();

    measurements.t0ckptSave = std::make_optional(high_resolution_clock::now());

    saveCheckpointImpl();

    measurements.t1ckptSave = std::make_optional(high_resolution_clock::now());

    fmt::println(
        "Took {:.4f}s",
        duration<double>(*measurements.t1ckptSave - *measurements.t0ckptSave).count());
}

//-------------------------------------------------------------------------

void CheckpointManager::cleanup()
{
    const auto dirs = ckptDirsSortedByWriteTime(m_dir);

    auto dirsToRemoveView = dirs
        | views::filter([&](auto&& f) {
            static const std::regex pattern{fmt::format("^\\d+\\{}$", s_dirExtension)};
            const auto name = f.filename();
            return std::regex_match(name.string(), pattern)
                && name != m_latestCkptDir.filename();
        })
        | views::take(std::max(0z, std::ssize(dirs) - m_numLastFilesToKeep));

    for (auto&& dir : dirsToRemoveView) {
        fs::remove_all(dir);
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::checkpoint

//-------------------------------------------------------------------------
