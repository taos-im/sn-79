// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT
//
// A small offline multi-asset run in a private temporary directory, for tests that run, checkpoint
// and resume one in process. Offline means no validator queues and no lifecycle notices, so nothing
// here can reach another engine's IPC.

#pragma once

#include <Simulation.hpp>
#include <taosim/checkpoint/CheckpointManager.hpp>
#include <taosim/checkpoint/serialization/helpers.hpp>
#include <taosim/message/serialization/MessageQueue.hpp>
#include <taosim/serialization/msgpack/common.hpp>
#include <taosim/simulation/multiasset/MultiAssetSimulationOrchestrator.hpp>

#include <fmt/format.h>
#include <msgpack.hpp>

extern "C" {
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
}

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

extern char** environ;

namespace taosim::test::multiasset_run
{

namespace fs = std::filesystem;
using Orchestrator = taosim::simulation::multiasset::MultiAssetSimulationOrchestrator;

inline constexpr uint64_t kStep = 1'000'000'000;

// Two books and two trading populations on a GBM fundamental, so a realization's state holds orders,
// trades and agent state worth restoring. The wrapper stamps the grid; these values are placeholders.
inline std::string backgroundXml()
{
    return R"(<Simulation start="0" duration="1" step="1" timescale="ns">
    <Agents>
        <MultiBookExchangeAgent name="EXCHANGE" gracePeriod="0" priceDecimals="2" volumeDecimals="4"
            baseDecimals="4" quoteDecimals="10" remoteAgentCount="1" maxLeverage="4" maxLoan="10000"
            maintenanceMargin="0.25" initialPrice="300.0" minOrderSize="0.25">
            <Books instanceCount="2" algorithm="PriceTime">
                <Processes>
                    <GBM name="fundamental" X0="300.0" mu="0.0" sigma="0.01" dt="0.0001" seed="1337"
                        updatePeriod="1000000000"/>
                    <MagneticField name="magneticfield" numRows="32" beta="0.5" alpha="20.0"
                        updatePeriod="10000000000"/>
                </Processes>
            </Books>
            <Balances>
                <Base total="100.0" maxLeverage="5" maxLoan="10000"/>
                <Quote total="5000.0" maxLeverage="5" maxLoan="10000"/>
            </Balances>
        </MultiBookExchangeAgent>
        <StylizedTraderAgent name="STYLIZED_TRADER_AGENT" instanceCount="4" exchange="EXCHANGE" debug="0"
            MFLmean="250000000" MFLstd="50000000" minOPLatency="90000000" maxOPLatency="300000000"
            opLatencyScaleRay="0.235" delayMean="1000000000" delaySTD="100000000" sigmaN="0.7" sigmaC="2.3"
            sigmaF="5.0" tau="600000000000" tauHist="50" tauF="1" volGuard="0.000001" sigmaEps="0.002"
            r_aversion="100000" HARA="2.0" regimeChangeFlag="0" acdOmega="5.0" acdAlpha="0.15"
            acdBeta="0.65" minDMD="200000000" maxDMD="1000000000" pO_alpha="0.0">
            <Balances>
                <Base total="500" maxLeverage="4" maxLoan="10000"/>
                <Quote total="150000.0" maxLeverage="4" maxLoan="10000"/>
            </Balances>
        </StylizedTraderAgent>
        <NoiseTraderAgent name="NOISE_TRADER_AGENT" instanceCount="4" exchange="EXCHANGE" debug="0"
            MFLmean="1500000000" MFLstd="167000000" minOPLatency="90000000" maxOPLatency="300000000"
            meanVolume="2.3" tau="1800000000000" balanceCoef="0.90" acdOmega="2.5" acdAlpha="0.45"
            acdBeta="0.45" acdGamma="15.0" sigmaExp="0.00000001" weight="1.0" minDMD="200000000"
            maxDMD="1000000000">
            <Balances>
                <Base total="500" maxLeverage="4" maxLoan="10000"/>
                <Quote total="150000.0" maxLeverage="4" maxLoan="10000"/>
            </Balances>
        </NoiseTraderAgent>
        <DistributedProxyAgent/>
    </Agents>
</Simulation>
)";
}

struct RunDesc
{
    uint32_t realizations{3};
    uint32_t threads{1};
    uint64_t steps{6};
    uint64_t ckptIntervalInSteps{2};
    uint64_t ckptNumLastFilesToKeep{100};
    uint64_t gracePeriod{0};
    std::string extraAttrs;
};

// One run directory under the system temp dir, removed on destruction. The run id is fixed, so the
// run root is base()/run whether the run is fresh or resumed.
class MultiAssetRun
{
public:
    MultiAssetRun(std::string_view name, RunDesc desc = {})
        : m_base{fs::temp_directory_path() / fmt::format("taosim-test-ma-{}-{}", ::getpid(), name)},
          m_desc{std::move(desc)}
    {
        fs::remove_all(m_base);
        fs::create_directories(m_base);
        std::ofstream{m_base / "background.xml"} << backgroundXml();
        std::ofstream{m_base / "wrapper.xml"} << fmt::format(
            R"(<MultiAssetSimulation id="run" threads="{}" seed="42" start="0" step="{}" duration="{}"
    gracePeriod="{}" ckptIntervalInSteps="{}" ckptNumLastFilesToKeep="{}" {}>
    <Background name="bg" path="background.xml" instanceCount="{}"/>
</MultiAssetSimulation>
)",
            m_desc.threads, kStep, m_desc.steps * kStep, m_desc.gracePeriod, m_desc.ckptIntervalInSteps,
            m_desc.ckptNumLastFilesToKeep, m_desc.extraAttrs, m_desc.realizations);
    }

    ~MultiAssetRun()
    {
        std::error_code ec;
        fs::remove_all(m_base, ec);
    }

    [[nodiscard]] uint64_t endTime() const noexcept { return m_desc.steps * kStep; }
    [[nodiscard]] fs::path root() const { return m_base / "run"; }
    [[nodiscard]] fs::path store() const { return root() / taosim::checkpoint::CheckpointManager::s_storeDirName; }
    [[nodiscard]] fs::path realizationDir(uint32_t idx) const { return root() / fmt::format("bg-{}", idx); }

    [[nodiscard]] std::unique_ptr<Orchestrator> build() const
    {
        return Orchestrator::fromConfig(m_base / "wrapper.xml", m_base);
    }

    // Every checkpoint directory, oldest grid time first.
    [[nodiscard]] std::vector<fs::path> checkpoints() const
    {
        std::vector<fs::path> dirs;
        if (!fs::exists(store())) return dirs;
        for (const auto& entry : fs::directory_iterator{store()}) {
            if (entry.is_directory() && entry.path().extension() == taosim::checkpoint::CheckpointManager::s_dirExtension) {
                dirs.push_back(entry.path());
            }
        }
        std::ranges::sort(dirs, {}, [](const fs::path& p) { return std::stoull(p.stem().string()); });
        return dirs;
    }

    [[nodiscard]] fs::path latestCheckpoint() const
    {
        const auto dirs = checkpoints();
        return dirs.empty() ? fs::path{} : dirs.back();
    }

private:
    fs::path m_base;
    RunDesc m_desc;
};

[[nodiscard]] inline fs::path commonFile(const fs::path& ckptDir)
{
    return ckptDir / fmt::format("common{}", taosim::checkpoint::CheckpointManager::s_fileExtension);
}

[[nodiscard]] inline std::string readBytes(const fs::path& path)
{
    std::ifstream ifs{path, std::ios::binary};
    return {std::istreambuf_iterator<char>{ifs}, std::istreambuf_iterator<char>{}};
}

[[nodiscard]] inline msgpack::object_handle readCommon(const fs::path& ckptDir)
{
    const auto bytes = readBytes(commonFile(ckptDir));
    return msgpack::unpack(bytes.data(), bytes.size());
}

// Rewrites the common file keeping every entry but the format keys, then packs `format` (the key
// count and a packer callback) after them. With no format this is the common file a checkpoint had
// before formats were declared.
template<typename PackFormat>
void rewriteCommon(const fs::path& ckptDir, uint32_t formatPairs, PackFormat&& packFormat)
{
    const auto oh = readCommon(ckptDir);
    const auto& map = oh.get().via.map;
    std::vector<const msgpack::object_kv*> kept;
    for (uint32_t i = 0; i < map.size; ++i) {
        const auto key = map.ptr[i].key.as<std::string>();
        if (key == "formatVersion" || key == "capabilities") continue;
        kept.push_back(&map.ptr[i]);
    }
    taosim::serialization::BinaryStream stream;
    msgpack::packer pk{stream};
    pk.pack_map(static_cast<uint32_t>(kept.size()) + formatPairs);
    for (const auto* kv : kept) {
        pk.pack(kv->key);
        pk.pack(kv->val);
    }
    packFormat(pk);
    std::ofstream{commonFile(ckptDir), std::ios::binary | std::ios::trunc}.write(stream.data(), stream.size());
}

inline void stripFormat(const fs::path& ckptDir)
{
    rewriteCommon(ckptDir, 0, [](auto&) {});
}

// One test of this binary run in a child process, so that a run whose outcome is compared with another's
// is the first of its process, or so that it can open what this process must not. The child inherits
// this environment without any GTEST_ variable or any name in `env`, then gets `env`; its output goes
// to `log`. A child still running at `limit` is killed.
struct ChildExit
{
    std::optional<int> status;
    std::string log;

    [[nodiscard]] bool clean() const { return status && WIFEXITED(*status) && WEXITSTATUS(*status) == 0; }
};

inline ChildExit runTestInChild(
    std::string_view test,
    const std::vector<std::pair<std::string, std::string>>& env,
    const fs::path& log,
    std::chrono::seconds limit = std::chrono::seconds{180})
{
    const auto self = fs::read_symlink("/proc/self/exe").string();
    std::vector<std::string> args{self, fmt::format("--gtest_filter={}", test), "--gtest_also_run_disabled_tests"};
    std::vector<std::string> vars;
    for (char** e = environ; *e != nullptr; ++e) {
        const std::string_view entry{*e};
        const auto overridden = std::ranges::any_of(env, [&](const auto& kv) {
            return entry.starts_with(kv.first + "=");
        });
        if (entry.starts_with("GTEST_") || overridden) continue;
        vars.emplace_back(entry);
    }
    for (const auto& [name, value] : env) vars.push_back(fmt::format("{}={}", name, value));
    auto pointers = [](std::vector<std::string>& strings) {
        std::vector<char*> out;
        for (auto& s : strings) out.push_back(s.data());
        out.push_back(nullptr);
        return out;
    };
    auto argv = pointers(args);
    auto envp = pointers(vars);

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
    pid_t pid{-1};
    const auto rc = ::posix_spawn(&pid, self.c_str(), &actions, nullptr, argv.data(), envp.data());
    posix_spawn_file_actions_destroy(&actions);

    ChildExit exit;
    if (rc == 0) {
        const auto deadline = std::chrono::steady_clock::now() + limit;
        int status{};
        while (std::chrono::steady_clock::now() < deadline) {
            if (::waitpid(pid, &status, WNOHANG) == pid) {
                exit.status = status;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{20});
        }
        if (!exit.status) {
            ::kill(pid, SIGKILL);
            ::waitpid(pid, nullptr, 0);
        }
    }
    exit.log = readBytes(log);
    return exit;
}

// Every realization's block state, packed as a checkpoint packs it.
[[nodiscard]] inline std::vector<std::string> packedBlocks(const Orchestrator& orchestrator)
{
    std::vector<std::string> blocks;
    for (const auto& sim : orchestrator.checkpointBlocks()) {
        taosim::serialization::BinaryStream stream;
        msgpack::packer pk{stream};
        taosim::checkpoint::serialization::packBlock(pk, *sim);
        blocks.emplace_back(stream.data(), stream.size());
    }
    return blocks;
}

}  // namespace taosim::test::multiasset_run
