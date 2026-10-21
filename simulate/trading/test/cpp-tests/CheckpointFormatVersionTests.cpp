// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT
//
// A checkpoint declares its format in its common file: a version, and the capabilities its content
// needs from a reader. A checkpoint written before the declaration carries neither key and loads
// exactly as it did, on the single-market path and the multi-asset one. A newer version, or a
// capability the binary lacks, is refused at load in one line naming both sides, before anything is
// restored or set aside.
//
// The single-market engine opens its validator queues as it is built, so its runs go to a child
// process with private queue names, never the production ones.

#include "MultiAssetRunFixture.hpp"

#include <taosim/checkpoint/CheckpointError.hpp>
#include <taosim/checkpoint/CheckpointFormat.hpp>
#include <taosim/ipc/PosixMessageQueue.hpp>
#include <taosim/serialization/msgpack/utils.hpp>
#include <taosim/simulation/SimulationManager.hpp>
#include <taosim/simulation/multiasset/helpers.hpp>

#include <gtest/gtest.h>

#include <cstdlib>

//-------------------------------------------------------------------------

using namespace taosim::test::multiasset_run;
using taosim::checkpoint::CheckpointError;
using taosim::checkpoint::kCapabilitiesKey;
using taosim::checkpoint::kFormatVersion;
using taosim::checkpoint::kFormatVersionKey;
using taosim::checkpoint::kLegacyFormatVersion;
using Packer = msgpack::packer<taosim::serialization::BinaryStream>;

//-------------------------------------------------------------------------

namespace
{

// A common map with the two entries every checkpoint has, then `formatPairs` more from `pack`.
template<typename Pack>
msgpack::object_handle commonWith(uint32_t formatPairs, Pack&& pack)
{
    taosim::serialization::BinaryStream stream;
    Packer pk{stream};
    pk.pack_map(2 + formatPairs);
    pk.pack("timestamp");
    pk.pack(uint64_t{5'000'000'000});
    pk.pack("logFileSizes");
    pk.pack_map(0);
    pack(pk);
    return msgpack::unpack(stream.data(), stream.size());
}

auto declaring(uint64_t version, std::vector<std::string> capabilities)
{
    return [version, capabilities = std::move(capabilities)](Packer& pk) {
        pk.pack(kFormatVersionKey);
        pk.pack(version);
        pk.pack(kCapabilitiesKey);
        pk.pack(capabilities);
    };
}

[[nodiscard]] std::string refusalOf(auto&& load)
{
    try {
        load();
    } catch (const CheckpointError& e) {
        return e.what();
    }
    return {};
}

[[nodiscard]] std::string newerVersionLine()
{
    return fmt::format("format version {}", kFormatVersion + 1);
}

[[nodiscard]] std::string ownVersionLine()
{
    return fmt::format("format version {}", kFormatVersion);
}

}  // namespace

//-------------------------------------------------------------------------
// The declaration itself.

TEST(CheckpointFormat, ACommonFileWithoutTheKeysReadsAsTheLegacyFormat)
{
    const auto common = commonWith(0, [](Packer&) {});

    const auto format = taosim::checkpoint::requireReadableFormat(common.get());

    EXPECT_EQ(format.version, kLegacyFormatVersion);
    EXPECT_TRUE(format.capabilities.empty());
}

TEST(CheckpointFormat, TheWritersOwnDeclarationReadsBack)
{
    const auto common = commonWith(taosim::checkpoint::kFormatKeyCount, [](Packer& pk) {
        taosim::checkpoint::packFormat(pk, {});
    });

    const auto format = taosim::checkpoint::requireReadableFormat(common.get());

    EXPECT_EQ(format.version, kFormatVersion);
    EXPECT_GT(format.version, kLegacyFormatVersion);
    EXPECT_TRUE(format.capabilities.empty());
}

TEST(CheckpointFormat, ANewerVersionIsRefusedInOneLineNamingBothVersions)
{
    const auto common = commonWith(2, declaring(kFormatVersion + 1, {}));

    const auto refusal = refusalOf([&] { (void)taosim::checkpoint::requireReadableFormat(common.get()); });

    ASSERT_FALSE(refusal.empty()) << "a newer checkpoint was accepted";
    EXPECT_TRUE(refusal.contains(newerVersionLine())) << refusal;
    EXPECT_TRUE(refusal.contains(ownVersionLine())) << refusal;
    EXPECT_FALSE(refusal.contains('\n')) << refusal;
}

TEST(CheckpointFormat, AnUnknownCapabilityIsRefusedNamingItAndTheBinarysVersion)
{
    const auto common = commonWith(2, declaring(kFormatVersion, {"derivatives"}));

    const auto refusal = refusalOf([&] { (void)taosim::checkpoint::requireReadableFormat(common.get()); });

    ASSERT_FALSE(refusal.empty()) << "a checkpoint needing an unknown capability was accepted";
    EXPECT_TRUE(refusal.contains("'derivatives'")) << refusal;
    EXPECT_TRUE(refusal.contains(ownVersionLine())) << refusal;
    EXPECT_FALSE(refusal.contains('\n')) << refusal;
}

TEST(CheckpointFormat, AVersionThatIsNotAnUnsignedIntegerIsRefused)
{
    const auto common = commonWith(1, [](Packer& pk) {
        pk.pack(kFormatVersionKey);
        pk.pack("1");
    });

    EXPECT_THROW((void)taosim::checkpoint::requireReadableFormat(common.get()), CheckpointError);
}

TEST(CheckpointFormat, CapabilitiesThatAreNotAListOfNamesAreRefused)
{
    const auto common = commonWith(2, [](Packer& pk) {
        pk.pack(kFormatVersionKey);
        pk.pack(kFormatVersion);
        pk.pack(kCapabilitiesKey);
        pk.pack(uint64_t{3});
    });

    EXPECT_THROW((void)taosim::checkpoint::requireReadableFormat(common.get()), CheckpointError);
}

//-------------------------------------------------------------------------
// The multi-asset path, in process: offline, so it opens no queue.

TEST(CheckpointFormat, AMultiAssetCheckpointDeclaresTheFormatAndRoundTrips)
{
    const MultiAssetRun run{"format-roundtrip"};
    run.build()->run();
    const auto ckpt = run.latestCheckpoint();
    ASSERT_FALSE(ckpt.empty()) << "the run wrote no checkpoint";
    const uint64_t ckptTime = std::stoull(ckpt.stem().string());

    const auto common = readCommon(ckpt);
    const auto version = taosim::serialization::msgpackFindMap<uint64_t>(common.get(), kFormatVersionKey);
    ASSERT_TRUE(version.has_value()) << "the common file declares no version";
    EXPECT_EQ(*version, kFormatVersion);
    const auto format = taosim::checkpoint::readFormat(common.get());
    EXPECT_TRUE(format.capabilities.empty()) << "a plain checkpoint needs no capability";

    const auto resumed = Orchestrator::fromCheckpoint(ckpt.string());
    for (const auto& sim : resumed->checkpointBlocks()) {
        EXPECT_EQ(sim->currentTimestamp(), ckptTime);
    }
    resumed->run();
    for (const auto& sim : resumed->checkpointBlocks()) {
        EXPECT_GE(sim->currentTimestamp(), run.endTime());
    }
}

TEST(CheckpointFormat, AMultiAssetCheckpointWithoutTheKeysLoadsExactlyAsWithThem)
{
    const MultiAssetRun run{"format-legacy"};
    run.build()->run();
    const auto ckpt = run.latestCheckpoint();
    ASSERT_FALSE(ckpt.empty()) << "the run wrote no checkpoint";

    const auto declared = packedBlocks(*Orchestrator::fromCheckpoint(ckpt.string()));
    stripFormat(ckpt);
    ASSERT_FALSE(taosim::serialization::msgpackFindMapObj(readCommon(ckpt).get(), kFormatVersionKey));
    const auto legacy = packedBlocks(*Orchestrator::fromCheckpoint(ckpt.string()));

    ASSERT_EQ(legacy.size(), declared.size());
    for (size_t i = 0; i < legacy.size(); ++i) {
        EXPECT_FALSE(legacy[i].empty());
        EXPECT_TRUE(legacy[i] == declared[i]) << "realization " << i << " restored differently";
    }
}

TEST(CheckpointFormat, AMultiAssetCheckpointOfANewerVersionIsRefusedBeforeAnythingIsSetAside)
{
    const MultiAssetRun run{"format-newer"};
    run.build()->run();
    const auto ckpt = run.latestCheckpoint();
    ASSERT_FALSE(ckpt.empty()) << "the run wrote no checkpoint";
    rewriteCommon(ckpt, 2, declaring(kFormatVersion + 1, {}));
    // A log file the checkpoint does not record, which a load sets aside before it rebuilds the run.
    const auto stray = run.realizationDir(0) / "L3_stray.log";
    std::ofstream{stray} << "x";

    const auto refusal = refusalOf([&] { (void)Orchestrator::fromCheckpoint(ckpt.string()); });

    ASSERT_FALSE(refusal.empty()) << "a newer checkpoint was loaded";
    EXPECT_TRUE(refusal.contains(newerVersionLine())) << refusal;
    EXPECT_TRUE(refusal.contains(ownVersionLine())) << refusal;
    EXPECT_TRUE(fs::exists(stray)) << "the refused load set a log file aside";
}

TEST(CheckpointFormat, AMultiAssetCheckpointNeedingAnUnknownCapabilityIsRefusedBeforeAnythingIsSetAside)
{
    const MultiAssetRun run{"format-capability"};
    run.build()->run();
    const auto ckpt = run.latestCheckpoint();
    ASSERT_FALSE(ckpt.empty()) << "the run wrote no checkpoint";
    rewriteCommon(ckpt, 2, declaring(kFormatVersion, {"derivatives"}));
    const auto stray = run.realizationDir(0) / "L3_stray.log";
    std::ofstream{stray} << "x";

    const auto refusal = refusalOf([&] { (void)Orchestrator::fromCheckpoint(ckpt.string()); });

    ASSERT_FALSE(refusal.empty()) << "a checkpoint needing an unknown capability was loaded";
    EXPECT_TRUE(refusal.contains("'derivatives'")) << refusal;
    EXPECT_TRUE(fs::exists(stray)) << "the refused load set a log file aside";
}

//-------------------------------------------------------------------------
// The single-market loader on a run directory with no config.xml, so no load of it can get as far
// as building an engine. What it does before that is what a refusal must not do.

namespace
{

struct SingleMarketCheckpointDir
{
    fs::path base{fs::temp_directory_path() / fmt::format("taosim-test-sm-format-{}", ::getpid())};
    fs::path run{base / "run"};
    fs::path ckpt{run / taosim::checkpoint::CheckpointManager::s_storeDirName / "5000000000.ckptd"};
    fs::path stray{run / "L3_stray.log"};

    template<typename Pack>
    SingleMarketCheckpointDir(uint32_t formatPairs, Pack&& pack)
    {
        fs::remove_all(base);
        fs::create_directories(ckpt);
        const auto oh = commonWith(formatPairs, std::forward<Pack>(pack));
        taosim::serialization::BinaryStream stream;
        Packer pk{stream};
        pk.pack(oh.get());
        std::ofstream{commonFile(ckpt), std::ios::binary}.write(stream.data(), stream.size());
        std::ofstream{stray} << "x";
    }

    ~SingleMarketCheckpointDir()
    {
        std::error_code ec;
        fs::remove_all(base, ec);
    }
};

}  // namespace

TEST(CheckpointFormat, ASingleMarketCheckpointOfANewerVersionIsRefusedBeforeAnythingIsSetAside)
{
    const SingleMarketCheckpointDir dir{2, declaring(kFormatVersion + 1, {})};
    ASSERT_FALSE(fs::exists(dir.run / "config.xml"));

    const auto refusal = refusalOf([&] {
        (void)taosim::simulation::SimulationManager::fromCheckpoint(dir.ckpt.string());
    });

    EXPECT_TRUE(refusal.contains(newerVersionLine())) << refusal;
    EXPECT_TRUE(refusal.contains(ownVersionLine())) << refusal;
    EXPECT_TRUE(fs::exists(dir.stray)) << "the refused load set a log file aside";
}

TEST(CheckpointFormat, ASingleMarketCheckpointWithoutTheKeysGetsPastTheCheckAsItDid)
{
    const SingleMarketCheckpointDir dir{0, [](Packer&) {}};
    ASSERT_FALSE(fs::exists(dir.run / "config.xml"));

    const auto refusal = refusalOf([&] {
        (void)taosim::simulation::SimulationManager::fromCheckpoint(dir.ckpt.string());
    });

    EXPECT_FALSE(refusal.contains("format version")) << refusal;
    EXPECT_FALSE(fs::exists(dir.stray)) << "the load stopped before setting the stray log aside";
}

//-------------------------------------------------------------------------
// The single-market writer and loader end to end, in child processes.

namespace
{

constexpr std::string_view kChildTest = "CheckpointFormatChild.DISABLED_RunsOrLoadsInItsEnvironment";
constexpr std::string_view kModeVar = "TAOSIM_TEST_CKPT_MODE";
constexpr std::string_view kSourceVar = "TAOSIM_TEST_CKPT_SOURCE";
constexpr std::string_view kTargetVar = "TAOSIM_TEST_CKPT_TARGET";
constexpr std::string_view kIpcSuffixVar = "TAOSIM_IPC_SUFFIX";

// Runs this binary's child test with `mode`: "run" runs the config `source` under the base directory
// `target`; "dump" loads the checkpoint `source` and writes every block's packed state to `target`.
ChildExit runChild(std::string_view mode, const fs::path& source, const fs::path& target, const fs::path& log)
{
    const auto suffix = fmt::format("-test-ckpt-format-{}", ::getpid());
    auto exit = runTestInChild(
        kChildTest,
        {{std::string{kIpcSuffixVar}, suffix},
         {std::string{kModeVar}, std::string{mode}},
         {std::string{kSourceVar}, source.string()},
         {std::string{kTargetVar}, target.string()}},
        log);
    for (const auto* base : {"taosim-req", "taosim-res"}) {
        taosim::ipc::PosixMessageQueue::remove(fmt::format("/{}{}", base, suffix));
    }
    return exit;
}

struct TempBase
{
    fs::path path;

    explicit TempBase(std::string_view name)
        : path{fs::temp_directory_path() / fmt::format("taosim-test-{}-{}", name, ::getpid())}
    {
        fs::remove_all(path);
        fs::create_directories(path);
    }

    ~TempBase()
    {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

std::string singleMarketXml()
{
    auto xml = backgroundXml();
    const std::string_view head = R"(<Simulation start="0" duration="1" step="1" timescale="ns">)";
    xml.replace(
        xml.find(head), head.size(),
        fmt::format(
            R"(<Simulation id="run" start="0" duration="{}" step="{}" timescale="ns" rngSeed="42" )"
            R"(ckptIntervalInSteps="2" ckptNumLastFilesToKeep="100">)",
            6 * kStep, kStep));
    return xml;
}

}  // namespace

TEST(CheckpointFormat, ASingleMarketCheckpointDeclaresTheFormatAndLoadsTheSameWithoutTheKeys)
{
    const TempBase temp{"sm-e2e"};
    const auto& base = temp.path;
    std::ofstream{base / "config.xml"} << singleMarketXml();

    const auto ran = runChild("run", base / "config.xml", base / "logs", base / "run.log");
    ASSERT_TRUE(ran.clean()) << ran.log;
    const auto store = base / "logs" / "run" / taosim::checkpoint::CheckpointManager::s_storeDirName;
    ASSERT_TRUE(fs::exists(store)) << ran.log;
    fs::path ckpt;
    for (const auto& entry : fs::directory_iterator{store}) {
        if (ckpt.empty() || std::stoull(entry.path().stem().string()) > std::stoull(ckpt.stem().string())) {
            ckpt = entry.path();
        }
    }
    ASSERT_FALSE(ckpt.empty()) << ran.log;

    const auto common = readCommon(ckpt);
    const auto version = taosim::serialization::msgpackFindMap<uint64_t>(common.get(), kFormatVersionKey);
    ASSERT_TRUE(version.has_value()) << "the common file declares no version";
    EXPECT_EQ(*version, kFormatVersion);
    EXPECT_TRUE(taosim::checkpoint::readFormat(common.get()).capabilities.empty());

    const auto declared = runChild("dump", ckpt, base / "declared.blocks", base / "declared.log");
    ASSERT_TRUE(declared.clean()) << declared.log;
    stripFormat(ckpt);
    const auto legacy = runChild("dump", ckpt, base / "legacy.blocks", base / "legacy.log");
    ASSERT_TRUE(legacy.clean()) << legacy.log;
    const auto declaredBlocks = readBytes(base / "declared.blocks");
    EXPECT_FALSE(declaredBlocks.empty());
    EXPECT_TRUE(readBytes(base / "legacy.blocks") == declaredBlocks) << "the versionless checkpoint restored differently";

    rewriteCommon(ckpt, 2, declaring(kFormatVersion + 1, {}));
    const auto newer = runChild("dump", ckpt, base / "newer.blocks", base / "newer.log");
    EXPECT_FALSE(newer.clean()) << "a newer checkpoint was loaded\n" << newer.log;
    EXPECT_TRUE(newer.log.contains(newerVersionLine())) << newer.log;
    EXPECT_FALSE(fs::exists(base / "newer.blocks"));
}

//-------------------------------------------------------------------------
// The child side, run only when re-executed by the test above. It refuses to run without a private
// queue suffix, so it can never open another engine's queues.

TEST(CheckpointFormatChild, DISABLED_RunsOrLoadsInItsEnvironment)
{
    const char* mode = std::getenv(kModeVar.data());
    const char* source = std::getenv(kSourceVar.data());
    const char* target = std::getenv(kTargetVar.data());
    const char* suffix = std::getenv(kIpcSuffixVar.data());
    if (mode == nullptr || source == nullptr || target == nullptr || suffix == nullptr || *suffix == '\0') {
        GTEST_SKIP() << "runs only as the child of a CheckpointFormat test";
    }
    ASSERT_TRUE(taosim::simulation::SimulationManager::s_validatorReqMessageQueueName.ends_with(suffix));
    ASSERT_TRUE(taosim::simulation::SimulationManager::s_validatorResMessageQueueName.ends_with(suffix));
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    if (std::string_view{mode} == "run") {
        taosim::simulation::multiasset::makeSimulationOrchestrator(source, target)->run();
        return;
    }
    ASSERT_EQ(std::string_view{mode}, "dump");
    const auto mngr = taosim::simulation::SimulationManager::fromCheckpoint(source);
    taosim::serialization::BinaryStream stream;
    msgpack::packer pk{stream};
    pk.pack_array(static_cast<uint32_t>(mngr->simulations().size()));
    for (const auto& sim : mngr->simulations()) {
        taosim::checkpoint::serialization::packBlock(pk, *sim);
    }
    std::ofstream{target, std::ios::binary}.write(stream.data(), stream.size());
}

//-------------------------------------------------------------------------
