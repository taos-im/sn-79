/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
// The engine's worker count can be set per host: TAOSIM_THREADS, when a positive integer, takes the place of the
// wrapper's threads attribute. How many workers step the realizations does not change what a seeded run publishes, so
// a test run at the shipped count covers a host that sets another.
#include "MultiAssetRunFixture.hpp"
#include <taosim/simulation/multiasset/helpers.hpp>

#include <gtest/gtest.h>
#include <msgpack.hpp>

#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

//-------------------------------------------------------------------------

using namespace taosim::test::multiasset_run;
using taosim::simulation::multiasset::threadsFromEnvironment;

namespace
{

constexpr std::string_view kChildTest = "EngineThreadsChild.DISABLED_RunsAndPacksEveryRealization";
constexpr std::string_view kOutVar = "TAOSIM_TEST_THREADS_OUT";
constexpr std::string_view kHostThreadsVar = "TAOSIM_THREADS";
// The count the child sets as its host's, under a name of its own: the environment below clears the host's variable in
// every process of this binary, the child's too, before any test runs.
constexpr std::string_view kChildThreadsVar = "TAOSIM_TEST_THREADS_COUNT";

// The tests of this binary run at the worker counts their configs name, so a host's TAOSIM_THREADS is cleared before
// the first of them, and with it every child a test starts inherits none.
class NoHostThreads : public testing::Environment
{
public:
    void SetUp() override { ::unsetenv(kHostThreadsVar.data()); }
};

[[maybe_unused]] testing::Environment* const s_noHostThreads = testing::AddGlobalTestEnvironment(new NoHostThreads);

// Every realization's packed state at the end of a seeded run in a child process, with TAOSIM_THREADS as given.
[[nodiscard]] std::vector<std::string> runInChild(std::string_view name, std::string_view threads)
{
    const auto dir = fs::temp_directory_path() / fmt::format("taosim-test-threads-{}-{}", ::getpid(), name);
    fs::remove_all(dir);
    fs::create_directories(dir);
    const auto exit = runTestInChild(
        kChildTest,
        {{std::string{kOutVar}, (dir / "blocks").string()}, {std::string{kChildThreadsVar}, std::string{threads}}},
        dir / "child.log");
    std::vector<std::string> blocks;
    if (exit.clean()) {
        const auto bytes = readBytes(dir / "blocks");
        msgpack::unpack(bytes.data(), bytes.size()).get().convert(blocks);
    } else {
        ADD_FAILURE() << exit.log;
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
    return blocks;
}

// Sets `name` to `value`, or unsets it when `value` is null, and restores what it held on leaving the scope.
class ScopedEnv
{
public:
    ScopedEnv(const char* name, const char* value) : m_name{name}
    {
        if (const char* previous = std::getenv(name)) m_previous = previous;
        if (value != nullptr) {
            ::setenv(name, value, 1);
        } else {
            ::unsetenv(name);
        }
    }
    ~ScopedEnv()
    {
        if (m_previous) {
            ::setenv(m_name, m_previous->c_str(), 1);
        } else {
            ::unsetenv(m_name);
        }
    }
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    const char* m_name;
    std::optional<std::string> m_previous;
};

}  // namespace

//-------------------------------------------------------------------------

TEST(EngineThreads, APositiveIntegerIsTheHostsRequestAndAnythingElseLeavesTheConfigs)
{
    EXPECT_EQ(threadsFromEnvironment("16"), std::optional<uint32_t>{16});
    EXPECT_EQ(threadsFromEnvironment("1"), std::optional<uint32_t>{1});
    EXPECT_EQ(threadsFromEnvironment(nullptr), std::nullopt) << "unset";
    EXPECT_EQ(threadsFromEnvironment(""), std::nullopt) << "empty";
    EXPECT_EQ(threadsFromEnvironment("0"), std::nullopt) << "0 is no count";
    EXPECT_EQ(threadsFromEnvironment("16 "), std::nullopt);
    EXPECT_EQ(threadsFromEnvironment("-4"), std::nullopt);
    EXPECT_EQ(threadsFromEnvironment("eight"), std::nullopt);
}

TEST(EngineThreads, TheHostsCountSetsTheWorkersWithinTheRealizationCount)
{
    const ScopedEnv none{kHostThreadsVar.data(), nullptr};
    const MultiAssetRun run{"override", {.realizations = 3, .threads = 1, .ckptIntervalInSteps = 0}};
    EXPECT_EQ(run.build()->allocation().workerCount, 1u) << "without the variable, the config's threads";
    {
        const ScopedEnv env{kHostThreadsVar.data(), "2"};
        EXPECT_EQ(run.build()->allocation().workerCount, 2u) << "the host's count";
    }
    {
        const ScopedEnv env{kHostThreadsVar.data(), "16"};
        EXPECT_EQ(run.build()->allocation().workerCount, 3u) << "never more workers than realizations";
    }
    {
        const ScopedEnv env{kHostThreadsVar.data(), "x"};
        EXPECT_EQ(run.build()->allocation().workerCount, 1u) << "a value that is no count leaves the config's";
    }
}

// Each run is the first of its own process, so the runs differ only in their worker count: one worker steps the
// realizations in turn, two split them unevenly, three give each its own.
TEST(EngineThreads, ASeededRunPublishesTheSameStateWhateverItsWorkerCount)
{
    const auto one = runInChild("one", "1");
    ASSERT_EQ(one.size(), 3u);
    EXPECT_FALSE(one[0] == one[1]) << "the realizations draw their own streams, so the comparison below has content";
    for (const auto* threads : {"2", "3"}) {
        const auto other = runInChild(threads, threads);
        ASSERT_EQ(other.size(), 3u);
        for (size_t i = 0; i < one.size(); ++i) {
            EXPECT_TRUE(one[i] == other[i])
                << "realization " << i << " ended differently on " << threads << " workers than on 1";
        }
    }
}

// The child side of the test above, run only when re-executed by it.
TEST(EngineThreadsChild, DISABLED_RunsAndPacksEveryRealization)
{
    const char* out = std::getenv(kOutVar.data());
    const char* threads = std::getenv(kChildThreadsVar.data());
    if (out == nullptr || threads == nullptr) {
        GTEST_SKIP() << "runs only as the child of an EngineThreads test";
    }
    const ScopedEnv host{kHostThreadsVar.data(), threads};
    // One name for every child, so that the runs differ only in their worker count; 60 steps so the traders trade.
    const MultiAssetRun run{"child", {.realizations = 3, .threads = 1, .steps = 60, .ckptIntervalInSteps = 0}};
    const auto orchestrator = run.build();
    orchestrator->run();
    taosim::serialization::BinaryStream stream;
    msgpack::packer pk{stream};
    pk.pack(packedBlocks(*orchestrator));
    std::ofstream{out, std::ios::binary}.write(stream.data(), stream.size());
}
