/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

// The fundamental's seed: where a realization takes it, what one seed step draws from it, and the offline
// mode in which a recorded seed file reproduces a run.
//
// The validator delivers each seed step's seed with its reply batch: the public randomness round published after
// the miners' reply deadline, which nobody holds before replying. A book takes a delivered seed once; a step with
// none new draws a random seed. One seed step's draws depend on that step's seed alone: every distribution starts
// afresh at the reseed and the fractional increment is drawn at its own step. Distinct books draw distinct streams.
// With seedFile set, the n-th seed step takes row n of the file, the rounds the validator recorded (or an older
// file's prices), so the same config, rngSeed and file give the same run.

#include "MultiAssetRunFixture.hpp"

#include <Simulation.hpp>
#include <taosim/checkpoint/serialization/process/FundamentalPrice.hpp>
#include <taosim/process/FundamentalPrice.hpp>
#include <taosim/process/FuturesSignal.hpp>
#include <taosim/process/helpers.hpp>
#include <taosim/serialization/msgpack/common.hpp>
#include <taosim/simulation/util.hpp>

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <msgpack.hpp>
#include <pugixml.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

//-------------------------------------------------------------------------

using namespace taosim;
using taosim::process::FundamentalPrice;
using taosim::process::FundamentalPriceDesc;

namespace fs = std::filesystem;

//-------------------------------------------------------------------------

namespace
{

static constexpr Timestamp s_interval = 30'000'000'000;
static constexpr int s_steps = 8;
static constexpr uint64_t s_simSeed = 7;

class TempDir
{
public:
    explicit TempDir(std::string_view name)
        : m_path{fs::temp_directory_path() / fmt::format("taosim-fpseed-{}-{}", ::getpid(), name)}
    {
        fs::remove_all(m_path);
        fs::create_directories(m_path);
    }
    ~TempDir() { std::error_code ec; fs::remove_all(m_path, ec); }
    [[nodiscard]] const fs::path& path() const noexcept { return m_path; }

private:
    fs::path m_path;
};

struct ProcessDesc
{
    uint64_t bookId{5};
    int ownRng{1};
    std::string seedFile{};
    fs::path runDir{};
    uint64_t simSeed{s_simSeed};
    Timestamp gracePeriod{};
};

// A fundamental on a bare simulation: the grid, the shared stream and the fBm factor it needs, no agents.
struct Process
{
    explicit Process(const ProcessDesc& desc)
        : sim{std::make_unique<Simulation>()},
          L{Eigen::MatrixXf::Zero(s_steps + 2, s_steps + 2)}
    {
        sim->time().start = 0;
        sim->time().duration = s_steps * s_interval;
        sim->time().step = 1;
        sim->rng().seed(desc.simSeed);
        if (!desc.runDir.empty()) sim->setRunDir(desc.runDir);
        process::helpers::precomputeFundamentalPriceL(L, 0.729);
        fp = std::make_unique<FundamentalPrice>(FundamentalPriceDesc{
            .simulation = sim.get(),
            .bookId = desc.bookId,
            .seedInterval = s_interval,
            .X0 = 300.0,
            .mu = 0.0,
            .sigma = 0.3,
            .dt = 1.0 / s_steps,
            .lambda = 2.0,
            .muJump = 0.0,
            .sigmaJump = 0.02,
            .hurst = 0.729,
            .epsilon = 0.2,
            .interpolate = 0,
            .ownRng = desc.ownRng,
            .gracePeriod = desc.gracePeriod,
            .proc = {.updatePeriod = s_interval},
            .L = &L,
            .seedFile = desc.seedFile
        });
    }

    std::unique_ptr<Simulation> sim;
    Eigen::MatrixXf L;
    std::unique_ptr<FundamentalPrice> fp;
};

// What one seed step drew: its Brownian increment, its jump increment and its fractional innovation. The
// increments are differences of running sums that differ between two runs, so they agree to rounding, not
// to the bit; the innovation is stored as drawn.
struct Draws
{
    double dW;
    double dJ;
    double v;
    bool operator==(const Draws& other) const
    {
        static constexpr double s_tolerance = 1e-12;
        return std::abs(dW - other.dW) < s_tolerance && std::abs(dJ - other.dJ) < s_tolerance && v == other.v;
    }
};

void PrintTo(const Draws& d, std::ostream* os)
{
    *os << fmt::format("{{dW={}, dJ={}, v={}}}", d.dW, d.dJ, d.v);
}

Draws stepOnce(Process& p, int step)
{
    const auto before = p.fp->state();
    p.fp->update(step * s_interval);
    const auto& after = p.fp->state();
    return {after.W - before.W, after.dJ - before.dJ, after.V(step)};
}

// The seed a validator delivers with its batch for the next seed step.
void deliver(Process& p, uint64_t round, uint64_t word)
{
    p.sim->setStepSeed(round, word);
}

// 32 bytes of randomness as the validator records them, whose first 8 are `word`.
std::string randomnessHex(uint64_t word)
{
    return fmt::format("{:016x}{:048x}", word, 0xC0FFEEull * word);
}

std::string recordedSeedFile(const fs::path& dir, int rows)
{
    const auto path = dir / "recorded_seed.csv";
    std::ofstream ofs{path};
    for (int row = 1; row <= rows; ++row) {
        ofs << fmt::format("{},{:.2f}\n", row, 67000.0 + 1.37 * row);
    }
    return path.generic_string();
}

static constexpr uint64_t s_words[] = {
    0, 0x8f3a1c2d4e5b6a79ull, 0x1b2c3d4e5f60718aull, 0xa9b8c7d6e5f40312ull, 0x0123456789abcdefull,
    0xfedcba9876543210ull, 0x5a5a5a5aa5a5a5a5ull, 0x13579bdf2468ace0ull, 0x7777888899990000ull};

}  // namespace

//-------------------------------------------------------------------------

TEST(FundamentalSeedTest, AStepTakesTheSeedTheValidatorDelivered)
{
    const TempDir root{"root"};
    Process p{{.runDir = root.path()}};
    // A seed file at the run root is not read: a step's seed is the round delivered with the batch.
    std::ofstream{root.path() / "fundamental_seed.csv"} << "1,67000.12\n2,67000.50\n";
    deliver(p, 41'234'567, s_words[1]);
    p.fp->update(s_interval);
    EXPECT_EQ(p.fp->state().lastCount, 41'234'567) << "the step did not take the delivered round";
    EXPECT_EQ(p.fp->state().lastSeed, FundamentalPrice::seedFor(static_cast<int64_t>(s_words[1]), 5));
    EXPECT_EQ(p.fp->randomSeedSteps(), 0u);
}

//-------------------------------------------------------------------------

TEST(FundamentalSeedTest, ADeliveredSeedIsTakenOnceAndAStepWithoutOneDrawsARandomSeed)
{
    Process p{{}};
    deliver(p, 7, s_words[2]);
    p.fp->update(1 * s_interval);
    const auto first = p.fp->state().lastSeed;
    p.fp->update(2 * s_interval);
    EXPECT_EQ(p.fp->randomSeedSteps(), 1u) << "the step reused the seed of the step before";
    EXPECT_EQ(p.fp->state().lastCount, 7) << "a random step names no round";
    deliver(p, 8, s_words[2]);
    p.fp->update(3 * s_interval);
    EXPECT_EQ(p.fp->state().lastCount, 8);
    EXPECT_EQ(p.fp->state().lastSeed, first) << "the same randomness seeds the same stream whatever its round";
    EXPECT_EQ(p.fp->randomSeedSteps(), 1u);
}

//-------------------------------------------------------------------------

// A step that finds no new seed is logged from the first such step after the grace period, in which no batch arrives
// to deliver one, and every thousandth after; every such step counts in randomSeedSteps, the grace period's too.
TEST(FundamentalSeedTest, TheFirstStepAfterTheGracePeriodWithoutASeedIsLogged)
{
    Process p{{.gracePeriod = 2 * s_interval}};
    auto printedAt = [&](int step) {
        testing::internal::CaptureStdout();
        p.fp->update(step * s_interval);
        std::fflush(stdout);
        return testing::internal::GetCapturedStdout();
    };
    static constexpr std::string_view s_warning = "no new fundamental seed delivered";
    const auto inGrace = printedAt(1);
    EXPECT_EQ(inGrace.find(s_warning), std::string::npos) << "a step in the grace period was logged: " << inGrace;
    const auto first = printedAt(2);
    EXPECT_NE(first.find(s_warning), std::string::npos) << "the first step after the grace period was not logged";
    EXPECT_NE(first.find("(1 so far)"), std::string::npos) << "the count is not the steps after the grace period: "
                                                            << first;
    const auto second = printedAt(3);
    EXPECT_EQ(second.find(s_warning), std::string::npos) << "the second step after the grace period was logged";
    EXPECT_EQ(p.fp->randomSeedSteps(), 3u);
}

//-------------------------------------------------------------------------

TEST(FundamentalSeedTest, EveryBookOfARealizationTakesTheOneDeliveredSeed)
{
    Process a{{.bookId = 5}};
    FundamentalPrice second{FundamentalPriceDesc{
        .simulation = a.sim.get(), .bookId = 6, .seedInterval = s_interval, .X0 = 300.0, .mu = 0.0, .sigma = 0.3,
        .dt = 1.0 / s_steps, .lambda = 2.0, .muJump = 0.0, .sigmaJump = 0.02, .hurst = 0.729, .epsilon = 0.2,
        .interpolate = 0, .ownRng = 1, .gracePeriod = 0, .proc = {.updatePeriod = s_interval}, .L = &a.L}};
    deliver(a, 9, s_words[3]);
    a.fp->update(s_interval);
    second.update(s_interval);
    EXPECT_EQ(a.fp->state().lastCount, 9);
    EXPECT_EQ(second.state().lastCount, 9) << "the second book of the realization did not take the seed";
    EXPECT_EQ(second.randomSeedSteps(), 0u);
    EXPECT_NE(a.fp->state().lastSeed, second.state().lastSeed) << "two books shared one stream";
}

//-------------------------------------------------------------------------

TEST(FundamentalSeedTest, EveryRealizationLooksForTheExternalSeedAtTheRunRoot)
{
    const taosim::test::multiasset_run::MultiAssetRun run{"fpseed-root"};
    const auto orchestrator = run.build();
    const auto realizations = orchestrator->checkpointBlocks();
    ASSERT_FALSE(realizations.empty());
    for (const auto& sim : realizations) {
        EXPECT_NE(sim->logDir(), run.root()) << "a realization logs into its own directory";
        EXPECT_EQ(sim->runDir(), run.root()) << sim->logDir().string() << " looks for its external seed elsewhere";
    }
}

//-------------------------------------------------------------------------

TEST(FundamentalSeedTest, AStepsDrawsDependOnItsOwnSeedAlone)
{
    // Every k: a cached draw carries over to alternate steps, so a single k can miss it.
    for (int ownRng : {0, 1}) {
        for (int k = 2; k <= 6; ++k) {
            const TempDir dirA{fmt::format("stepA{}{}", ownRng, k)}, dirB{fmt::format("stepB{}{}", ownRng, k)};
            Process a{{.ownRng = ownRng, .runDir = dirA.path()}};
            Process b{{.ownRng = ownRng, .runDir = dirB.path()}};
            std::vector<Draws> drawsA, drawsB;
            for (int step = 1; step <= k; ++step) {
                deliver(a, step, s_words[step]);
                deliver(b, step, step == k - 1 ? 0x6123456000000000ull : s_words[step]);
                drawsA.push_back(stepOnce(a, step));
                drawsB.push_back(stepOnce(b, step));
            }
            EXPECT_NE(drawsA[k - 2], drawsB[k - 2])
                << "ownRng=" << ownRng << " k=" << k << ": the two seeds of step k-1 drew alike";
            EXPECT_EQ(drawsA[k - 1], drawsB[k - 1])
                << "ownRng=" << ownRng << " k=" << k << ": step k's draws changed with the seed of step k-1";
        }
    }
}

//-------------------------------------------------------------------------

TEST(FundamentalSeedTest, DistinctBooksDrawDistinctStreams)
{
    const TempDir dirA{"bookA"}, dirB{"bookB"};
    Process a{{.bookId = 5, .runDir = dirA.path()}};
    Process b{{.bookId = 6, .runDir = dirB.path()}};
    deliver(a, 1, s_words[4]);
    deliver(b, 1, s_words[4]);
    const auto drawsA = stepOnce(a, 1), drawsB = stepOnce(b, 1);
    EXPECT_NE(drawsA.dW, drawsB.dW) << "two books shared one stream";
    EXPECT_NE(a.fp->state().lastSeed, b.fp->state().lastSeed);
}

//-------------------------------------------------------------------------

TEST(FundamentalSeedTest, ARecordedSeedFileReproducesTheFundamental)
{
    const TempDir dir{"recorded"};
    const auto file = recordedSeedFile(dir.path(), s_steps);
    // The shared stream differs between the two: with ownRng the fundamental does not draw from it.
    Process a{{.seedFile = file, .simSeed = 1}};
    Process b{{.seedFile = file, .simSeed = 2}};
    for (int step = 1; step <= s_steps; ++step) {
        a.fp->update(step * s_interval);
        b.fp->update(step * s_interval);
        EXPECT_EQ(a.fp->state().lastCount, step) << "the step at " << step << " intervals took another row";
        EXPECT_EQ(a.fp->value(), b.fp->value()) << "the two runs parted at step " << step;
    }
}

//-------------------------------------------------------------------------

TEST(FundamentalSeedTest, TheRecordedRoundsReplayTheRunTheirDeliveriesSeeded)
{
    const TempDir dir{"rounds"};
    const auto path = dir.path() / "recorded_rounds.csv";
    {
        std::ofstream ofs{path};
        for (int step = 1; step <= s_steps; ++step) {
            ofs << fmt::format("{},{},{}\n", 41'000'000 + step, randomnessHex(s_words[step]), step * s_interval);
        }
    }
    Process live{{.simSeed = 1}};
    Process replay{{.seedFile = path.generic_string(), .simSeed = 2}};
    for (int step = 1; step <= s_steps; ++step) {
        deliver(live, 41'000'000 + step, s_words[step]);
        live.fp->update(step * s_interval);
        replay.fp->update(step * s_interval);
        EXPECT_EQ(replay.fp->state().lastCount, 41'000'000 + step);
        EXPECT_EQ(live.fp->value(), replay.fp->value()) << "the replay parted from the live run at step " << step;
    }
}

//-------------------------------------------------------------------------

TEST(FundamentalSeedTest, TheBatchCarriesTheRoundAndItsRandomness)
{
    const auto packed = [](auto&& packSeed) {
        msgpack::sbuffer buf;
        msgpack::packer<msgpack::sbuffer> pk{buf};
        pk.pack_map(2);
        pk.pack(std::string{"responses"});
        pk.pack_array(0);
        pk.pack(std::string{"seed"});
        packSeed(pk);
        return msgpack::unpack(buf.data(), buf.size());
    };
    const std::string bytes{"\x8f\x3a\x1c\x2d\x4e\x5b\x6a\x79" "0123456789abcdefghijklmnopqrstuv", 40};
    auto bin = packed([&](auto& pk) {
        pk.pack_array(2);
        pk.pack(uint64_t{41'234'567});
        pk.pack_bin(32);
        pk.pack_bin_body(bytes.data(), 32);
    });
    const auto seed = taosim::simulation::readStepSeed(bin.get());
    ASSERT_TRUE(seed.has_value());
    EXPECT_EQ(seed->round, 41'234'567u);
    EXPECT_EQ(seed->word, s_words[1]) << "the first 8 bytes, big-endian";

    auto hex = packed([&](auto& pk) {
        pk.pack_array(2);
        pk.pack(uint64_t{5});
        pk.pack(randomnessHex(s_words[3]));
    });
    ASSERT_TRUE(taosim::simulation::readStepSeed(hex.get()).has_value());
    EXPECT_EQ(taosim::simulation::readStepSeed(hex.get())->word, s_words[3]);

    auto shortBin = packed([&](auto& pk) {
        pk.pack_array(2);
        pk.pack(uint64_t{5});
        pk.pack_bin(4);
        pk.pack_bin_body(bytes.data(), 4);
    });
    EXPECT_FALSE(taosim::simulation::readStepSeed(shortBin.get()).has_value()) << "fewer than 8 bytes is no seed";
    auto notAPair = packed([](auto& pk) { pk.pack(uint64_t{5}); });
    EXPECT_FALSE(taosim::simulation::readStepSeed(notAPair.get()).has_value());

    msgpack::sbuffer buf;
    msgpack::packer<msgpack::sbuffer> pk{buf};
    pk.pack_map(1);
    pk.pack(std::string{"responses"});
    pk.pack_array(0);
    const auto none = msgpack::unpack(buf.data(), buf.size());
    EXPECT_FALSE(taosim::simulation::readStepSeed(none.get()).has_value()) << "a batch without the member";
}

//-------------------------------------------------------------------------

// Randomness sent as text is read from its first 16 characters, every one of them a hex digit: a sign, white space, a
// 0x prefix or any other character there makes the member no seed, rather than a seed read from part of it.
TEST(FundamentalSeedTest, RandomnessSentAsTextIsSixteenHexDigitsOrNoSeed)
{
    const auto batch = [](std::string_view randomness) {
        msgpack::sbuffer buf;
        msgpack::packer<msgpack::sbuffer> pk{buf};
        pk.pack_map(1);
        pk.pack(std::string{"seed"});
        pk.pack_array(2);
        pk.pack(uint64_t{5});
        pk.pack(std::string{randomness});
        return msgpack::unpack(buf.data(), buf.size());
    };
    const auto upper = batch("8F3A1C2D4E5B6A79" + std::string(48, '0'));
    ASSERT_TRUE(taosim::simulation::readStepSeed(upper.get()).has_value()) << "upper-case hex is hex";
    EXPECT_EQ(taosim::simulation::readStepSeed(upper.get())->word, s_words[1]);
    for (const std::string_view head : {"0x8f3a1c2d4e5b6a", " 8f3a1c2d4e5b6a7", "+8f3a1c2d4e5b6a7", "-000000000000001",
                                        "8f3a1c2dzzzzzzzz"}) {
        const auto bad = batch(std::string{head} + std::string(48, '0'));
        EXPECT_FALSE(taosim::simulation::readStepSeed(bad.get()).has_value())
            << "randomness starting \"" << head << "\" was read as a seed";
    }
}

//-------------------------------------------------------------------------

TEST(FundamentalSeedTest, ARecordedSeedFileThatEndsStopsTheRun)
{
    const TempDir dir{"short"};
    Process p{{.seedFile = recordedSeedFile(dir.path(), 2)}};
    p.fp->update(1 * s_interval);
    p.fp->update(2 * s_interval);
    EXPECT_THROW(p.fp->update(3 * s_interval), std::runtime_error)
        << "a recorded run went on without its seed";
}

//-------------------------------------------------------------------------

TEST(FundamentalSeedTest, AMalformedRowOfARecordedSeedFileNamesBothForms)
{
    const TempDir dir{"malformed"};
    const auto path = dir.path() / "malformed.csv";
    std::ofstream{path} << "1,not-a-seed\n";
    Process p{{.seedFile = path.generic_string()}};
    try {
        p.fp->update(s_interval);
        ADD_FAILURE() << "a malformed row was read";
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("round,randomness"), std::string::npos) << what;
        EXPECT_NE(what.find("count,price"), std::string::npos) << what;
    }
}

//-------------------------------------------------------------------------

TEST(FundamentalSeedTest, ARestoreReseedsTheStreamTheProcessDrawsFrom)
{
    const TempDir dir{"restore"};
    const auto file = recordedSeedFile(dir.path(), s_steps);
    for (int ownRng : {0, 1}) {
        Process before{{.ownRng = ownRng, .seedFile = file}};
        before.fp->update(s_interval);
        serialization::BinaryStream stream;
        msgpack::packer<serialization::BinaryStream> packer{stream};
        packer.pack(*before.fp);

        Process after{{.ownRng = ownRng, .seedFile = file, .simSeed = 99}};
        auto handle = msgpack::unpack(stream.data(), stream.size());
        handle.get().convert(*after.fp);

        std::mt19937 expected{ownRng ? 99u : static_cast<uint32_t>(before.fp->state().lastSeed)};
        EXPECT_EQ(after.sim->rng()(), expected())
            << "ownRng=" << ownRng << ": the restore reseeded the wrong stream";
    }
}

//-------------------------------------------------------------------------

TEST(FundamentalSeedTest, TheFuturesSignalReadsItsSeedAtTheRunRoot)
{
    static constexpr Timestamp s_signalInterval = 10'000'000'000;
    const TempDir root{"futures"};
    const auto sim = std::make_unique<Simulation>();
    sim->setRunDir(root.path());
    std::ofstream{root.path() / "external_seed_sampled.csv"} << "1,67000.5\n";
    process::FuturesSignal signal{process::FuturesSignalDesc{
        .simulation = sim.get(),
        .bookId = 3,
        .seedInterval = s_signalInterval,
        .X0 = 300.0,
        .lambda = 0.001,
        .staleIntervals = 3,
        .proc = {.updatePeriod = s_signalInterval}
    }};
    signal.update(s_signalInterval);
    EXPECT_EQ(signal.state().lastCount, 1u) << "the signal did not read the external seed at the run root";
    EXPECT_DOUBLE_EQ(signal.value(), 67000.5);
}

//-------------------------------------------------------------------------

// The pair check: one config, one wrapper seed and one recorded seed file, run twice offline, give the same
// fundamental in every realization.
TEST(FundamentalSeedTest, TwoOfflineRunsFromOneSeedFileMatch)
{
    namespace mar = taosim::test::multiasset_run;
    const TempDir seeds{"pairseeds"};
    const auto file = recordedSeedFile(seeds.path(), 16);

    auto runOnce = [&](std::string_view name) {
        const TempDir base{name};
        auto background = mar::backgroundXml();
        const std::string gbm = R"(<GBM name="fundamental" X0="300.0" mu="0.0" sigma="0.01" dt="0.0001" seed="1337"
                        updatePeriod="1000000000"/>)";
        const auto at = background.find(gbm);
        EXPECT_NE(at, std::string::npos) << "the fixture's background changed shape";
        background.replace(at, gbm.size(), fmt::format(
            R"(<FundamentalPrice name="fundamental" seedInterval="1000000000" mu="0.0" sigma="0.0277"
                lambda="0.000206" muJump="0.0" sigmaJump="0.025" updatePeriod="1000000000" Hurst="0.729"
                epsilon="0.07" ownRng="1" seedFile="{}"/>)", file));
        std::ofstream{base.path() / "background.xml"} << background;
        std::ofstream{base.path() / "wrapper.xml"} << fmt::format(
            R"(<MultiAssetSimulation id="run" threads="1" seed="42" start="0" step="{}" duration="{}" gracePeriod="0">
    <Background name="bg" path="background.xml" instanceCount="2"/>
</MultiAssetSimulation>
)", mar::kStep, 8 * mar::kStep);
        auto orchestrator = taosim::simulation::multiasset::MultiAssetSimulationOrchestrator::fromConfig(
            base.path() / "wrapper.xml", base.path());
        orchestrator->run();
        std::vector<std::string> fundamentals;
        for (uint32_t idx = 0; idx < 2; ++idx) {
            const auto dir = base.path() / "run" / fmt::format("bg-{}", idx);
            for (const auto& entry : fs::directory_iterator{dir}) {
                if (entry.path().filename().string().starts_with("fundamental")) {
                    std::ifstream ifs{entry.path()};
                    std::ostringstream oss;
                    oss << ifs.rdbuf();
                    fundamentals.push_back(entry.path().filename().string() + "\n" + oss.str());
                }
            }
        }
        std::ranges::sort(fundamentals);
        return fundamentals;
    };

    const auto first = runOnce("pair1");
    const auto second = runOnce("pair2");
    ASSERT_EQ(first.size(), 2u) << "each realization logs its fundamental";
    EXPECT_EQ(first, second) << "two runs of one config, seed and seed file gave different fundamentals";
    EXPECT_NE(first[0].substr(first[0].find('\n')), first[1].substr(first[1].find('\n')))
        << "the two realizations drew one path";
}

//-------------------------------------------------------------------------

namespace
{

// A FundamentalPrice element nested as a class file nests it, with the given ownRng text or none when it is
// empty: the fundamental the loader builds from it and what the loader printed.
std::pair<std::unique_ptr<FundamentalPrice>, std::string> loadElement(Process& p, std::string_view ownRng)
{
    const auto attr = ownRng.empty() ? std::string{} : fmt::format(R"( ownRng="{}")", ownRng);
    pugi::xml_document doc;
    doc.load_string(fmt::format(
        R"(<MultiBookExchangeAgent gracePeriod="0"><Books><Processes>)"
        R"(<FundamentalPrice name="fundamental" seedInterval="{0}" updatePeriod="{0}" mu="0.0" sigma="0.3")"
        R"( lambda="2.0" muJump="0.0" sigmaJump="0.02" Hurst="0.729" epsilon="0.2"{1}/>)"
        R"(</Processes></Books></MultiBookExchangeAgent>)",
        s_interval, attr).c_str());
    testing::internal::CaptureStdout();
    auto fp = FundamentalPrice::fromXML(p.sim.get(), doc.select_node("//FundamentalPrice").node(), 5, 300.0, &p.L);
    std::fflush(stdout);
    return {std::move(fp), testing::internal::GetCapturedStdout()};
}

}  // namespace

TEST(FundamentalSeedTest, AnElementWithoutOwnRngDrawsFromItsOwnGeneratorAndSaysSo)
{
    Process p{{}};
    const auto [fp, printed] = loadElement(p, "");
    EXPECT_EQ(fp->ownRng(), 1) << "an element without the attribute draws from the shared stream";
    EXPECT_NE(printed.find("ownRng"), std::string::npos) << "the loader did not name the missing attribute: " << printed;
}

TEST(FundamentalSeedTest, AnExplicitOwnRngIsKeptWithoutAWarning)
{
    for (const std::string_view value : {"0", "1"}) {
        Process p{{}};
        const auto [fp, printed] = loadElement(p, value);
        EXPECT_EQ(fp->ownRng(), value == "1" ? 1 : 0) << "ownRng=\"" << value << "\" was not kept";
        EXPECT_EQ(printed.find("ownRng"), std::string::npos) << "ownRng=\"" << value << "\" warned: " << printed;
    }
}

//-------------------------------------------------------------------------
