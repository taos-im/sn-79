/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

// The fundamental's seed: where a realization finds it, what one seed step draws from it, and the offline
// mode in which a recorded seed file reproduces a run.
//
// A multi-asset realization looks for the seed file at the run root, where the validator writes it. One
// seed step's draws depend on that step's seed alone: every distribution starts afresh at the reseed and the
// fractional increment is drawn at its own step. Distinct books draw distinct streams. With seedFile set, the
// n-th seed step takes row n of the file, so the same config, rngSeed and file give the same run.

#include "MultiAssetRunFixture.hpp"

#include <Simulation.hpp>
#include <taosim/checkpoint/serialization/process/FundamentalPrice.hpp>
#include <taosim/process/FundamentalPrice.hpp>
#include <taosim/process/FuturesSignal.hpp>
#include <taosim/process/helpers.hpp>
#include <taosim/serialization/msgpack/common.hpp>

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
            .gracePeriod = 0,
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

// The live service's file as the engine reads it: its newest complete row.
void writeLiveSeed(const fs::path& dir, int count, std::string_view price)
{
    std::ofstream{dir / "fundamental_seed.csv"} << fmt::format("{},{}\n", count, price);
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

static constexpr std::string_view s_prices[] = {
    "", "67000.12", "67001.40", "67000.95", "67002.33", "67003.10", "67001.88", "67004.20", "67005.05"};

}  // namespace

//-------------------------------------------------------------------------

TEST(FundamentalSeedTest, ARealizationReadsTheSeedAtTheRunRoot)
{
    const TempDir root{"root"};
    Process p{{.runDir = root.path()}};
    EXPECT_EQ(fs::path{p.fp->seedFile()}, root.path() / "fundamental_seed.csv");

    std::ofstream{root.path() / "fundamental_seed.csv"} << "1,67000.12\n2,67000.50\n";
    p.fp->update(s_interval);
    EXPECT_EQ(p.fp->state().lastCount, 2) << "the newest complete row at the run root was not read";
    EXPECT_EQ(p.fp->state().lastSeed, FundamentalPrice::seedFor(6'700'050, 5));
}

//-------------------------------------------------------------------------

TEST(FundamentalSeedTest, EveryRealizationLooksForTheSeedAtTheRunRoot)
{
    const taosim::test::multiasset_run::MultiAssetRun run{"fpseed-root"};
    const auto orchestrator = run.build();
    const auto realizations = orchestrator->checkpointBlocks();
    ASSERT_FALSE(realizations.empty());
    for (const auto& sim : realizations) {
        EXPECT_NE(sim->logDir(), run.root()) << "a realization logs into its own directory";
        EXPECT_EQ(sim->runDir(), run.root()) << sim->logDir().string() << " looks for its seed elsewhere";
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
                writeLiveSeed(dirA.path(), step, s_prices[step]);
                writeLiveSeed(dirB.path(), step, step == k - 1 ? "61234.56" : s_prices[step]);
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
    writeLiveSeed(dirA.path(), 1, "67000.20");
    writeLiveSeed(dirB.path(), 1, "67000.10");
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
