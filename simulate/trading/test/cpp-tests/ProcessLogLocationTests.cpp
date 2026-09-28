/**
 * An unconfigured simulation must not write its process logs into the current directory.
 *
 * BookProcessManager names each process log `<name>.<firstBook>-<lastBook>.csv` and writes it to
 * `simulation->logDir() / <that name>`. The bare `Simulation()` constructor -- the one every test
 * fixture uses -- never assigned m_logDir, so it stayed a default-constructed empty path and that
 * join produced a RELATIVE filename. A relative path resolves against the process working
 * directory, and the gtest suite runs as `cd simulate/trading && ./build/test/cpp-tests/taosim-tests`.
 *
 * So running the unit tests writes files into the source checkout: external.0-1.csv,
 * fundamental.0-1.csv, magneticfield.0-1.csv and MagneticField-{0,1}.csv all appear beside the
 * sources. The `0-1` is `<Books instanceCount="2">` from test/cpp-tests/data/WakeupChain.xml, and
 * the 300.0 first row is that fixture's `<GBM name="fundamental" X0="300.0">`.
 *
 * Why a test rather than an ignore rule. Ignoring the files hides them from source control while
 * still leaving them on disk, so anything that copies the tree wholesale copies them too. The only
 * fix that closes the class is for the engine never to write there in the first place.
 *
 * The invariant pinned here is the one that makes the whole class impossible: an unconfigured
 * simulation's log directory is ABSOLUTE, so nothing it writes can ever land in the caller's
 * working directory, whatever that happens to be.
 */

#include <taosim/simulation/util.hpp>

#include "Simulation.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>

namespace fs = std::filesystem;

static const fs::path s_dataPath = fs::path{__FILE__}.parent_path() / "data";

//-------------------------------------------------------------------------

TEST(ProcessLogLocationTests, BareSimulationHasALogDirectory)
{
    const auto simulation = std::make_unique<Simulation>();
    EXPECT_FALSE(simulation->logDir().empty())
        << "an empty logDir() makes every process-log path relative to the working directory";
}

//-------------------------------------------------------------------------

TEST(ProcessLogLocationTests, BareSimulationLogDirectoryIsAbsolute)
{
    const auto simulation = std::make_unique<Simulation>();
    EXPECT_TRUE(simulation->logDir().is_absolute())
        << "logDir() is '" << simulation->logDir().string()
        << "', which resolves against the caller's working directory";
}

//-------------------------------------------------------------------------

TEST(ProcessLogLocationTests, BareSimulationDoesNotLogIntoTheCurrentDirectory)
{
    const auto simulation = std::make_unique<Simulation>();
    const auto logDir = simulation->logDir();
    ASSERT_TRUE(logDir.is_absolute());

    std::error_code ec;
    const auto here = fs::current_path(ec);
    ASSERT_FALSE(ec) << "could not read the current directory";

    // The join BookProcessManager actually performs, with a name that fixture really produces.
    const auto wouldWrite = logDir / "external.0-1.csv";
    EXPECT_NE(wouldWrite.parent_path(), here)
        << "process logs would be written into " << here.string()
        << ", which under the gtest runner is the source checkout";
}

//-------------------------------------------------------------------------

/**
 * THE ONE THAT ACTUALLY CAUGHT IT. configureLogging() does `m_logDir = m_baseLogDir`, so a fixture
 * that loads XML -- which is most of them, and every fixture that builds books and therefore
 * process logs -- reset m_logDir to whatever the BASE was. A first fix set only m_logDir in the
 * constructor: all three tests above passed, and the full suite still wrote five CSVs into the
 * checkout. Configuring is the state that matters, so it is the state under test.
 */
TEST(ProcessLogLocationTests, LogDirectorySurvivesConfiguration)
{
    auto nodes = taosim::util::parseSimulationFile(s_dataPath / "WakeupChain.xml");
    auto simulation = std::make_unique<Simulation>();
    simulation->setDebug(false);
    simulation->configure(nodes.simulation);

    EXPECT_FALSE(simulation->logDir().empty())
        << "configuring emptied logDir(), so every process log is written relative to the caller";
    EXPECT_TRUE(simulation->logDir().is_absolute())
        << "after configure() logDir() is '" << simulation->logDir().string() << "'";

    std::error_code ec;
    const auto here = fs::current_path(ec);
    ASSERT_FALSE(ec);
    EXPECT_NE((simulation->logDir() / "external.0-1.csv").parent_path(), here)
        << "a configured simulation would still write its process logs into " << here.string();
}

//-------------------------------------------------------------------------
