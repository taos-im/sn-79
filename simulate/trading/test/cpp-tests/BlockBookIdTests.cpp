/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

/*
 * Books have TWO ids and the difference is invisible in every other test in this suite.
 *
 * A run is split into `blockCount` blocks, each a whole Simulation with its own exchange,
 * agents and books. Inside a block books are numbered 0..dimension-1, and that LOCAL id is what
 * every agent, signal and per-book container uses. Anything crossing to the validator carries
 * the CANONICAL id, `blockIdx * dimension + localId`, so that book 0 of block 3 is not confused
 * with book 0 of block 0.
 *
 * Every other fixture here is driven through the bare `Simulation()` constructor, which leaves
 * blockIdx and blockDim at ZERO. Canonical then equals local for every book, so a path that
 * confused the two would pass the entire suite. That is the gap these tests exist to close, and
 * it is not hypothetical: this distinction has produced a wakeup defect before.
 *
 * The block-aware constructor is used directly, so no SimulationManager and no POSIX IPC object
 * is created and these cannot collide with another taosim on the host.
 */

#include <taosim/simulation/util.hpp>
#include <taosim/xml/helpers.hpp>

#include "MultiBookExchangeAgent.hpp"
#include "Simulation.hpp"

#include <fmt/format.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>

//-------------------------------------------------------------------------

using namespace taosim;
using namespace testing;

namespace fs = std::filesystem;

namespace
{

const auto s_fixture = fs::path{__FILE__}.parent_path() / "data" / "WakeupChain.xml";

// The fixture declares two books per block, which is `dimension`.
constexpr uint32_t kBooksPerBlock = 2;

std::unique_ptr<Simulation> configureBlock(uint32_t blockIdx, pugi::xml_document& doc)
{
    EXPECT_TRUE(doc.load_file(s_fixture.c_str()));
    auto simulation = std::make_unique<Simulation>(
        blockIdx, kBooksPerBlock, fs::temp_directory_path(), nullptr);
    simulation->setDebug(false);
    simulation->configure(taosim::xml::rootNode(doc));
    return simulation;
}

}  // namespace

//-------------------------------------------------------------------------

// The mapping itself, on a block that is not the first. `blockIdx * dimension + localId`, and
// the local ids restart at zero in every block rather than continuing.
TEST(BlockBookIdTest, CanonicalIdsAreOffsetByTheBlockButLocalIdsRestart)
{
    for (uint32_t blockIdx : {0u, 1u, 3u}) {
        pugi::xml_document doc;
        auto simulation = configureBlock(blockIdx, doc);

        ASSERT_EQ(simulation->exchange()->books().size(), kBooksPerBlock)
            << "block " << blockIdx << " built a different number of books than `dimension`, "
               "which breaks the canonical mapping for every block after it";

        for (BookId local{}; local < kBooksPerBlock; ++local) {
            EXPECT_EQ(simulation->exchange()->books()[local]->id(), local)
                << "book " << local << " of block " << blockIdx
                << " carries a canonical id where a local one belongs";
            EXPECT_EQ(simulation->bookIdCanon(local), blockIdx * kBooksPerBlock + local);
        }
    }
}

//-------------------------------------------------------------------------

// The inverse, which is what the manager uses to route an incoming canonical id back to a
// block, and what the seed delivery uses to find the right FuturesSignal. Asserted against
// `bookIdCanon` rather than restating the arithmetic, so the two cannot drift apart.
TEST(BlockBookIdTest, TheRoundTripThroughCanonicalIdsIsExact)
{
    for (uint32_t blockIdx : {0u, 1u, 3u}) {
        pugi::xml_document doc;
        auto simulation = configureBlock(blockIdx, doc);

        for (BookId local{}; local < kBooksPerBlock; ++local) {
            const BookId canon = simulation->bookIdCanon(local);
            EXPECT_EQ(canon / kBooksPerBlock, blockIdx) << "canonical " << canon
                << " routes to the wrong block";
            EXPECT_EQ(canon % kBooksPerBlock, local) << "canonical " << canon
                << " routes to the wrong book within its block";
        }
    }
}

//-------------------------------------------------------------------------

// Every per-book container the exchange owns is sized and keyed LOCALLY. A container sized by
// the local count but keyed canonically would read out of bounds on any block past the first,
// and a container sized canonically would silently waste most of itself; neither shows up on
// block 0, which is the only block the rest of this suite ever builds.
TEST(BlockBookIdTest, PerBookContainersAreSizedAndKeyedLocallyOnEveryBlock)
{
    for (uint32_t blockIdx : {0u, 1u, 3u}) {
        pugi::xml_document doc;
        auto simulation = configureBlock(blockIdx, doc);
        auto* exchange = simulation->exchange();

        EXPECT_EQ(exchange->wakeupChains().size(), kBooksPerBlock)
            << "block " << blockIdx << ": the wakeup registry is not one per local book";
        EXPECT_EQ(exchange->acdClocks().size(), kBooksPerBlock)
            << "block " << blockIdx << ": the ACD clocks are not one per local book";

        // And the local ids index them, on a block whose canonical ids start well past them.
        for (BookId local{}; local < kBooksPerBlock; ++local) {
            EXPECT_NO_FATAL_FAILURE((void)exchange->wakeupChains().at(local));
            EXPECT_GT(exchange->statsHub()->barPeriod(), 0u);
            EXPECT_NO_FATAL_FAILURE((void)exchange->statsHub()->l1(local));
        }
    }
}

//-------------------------------------------------------------------------

// The chains have to actually run on a block that is not the first. If any wakeup carried a
// canonical id into a locally-keyed structure, the class would either never be woken again or
// would wake against the wrong book, and both look like a quiet market from the outside.
//
// This is the assertion the previous cohort-wakeup defect would have failed.
TEST(BlockBookIdTest, WakeupChainsRunOnABlockThatIsNotTheFirst)
{
    static constexpr auto kStylized = "STYLIZED_TRADER_AGENT";

    for (uint32_t blockIdx : {0u, 3u}) {
        pugi::xml_document doc;
        auto simulation = configureBlock(blockIdx, doc);
        auto* exchange = simulation->exchange();

        simulation->simulate();

        for (BookId local{}; local < kBooksPerBlock; ++local) {
            const auto* chain = exchange->wakeupChains().at(local).find(kStylized);
            ASSERT_NE(chain, nullptr)
                << "block " << blockIdx << " book " << local << " never registered a chain";
            EXPECT_GT(chain->wakeCount, 0u)
                << "block " << blockIdx << " book " << local << " (canonical "
                << simulation->bookIdCanon(local) << ") never hopped, so the class is silent on "
                   "this block while block 0 would look healthy";
            EXPECT_EQ(chain->reseedCount, 0u)
                << "block " << blockIdx << " book " << local
                << ": a healthy chain was repaired, which means a wake went missing";
            EXPECT_EQ(chain->duplicateCount, 0u)
                << "block " << blockIdx << " book " << local << ": the chain forked";
        }
    }
}


//-------------------------------------------------------------------------

// Diagnostics have to name a book the same way everywhere, and that way has to be CANONICAL.
// Local ids restart at zero in every block, so on an 8-block production run eight different
// books each emit their own "book 0" and no AGENTDIAG line can be attributed to a book. The
// maker and the WAKEUPCHAIN lines already canonized; StylizedTrader did not, and that was
// invisible because every other fixture here runs as block 0 where the two coincide.
//
// This matters beyond tidiness: §12 of notes/direct_state_and_wakeup.md tells the next person
// to read AGENTDIAG hop counts per class as the first check on a real run, and on a real run
// that means several blocks.
TEST(BlockBookIdTest, DiagnosticsNameBooksCanonicallyOnEveryBlock)
{
    static constexpr uint32_t kBlockIdx = 3;

    testing::internal::CaptureStdout();
    {
        pugi::xml_document doc;
        auto simulation = configureBlock(kBlockIdx, doc);
        simulation->simulate();
    }
    const std::string out = testing::internal::GetCapturedStdout();

    // The fixture's block 3 owns canonical books 6 and 7 and no others.
    for (BookId local{}; local < kBooksPerBlock; ++local) {
        const BookId canon = kBlockIdx * kBooksPerBlock + local;
        EXPECT_THAT(out, HasSubstr(fmt::format("\"book\":{},\"n\":", canon)))
            << "no stylized-trader diagnostic named canonical book " << canon;
    }
    // And no line claims a book this block does not own. Local ids 0 and 1 are books 0 and 1
    // of block 0, which is a different market entirely.
    for (BookId local{}; local < kBooksPerBlock; ++local) {
        EXPECT_THAT(out, Not(HasSubstr(fmt::format("\"book\":{},\"n\":", local))))
            << "a diagnostic on block " << kBlockIdx << " named book " << local
            << ", which belongs to block 0";
    }
}
