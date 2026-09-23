/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

/*
 * A token-passing agent class has exactly one live wakeup token per book. If that one
 * message is ever dropped, the class stops trading for the rest of the run and nothing
 * reports it: an undeliverable message is discarded with a debug line
 * (Simulation.cpp:409), and a class that has gone quiet looks like a class that decided
 * not to trade. These tests cover the two ways the token can be lost and the repair.
 *
 * The unit tests drive WakeupChainRegistry directly, so they assert the state machine
 * without depending on any agent's parameters, on the rng, or on the market. The
 * integration tests drive a bare Simulation (not SimulationManager), so no POSIX IPC
 * object is created and they cannot collide with another taosim on the host.
 */

#include <taosim/book/WakeupChainRegistry.hpp>

#include "MultiBookExchangeAgent.hpp"
#include "Simulation.hpp"
#include "util.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

//-------------------------------------------------------------------------

using namespace taosim::book;
using namespace testing;

namespace fs = std::filesystem;

//-------------------------------------------------------------------------

static const auto s_testDataPath = fs::path{__FILE__}.parent_path() / "data";

static constexpr auto s_chain = "TEST_CHAIN";
static constexpr Timestamp s_maxDelay = 10;
static constexpr Timestamp s_grace = 1;
static constexpr uint32_t s_instances = 3;

[[nodiscard]] static WakeupChainRegistry makeRegistry(Timestamp now = 0)
{
    WakeupChainRegistry registry;
    registry.registerChain(s_chain, now, s_maxDelay, s_instances);
    return registry;
}

[[nodiscard]] static WakeupChainRegistry makeSelfTimerRegistry(Timestamp now = 0)
{
    WakeupChainRegistry registry;
    registry.registerSelfTimer(s_chain, now, s_maxDelay);
    return registry;
}

//-------------------------------------------------------------------------

// A chain hopping normally is never touched, however many times it is swept. This is the
// false-positive guard: a watchdog that repairs a healthy chain would silently double the
// class's order rate, which is worse than the fault it exists to fix.
TEST(WakeupChainRegistry, HealthyChainIsNeverReseeded)
{
    auto registry = makeRegistry();
    Timestamp now = 0;

    ASSERT_TRUE(registry.arm(s_chain, now, 4));
    for (int hop = 0; hop < 50; ++hop) {
        // Sweeps while the token is still in flight must not fire.
        for (Timestamp t = now; t < now + 4; ++t) {
            EXPECT_THAT(registry.collectOverdue(t, s_grace), IsEmpty());
        }
        now += 4;
        registry.noteWake(s_chain, now);
        ASSERT_TRUE(registry.arm(s_chain, now, 4));
    }

    const auto* chain = registry.find(s_chain);
    ASSERT_NE(chain, nullptr);
    EXPECT_EQ(chain->reseedCount, 0u);
    EXPECT_EQ(chain->duplicateCount, 0u);
    EXPECT_EQ(chain->wakeCount, 50u);
}

//-------------------------------------------------------------------------

// Loss mode one: the token was handed on and never arrived.
TEST(WakeupChainRegistry, TokenLostInFlightIsReseeded)
{
    auto registry = makeRegistry();
    ASSERT_TRUE(registry.arm(s_chain, 0, 4));

    // Due at 4, so 4 + grace is still not overdue.
    EXPECT_THAT(registry.collectOverdue(4 + s_grace, s_grace), IsEmpty());

    const auto overdue = registry.collectOverdue(4 + s_grace + 1, s_grace);
    ASSERT_THAT(overdue, SizeIs(1));
    EXPECT_EQ(overdue.front().key, s_chain);
    EXPECT_EQ(registry.find(s_chain)->reseedCount, 1u);
}

//-------------------------------------------------------------------------

// Loss mode two: the token arrived and was consumed, but the handler returned early or
// threw before passing it on. Nothing is in flight and nothing ever will be.
TEST(WakeupChainRegistry, TokenConsumedButNeverPassedOnIsReseeded)
{
    auto registry = makeRegistry();
    registry.noteWake(s_chain, 100);

    // The wake grants the handler the chain's own widest delay before it is accused.
    EXPECT_THAT(registry.collectOverdue(100 + s_maxDelay, s_grace), IsEmpty());
    EXPECT_THAT(registry.collectOverdue(100 + s_maxDelay + s_grace + 1, s_grace), SizeIs(1));
}

//-------------------------------------------------------------------------

// The repair must not itself become a source of duplicates: after reseeding once, the
// chain is re-armed on its own bound, so sweeping every step does not keep firing.
TEST(WakeupChainRegistry, ReseedIsNotRepeatedOnEveryStep)
{
    auto registry = makeRegistry();
    ASSERT_TRUE(registry.arm(s_chain, 0, 4));

    const Timestamp firstFire = 4 + s_grace + 1;
    ASSERT_THAT(registry.collectOverdue(firstFire, s_grace), SizeIs(1));

    for (Timestamp t = firstFire + 1; t <= firstFire + s_maxDelay + s_grace; ++t) {
        EXPECT_THAT(registry.collectOverdue(t, s_grace), IsEmpty()) << "at t=" << t;
    }
    // If the reseed is lost too, the next bound catches it.
    EXPECT_THAT(registry.collectOverdue(firstFire + s_maxDelay + s_grace + 1, s_grace), SizeIs(1));
    EXPECT_EQ(registry.find(s_chain)->reseedCount, 2u);
}

//-------------------------------------------------------------------------

// Two live tokens in one chain double the class's order rate silently, which is harder to
// notice than a chain that stopped. arm() refuses rather than dispatches.
TEST(WakeupChainRegistry, SecondTokenIsRefused)
{
    auto registry = makeRegistry();

    ASSERT_TRUE(registry.arm(s_chain, 0, 4));
    EXPECT_FALSE(registry.arm(s_chain, 0, 4));
    EXPECT_EQ(registry.find(s_chain)->duplicateCount, 1u);
    // The refusal leaves the original token's deadline in place.
    EXPECT_THAT(registry.collectOverdue(4 + s_grace, s_grace), IsEmpty());

    // Consuming the token clears the way for the next hop.
    registry.noteWake(s_chain, 4);
    EXPECT_TRUE(registry.arm(s_chain, 4, 4));
}

//-------------------------------------------------------------------------

// The reseed target rotates deterministically. Drawing it from the simulation rng would
// mean one dropped message shifted every subsequent draw in the run, so a repaired run
// could no longer be compared against a healthy one.
TEST(WakeupChainRegistry, ReseedTargetRotatesWithoutTouchingTheRng)
{
    auto registry = makeRegistry();
    std::vector<std::string> targets;

    Timestamp now = 0;
    for (auto i = 0z; i < 2 * s_instances; ++i) {
        registry.noteWake(s_chain, now);
        now += s_maxDelay + s_grace + 1;
        const auto overdue = registry.collectOverdue(now, s_grace);
        ASSERT_THAT(overdue, SizeIs(1));
        targets.push_back(overdue.front().target);
    }

    EXPECT_THAT(
        targets,
        ElementsAre(
            "TEST_CHAIN_0", "TEST_CHAIN_1", "TEST_CHAIN_2",
            "TEST_CHAIN_0", "TEST_CHAIN_1", "TEST_CHAIN_2"));
}

//-------------------------------------------------------------------------

// A self timer is a chain of length one: the reseed goes back to the agent itself rather
// than to an indexed instance of a class, and the loss it guards against is the same one.
TEST(WakeupChainRegistry, SelfTimerReseedsToItself)
{
    auto registry = makeSelfTimerRegistry();
    ASSERT_TRUE(registry.arm(s_chain, 0, 4));

    const auto overdue = registry.collectOverdue(4 + s_grace + 1, s_grace);
    ASSERT_THAT(overdue, SizeIs(1));
    EXPECT_EQ(overdue.front().target, s_chain);

    // And repeatedly, because there is no other instance to rotate to.
    registry.noteWake(s_chain, 100);
    const auto again = registry.collectOverdue(100 + s_maxDelay + s_grace + 1, s_grace);
    ASSERT_THAT(again, SizeIs(1));
    EXPECT_EQ(again.front().target, s_chain);
}

//-------------------------------------------------------------------------

// A class that never registers a chain keeps working exactly as before: arm() lets the
// dispatch through and nothing is ever swept. This is what makes the registry safe to
// call from an agent whose class does not use a chain at all.
TEST(WakeupChainRegistry, UnregisteredChainIsInert)
{
    WakeupChainRegistry registry;

    EXPECT_TRUE(registry.arm("NOBODY", 0, 4));
    registry.noteWake("NOBODY", 4);
    EXPECT_EQ(registry.find("NOBODY"), nullptr);
    EXPECT_THAT(registry.collectOverdue(1'000'000, s_grace), IsEmpty());
}

//-------------------------------------------------------------------------

// A chain that has not declared a bound cannot be judged late, so it is left alone rather
// than reseeded on a guess.
TEST(WakeupChainRegistry, ChainWithoutABoundIsNeverReseeded)
{
    WakeupChainRegistry registry;
    registry.registerChain(s_chain, 0, Timestamp{}, s_instances);
    ASSERT_TRUE(registry.arm(s_chain, 0, 4));

    EXPECT_THAT(registry.collectOverdue(1'000'000, s_grace), IsEmpty());
    EXPECT_EQ(registry.find(s_chain)->reseedCount, 0u);
}

//-------------------------------------------------------------------------

// registerChain runs on the same path that seeds, once per book, and must not reset a
// chain that is already running.
TEST(WakeupChainRegistry, ReRegisteringDoesNotDisturbALiveChain)
{
    auto registry = makeRegistry();
    ASSERT_TRUE(registry.arm(s_chain, 0, 4));
    registry.noteWake(s_chain, 4);
    ASSERT_TRUE(registry.arm(s_chain, 4, 4));

    registry.registerChain(s_chain, 100, s_maxDelay, s_instances);

    const auto* chain = registry.find(s_chain);
    EXPECT_TRUE(chain->armed);
    EXPECT_EQ(chain->dueBy, 8u);
    EXPECT_EQ(chain->wakeCount, 1u);
}

//-------------------------------------------------------------------------

class WakeupChainSweepTest : public Test
{
protected:
    static constexpr auto kStylized = "STYLIZED_TRADER_AGENT";
    static constexpr auto kRandom = "RANDOM_TRADER_AGENT";
    // Every class in the fixture that wakes by passing a token around.
    static constexpr std::array kChainClasses{
        "STYLIZED_TRADER_AGENT", "NOISE_TRADER_AGENT", "FUTURES_TRADER_AGENT",
        "HIGH_FREQUENCY_TRADER_AGENT"};
    static constexpr BookId kHealthyBook = 0;
    static constexpr BookId kBrokenBook = 1;

    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange{};

    void configure()
    {
        nodes = taosim::util::parseSimulationFile(s_testDataPath / "WakeupChain.xml");
        simulation = std::make_unique<Simulation>();
        simulation->setDebug(false);
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        ASSERT_NE(exchange, nullptr);
        ASSERT_EQ(exchange->books().size(), 2u);
    }

    const WakeupChain* chain(BookId bookId) const
    {
        return exchange->wakeupChains().at(bookId).find(kStylized);
    }

    // Removes every queued WAKEUP addressed to `baseName` for `bookId`, the way an
    // undeliverable message is discarded, and returns how many went. Rebuilding the queue
    // is the only way to take one out: PriorityQueue exposes its container read-only. The
    // id counter is carried over so the messages pushed afterwards keep tie-breaking
    // against the survivors the way they would have.
    int dropTokensFor(std::string_view baseName, BookId bookId)
    {
        auto& messageQueue = simulation->messageQueue();
        const auto savedCounter = messageQueue.idCounter();
        std::vector<taosim::message::PrioritizedMessageWithId> kept;
        int removed = 0;
        for (const auto& entry : messageQueue.queue().underlying()) {
            const auto& message = entry.pmsg.msg;
            const auto payload =
                std::dynamic_pointer_cast<WakeupPayload>(message->payload);
            const bool addressedToClass = ranges::any_of(
                message->targets,
                [&](std::string_view target) { return target.starts_with(baseName); });
            if (message->type == "WAKEUP"
                && addressedToClass
                && payload != nullptr
                && payload->bookId == bookId) {
                ++removed;
                continue;
            }
            kept.push_back(entry);
        }
        messageQueue = taosim::message::MessageQueue{std::move(kept)};
        messageQueue.idCounter() = savedCounter;
        return removed;
    }
};

//-------------------------------------------------------------------------

// The wiring test. Every hop calls noteWake then arm, so if either is missing or paired
// wrongly the chain either accumulates duplicates or trips the watchdog. Asserting zero
// of both is what says the agent-side wiring agrees with the registry's state machine.
//
// Deliberately says nothing about orders placed: a chain has to keep running even when
// every decision on it bails out, and on an empty book many of them do.
TEST_F(WakeupChainSweepTest, HealthyChainsRunWithoutRepair)
{
    ASSERT_NO_FATAL_FAILURE(configure());
    simulation->simulate();

    for (std::string_view className : kChainClasses) {
        for (BookId bookId : {kHealthyBook, kBrokenBook}) {
            const auto* c = exchange->wakeupChains().at(bookId).find(std::string{className});
            ASSERT_NE(c, nullptr)
                << className << " book " << bookId << " has no registered chain, so "
                   "instance 0 never seeded it";
            EXPECT_GT(c->wakeCount, 0u)
                << className << " book " << bookId << " chain never woke anyone";
            EXPECT_EQ(c->reseedCount, 0u)
                << className << " book " << bookId << " healthy chain was repaired";
            EXPECT_EQ(c->duplicateCount, 0u)
                << className << " book " << bookId << " chain forked";
        }
    }
}

//-------------------------------------------------------------------------

// End to end, and the closest a test can get to the fault itself: take the book's WAKEUP
// out of the message queue, exactly as an undeliverable message is dropped, and the sweep
// hands the chain back to a real instance which resumes hopping. The other book is
// untouched, which is the property that keeps one dropped message from being a whole-class
// outage.
//
// Poking the registry's own state instead would not test this. Tried that way first, and
// the still-live token simply arrived and cleared the sabotage, so whether the test saw a
// repair came down to whether the next hop happened to land inside the same step.
TEST_F(WakeupChainSweepTest, DroppedTokenIsRevivedAndOtherBooksAreUntouched)
{
    ASSERT_NO_FATAL_FAILURE(configure());

    static constexpr Timestamp kDropAt = 20'000'000'000;
    int dropped = 0;
    uint64_t wakesAtDrop = 0;

    // Connected after the exchange's own step subscriber, so each step sweeps first and the
    // drop below lands on a chain the sweep has already accepted as healthy.
    simulation->signals().step.connect([&] {
        if (dropped > 0 || simulation->currentTimestamp() < kDropAt) return;
        wakesAtDrop = chain(kBrokenBook)->wakeCount;
        dropped = dropTokensFor(kStylized, kBrokenBook);
    });

    simulation->simulate();

    // Exactly one, which is also the single-token invariant observed from outside: a chain
    // carrying two would let the class quote twice as often with nothing reporting it.
    ASSERT_EQ(dropped, 1) << "expected exactly one live token for the class on that book";

    const auto* healthy = chain(kHealthyBook);
    const auto* revived = chain(kBrokenBook);
    ASSERT_NE(revived, nullptr);

    EXPECT_EQ(revived->reseedCount, 1u) << "the dropped token was never repaired";
    EXPECT_GT(revived->wakeCount, wakesAtDrop)
        << "the chain was reseeded but never woke again, so the repair did not reach a "
           "live instance";
    EXPECT_EQ(healthy->reseedCount, 0u)
        << "dropping one book's token dragged another book's chain into a repair";
}

//-------------------------------------------------------------------------

// The other wakeup shape. A self timer is keyed per agent per book rather than per class,
// so one stalled instance cannot be hidden by a sibling that is still going.
TEST_F(WakeupChainSweepTest, SelfTimersRunPerInstancePerBook)
{
    ASSERT_NO_FATAL_FAILURE(configure());
    simulation->simulate();

    const auto agentName = fmt::format("{}_0", kRandom);
    for (BookId bookId : {kHealthyBook, kBrokenBook}) {
        const auto* timer = exchange->wakeupChains().at(bookId).find(agentName);
        ASSERT_NE(timer, nullptr)
            << agentName << " book " << bookId << " never registered a self timer";
        EXPECT_GT(timer->wakeCount, 0u)
            << agentName << " book " << bookId << " never woke itself";
        EXPECT_EQ(timer->reseedCount, 0u)
            << agentName << " book " << bookId << " healthy timer was repaired";
        EXPECT_EQ(timer->duplicateCount, 0u)
            << agentName << " book " << bookId << " scheduled two wakes at once";
    }
}
