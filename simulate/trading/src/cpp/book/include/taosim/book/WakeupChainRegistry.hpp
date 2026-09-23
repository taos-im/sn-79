/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <Timestamp.hpp>

#include <fmt/format.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

//-------------------------------------------------------------------------

namespace taosim::book
{

//-------------------------------------------------------------------------

// One live token per chain per book. A class's whole order flow rides on that token, so a
// dropped WAKEUP silences the class with no other symptom; the watchdog reseeds it.
struct WakeupChain
{
    // When the next event is due. arm() sets it to the scheduled hop; noteWake() sets it
    // to the conservative bound within which the woken instance must arm the next hop.
    Timestamp dueBy{};
    Timestamp lastWake{};
    // The widest delay this chain may schedule, from the agent's own clamp.
    Timestamp maxDelay{};
    // Instances of the class, for picking a reseed target.
    uint32_t instanceCount{};
    // A self timer's key IS its agent name, so the reseed goes straight back to it rather
    // than to an indexed instance of a class.
    bool targetIsKey{};
    bool armed{};
    uint64_t wakeCount{};
    uint64_t hopCount{};
    // Nonzero means the chain broke and was repaired. It is a fault counter, not a
    // routine one, and belongs in the end-of-run diagnostics.
    uint64_t reseedCount{};
    // arm() called while a token was already live. Refused rather than dispatched: two
    // tokens in one chain double the class's order rate silently, which is harder to
    // notice than a chain that stopped.
    uint64_t duplicateCount{};
    // Rotates the reseed target. Deliberately not drawn from the simulation rng: a repair
    // must not shift the rng stream, or one dropped message would change every subsequent
    // draw in the run and make the repaired run incomparable to the healthy one.
    uint64_t reseedCursor{};
};

//-------------------------------------------------------------------------

// A clock that missed its deadline, and the agent the wake should be sent to. The registry
// formats the name so that a chain and a self timer are one case at the call site.
struct WakeupReseed
{
    std::string key;
    std::string target;
};

//-------------------------------------------------------------------------

// Per-book registry of wakeup chains, keyed by the agent class base name, alongside
// AcdClockRegistry and for the same reason: scheduling state is exchange-side and is
// never copied into a response payload.
class WakeupChainRegistry
{
public:
    // Idempotent, so a class whose instance 0 seeds one chain per book can call it on the
    // path that also seeds. maxDelay of 0 disables the watchdog for that chain, which is
    // the honest reading of "this class has not told us what its bound is".
    void registerChain(
        const std::string& baseName,
        Timestamp now,
        Timestamp maxDelay,
        uint32_t instanceCount)
    {
        registerImpl(baseName, now, maxDelay, instanceCount, false);
    }

    // A single agent waking itself. Keyed by the agent's own name, so instances of the same
    // class each get their own entry and one stalled instance is not hidden by its siblings.
    void registerSelfTimer(const std::string& agentName, Timestamp now, Timestamp maxDelay)
    {
        registerImpl(agentName, now, maxDelay, 1, true);
    }

    // The token was passed on. Returns false if one was already live, in which case the
    // caller must not dispatch: refusing here is what keeps a chain single-token.
    [[nodiscard]] bool arm(const std::string& baseName, Timestamp now, Timestamp delay)
    {
        auto it = m_chains.find(baseName);
        if (it == m_chains.end()) return true;
        auto& chain = it->second;
        if (chain.armed) {
            ++chain.duplicateCount;
            return false;
        }
        chain.armed = true;
        chain.dueBy = now + delay;
        ++chain.hopCount;
        return true;
    }

    // The token arrived and is about to be consumed.
    void noteWake(const std::string& baseName, Timestamp now)
    {
        auto it = m_chains.find(baseName);
        if (it == m_chains.end()) return;
        auto& chain = it->second;
        chain.armed = false;
        chain.lastWake = now;
        chain.dueBy = now + chain.maxDelay;
        ++chain.wakeCount;
    }

    // Chains overdue at `now`, marked repaired and re-armed so the same chain is not
    // reported again on the next sweep. The caller owns the dispatch.
    [[nodiscard]] std::vector<WakeupReseed> collectOverdue(Timestamp now, Timestamp grace)
    {
        std::vector<WakeupReseed> overdue;
        for (auto& [baseName, chain] : m_chains) {
            if (chain.maxDelay == Timestamp{} || chain.instanceCount == 0) continue;
            if (now <= chain.dueBy + grace) continue;
            overdue.push_back({
                .key = baseName,
                .target = chain.targetIsKey
                    ? baseName
                    : fmt::format("{}_{}", baseName, chain.reseedCursor % chain.instanceCount)
            });
            ++chain.reseedCursor;
            ++chain.reseedCount;
            // Re-arm on the same bound the chain itself uses, so a reseed that is also
            // lost is caught on the next bound rather than every step until it lands.
            chain.armed = true;
            chain.dueBy = now + chain.maxDelay;
        }
        return overdue;
    }

    [[nodiscard]] const WakeupChain* find(const std::string& baseName) const noexcept
    {
        const auto it = m_chains.find(baseName);
        return it != m_chains.end() ? &it->second : nullptr;
    }

    [[nodiscard]] auto&& chains(this auto&& self) noexcept { return self.m_chains; }

private:
    void registerImpl(
        const std::string& key,
        Timestamp now,
        Timestamp maxDelay,
        uint32_t instanceCount,
        bool targetIsKey)
    {
        auto& chain = m_chains[key];
        chain.maxDelay = maxDelay;
        chain.instanceCount = instanceCount;
        chain.targetIsKey = targetIsKey;
        if (chain.lastWake == Timestamp{} && !chain.armed) {
            chain.lastWake = now;
            chain.dueBy = now + maxDelay;
        }
    }

    std::map<std::string, WakeupChain> m_chains;
};

//-------------------------------------------------------------------------

}  // namespace taosim::book

//-------------------------------------------------------------------------
