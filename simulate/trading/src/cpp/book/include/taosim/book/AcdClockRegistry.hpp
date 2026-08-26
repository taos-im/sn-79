/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <string>

//-------------------------------------------------------------------------

namespace taosim::book
{

//-------------------------------------------------------------------------

// One ACD (autoregressive conditional duration) wakeup chain's state.
//
// `psi` is the conditional log-duration, `delay` the log of the realized delay that
// produced it; together they carry the recursion
// psi_next = omega + alpha*delay + beta*psi.
struct AcdClock
{
    float delay{};
    float psi{};
};

// Running moments of a chain's realized durations. Carried over from the DurationStats
// that MagneticField::insertDurationComp accumulated, so relocating a clock out of the
// field does not silently drop its AGENTDIAG line.
struct AcdClockStats
{
    uint64_t n{};
    double delaySum{}, delaySumSq{};
    double delayMin{std::numeric_limits<double>::infinity()};
    double delayMax{-std::numeric_limits<double>::infinity()};
    double psiSum{}, psiSumSq{};
};

//-------------------------------------------------------------------------

// Per-book registry of ACD chains, keyed by the agent class base name of the group that
// shares one token-passed clock.
//
// WHY THIS EXISTS SEPARATELY. This state used to live inside the MagneticField process,
// which made every agent with an ACD clock depend on the Ising herding field — including
// StylizedTrader, which reads no magnetism at all and only needed somewhere shared to
// keep its own timers. A trader wanting zero herding still could not run without the
// herding process, and the deref sites on the decision path were unchecked.
//
// It is deliberately NOT a member of BookTradeStats, despite living next to it on the
// exchange and being reached the same way. BookTradeStats is copied by value into every
// RETRIEVE_L1_EXT response, so a map in it would allocate on every poll and, worse, would
// ship every agent class's wakeup schedule to every subscriber — including the
// distributed-proxy path. Scheduling state must stay exchange-side.
class AcdClockRegistry
{
public:
    // Absent keys read as a zeroed chain, matching the map::operator[] behaviour the
    // MagneticField version had: the first decision of a run finds {0, 0} and the
    // caller's own initialization supplies the starting psi.
    [[nodiscard]] AcdClock get(const std::string& key) const noexcept
    {
        const auto it = m_clocks.find(key);
        return it != m_clocks.end() ? it->second : AcdClock{};
    }

    void insert(const std::string& key, AcdClock clock)
    {
        auto& s = m_stats[key];
        ++s.n;
        s.delaySum += clock.delay;
        s.delaySumSq += static_cast<double>(clock.delay) * clock.delay;
        s.delayMin = std::min(s.delayMin, static_cast<double>(clock.delay));
        s.delayMax = std::max(s.delayMax, static_cast<double>(clock.delay));
        s.psiSum += clock.psi;
        s.psiSumSq += static_cast<double>(clock.psi) * clock.psi;
        m_clocks[key] = clock;
    }

    [[nodiscard]] AcdClockStats stats(const std::string& key) const noexcept
    {
        const auto it = m_stats.find(key);
        return it != m_stats.end() ? it->second : AcdClockStats{};
    }

    [[nodiscard]] auto&& clocks(this auto&& self) noexcept { return self.m_clocks; }

private:
    std::map<std::string, AcdClock> m_clocks;
    std::map<std::string, AcdClockStats> m_stats;
};

//-------------------------------------------------------------------------

}  // namespace taosim::book

//-------------------------------------------------------------------------
