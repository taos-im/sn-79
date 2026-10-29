/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/simulation/TimeConfig.hpp>

#include <pugixml.hpp>

#include <limits>
#include <stdexcept>

//-------------------------------------------------------------------------

namespace taosim::simulation
{

//-------------------------------------------------------------------------

class SimulationConfig
{
public:
    // The longest a validator instruction may be scheduled ahead, when the <Simulation> node does
    // not set maxInstructionDelay (in its own time units). The protocol's bound is the same.
    static constexpr double kDefaultMaxInstructionDelaySeconds = 600.0;

    SimulationConfig() noexcept = default;
    SimulationConfig(TimeConfig time);

    [[nodiscard]] auto&& time(this auto&& self) noexcept
    {
        return std::forward_like<decltype(self)>(self.m_time);
    }

    [[nodiscard]] Timestamp maxInstructionDelay() const noexcept { return m_maxInstructionDelay; }

    [[nodiscard]] static SimulationConfig fromXML(pugi::xml_node node);

private:
    TimeConfig m_time;
    Timestamp m_maxInstructionDelay{std::numeric_limits<Timestamp>::max()};
};

//-------------------------------------------------------------------------

struct InstructionDelayRefused : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

// Where validator instructions enter, before they are queued: an instruction delayed beyond the
// bound is refused on its own, and the rest of the response is kept.
void checkInstructionDelay(Timestamp delay, Timestamp bound);

//-------------------------------------------------------------------------

}  // namespace taosim::simulation

//-------------------------------------------------------------------------
