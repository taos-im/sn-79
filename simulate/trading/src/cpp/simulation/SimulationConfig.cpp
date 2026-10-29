/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/simulation/SimulationConfig.hpp>

#include <fmt/format.h>

//-------------------------------------------------------------------------

namespace taosim::simulation
{

//-------------------------------------------------------------------------

SimulationConfig SimulationConfig::fromXML(pugi::xml_node node)
{
    SimulationConfig config{TimeConfig::fromXML(node)};
    const auto attr = node.attribute("maxInstructionDelay");
    config.m_maxInstructionDelay = attr.empty()
        ? kDefaultMaxInstructionDelaySeconds * config.m_time.scale
        : attr.as_ullong();
    return config;
}

//-------------------------------------------------------------------------

SimulationConfig::SimulationConfig(TimeConfig time)
{
    m_time = time;
}

//-------------------------------------------------------------------------

void checkInstructionDelay(Timestamp delay, Timestamp bound)
{
    if (delay > bound) {
        throw InstructionDelayRefused{
            fmt::format("delay {} is beyond maxInstructionDelay {}", delay, bound)};
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::simulation

//-------------------------------------------------------------------------
