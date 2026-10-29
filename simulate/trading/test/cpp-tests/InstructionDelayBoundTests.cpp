// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT
//
// A validator instruction carries a delay and is queued until its arrival. The engine takes no delay
// beyond its own bound, maxInstructionDelay on the <Simulation> node, 600 seconds of simulation time
// when unset: the instruction is refused on its own where it enters, and the rest of the response is
// kept. The two entry points (SimulationManager's and the multi-asset orchestrator's) read validator
// replies over IPC and are not driven here; both call the check tested below.

#include <taosim/simulation/SimulationConfig.hpp>

#include <gtest/gtest.h>
#include <pugixml.hpp>

#include <string>

//-------------------------------------------------------------------------

using taosim::simulation::checkInstructionDelay;
using taosim::simulation::InstructionDelayRefused;
using taosim::simulation::SimulationConfig;

//-------------------------------------------------------------------------

namespace
{

[[nodiscard]] SimulationConfig parse(const std::string& xml)
{
    pugi::xml_document doc;
    const auto result = doc.load_string(xml.c_str());
    EXPECT_TRUE(result) << result.description();
    return SimulationConfig::fromXML(doc.child("Simulation"));
}

}  // namespace

//-------------------------------------------------------------------------

TEST(InstructionDelayBoundTest, DefaultsTo600SecondsInTheSimulationsOwnUnits)
{
    EXPECT_EQ(
        parse(R"(<Simulation start="0" duration="1" step="1" timescale="ns"/>)").maxInstructionDelay(),
        600'000'000'000u);
    EXPECT_EQ(
        parse(R"(<Simulation start="0" duration="1" step="1" timescale="ms"/>)").maxInstructionDelay(),
        600'000u);
}

//-------------------------------------------------------------------------

TEST(InstructionDelayBoundTest, ADeclaredBoundIsTakenInTheSimulationsUnits)
{
    EXPECT_EQ(
        parse(R"(<Simulation start="0" duration="1" step="1" timescale="ns" maxInstructionDelay="2500000000"/>)")
            .maxInstructionDelay(),
        2'500'000'000u);
}

//-------------------------------------------------------------------------

TEST(InstructionDelayBoundTest, OnlyADelayBeyondTheBoundIsRefused)
{
    constexpr Timestamp kBound = 600'000'000'000;
    EXPECT_NO_THROW(checkInstructionDelay(0, kBound));
    EXPECT_NO_THROW(checkInstructionDelay(kBound, kBound));
    EXPECT_THROW(checkInstructionDelay(kBound + 1, kBound), InstructionDelayRefused);
    // One simulated day.
    EXPECT_THROW(checkInstructionDelay(86'400'000'000'000, kBound), InstructionDelayRefused);
}

//-------------------------------------------------------------------------
