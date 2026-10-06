/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

//-------------------------------------------------------------------------

namespace taosim::simulation
{

//-------------------------------------------------------------------------
// Common entry point for every run mode. Each implementation owns its own setup
// (single config, checkpoint, replay, multi-asset) and exposes one verb, so the
// caller drives a run without branching on the concrete kind.

class SimulationOrchestrator
{
public:
    virtual ~SimulationOrchestrator() = default;

    virtual void run() = 0;

protected:
    SimulationOrchestrator() noexcept = default;
};

//-------------------------------------------------------------------------

}  // namespace taosim::simulation

//-------------------------------------------------------------------------
