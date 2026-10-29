/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/process/Process.hpp>
#include <taosim/process/RNG.hpp>
#include "common.hpp"

#include <pugixml.hpp>

//-------------------------------------------------------------------------

namespace taosim::process
{

//-------------------------------------------------------------------------

struct JumpDiffusionDesc
{
    double X0;
    double mu;
    double sigma;
    double dt;
    double lambda;
    double muJump;
    double sigmaJump;
    std::optional<uint64_t> seed;
    ProcessDesc proc;
};

struct JumpDiffusionState
{
    RNG rng;
    double dJ{};
    double t{};
    double W{};
    double value;
};

//-------------------------------------------------------------------------

class JumpDiffusion : public Process
{
public:
    JumpDiffusion() noexcept = default;
    JumpDiffusion(const JumpDiffusionDesc& desc) noexcept;

    [[nodiscard]] auto&& state(this auto&& self) noexcept { return self.m_state; }
    [[nodiscard]] auto&& gaussian(this auto&& self) noexcept { return self.m_gaussian; }
    [[nodiscard]] auto&& jump(this auto&& self) noexcept { return self.m_jump; }
    [[nodiscard]] auto&& poisson(this auto&& self) noexcept { return self.m_poisson; }

    virtual double value() const override { return m_state.value; }

    virtual void update(Timestamp timestamp) override;

    [[nodiscard]] static std::unique_ptr<JumpDiffusion> fromXML(pugi::xml_node node, uint64_t bookId);

private:
    double m_X0, m_mu, m_sigma, m_dt;
    JumpDiffusionState m_state;
    std::normal_distribution<double> m_gaussian;
    std::normal_distribution<double> m_jump;
    std::poisson_distribution<int> m_poisson;
};

//-------------------------------------------------------------------------

}  // namespace taosim::process

//-------------------------------------------------------------------------
