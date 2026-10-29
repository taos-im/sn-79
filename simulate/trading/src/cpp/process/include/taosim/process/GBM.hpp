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

struct GBMDesc
{
    double X0;
    double mu;
    double sigma;
    double dt;
    std::optional<uint64_t> seed;
    ProcessDesc proc;
};

struct GBMState
{
    RNG rng;
    double t{};
    double W{};
};

//-------------------------------------------------------------------------

class GBM : public Process
{
public:
    GBM() noexcept = default;
    GBM(const GBMDesc& desc) noexcept;

    [[nodiscard]] auto&& state(this auto&& self) noexcept { return self.m_state; }
    [[nodiscard]] auto&& val(this auto&& self) noexcept { return self.m_value; }
    [[nodiscard]] auto&& gaussian(this auto&& self) noexcept { return self.m_gaussian; }

    virtual double value() const override { return m_value; }

    virtual void update(Timestamp timestamp) override;

    [[nodiscard]] static std::unique_ptr<GBM> fromXML(pugi::xml_node node, uint64_t bookId);

private:
    double m_X0, m_mu, m_sigma, m_dt;
    GBMState m_state;
    std::normal_distribution<double> m_gaussian;
    double m_value;
};

//-------------------------------------------------------------------------

}  // namespace taosim::process

//-------------------------------------------------------------------------
