/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/process/JumpDiffusion.hpp>

//-------------------------------------------------------------------------

namespace taosim::process
{

//-------------------------------------------------------------------------

JumpDiffusion::JumpDiffusion(const JumpDiffusionDesc& desc) noexcept
    : m_X0{desc.X0},
      m_mu{desc.mu},
      m_sigma{desc.sigma},
      m_dt{desc.dt},
      m_gaussian{0.0, std::sqrt(desc.dt)},
      m_poisson{desc.lambda},
      m_jump{desc.muJump, desc.sigmaJump}
{
    m_updatePeriod = desc.proc.updatePeriod;
    if (desc.seed) { m_state.rng = RNG{*desc.seed}; }
    m_state.value = desc.X0;
}

//-------------------------------------------------------------------------

void JumpDiffusion::update(Timestamp timestamp)
{
    if (m_values.empty()) {
        m_state.t += m_dt;
        m_state.W += m_gaussian(m_state.rng);
        m_state.dJ += m_poisson(m_state.rng) * m_jump(m_state.rng);
        m_state.value =
            m_X0 * std::exp((m_mu - 0.5 * m_sigma * m_sigma) * m_state.t + m_sigma * m_state.W + m_state.dJ);
    }
    else {
        m_state.value = m_values.at(m_valueIdx);
        m_valueIdx = std::min(m_valueIdx + 1, m_values.size() - 1);
    }
    m_valueSignal(m_state.value);
}

//-------------------------------------------------------------------------

std::unique_ptr<JumpDiffusion> JumpDiffusion::fromXML(pugi::xml_node node, uint64_t seedShift)
{
    static constexpr auto ctx = std::source_location::current().function_name();

    auto getNonNegativeAttribute = [&](pugi::xml_node node, const char* name) {
        pugi::xml_attribute attr = node.attribute(name);
        if (double value = attr.as_double(); attr.empty() || value < 0.0) {
            throw std::invalid_argument(fmt::format(
                "{}: Attribute '{}' must be non-negative", ctx, name));
        } else {
            return value;
        }
    };

    const uint64_t seed = [&] {
        pugi::xml_attribute attr;
        if (attr = node.attribute("seed"); attr.empty()) {
            throw std::invalid_argument(fmt::format(
                "{}: Missing required attribute '{}'", ctx, "seed"));
        }
        return attr.as_ullong();
    }();

    // Required and strictly positive — see FundamentalPrice::fromXML: a defaulted ~1ns
    // period turns the per-step update loop into ~1e9 iterations and an OOM.
    const auto updatePeriod = node.attribute("updatePeriod").as_ullong();
    if (updatePeriod == 0) {
        throw std::invalid_argument{
            "JumpDiffusion::fromXML: attribute 'updatePeriod' must be present and > 0"};
    }
    const float dt = updatePeriod / 86'400'000'000'000.0;

    return std::make_unique<JumpDiffusion>(JumpDiffusionDesc{
        .X0 = getNonNegativeAttribute(node, "X0"),
        .mu = getNonNegativeAttribute(node, "mu"),
        .sigma = getNonNegativeAttribute(node, "sigma"),
        .dt = dt,
        .lambda = getNonNegativeAttribute(node, "lambda"),
        .muJump = getNonNegativeAttribute(node, "muJump"),
        .sigmaJump = getNonNegativeAttribute(node, "sigmaJump"),
        .seed = seed + seedShift,
        .proc = {
            .updatePeriod = updatePeriod
        }
    });
}

//-------------------------------------------------------------------------

}  // namespace taosim::process

//-------------------------------------------------------------------------