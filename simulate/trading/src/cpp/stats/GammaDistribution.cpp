/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "GammaDistribution.hpp"

#include <fmt/format.h>

#include <source_location>

//-------------------------------------------------------------------------

namespace taosim::stats
{

//-------------------------------------------------------------------------

GammaDistribution::GammaDistribution(double shape, double scale)
{
    static constexpr auto ctx = std::source_location::current().function_name();

    auto checkArg = [&](double arg, const char* name) {
        if (arg <= 0.0) {
            throw std::invalid_argument{fmt::format(
                "{}: parameter '{}' should be > 0.0, was {}", ctx, name, arg)};
        }
        return arg;
    };

    shape = checkArg(shape, "shape");
    scale = checkArg(scale, "scale");

    m_samplingDistribution = decltype(m_samplingDistribution){shape, scale};
    m_distribution = decltype(m_distribution){shape, scale};
}

//-------------------------------------------------------------------------

double GammaDistribution::sample(std::mt19937& rng) noexcept
{
    return m_samplingDistribution(rng);
}

//-------------------------------------------------------------------------

double GammaDistribution::quantile(double p) noexcept
{
    return boost::math::quantile(m_distribution, p);
}

//-------------------------------------------------------------------------

std::unique_ptr<GammaDistribution> GammaDistribution::fromXML(pugi::xml_node node)
{
    static constexpr auto ctx = std::source_location::current().function_name();

    auto getAttr = [&](const char* name) {
        if (pugi::xml_attribute attr = node.attribute(name)) {
            return attr.as_double();
        }
        throw std::invalid_argument{fmt::format(
            "{}: missing required attribute '{}'", ctx, name)};
    };

    return std::make_unique<GammaDistribution>(getAttr("shape"), getAttr("scale"));
}

//-------------------------------------------------------------------------

}  // namespace taosim::stats

//-------------------------------------------------------------------------
