/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "ExchangeAgentConfig.hpp"

#include <fmt/format.h>

//-------------------------------------------------------------------------

namespace taosim::config
{

//-------------------------------------------------------------------------

void ExchangeAgentConfig::configure(pugi::xml_node node)
{
    try {
        setPriceIncrement(node);
        setVolumeIncrement(node);
        setBaseDecimals(node);
        setQuoteDecimals(node);
        setPriceBand(node);
    }
    catch (...) {
        handleException();
    }
}

//-------------------------------------------------------------------------

const ExchangeAgentConfig::Parameters& ExchangeAgentConfig::parameters() const noexcept
{
    return m_parameters;
}

//-------------------------------------------------------------------------

void ExchangeAgentConfig::setPriceIncrement(pugi::xml_node node)
{
    static constexpr const char* attrName = "priceDecimals";

    pugi::xml_attribute attr = node.attribute(attrName);
    if (attr.empty()) return;

    if (uint32_t value = attr.as_uint(); value < Parameters::kMinimumPriceIncrementDecimals) {
        throw ExchangeAgentConfigException{fmt::format(
            "Value of attribute '{}' should be at least {}, was {}",
            attrName,
            Parameters::kMinimumPriceIncrementDecimals,
            value)};
    } else {
        m_parameters.priceIncrementDecimals = value;
    }
}

//-------------------------------------------------------------------------

void ExchangeAgentConfig::setVolumeIncrement(pugi::xml_node node)
{
    static constexpr const char* attrName = "volumeDecimals";

    pugi::xml_attribute attr = node.attribute(attrName);
    if (attr.empty()) return;

    if (uint32_t value = attr.as_uint(); value < Parameters::kMinimumVolumeIncrementDecimals) {
        throw ExchangeAgentConfigException{fmt::format(
            "Value of attribute '{}' should be at least {}, was {}",
            attrName,
            Parameters::kMinimumVolumeIncrementDecimals,
            value)};
    } else {
        m_parameters.volumeIncrementDecimals = value;
    }
}

//-------------------------------------------------------------------------

void ExchangeAgentConfig::setBaseDecimals(pugi::xml_node node)
{
    static constexpr const char* attrName = "baseDecimals";

    pugi::xml_attribute attr = node.attribute(attrName);
    if (attr.empty()) return;

    if (uint32_t value = attr.as_uint(); value < Parameters::kMinimumVolumeIncrementDecimals) {
        throw ExchangeAgentConfigException{fmt::format(
            "Value of attribute '{}' should be at least {}, was {}",
            attrName,
            Parameters::kMinimumVolumeIncrementDecimals,
            value)};
    } else {
        m_parameters.baseIncrementDecimals = value;
    }
}

//-------------------------------------------------------------------------

void ExchangeAgentConfig::setQuoteDecimals(pugi::xml_node node)
{
    static constexpr const char* attrName = "quoteDecimals";

    pugi::xml_attribute attr = node.attribute(attrName);
    if (attr.empty()) return;

    if (uint32_t value = attr.as_uint(); value < Parameters::kMinimumPriceIncrementDecimals) {
        throw ExchangeAgentConfigException{fmt::format(
            "Value of attribute '{}' should be at least {}, was {}",
            attrName,
            Parameters::kMinimumPriceIncrementDecimals,
            value)};
    } else {
        m_parameters.quoteIncrementDecimals = value;
    }
}

//-------------------------------------------------------------------------

void ExchangeAgentConfig::handleException()
{
    try {
        throw;
    }
    catch (const ExchangeAgentConfigException& exc) {
        fmt::println("{}", exc.what());
        throw;
    }
}

//-------------------------------------------------------------------------

ExchangeAgentConfigException::ExchangeAgentConfigException(std::string msg) noexcept
    : m_msg{std::move(msg)}
{}

//-------------------------------------------------------------------------

const char* ExchangeAgentConfigException::what() const noexcept
{
    return m_msg.c_str();
}

//-------------------------------------------------------------------------


//-------------------------------------------------------------------------

void ExchangeAgentConfig::setPriceBand(pugi::xml_node node)
{
    // maxPriceBand: max fractional deviation a match may reach from the book's slow trailing reference.
    // Absent or <=0 leaves the band disabled, so an existing config behaves exactly as before.
    if (pugi::xml_attribute attr = node.attribute("maxPriceBand"); !attr.empty()) {
        const double value = attr.as_double();
        if (value < 0.0 || value > 1.0) {
            throw ExchangeAgentConfigException{fmt::format(
                "Value of attribute 'maxPriceBand' should be in [0,1], was {}", value)};
        }
        m_parameters.maxPriceBand = value;
    }
    if (pugi::xml_attribute attr = node.attribute("bandRefInterval"); !attr.empty()) {
        const int64_t value = attr.as_llong();
        if (value <= 0) {
            throw ExchangeAgentConfigException{fmt::format(
                "Value of attribute 'bandRefInterval' should be > 0 (sim ns), was {}", value)};
        }
        m_parameters.bandRefInterval = value;
    }
    if (pugi::xml_attribute attr = node.attribute("bandRefWindow"); !attr.empty()) {
        const int64_t value = attr.as_llong();
        if (value <= 0) {
            throw ExchangeAgentConfigException{fmt::format(
                "Value of attribute 'bandRefWindow' should be > 0 (sim ns), was {}", value)};
        }
        m_parameters.bandRefWindow = value;
    }
}

}  // namespace taosim::config
