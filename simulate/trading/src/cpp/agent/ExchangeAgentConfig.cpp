/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "ExchangeAgentConfig.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>

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
    if (pugi::xml_attribute attr = node.attribute("bandReleaseAfter"); !attr.empty()) {
        const int64_t value = attr.as_llong();
        if (value < 0) {
            throw ExchangeAgentConfigException{fmt::format(
                "Value of attribute 'bandReleaseAfter' should be >= 0 (sim ns, 0 disables), was {}", value)};
        }
        m_parameters.bandReleaseAfter = value;
    }
    if (pugi::xml_attribute attr = node.attribute("bandReleaseStep"); !attr.empty()) {
        const int64_t value = attr.as_llong();
        if (value <= 0) {
            throw ExchangeAgentConfigException{fmt::format(
                "Value of attribute 'bandReleaseStep' should be > 0 (sim ns), was {}", value)};
        }
        m_parameters.bandReleaseStep = value;
    }
    if (pugi::xml_attribute attr = node.attribute("bandReleaseMax"); !attr.empty()) {
        const double value = attr.as_double();
        if (value <= 0.0 || value > 1.0 || value < m_parameters.maxPriceBand) {
            throw ExchangeAgentConfigException{fmt::format(
                "Value of attribute 'bandReleaseMax' should be in (0,1] and at least maxPriceBand, was {}", value)};
        }
        m_parameters.bandReleaseMax = value;
    }
    const auto checkWidth = [](std::string_view name, double value) {
        if (value < 0.0 || value >= 1.0) {
            throw ExchangeAgentConfigException{fmt::format(
                "A width in attribute '{}' should be in [0,1), was {}", name, value)};
        }
        return value;
    };
    if (pugi::xml_attribute attr = node.attribute("offGridOrders"); !attr.empty()) {
        const std::string_view value = attr.as_string();
        if (value != "cut" && value != "refuse") {
            throw ExchangeAgentConfigException{fmt::format(
                "Value of attribute 'offGridOrders' should be 'cut' or 'refuse', was '{}'", value)};
        }
        m_parameters.refuseOffGrid = value == "refuse";
    }
    if (pugi::xml_attribute attr = node.attribute("poolPriceBand"); !attr.empty()) {
        m_parameters.poolPriceBand = checkWidth("poolPriceBand", attr.as_double());
    }
    if (pugi::xml_attribute attr = node.attribute("poolPriceBandBooks"); !attr.empty()) {
        // "book:width,book:width"; a malformed entry stops the load rather than leaving a book unbanded unseen.
        const std::string text = attr.as_string();
        size_t pos = 0;
        while (pos < text.size()) {
            const auto end = std::min(text.find(',', pos), text.size());
            const auto entry = text.substr(pos, end - pos);
            const auto colon = entry.find(':');
            if (colon == std::string::npos) {
                throw ExchangeAgentConfigException{fmt::format(
                    "Entry '{}' of attribute 'poolPriceBandBooks' should read book:width", entry)};
            }
            try {
                const auto book = static_cast<uint32_t>(std::stoul(entry.substr(0, colon)));
                m_parameters.poolPriceBandBooks[book] =
                    checkWidth("poolPriceBandBooks", std::stod(entry.substr(colon + 1)));
            } catch (const std::logic_error&) {
                throw ExchangeAgentConfigException{fmt::format(
                    "Entry '{}' of attribute 'poolPriceBandBooks' should read book:width", entry)};
            }
            pos = end + 1;
        }
    }
}

}  // namespace taosim::config
