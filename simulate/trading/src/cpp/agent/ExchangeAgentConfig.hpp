/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <pugixml.hpp>

#include <cstdint>
#include <exception>
#include <source_location>
#include <string>

//-------------------------------------------------------------------------

namespace taosim::config
{

//-------------------------------------------------------------------------

class ExchangeAgentConfig
{
public:
    struct Parameters
    {
        ///###
        static inline constexpr uint32_t kMinimumPriceIncrementDecimals = 2;
        static inline constexpr uint32_t kMinimumVolumeIncrementDecimals = 2;

        uint32_t priceIncrementDecimals = kMinimumPriceIncrementDecimals;
        uint32_t volumeIncrementDecimals = kMinimumVolumeIncrementDecimals;
        uint32_t baseIncrementDecimals = kMinimumVolumeIncrementDecimals;
        uint32_t quoteIncrementDecimals = kMinimumPriceIncrementDecimals;

        // PRICE BAND (0 = disabled, the pre-band behaviour, so this is inert until configured).
        // An uncapped market order currently sweeps with maxPrice = numeric_limits::max(), which is how a
        // single order manufactures a +20% excursion. Bounding it is not enough on its own: a band
        // referenced to bestAsk RATCHETS, because each order consumes the level, bestAsk moves up, and the
        // next order gets fresh headroom (four 5% steps reach +21%). So the band is referenced to a SLOW
        // trailing price that a burst cannot drag with it -- the shape real limit-up/limit-down rules use.
        // bandRefWindow is the trailing SIM-TIME window (ns) whose mean trade price is the reference --
        // the LULD construction (US equities uses a 5-minute mean). A per-trade EMA was tried first and
        // rejected: many matches inside one batch can still drag it, which is the ratchet the time window
        // exists to prevent. Moving a time-windowed mean requires elapsed time, not just more orders.
        double maxPriceBand = 0.0;
        int64_t bandRefWindow = 300'000'000'000;     // 5 minutes of SIM time (m_time.current, never wall clock)
        int64_t bandRefInterval = 1'000'000'000;     // sample the prevailing price once per sim-second
        // ...
    };

    ExchangeAgentConfig() noexcept = default;

    void configure(pugi::xml_node node);

    [[nodiscard]] const Parameters& parameters() const noexcept;

private:
    void setPriceIncrement(pugi::xml_node node);
    void setVolumeIncrement(pugi::xml_node node);
    void setBaseDecimals(pugi::xml_node node);
    void setQuoteDecimals(pugi::xml_node node);
    void setPriceBand(pugi::xml_node node);
    void handleException();

    Parameters m_parameters;
};

//-------------------------------------------------------------------------

class ExchangeAgentConfigException : public std::exception
{
public:
    explicit ExchangeAgentConfigException(std::string msg) noexcept;

    virtual const char* what() const noexcept override;

private:
    std::string m_msg;
};

//-------------------------------------------------------------------------

}  // namespace taosim::config

//-------------------------------------------------------------------------
