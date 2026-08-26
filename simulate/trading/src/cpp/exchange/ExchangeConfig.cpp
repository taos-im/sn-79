/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/exchange/ExchangeConfig.hpp>

#include <fmt/format.h>

//-------------------------------------------------------------------------

namespace taosim::exchange
{

ExchangeConfig makeExchangeConfig(pugi::xml_node node)
{
    using namespace literals;
    using accounting::validateDecimalPlaces;

    static constexpr auto sl = std::source_location::current();

    const auto priceDecimals = validateDecimalPlaces(node.attribute("priceDecimals").as_uint(), sl);
    const auto volumeDecimals = validateDecimalPlaces(node.attribute("volumeDecimals").as_uint(), sl);

    // TODO: Validation for leverage.
    const auto maxLeverage = util::double2decimal(
        node.attribute("maxLeverage").as_double(), volumeDecimals);
    const auto maintenanceMargin = [&] {
        const auto maintenanceMargin = util::double2decimal(
            node.attribute("maintenanceMargin").as_double(), volumeDecimals);
        const auto maxAllowedMaintenance = util::round(
            1_dec / (2_dec * util::dec1p(maxLeverage), volumeDecimals));
        if (maintenanceMargin > maxAllowedMaintenance){
            throw std::invalid_argument{fmt::format(
                "{}: 'maintenanceMargin' {} cannot be less than {} when maxLeverage is {}",
                sl.function_name(),
                maintenanceMargin,
                maxAllowedMaintenance,
                maxLeverage)};
        }
        return maintenanceMargin;
    }();

    return {
        .priceDecimals = priceDecimals,
        .volumeDecimals = volumeDecimals,
        .baseDecimals = validateDecimalPlaces(node.attribute("baseDecimals").as_uint(), sl),
        .quoteDecimals = validateDecimalPlaces(node.attribute("quoteDecimals").as_uint(), sl),
        .maxLeverage = maxLeverage,
        // TODO: Validation for loan.
        .maxLoan = decimal_t{node.attribute("maxLoan").as_double()},
        .maintenanceMargin = maintenanceMargin,
        // TODO: Validation for price.
        .initialPrice = decimal_t{node.attribute("initialPrice").as_double()},
        .maxOpenOrders = node.attribute("maxOpenOrders")
            .as_ullong(std::numeric_limits<decltype(ExchangeConfig::maxOpenOrders)>::max()),
        .minOrderSize = std::max(
            util::double2decimal(node.attribute("minOrderSize").as_double()),
            util::pow(10_dec, -decimal_t{node.attribute("volumeDecimals").as_uint()})),
        // Unlike minOrderSize this takes no representational floor: absent means absent, so the check
        // stays inert wherever the attribute is not declared, which is every simulation config.
        .minQuoteOrderSize = util::double2decimal(node.attribute("minQuoteOrderSize").as_double(0.0))
    };
}

}  // namespace taosim::exchange

//-------------------------------------------------------------------------
