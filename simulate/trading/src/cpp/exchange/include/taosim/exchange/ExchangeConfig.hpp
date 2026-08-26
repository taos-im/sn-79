/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <pugixml.hpp>

#include <taosim/accounting/common.hpp>
#include <taosim/decimal/decimal.hpp>

#include <source_location>

//-------------------------------------------------------------------------

namespace taosim::exchange
{

struct ExchangeConfig
{
    uint32_t priceDecimals;
    uint32_t volumeDecimals;
    uint32_t baseDecimals;
    uint32_t quoteDecimals;
    decimal_t maxLeverage;
    decimal_t maxLoan;
    decimal_t maintenanceMargin;
    decimal_t initialPrice;
    size_t maxOpenOrders;
    // Floor on the BASE (alpha) amount of an order.
    decimal_t minOrderSize;
    // Floor on the QUOTE (TAO) notional of an order, mirroring the chain's minimum stake amount
    // (SubtensorModule::InitialMinStake). Zero disables the check, which is the case for every
    // simulation config: only exchange mode settles on chain, so only it carries the constraint.
    decimal_t minQuoteOrderSize;
};

[[nodiscard]] ExchangeConfig makeExchangeConfig(pugi::xml_node node);

}  // namespace taosim::exchange

//-------------------------------------------------------------------------