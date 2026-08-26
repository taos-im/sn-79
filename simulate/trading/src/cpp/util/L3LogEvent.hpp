/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "Order.hpp"
#include "Trade.hpp"
#include "Cancellation.hpp"
#include "InstructionLogger.hpp"

#include <cstdint>
#include <variant>

//-------------------------------------------------------------------------

namespace taosim
{

//-------------------------------------------------------------------------

struct L3LogEvent
{
    std::variant<
        InstructionLogContext,
        OrderWithLogContext,
        TradeWithLogContext,
        CancellationWithLogContext> item;
    uint64_t id;
};

//-------------------------------------------------------------------------

}  // namespace taosim

//-------------------------------------------------------------------------