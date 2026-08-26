/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "FeeLogEvent.hpp"
#include "L3LogEvent.hpp"

//-------------------------------------------------------------------------

namespace taosim::matching
{

//-------------------------------------------------------------------------

class FeePolicyWrapper;

//-------------------------------------------------------------------------

struct ExchangeSignals
{
    UnsyncSignal<void(InstructionLogContext)> instructionLog;
    UnsyncSignal<void(OrderWithLogContext)> orderLog;
    UnsyncSignal<void(TradeWithLogContext)> tradeLog;
    UnsyncSignal<void(CancellationWithLogContext)> cancelLog;
    UnsyncSignal<void(taosim::L3LogEvent)> L3;
    UnsyncSignal<void(const FeePolicyWrapper*, taosim::FeeLogEvent)> feeLog;
    // uint64 to match TradeID, which was widened while this was missed. It is the identity of every L3
    // record (the "k" field), so it belongs in the same width as the other engine ids. The checkpoint
    // format is unaffected: it packs as a positive integer and converts back into either width.
    uint64_t eventCounter{};

    ExchangeSignals() noexcept;
};

//-------------------------------------------------------------------------

}  // namespace taosim::matching

//-------------------------------------------------------------------------