/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "AgentReset.hpp"
#include "Cancellation.hpp"
#include "InstructionLogger.hpp"
#include "Order.hpp"
#include "Trade.hpp"

#include <fmt/format.h>

//-------------------------------------------------------------------------
// fmt-based emission of the L3 event-log lines, replacing the rapidjson path
// (L3Serialize + json2str) on the hot logging path: no rapidjson Documents, no
// decimal->double conversions. Structure, key order, enum spellings and null
// for empty optionals match the legacy lines exactly; numbers print from the
// decimal digits (fmt::formatter<decimal_t>), which keeps digits past
// rapidjson's 8-decimal clamp and renders tiny values in scientific notation
// instead of clipping them to 0.0. The L3Serialize implementations remain the
// reference; L3FmtTests asserts the equivalence differentially on the value
// domain where the two conventions coincide.

namespace taosim::l3fmt
{

// One overload per L3LogEvent alternative. `bookIdCanon` replaces the log
// context's local book id (the logger's canonicalization); `eventId` is the
// trailing "k" member.
void formatItem(
    fmt::memory_buffer& out, const OrderWithLogContext& item,
    BookId bookIdCanon, uint64_t eventId);
void formatItem(
    fmt::memory_buffer& out, const CancellationWithLogContext& item,
    BookId bookIdCanon, uint64_t eventId);
void formatItem(
    fmt::memory_buffer& out, const TradeWithLogContext& item,
    BookId bookIdCanon, uint64_t eventId);
void formatItem(
    fmt::memory_buffer& out, const AgentResetLogContext& item,
    BookId bookIdCanon, uint64_t eventId);
void formatItem(
    fmt::memory_buffer& out, const taosim::InstructionLogContext& item, uint64_t eventId);

}  // namespace taosim::l3fmt

//-------------------------------------------------------------------------
