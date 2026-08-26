/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/matching/ExchangeSignals.hpp>

#include <fmt/core.h>

//-------------------------------------------------------------------------

namespace taosim::matching
{

//-------------------------------------------------------------------------
// A fixed defect this design guards against: in exchange mode every L3 log HELD its header and
// nothing else, on both the prediction-time
// logger and the deferred (reconciled) one, while the in-memory L3Record demonstrably carries the
// same events. Four candidate causes were refuted by inspection: scoped_connection reallocation
// (Boost resets the moved-from source), a second configure pass, an event backlog outliving its signals,
// and clear() severing feeds. What is left is which hop stops the event, and that cannot be read
// off the source. These announce once per process so one run names the dead hop.

namespace
{

void announceOnce(std::string_view what)
{
    fmt::println("ExchangeSignals: {} fired for the first time", what);
}

}  // namespace

//-------------------------------------------------------------------------

ExchangeSignals::ExchangeSignals() noexcept
{
    instructionLog.connect([this](InstructionLogContext item) {
        L3({ .item = item, .id = eventCounter++ });
    });
    orderLog.connect([this](OrderWithLogContext item) {
        [[maybe_unused]] static const bool once =
            [] { announceOnce("orderLog -> L3 relay"); return true; }();
        L3({.item = item, .id = eventCounter++});
    });
    tradeLog.connect([this](TradeWithLogContext item) {
        [[maybe_unused]] static const bool once =
            [] { announceOnce("tradeLog -> L3 relay"); return true; }();
        L3({.item = item, .id = eventCounter++});
    });
    cancelLog.connect([this](CancellationWithLogContext item) {
        L3({.item = item, .id = eventCounter++});
    });
}

//-------------------------------------------------------------------------

}  // namespace taosim::matching

//-------------------------------------------------------------------------