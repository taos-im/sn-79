/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/book/FeeLogger.hpp>

#include "Simulation.hpp"
#include "util.hpp"

#include <fmt/chrono.h>

//-------------------------------------------------------------------------

namespace taosim::book
{

//-------------------------------------------------------------------------

FeeLogger::FeeLogger(
    const fs::path &filepath, 
    std::chrono::system_clock::time_point startTimePoint,
    decltype(matching::ExchangeSignals::feeLog)& signal,
    Simulation *simulation) noexcept
    : logging::RotatingLoggerBase(logging::RotatingLoggerBaseDesc{
        .name = "FeeLogger",
        .simulation = simulation,
        .filepath = filepath,
        .startTimePoint = startTimePoint,
        .header = std::string{FeeLogger::s_header}
      })
{
    m_feed = signal.connect(
        [this](const matching::FeePolicyWrapper* feePolicyWrapper, const FeeLogEvent& event) {
            log(feePolicyWrapper, event); 
        });
}

//-------------------------------------------------------------------------

void FeeLogger::log(const matching::FeePolicyWrapper* feePolicyWrapper, const FeeLogEvent& event)
{
    updateSink();

    const auto time = m_startTimePoint + m_timeConverter(m_simulation->currentTimestamp());

    const auto aggressingEntry = fmt::format(
        "{:%Y-%m-%d,%H:%M:%S},{},{},{},{},{},{},{}",
        time,
        event.aggressingAgentId,
        "Taker",
        event.fees.taker,
        feePolicyWrapper->getRates(event.bookId, event.aggressingAgentId).taker,
        event.price,
        event.volume,
        event.aggressingRatio);
    const auto restingEntry = fmt::format(
        "{:%Y-%m-%d,%H:%M:%S},{},{},{},{},{},{},{}",
        time,
        event.restingAgentId,
        "Maker",
        event.fees.maker,
        feePolicyWrapper->getRates(event.bookId, event.restingAgentId).maker,
        event.price,
        event.volume,
        event.restingRatio);

    m_logger->trace(aggressingEntry);
    m_logger->trace(restingEntry);
    m_logger->flush();
}

//-------------------------------------------------------------------------

}  // namespace taosim::book

//-------------------------------------------------------------------------