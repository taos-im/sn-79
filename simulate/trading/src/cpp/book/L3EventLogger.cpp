/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/book/L3EventLogger.hpp>

#include "L3Fmt.hpp"

#include "Simulation.hpp"
#include "util.hpp"

#include <fmt/chrono.h>

//-------------------------------------------------------------------------

namespace taosim::book
{

//-------------------------------------------------------------------------

L3EventLogger::L3EventLogger(
    const fs::path& filepath,
    std::chrono::system_clock::time_point startTimePoint,
    decltype(matching::ExchangeSignals::L3)& signal,
    Simulation* simulation) noexcept
    : logging::RotatingLoggerBase(logging::RotatingLoggerBaseDesc{
        .name = "L3Logger",
        .simulation = simulation,
        .filepath = filepath,
        .startTimePoint = startTimePoint,
        .header = std::string{s_header}
      }),
      m_feed{signal.connect([this](taosim::L3LogEvent event) { log(event); })}
{}

//-------------------------------------------------------------------------

void L3EventLogger::log(taosim::L3LogEvent event)
{
    updateSink();

    // Exchange mode: stamp the record with the block's wall-clock time, the same clock the trades tape,
    // agent_fills and the events stream report against. Without it startTimePoint resolves to the epoch and
    // the added value is a sim-relative offset, so the record reads as 1970 and cannot be lined up
    // with any other surface. Simulation is untouched: it has no blocks, its origin is
    // the configured startDate and its records are already correct.
    const auto blockTs = m_simulation->blockTimestamp();
    const auto time = blockTs != 0
        ? std::chrono::system_clock::time_point{std::chrono::nanoseconds{blockTs}}
        : m_startTimePoint + m_timeConverter(m_simulation->currentTimestamp());

    // Emitted via fmt straight from the event objects (no rapidjson Document, no
    // decimal->double conversion); same lines as the previous L3Serialize + json2str
    // path except numbers keep their full decimal digits — see util/L3Fmt.hpp and
    // L3FmtTests.
    fmt::memory_buffer buf;
    std::visit(
        [&](auto&& item) {
            using T = std::remove_cvref_t<decltype(item)>;
            if constexpr (std::same_as<T, taosim::InstructionLogContext>) {
                taosim::l3fmt::formatItem(buf, item, event.id);
            } else if constexpr (std::same_as<T, AgentResetLogContext>) {
                taosim::l3fmt::formatItem(
                    buf, item, m_simulation->bookIdCanon(item.bookId), event.id);
            } else {
                taosim::l3fmt::formatItem(
                    buf, item, m_simulation->bookIdCanon(item.logContext->bookId), event.id);
            }
        },
        event.item);

    const auto line = fmt::format(
        "{:%Y-%m-%d,%H:%M:%S},{}", time, fmt::string_view{buf.data(), buf.size()});
    m_logger->trace(line);
    m_logger->flush();
    m_loggedSignal(line);
}

//-------------------------------------------------------------------------

}  // namespace taosim::book

//-------------------------------------------------------------------------