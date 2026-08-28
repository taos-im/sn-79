/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/book/L3EventLogger.hpp>

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
    // agent_fills and the events stream report against. Before this, the record read
    // "1970-01-01,00:00:00.012413650": startTimePoint resolved to the epoch and the added value was a
    // 12 ms sim-relative clock, so an event that happened on 2026-08-04 was filed under 1970 and could
    // not be lined up with any other surface. Simulation is untouched: it has no blocks, its origin is
    // the configured startDate and its records are already correct.
    const auto blockTs = m_simulation->blockTimestamp();
    const auto time = blockTs != 0
        ? std::chrono::system_clock::time_point{std::chrono::nanoseconds{blockTs}}
        : m_startTimePoint + m_timeConverter(m_simulation->currentTimestamp());

    rapidjson::Document json = std::visit(
        [&](auto&& item) {
            using T = std::remove_cvref_t<decltype(item)>;
            static_assert(taosim::json::IsL3Serializable<T>);
            rapidjson::Document json;
            item.L3Serialize(json);
            if constexpr (!std::same_as<T, taosim::InstructionLogContext>) {
                json["g"]["b"].SetUint(
                    m_simulation->bookIdCanon(json["g"]["b"].GetUint()));
            }
            json.AddMember("k", rapidjson::Value{event.id}, json.GetAllocator());
            return json;
        },
        event.item);

    const auto line =
        fmt::format("{:%Y-%m-%d,%H:%M:%S},{}", time, taosim::json::json2str(json));
    m_logger->trace(line);
    m_logger->flush();
    m_loggedSignal(line);
}

//-------------------------------------------------------------------------

}  // namespace taosim::book

//-------------------------------------------------------------------------