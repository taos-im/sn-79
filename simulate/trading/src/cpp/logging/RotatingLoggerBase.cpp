/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/logging/RotatingLoggerBase.hpp>

#include "Simulation.hpp"

#include <fmt/chrono.h>

//-------------------------------------------------------------------------

namespace taosim::logging
{

//-------------------------------------------------------------------------

RotatingLoggerBase::RotatingLoggerBase(const RotatingLoggerBaseDesc& desc) noexcept
    : m_simulation{desc.simulation},
      m_filepath{desc.filepath},
      m_startTimePoint{desc.startTimePoint},
      m_header{desc.header}
{
    m_timeConverter = simulation::timescaleToConverter(m_simulation->config().time().scale);

    m_currentWindowBegin = m_simulation->logWindow()
        ? m_simulation->currentTimestamp() / m_simulation->logWindow() * m_simulation->logWindow()
        : taosim::simulation::kLogWindowMax;

    auto [sink, fileExisted] = makeFileSink();

    m_logger = std::make_unique<spdlog::logger>(desc.name, std::move(sink));
    m_logger->set_level(spdlog::level::trace);
    m_logger->set_pattern("%v");

    if (!fileExisted) {
        m_logger->trace(m_header);
        m_logger->flush();
    }
}

//-------------------------------------------------------------------------

void RotatingLoggerBase::updateSink(std::optional<Timestamp> currentTime)
{
    if (!m_simulation->logWindow()) return;

    // Exchange mode rotates on the block clock, so a file's name and the records inside it are stamped
    // from the same source. Simulation keeps rotating on its own clock, unchanged.
    const auto blockTs = m_simulation->blockTimestamp();
    const auto t = blockTs != 0 ? blockTs : currentTime.value_or(m_simulation->currentTimestamp());
    const auto window = m_simulation->logWindow();
    if (t < m_currentWindowBegin + window) [[likely]] return;

    // Re-bucket from t directly so multi-window jumps (common with
    // wall-clock-gapped batch timestamps) land in the correct window in
    // one step rather than chasing them.
    m_currentWindowBegin = t / window * window;

    m_logger->sinks().clear();
    m_logger->sinks().push_back(makeFileSink().sink);
    m_logger->set_pattern("%v");
    m_logger->trace(m_header);
    m_logger->flush();
}

//-------------------------------------------------------------------------

FileSinkWithInfo RotatingLoggerBase::makeFileSink()
{
    m_currentFilepath = [this] {
        if (!m_simulation->logWindow()) return m_filepath;
        // Exchange mode only: name the window by the actual calendar date and hour it covers.
        //
        // logFormatTime emits DDHHMMSS counted from the simulation's own origin, which reads as
        // "20669100000-20669110000" for an hour of a real trading day: correct arithmetically (day
        // 20669 since the epoch, hour 10 to hour 11) and useless to anyone looking for a file by when
        // it happened. In simulation that origin IS the meaningful clock, so this leaves sim names
        // exactly as they were and only reformats when a chain block timestamp exists.
        // Decided from the window itself rather than from blockTimestamp(), because the first sink is
        // opened while the simulation is still being built and no batch has arrived, so keying on the
        // block timestamp named the first hour in the old format and only switched at the next
        // rotation. A window at or past kWallClockEpochFloor is a real Unix time; simulation windows
        // start at zero and a long run never reaches 1e18 ns (31.7 years). The data service already
        // separates the two modes by exactly this kind of date threshold.
        if (m_currentWindowBegin >= taosim::simulation::kWallClockEpochFloor) {
            const auto window = m_simulation->logWindow();
            const auto begin = std::chrono::system_clock::time_point{
                std::chrono::nanoseconds{m_currentWindowBegin / window * window}};
            const auto end = begin + std::chrono::nanoseconds{window};
            return fs::path{fmt::format(
                "{}.{:%Y%m%d-%H%M}-{:%H%M}.log",
                (m_filepath.parent_path() / m_filepath.stem()).generic_string(), begin, end)};
        }
        return fs::path{fmt::format(
            "{}.{}-{}.log",
            (m_filepath.parent_path() / m_filepath.stem()).generic_string(),
            taosim::simulation::logFormatTime(m_timeConverter(m_currentWindowBegin)),
            taosim::simulation::logFormatTime(
                m_timeConverter(m_currentWindowBegin + m_simulation->logWindow())))};
    }();

    const bool fileExisted = fs::exists(m_currentFilepath);

    return {
        .sink = std::make_unique<spdlog::sinks::basic_file_sink_st>(m_currentFilepath),
        .fileExisted = fileExisted
    };
}

//-------------------------------------------------------------------------

}  // namespace taosim::logging

//-------------------------------------------------------------------------
