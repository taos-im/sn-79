/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "Timestamp.hpp"

#include <fmt/format.h>
#include <magic_enum/magic_enum.hpp>
#include <pugixml.hpp>

#include <array>
#include <chrono>
#include <concepts>

//-------------------------------------------------------------------------

namespace taosim::simulation
{

//-------------------------------------------------------------------------

enum class Timescale { s, ms, us, ns };

inline constexpr auto kTimescaleCount = magic_enum::enum_count<Timescale>();

inline constexpr std::array<Timestamp, kTimescaleCount> timescaleFactor{
    1,
    1'000,
    1'000'000,
    1'000'000'000
};

[[nodiscard]] inline constexpr auto timescaleToFactor(Timescale ts) noexcept
{
    return timescaleFactor.at(std::to_underlying(ts));
}

template<typename T1, typename T2>
requires (
    std::same_as<T1, Timescale> && std::floating_point<T2>
    || std::floating_point<T1> && std::same_as<T2, Timescale>)
[[nodiscard]] inline constexpr Timestamp operator*(T1 lhs, T2 rhs) noexcept
{
    if constexpr (std::same_as<T1, Timescale>) {
        return static_cast<Timestamp>(timescaleToFactor(lhs) * rhs);
    } else {
        return static_cast<Timestamp>(lhs * timescaleToFactor(rhs));
    }
}

//-------------------------------------------------------------------------

[[nodiscard]] inline constexpr auto timestampAsSeconds(Timestamp t)
{
    return std::chrono::system_clock::duration{std::chrono::seconds{t}};
}

[[nodiscard]] inline constexpr auto timestampAsMilliseconds(Timestamp t)
{
    return std::chrono::system_clock::duration{std::chrono::milliseconds{t}};
}

[[nodiscard]] inline constexpr auto timestampAsMicroseconds(Timestamp t)
{
    return std::chrono::system_clock::duration{std::chrono::microseconds{t}};
}

[[nodiscard]] inline constexpr auto timestampAsNanoseconds(Timestamp t)
{
    return std::chrono::system_clock::duration{std::chrono::nanoseconds{t}};
}

using TimestampConversionFn = decltype(&timestampAsSeconds);

inline constexpr std::array<TimestampConversionFn, kTimescaleCount> timescaleConverter{
    timestampAsSeconds,
    timestampAsMilliseconds,
    timestampAsMicroseconds,
    timestampAsNanoseconds
};

[[nodiscard]] inline constexpr auto timescaleToConverter(Timescale ts) noexcept
{
    return timescaleConverter[std::to_underlying(ts)];
}

inline constexpr Timestamp kLogWindowMin =
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)).count();

inline constexpr Timestamp kLogWindowMax =
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::days(99)).count();

// Above this, a timestamp is a real Unix time in nanoseconds; below it, simulation-relative.
//
// 1e18 ns is 2001-09-09. Simulation clocks start at zero and kLogWindowMax caps a run's windows at 99
// days (8.6e15 ns), two orders of magnitude below the floor, so the two domains cannot be confused.
// Used to decide whether a log file's window should be named as a calendar date or as the simulation's
// own DDHHMMSS offset, and deliberately NOT a mode flag: the first log sink is opened while the
// simulation is still being constructed, before any chain block has been seen.
inline constexpr Timestamp kWallClockEpochFloor = 1'000'000'000'000'000'000ull;

[[nodiscard]] std::string logFormatTime(auto t)
{
    const auto days = std::chrono::duration_cast<std::chrono::days>(t);
    const std::chrono::hh_mm_ss rem{t - days};
    return fmt::format(
        "{:02}{:02}{:02}{:02}",
        days.count(), rem.hours().count(), rem.minutes().count(), rem.seconds().count());
}

//-------------------------------------------------------------------------

struct TimeConfig
{
    Timestamp start{}, duration{}, step{};
    Timescale scale{};

    TimeConfig() noexcept = default;
    TimeConfig(Timestamp start, Timestamp duration, Timestamp step, Timescale scale);

    [[nodiscard]] static TimeConfig fromXML(pugi::xml_node node);
};

//-------------------------------------------------------------------------

}  // namespace taosim::simulation

//-------------------------------------------------------------------------

template<>
struct fmt::formatter<taosim::simulation::Timescale>
{
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(taosim::simulation::Timescale ts, FormatContext& ctx) const
    {
        return fmt::format_to(ctx.out(), "{}", magic_enum::enum_name(ts));
    }
};

//-------------------------------------------------------------------------