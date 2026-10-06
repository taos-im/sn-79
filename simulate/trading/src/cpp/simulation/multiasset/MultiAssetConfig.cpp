/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/simulation/multiasset/MultiAssetConfig.hpp>
#include <taosim/simulation/multiasset/MultiAssetError.hpp>

#include <fmt/format.h>

#include <charconv>
#include <optional>
#include <string_view>
#include <system_error>
#include <utility>

//-------------------------------------------------------------------------

namespace taosim::simulation::multiasset
{

//-------------------------------------------------------------------------

MultiAssetConfig MultiAssetConfig::fromXML(
    const pugi::xml_node& node, const std::filesystem::path& configDir)
{
    auto backgrounds = [&] {
        std::vector<BackgroundDesc> bgs;
        for (const pugi::xml_node bg : node.children("Background")) {
            const std::filesystem::path rel = bg.attribute("path").as_string();
            if (rel.empty()) {
                throw MultiAssetError{"<Background> missing required attribute 'path'"};
            }
            const std::filesystem::path resolved = rel.is_absolute() ? rel : configDir / rel;
            bgs.push_back(BackgroundDesc{
                .name = bg.attribute("name").as_string(),
                .path = std::filesystem::weakly_canonical(resolved),
                .instanceCount = bg.attribute("instanceCount").as_uint(1)
            });
        }
        if (bgs.empty()) {
            throw MultiAssetError{
                fmt::format("'{}' declares no <Background> assets", kRootElement)};
        }
        return bgs;
    }();

    return MultiAssetConfig{
        .backgrounds = std::move(backgrounds),
        .requestedWorkers = node.attribute("threads").as_uint(0),
        .masterSeed = [&]() -> std::optional<uint64_t> {
            // Presence is the signal: seed="0" is a declared seed, absence leaves the
            // backgrounds' own rngSeed (or device seeding) in force. A declared value that
            // is not an unsigned integer is refused rather than read as 0, which would pin
            // every realization to realizationSeed(0, ...) on a typo, silently.
            const auto attr = node.attribute("seed");
            if (attr.empty()) return std::nullopt;
            const std::string_view text = attr.as_string();
            uint64_t value{};
            const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (ec != std::errc{} || end != text.data() + text.size()) {
                throw MultiAssetError{fmt::format(
                    "'{}' attribute 'seed' must be an unsigned integer, got \"{}\"",
                    kRootElement, text)};
            }
            return value;
        }(),
        .start = node.attribute("start").as_ullong(0),
        .step = node.attribute("step").as_ullong(0),
        .duration = node.attribute("duration").as_ullong(0),
        .useMessagePack = node.attribute("useMessagePack").as_bool(),
        .gracePeriod = node.attribute("gracePeriod").as_ullong(0),
        .ckptIntervalInSteps = node.attribute("ckptIntervalInSteps").as_ullong(0),
        .ckptNumLastFilesToKeep = node.attribute("ckptNumLastFilesToKeep").as_ullong(0),
        .ckptMeasureWallClockTime = node.attribute("ckptMeasureWallClockTime").as_bool(),
        .ckptNumWorkers = node.attribute("ckptNumWorkers").as_uint(1),
        // The validator's address for the lifecycle notices, read from the wrapper root
        // with SimulationManager's attribute names and defaults (SimulationManager.cpp
        // fromConfig); a background's own host/port is ignored, as documented.
        .netInfo = net::NetworkingInfo{
            .host = node.attribute("host").as_string(),
            .port = node.attribute("port").as_string(),
            .resolveTimeout = node.attribute("resolveTimeout").as_llong(1),
            .connectTimeout = node.attribute("connectTimeout").as_llong(3),
            .writeTimeout = node.attribute("writeTimeout").as_llong(15),
            .readTimeout = node.attribute("readTimeout").as_llong(60)
        },
        .generalMsgEndpoint = node.attribute("generalMsgEndpoint").as_string("/")
    };
}

//-------------------------------------------------------------------------

}  // namespace taosim::simulation::multiasset

//-------------------------------------------------------------------------
