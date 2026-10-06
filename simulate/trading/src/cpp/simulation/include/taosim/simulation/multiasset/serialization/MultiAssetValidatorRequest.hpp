/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/simulation/multiasset/MultiAssetConfig.hpp>
#include <taosim/simulation/serialization/ValidatorRequest.hpp>

#include <msgpack.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

//-------------------------------------------------------------------------

class Simulation;

//-------------------------------------------------------------------------

namespace taosim::simulation::multiasset::serialization
{

// The multi-asset analogue of serialization::ValidatorRequest. Where the single
// background was published as one flat state, a multi-asset run is published as a
// grouping over backgrounds (some with several instances), and each instance is a
// whole single-background ValidatorRequest over its one realization — i.e. this is
// literally a composition of the old request. Book ids run consecutively across the
// realizations (each numbered from the sum of the book counts before it), so the flat
// id space is 0..N-1 and decanonize() routes a validator response back to the
// realization whose range holds the id.
struct MultiAssetValidatorRequest
{
    std::span<const std::unique_ptr<Simulation>> simulations;  // flat, all realizations
    const MultiAssetConfig* config;                            // backgrounds + instanceCount
    std::filesystem::path logDir;                              // run root
};

}  // namespace taosim::simulation::multiasset::serialization

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct pack<taosim::simulation::multiasset::serialization::MultiAssetValidatorRequest>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o,
        const taosim::simulation::multiasset::serialization::MultiAssetValidatorRequest& v) const
    {
        using namespace std::string_literals;
        namespace single = taosim::simulation::serialization;

        o.pack_map(4);

        // Run root (each realization additionally carries its own log directory).
        o.pack("logDir"s);
        o.pack(v.logDir.string());

        // All realizations advance on one shared grid, so any of them dates the run.
        o.pack("timestamp"s);
        o.pack(v.simulations.front()->currentTimestamp());

        o.pack("model"s);
        o.pack("im"s);

        // One entry per background; each holds its instances, and each instance is a
        // full single-background request over that realization. (background, instance)
        // is positional and follows the orchestrator's flat order.
        o.pack("backgrounds"s);
        o.pack_array(static_cast<uint32_t>(v.config->backgrounds.size()));
        uint32_t flatIdx = 0;
        for (const auto& [bgIdx, bg] : views::enumerate(v.config->backgrounds)) {
            o.pack_map(3);

            o.pack("bg"s);
            o.pack(static_cast<uint32_t>(bgIdx));

            o.pack("name"s);
            // An unnamed background is "bg<index>" on the wire as well: the name its
            // realization directories and the validator model already carry.
            o.pack(bg.name.empty() ? "bg" + std::to_string(bgIdx) : bg.name);

            o.pack("instances"s);
            o.pack_array(bg.instanceCount);
            for (uint32_t instance = 0; instance < bg.instanceCount; ++instance) {
                const auto& simulation = v.simulations[flatIdx];
                o.pack(single::ValidatorRequest{
                    .simulations = v.simulations.subspan(flatIdx, 1),
                    .logDir = simulation->logDir()
                });
                ++flatIdx;
            }
        }

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
