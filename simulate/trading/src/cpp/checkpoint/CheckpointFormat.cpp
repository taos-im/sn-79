/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/checkpoint/CheckpointFormat.hpp>

#include <taosim/checkpoint/CheckpointError.hpp>
#include <taosim/serialization/msgpack/utils.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <span>

//-------------------------------------------------------------------------

namespace taosim::checkpoint
{

//-------------------------------------------------------------------------

CheckpointFormat readFormat(const msgpack::object& common)
{
    CheckpointFormat format;

    if (const auto version = serialization::msgpackFindMapObj(common, kFormatVersionKey)) {
        const auto& obj = version->get();
        if (obj.type != msgpack::type::POSITIVE_INTEGER || obj.via.u64 > UINT32_MAX) {
            throw CheckpointError{fmt::format("checkpoint '{}' is not an unsigned integer", kFormatVersionKey)};
        }
        format.version = static_cast<uint32_t>(obj.via.u64);
    }

    if (const auto capabilities = serialization::msgpackFindMapObj(common, kCapabilitiesKey)) {
        const auto& obj = capabilities->get();
        if (obj.type != msgpack::type::ARRAY) {
            throw CheckpointError{fmt::format("checkpoint '{}' is not a list of names", kCapabilitiesKey)};
        }
        for (const auto& name : std::span{obj.via.array.ptr, obj.via.array.size}) {
            if (name.type != msgpack::type::STR) {
                throw CheckpointError{fmt::format("checkpoint '{}' is not a list of names", kCapabilitiesKey)};
            }
            format.capabilities.push_back(name.as<std::string>());
        }
    }

    return format;
}

//-------------------------------------------------------------------------

CheckpointFormat requireReadableFormat(const msgpack::object& common)
{
    auto format = readFormat(common);

    if (format.version > kFormatVersion) {
        throw CheckpointError{fmt::format(
            "the checkpoint has format version {}, newer than this binary reads (format version {}); "
            "resume it with a binary that reads format version {}",
            format.version, kFormatVersion, format.version)};
    }

    for (const auto& capability : format.capabilities) {
        if (std::ranges::find(kKnownCapabilities, capability) != kKnownCapabilities.end()) continue;
        throw CheckpointError{fmt::format(
            "the checkpoint needs capability '{}', which this binary (format version {}, capabilities: {}) "
            "does not have; resume it with a binary that has it",
            capability, kFormatVersion,
            kKnownCapabilities.empty() ? std::string{"none"} : fmt::format("{}", fmt::join(kKnownCapabilities, ", ")))};
    }

    return format;
}

//-------------------------------------------------------------------------

}  // namespace taosim::checkpoint

//-------------------------------------------------------------------------
