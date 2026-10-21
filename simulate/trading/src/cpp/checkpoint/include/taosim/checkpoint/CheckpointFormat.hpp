/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <msgpack.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

//-------------------------------------------------------------------------

namespace taosim::checkpoint
{

//-------------------------------------------------------------------------
// The format a checkpoint declares in its common file: a version, and the capabilities its content
// needs from a reader. A common file without the keys predates them and reads as the legacy version
// with no capabilities, so it loads as it always has.
//
// Bump kFormatVersion when state every checkpoint carries changes so that an older reader would
// misread it. State only some checkpoints carry is a capability instead: the writer lists it only
// when that state is present, so a checkpoint without it still loads on a binary that lacks it.

inline constexpr uint32_t kLegacyFormatVersion = 0;
inline constexpr uint32_t kFormatVersion = 1;

inline constexpr std::string_view kFormatVersionKey{"formatVersion"};
inline constexpr std::string_view kCapabilitiesKey{"capabilities"};

// The key/value pairs packFormat adds to the common map.
inline constexpr uint32_t kFormatKeyCount = 2;

// A multi-asset checkpoint that marks failed realizations, which a resume keeps stopped.
inline constexpr std::string_view kFailedRealizationsCapability{"failedRealizations"};

// Every capability this binary reads.
inline constexpr std::array<std::string_view, 1> kKnownCapabilities{kFailedRealizationsCapability};

struct CheckpointFormat
{
    uint32_t version{kLegacyFormatVersion};
    std::vector<std::string> capabilities;
};

// The format the common object declares. Throws CheckpointError when either key is malformed.
[[nodiscard]] CheckpointFormat readFormat(const msgpack::object& common);

// The format the common object declares, when this binary reads it; otherwise a CheckpointError
// whose one line names the checkpoint's side and this binary's. Call it before anything is restored
// or set aside, so a refused checkpoint leaves the run directory as it found it.
CheckpointFormat requireReadableFormat(const msgpack::object& common);

template<typename Stream>
void packFormat(msgpack::packer<Stream>& o, std::span<const std::string_view> capabilities)
{
    o.pack(kFormatVersionKey);
    o.pack(kFormatVersion);
    o.pack(kCapabilitiesKey);
    o.pack_array(static_cast<uint32_t>(capabilities.size()));
    for (const auto capability : capabilities) {
        o.pack(capability);
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::checkpoint

//-------------------------------------------------------------------------
