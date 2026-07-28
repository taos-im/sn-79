/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/checkpoint/serialization/matching/DynamicFeePolicy.hpp>
#include <taosim/checkpoint/serialization/matching/TieredFeePolicy.hpp>
#include <taosim/matching/ZeroFeePolicy.hpp>
#include <taosim/serialization/msgpack/utils.hpp>

//-------------------------------------------------------------------------

namespace taosim::checkpoint::serialization
{

//-------------------------------------------------------------------------

void packFeePolicy(auto& o, const taosim::matching::FeePolicy& feePolicy)
{
    if (auto fp = dynamic_cast<const taosim::matching::DynamicFeePolicy*>(&feePolicy)) {
        o.pack(*fp);
    }
    else if (auto fp = dynamic_cast<const taosim::matching::TieredFeePolicy*>(&feePolicy)) {
        o.pack(*fp);
    }
    else {
        // ZeroFeePolicy (and any other stateless policy): pack a tagged, stateless map
        // so unpackFeePolicy can round-trip it. Packing nil here made the loader throw on
        // restore (nil carries no "type"), which aborted the entire clearingManager section.
        o.pack_map(1);
        o.pack("type");
        o.pack("zero");
    }
}

void unpackFeePolicy(const auto& o, taosim::matching::FeePolicy& feePolicy)
{
    const auto typeOpt = taosim::serialization::msgpackFindMap<std::string_view>(o, "type");
    if (!typeOpt) {
        // Legacy checkpoints packed stateless/unknown policies as nil (no "type"). A
        // stateless policy has nothing to restore, so treat a missing type as a no-op
        // rather than fatal — this is what previously reset the whole exchange.
        return;
    }
    auto type = *typeOpt;

    // Guard the dynamic_cast: if the live policy type differs from the checkpoint's
    // (fee-policy config changed between runs), the cast is null — dereferencing it
    // would segfault (uncatchable), so skip and keep the constructed default instead.
    if (type == "dynamic") {
        if (auto ptr = dynamic_cast<taosim::matching::DynamicFeePolicy*>(&feePolicy)) {
            o.convert(*ptr);
        }
    }
    else if (type == "tiered") {
        if (auto ptr = dynamic_cast<taosim::matching::TieredFeePolicy*>(&feePolicy)) {
            o.convert(*ptr);
        }
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::matching::serialization

//-------------------------------------------------------------------------