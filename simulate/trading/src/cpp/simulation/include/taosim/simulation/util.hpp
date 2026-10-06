/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/message/Message.hpp>

#include <cstdint>
#include <optional>
#include <span>

//-------------------------------------------------------------------------

namespace taosim::simulation
{

//-------------------------------------------------------------------------

Message::Ptr canonize(Message::Ptr msg, uint32_t blockIdx, uint32_t blockDim);

struct DecanonizeResult
{
    Message::Ptr msg;
    std::optional<uint32_t> blockIdx;
    // The message named a book that no block or realization owns; it is left canonical
    // and belongs nowhere, unlike a message that names no book at all (blockIdx empty,
    // unrouted false), which every block receives.
    bool unrouted{};
};

// Uniform blocks of `blockDim` books, the single-config manager's layout: block
// blockIdx owns canonical ids [blockIdx * blockDim, (blockIdx + 1) * blockDim).
DecanonizeResult decanonize(Message::Ptr msg, uint32_t blockDim);

// Realizations of uneven size numbered consecutively, the multi-asset layout:
// realization i owns canonical ids [bases[i], bases[i] + counts[i]), `bases` ascending
// from zero, each the sum of the book counts before it.
struct RealizationRoute
{
    uint32_t idx;
    uint32_t local;
};

std::optional<RealizationRoute> routeCanonicalBookId(
    uint32_t canon, std::span<const uint32_t> bases, std::span<const uint32_t> counts);

DecanonizeResult decanonize(
    Message::Ptr msg, std::span<const uint32_t> bases, std::span<const uint32_t> counts);

}  // namespace taosim::simulation

//-------------------------------------------------------------------------
