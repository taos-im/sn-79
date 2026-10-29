/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/message/Message.hpp>

#include <msgpack.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

class Simulation;

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
// blockIdx owns canonical ids [blockIdx * blockDim, (blockIdx + 1) * blockDim), and an id
// past the last of `blockCount` blocks belongs to none.
DecanonizeResult decanonize(Message::Ptr msg, uint32_t blockDim, uint32_t blockCount);

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

// Queues a validator response on the block that owns the book it names, or on every block when it names
// no book. One naming a book no block owns is dropped with one line.
void queueValidatorResponse(
    Message::Ptr response, const std::vector<std::unique_ptr<Simulation>>& blocks, uint32_t blockDim);

// An (agent, book) pair the validator reports at or over its volume cap.
using CappedPair = std::pair<AgentId, BookId>;

// The reply batch's "capped" member, [[agent, canonical book], ...], the validator's full set for the round. No
// value when the batch carries none (the engine keeps its last set). A member that is not a list reads as the empty set,
// and an entry that is not a pair of non-negative integers is skipped.
[[nodiscard]] std::optional<std::vector<CappedPair>> readCappedPairs(const msgpack::object& batch);

// The pairs split over the realizations that own their books, with local book ids; every realization receives a
// set, empty where none of its books is capped, and a pair naming a book no realization owns is dropped.
[[nodiscard]] std::vector<std::vector<CappedPair>> splitCappedPairs(
    std::span<const CappedPair> pairs, std::span<const uint32_t> bases, std::span<const uint32_t> counts);

// The same over uniform blocks of `blockDim` books.
[[nodiscard]] std::vector<std::vector<CappedPair>> splitCappedPairs(
    std::span<const CappedPair> pairs, uint32_t blockDim, uint32_t blockCount);

}  // namespace taosim::simulation

//-------------------------------------------------------------------------
