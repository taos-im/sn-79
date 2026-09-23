/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/move.hpp>
#include <range/v3/view/subrange.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/cache1.hpp>
#include <range/v3/view/transform.hpp>

#include <filesystem>
#include <span>
#include <vector>

//-------------------------------------------------------------------------

namespace taosim::filesystem
{

[[nodiscard]] std::vector<std::filesystem::path> collectMatchingPaths(
    const std::filesystem::path& dir, std::predicate<const std::filesystem::path&> auto criterion);

[[nodiscard]] std::vector<std::filesystem::path> collectPaths(const std::filesystem::path& dir);

// Publishes `bytes` to `path` so the file is only ever observed absent, with its previous
// contents, or with `bytes` in full: the data is written to the sibling `<path>.tmp` (same
// filesystem) and renamed over `path`, which std::filesystem::rename does atomically as POSIX
// rename(). On any failure the temp file is removed and a std::filesystem::filesystem_error
// naming both paths is thrown. Crash-safe but not fsync-backed, so a power loss may still lose
// the data. The temp name is deterministic: concurrent writers of one `path` must serialize.
void atomicWrite(const std::filesystem::path& path, std::span<const char> bytes);

}  // namespace taosim::filesystem

//-------------------------------------------------------------------------

namespace taosim::filesystem
{

//-------------------------------------------------------------------------

std::vector<std::filesystem::path> collectMatchingPaths(
    const std::filesystem::path& dir, std::predicate<const std::filesystem::path&> auto criterion)
{
    using std::filesystem::directory_iterator;

    return ranges::subrange(directory_iterator{dir}, directory_iterator{})
        | ranges::views::transform([](auto&& entry) { return entry.path(); })
        | ranges::views::cache1
        | ranges::views::filter(criterion)
        | ranges::views::move
        | ranges::to<std::vector>;
}

//-------------------------------------------------------------------------

}  // namespace taosim::filesystem

//-------------------------------------------------------------------------
