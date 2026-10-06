/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <functional>
#include <string>
#include <string_view>

//-------------------------------------------------------------------------

namespace taosim::util
{

// Transparent hash for heterogeneous unordered_map<std::string, ...> lookup
// (find by string_view / const char* without materializing a std::string).
struct TransparentStringHash
{
    using is_transparent = void;

    std::size_t operator()(std::string_view s) const noexcept
    {
        return std::hash<std::string_view>{}(s);
    }
    std::size_t operator()(const std::string& s) const noexcept
    {
        return std::hash<std::string_view>{}(s);
    }
    std::size_t operator()(const char* s) const noexcept
    {
        return std::hash<std::string_view>{}(s);
    }
};

}  // namespace taosim::util

//-------------------------------------------------------------------------
