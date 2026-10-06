/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "json_util.hpp"
#include <taosim/mp/mp.hpp>

#include <fmt/format.h>

#include <optional>

//-------------------------------------------------------------------------
// TODO: Make this a C++20 concept instead.

class JsonSerializable
{
public:
    virtual ~JsonSerializable() noexcept = default;

    virtual void jsonSerialize(rapidjson::Document& json, const std::string& key = {}) const = 0;

protected:
    JsonSerializable() noexcept = default;
};

//-------------------------------------------------------------------------

namespace taosim::json
{

namespace
{

template<typename T>
concept IsJsonSerializableValue =
    requires (T t, rapidjson::Document& json, const std::string& key) {
        { t.jsonSerialize(json, key) };
    };

template<typename T>
concept IsJsonSerializablePointer =
    (mp::IsPointer<T> && requires (T t, rapidjson::Document& json, const std::string& key) {
        { t->jsonSerialize(json, key) };
    });

}  // namespace

template<typename T>
concept IsJsonSerializable = IsJsonSerializableValue<T> || IsJsonSerializablePointer<T>;

[[nodiscard]] std::string jsonSerializable2str(
    const IsJsonSerializable auto& serializable, const FormatOptions& formatOptions = {})
{
    rapidjson::Document json;
    if constexpr (mp::IsPointer<decltype(serializable)>) {
        serializable->jsonSerialize(json);
    } else {
        serializable.jsonSerialize(json);
    }
    return json2str(json, formatOptions);
}

}  // namespace taosim::json

//-------------------------------------------------------------------------
// Lazy fmt formatting for json-serializable pointers. Passing the pointer to a
// formatting call (e.g. logDebug("{}", ptr)) defers the JSON serialization to
// the moment the text is actually emitted, instead of building — and then
// discarding — the string eagerly at the call site when logging is disabled.
// Constrained to pointers so it never collides with value formatters.

template<typename T>
    requires (taosim::mp::IsPointer<T> && taosim::json::IsJsonSerializable<T>)
struct fmt::formatter<T>
{
    constexpr auto parse(fmt::format_parse_context& ctx) const noexcept { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const T& serializable, FormatContext& ctx) const
    {
        return fmt::format_to(ctx.out(), "{}", taosim::json::jsonSerializable2str(serializable));
    }
};

//-------------------------------------------------------------------------
