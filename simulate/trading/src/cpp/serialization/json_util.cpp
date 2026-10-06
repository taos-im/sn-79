/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "json_util.hpp"

#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <source_location>

//-------------------------------------------------------------------------

namespace taosim::json
{

//-------------------------------------------------------------------------

std::string json2str(const rapidjson::Value& json, const FormatOptions& formatOptions)
{
    const auto& [indent, decimals] = formatOptions;
    rapidjson::StringBuffer buffer;
    if (indent.has_value()) {
        const auto& opts = indent.value();
        rapidjson::PrettyWriter writer{buffer};
        writer.SetIndent(opts.indentChar, opts.indentCharCount);
        writer.SetMaxDecimalPlaces(decimals);
        json.Accept(writer);
    } else {
        rapidjson::Writer writer{buffer};
        writer.SetMaxDecimalPlaces(decimals);
        json.Accept(writer);
    }
    return buffer.GetString();
}

//-------------------------------------------------------------------------

rapidjson::Document str2json(const std::string& str)
{
    rapidjson::Document json;
    if (json.Parse(str.c_str()).HasParseError()) {
        static constexpr size_t maxCharsShown = 200uz;
        std::string_view facade{str.data(), std::min(maxCharsShown, str.size())};
        throw std::invalid_argument{fmt::format(
            "{}: Error parsing Json string: {}{}",
            std::source_location::current().function_name(),
            facade,
            facade.size() < str.size() ? "..." : "")};
    }
    return json;
}

//-------------------------------------------------------------------------

void dumpJson(
    const rapidjson::Value& json,
    std::ofstream& ofs,
    const FormatOptions& formatOptions)
{
    const auto& [indent, decimals] = formatOptions;
    rapidjson::OStreamWrapper osw{ofs};
    if (indent.has_value()) {
        const auto& opts = indent.value();
        rapidjson::PrettyWriter writer{osw};
        writer.SetIndent(opts.indentChar, opts.indentCharCount);
        writer.SetMaxDecimalPlaces(decimals);
        json.Accept(writer);
        return;
    }
    rapidjson::Writer writer{osw};
    writer.SetMaxDecimalPlaces(decimals);
    json.Accept(writer);
}

//-------------------------------------------------------------------------

rapidjson::Document loadJson(const std::filesystem::path& path)
{
    static constexpr auto ctx = std::source_location::current().function_name();
    if (!std::filesystem::exists(path)) {
        throw std::invalid_argument{fmt::format("{}: No such file '{}'", ctx, path.c_str())};
    }
    std::ifstream ifs{path};
    rapidjson::IStreamWrapper isw{ifs};
    rapidjson::Document json;
    if (json.ParseStream(isw).HasParseError()) {
        throw std::invalid_argument{fmt::format(
            "{}: Unable to parse Json data from '{}'", ctx, path.c_str())};
    }
    return json;
}

//-------------------------------------------------------------------------

decimal_t getDecimal(const rapidjson::Value& json)
{
    if (json.IsArray() && json.GetArray().Size() == 2) [[likely]] {
        const auto& arrJson = json.GetArray();
        const uint64_t left = arrJson[0].GetUint64();
        const uint64_t right = arrJson[1].GetUint64();
        PackedDecimal packed;
        std::memcpy(
            packed.data,
            std::bit_cast<const uint8_t*>(&left),
            sizeof(uint64_t));
        std::memcpy(
            packed.data + sizeof(uint64_t),
            std::bit_cast<const uint8_t*>(&right),
            sizeof(uint64_t));
        return util::unpackDecimal(packed);
    }
    else if (json.IsUint64()) [[unlikely]] {
        return decimal_t{[&] {
            const uint64_t packed = json.GetUint64();
            BloombergLP::bdldfp::Decimal64 res{};
            BloombergLP::bdldfp::DecimalConvertUtil::decimalFromDPD(
                &res, reinterpret_cast<const uint8_t*>(&packed));
            return res;
        }()};
    }
    else if (json.IsDouble()) [[unlikely]] {
        return decimal_t{json.GetDouble()};
    }
    else {
        throw std::invalid_argument{fmt::format(
            "{}: Ill-formed Json value to form a decimal with: {}",
            std::source_location::current().function_name(),
            json2str(json))};
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::json

//-------------------------------------------------------------------------