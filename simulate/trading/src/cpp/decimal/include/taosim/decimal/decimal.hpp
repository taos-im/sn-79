/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <bdldfp_decimal.h>
#include <bdldfp_decimalconvertutil.h>
#include <bdldfp_decimalutil.h>
#include <fmt/format.h>

#include <spanstream>

//-------------------------------------------------------------------------

#define DEC(lit) BDLDFP_DECIMAL_DL(lit)

//-------------------------------------------------------------------------

namespace taosim
{

using decimal_t = BloombergLP::bdldfp::Decimal128;

struct PackedDecimal
{
    uint8_t data[sizeof(decimal_t)]{};
};

}  // namespace taosim

//-------------------------------------------------------------------------

namespace taosim::util
{

inline constexpr uint32_t kDefaultDecimalPlaces = 8;

[[nodiscard]] inline decimal_t round(
    decimal_t val, uint32_t decimalPlaces = kDefaultDecimalPlaces)
{
    return BloombergLP::bdldfp::DecimalUtil::trunc(val, decimalPlaces);
}

[[nodiscard]] inline decimal_t roundUp(decimal_t val, uint32_t decimalPlaces)
{
    using namespace BloombergLP::bdldfp;
    const auto factor = DecimalUtil::multiplyByPowerOf10(decimal_t{1}, decimalPlaces);
    return DecimalUtil::ceil(val * factor) / factor;
}

[[nodiscard]] inline double decimal2double(decimal_t val)
{
    return BloombergLP::bdldfp::DecimalConvertUtil::decimalToDouble(val);
}

[[nodiscard]] inline decimal_t double2decimal(
    double val, uint32_t decimalPlaces = kDefaultDecimalPlaces)
{
    return round(decimal_t{val}, decimalPlaces);
}

//-------------------------------------------------------------------------

// An on-chain amount is an integer count of rao, and rao is 1e-9, so NINE decimal places are required
// to hold one exactly. Eight cannot, and the loss is not theoretical: a settled 0.999496465 alpha was
// recorded as 0.99949646 and a settled 0.999496453 as 0.99949645.
inline constexpr uint32_t kChainDecimalPlaces = 9;

//-------------------------------------------------------------------------

// Convert a double that carries an on-chain amount.
//
// Deliberately ROUNDS where double2decimal truncates, and that is the substance of this function. The
// value reaches us as `rao / 1e9` computed in Python, and that double can sit a hair BELOW the exact
// decimal: 6835900/1e9 is 0.006835899999999999789..., not 0.0068359. Truncation therefore drops a whole
// unit in the last place and turns an exactly-settled 0.0068359 into 0.00683589, a 1e-8 error on a
// number the chain knew exactly. Truncating at nine places is no better; it yields 0.006835899.
//
// Rounding recovers 0.006835900. Chain amounts are non-negative, so trunc(x + half a quantum) is
// round-half-up, which reuses the existing helper rather than introducing a second rounding mode.
//
// double2decimal is left alone. Its truncation is the order-grid behaviour the simulation relies on,
// and this concerns settled chain amounts only.
[[nodiscard]] inline decimal_t chain2decimal(double val)
{
    static const decimal_t halfQuantum = BloombergLP::bdldfp::DecimalUtil::multiplyByPowerOf10(
        decimal_t{5}, -static_cast<int>(kChainDecimalPlaces + 1));
    return round(decimal_t{val} + halfQuantum, kChainDecimalPlaces);
}

[[nodiscard]] inline PackedDecimal packDecimal(decimal_t val)
{
    PackedDecimal packed;
    BloombergLP::bdldfp::DecimalConvertUtil::decimalToDPD(packed.data, val);
    return packed;
}

[[nodiscard]] inline decimal_t unpackDecimal(PackedDecimal val)
{
    decimal_t unpacked;
    BloombergLP::bdldfp::DecimalConvertUtil::decimalFromDPD(&unpacked, val.data);
    return unpacked;
}

[[nodiscard]] inline decimal_t fma(decimal_t a, decimal_t b, decimal_t c) noexcept
{
    return BloombergLP::bdldfp::DecimalUtil::fma(a, b, c);
}

[[nodiscard]] inline decimal_t pow(decimal_t a, decimal_t b)
{
    return BloombergLP::bdldfp::DecimalUtil::pow(a, b);
}

[[nodiscard]] inline decimal_t dec1p(decimal_t val) noexcept
{
    return 1 + val;
}

[[nodiscard]] inline decimal_t dec1m(decimal_t val) noexcept
{
    return 1 - val;
}

[[nodiscard]] inline decimal_t decInv1p(decimal_t val) noexcept
{
    return 1 / dec1p(val);
}

[[nodiscard]] inline decimal_t abs(decimal_t val) noexcept
{
    return val < decimal_t{} ? -val : val;
}

}  // namespace taosim::util

//-------------------------------------------------------------------------

namespace taosim::literals
{

[[nodiscard]] constexpr decimal_t operator"" _dec(unsigned long long int val)
{
    return decimal_t{val};
}

}  // namespace taosim::literals

//-------------------------------------------------------------------------

static inline void trim(std::span<char> span)
{
    const size_t len = std::strlen(span.data());
    if (len <= 3uz) return;
    size_t i = len - 1;
    while (i > 1 && span[i] == '0' && span[i - 1] != '.') {
        --i;
    }
    span[i + 1] = '\0';
}

template<>
struct fmt::formatter<taosim::decimal_t>
{
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(taosim::decimal_t val, FormatContext& ctx) const
    {
        using namespace taosim::literals;
        char buf[64]{};
        std::ospanstream oss{buf};
        if (val == 0_dec) [[unlikely]] {
            oss << "0.0";
        } else {
            oss << val;
            trim(buf);
        }
        return fmt::format_to(ctx.out(), "{}", buf);
    }
};

//-------------------------------------------------------------------------
