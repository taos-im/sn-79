/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "taosim/decimal/decimal.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <iomanip>
#include <spanstream>
#include <string>
#include <vector>

//-------------------------------------------------------------------------

using namespace taosim;
using namespace taosim::literals;

using namespace testing;

//-------------------------------------------------------------------------

struct RoundUpTestParams
{
    decimal_t value;
    uint32_t decimalPlaces;
    decimal_t refValue;
};

void PrintTo(const RoundUpTestParams& params, std::ostream* os)
{
    *os << fmt::format(
        "{{.value = {}, .decimalPlaces = {}, .refValue = {}}}",
        params.value,
        params.decimalPlaces,
        params.refValue);
}

struct RoundUpTest : TestWithParam<RoundUpTestParams> {};

TEST_P(RoundUpTest, WorksCorrectly)
{
    const auto [value, decimalPlaces, refValue] = GetParam();
    EXPECT_EQ(util::roundUp(value, decimalPlaces), refValue);
}

INSTANTIATE_TEST_SUITE_P(
    DecimalTests,
    RoundUpTest,
    Values(
        RoundUpTestParams{
            .value = DEC(42.32125839), .decimalPlaces = 3, .refValue = DEC(42.322)
        },
        RoundUpTestParams{
            .value = DEC(0.00005100), .decimalPlaces = 4, .refValue = DEC(0.0001)
        },
        RoundUpTestParams{
            .value = DEC(420.6921), .decimalPlaces = 2, .refValue = DEC(420.70)
        },
        RoundUpTestParams{
            .value = DEC(0.0), .decimalPlaces = 10, .refValue = DEC(0.0)
        },
        RoundUpTestParams{
            .value = DEC(-29358.2416619814), .decimalPlaces = 7, .refValue = DEC(-29358.2416619)
        },
        RoundUpTestParams{
            .value = DEC(-420.6921), .decimalPlaces = 2, .refValue = DEC(-420.69)
        },
        RoundUpTestParams{
            .value = DEC(10000.1), .decimalPlaces = 0, .refValue = DEC(10001.0)
        }
    ));

//-------------------------------------------------------------------------

struct PackUnpackTest : TestWithParam<decimal_t> {};

TEST_P(PackUnpackTest, WorksCorrectly)
{
    const decimal_t packee = GetParam();
    const auto packed = util::packDecimal(packee);
    const decimal_t unpacked = util::unpackDecimal(packed);
    EXPECT_EQ(packee, unpacked);
}

INSTANTIATE_TEST_SUITE_P(
    DecimalTests,
    PackUnpackTest,
    Values(
        DEC(0.0),
        DEC(1.337),
        DEC(-32.2),
        DEC(42.0),
        DEC(-69420.0),
        DEC(1.234567890123456e-42)));

//-------------------------------------------------------------------------
// fmt::formatter<decimal_t> used to render inline exactly like this; decimalToChars is the
// extraction of that body and must keep every spelling (relocated here from the retired
// PriceVolume level-text cache's tests).

TEST(DecimalToCharsTest, MatchesLegacyFormatter)
{
    auto legacyFormat = [](decimal_t val) {
        char buf[64]{};
        std::ospanstream oss{buf};
        if (val == 0_dec) {
            oss << "0.0";
        } else {
            oss << std::setprecision(34) << val;
            const size_t len = std::strlen(buf);
            if (len > 3uz && std::memchr(buf, '.', len) != nullptr) {
                size_t i = len - 1;
                while (i > 1 && buf[i] == '0' && buf[i - 1] != '.') {
                    --i;
                }
                buf[i + 1] = '\0';
            }
        }
        return std::string{buf};
    };

    const std::vector<decimal_t> values{
        0_dec, -0_dec, 1_dec, 3000_dec, 260_dec, DEC(300.25), DEC(300.50), DEC(2.0), DEC(0.0001),
        DEC(-0.0001), DEC(-1.5), DEC(123456789.12345678), DEC(0.00000001),
        DEC(0.1234567890123456789012345678901234), DEC(1234567890.123456789012345678901234),
        DEC(0.000000000000000000001), DEC(1000000000000000000000000000.0),
        decimal_t{1} / 3_dec, DEC(2.5) * DEC(1.1), DEC(39.992),
    };
    for (const auto value : values) {
        char buf[util::kDecimalTextCapacity];
        EXPECT_EQ(util::decimalToChars(buf, value), legacyFormat(value));
        EXPECT_EQ(fmt::format("{}", value), legacyFormat(value));
    }
}

//-------------------------------------------------------------------------
