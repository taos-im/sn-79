/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "taosim/decimal/serialization/decimal.hpp"
#include "taosim/serialization/msgpack/common.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

//-------------------------------------------------------------------------

using namespace taosim;

using namespace testing;

//-------------------------------------------------------------------------

struct DecimalSerializationTest : TestWithParam<decimal_t>
{
    virtual void SetUp() override
    {
        refValue = GetParam();
    }

    decimal_t refValue;
};

//-------------------------------------------------------------------------

TEST_P(DecimalSerializationTest, Double)
{
    static const decimal_t epsilon = DEC(1e-7);
    serialization::HumanReadableStream stream;
    msgpack::pack(stream, refValue);
    msgpack::object_handle oh = msgpack::unpack(stream.data(), stream.size());
    msgpack::object deserialized = oh.get();
    EXPECT_TRUE(util::abs(deserialized.as<decimal_t>() - refValue) < epsilon);
}

TEST_P(DecimalSerializationTest, Packed)
{
    serialization::BinaryStream stream;
    msgpack::pack(stream, refValue);
    msgpack::object_handle oh = msgpack::unpack(stream.data(), stream.size());
    msgpack::object deserialized = oh.get();
    EXPECT_EQ(deserialized.as<decimal_t>(), refValue);
}

INSTANTIATE_TEST_SUITE_P(
    DecimalSerializationTests,
    DecimalSerializationTest,
    Values(
        DEC(-293.497),
        DEC(-4.2e-18),
        DEC(3.22),
        DEC(13.37),
        DEC(6.8392581e8)
    ));

//-------------------------------------------------------------------------
// AN INTEGER IS A VALID DECIMAL ON THE WIRE, and refusing one cost a user-visible wrong state.
//
// The adaptor took FLOAT64 or BIN and threw MsgPackError otherwise. A client that sent a whole-number
// volume as an integer (which every msgpack encoder does when the value has no fractional part, and which
// json-derived payloads do routinely) therefore threw during decode. Because the throw happens while
// unpacking the INSTRUCTION, the engine discarded the entire CANCEL_ORDERS batch it belonged to: the UI
// reported the order CANCELLED while it stayed resting with its TAO still held. Same class as the earlier
// ExecutedFill.tradeId bad_cast.
//
// Exactness matters here, so integers are converted directly rather than via double: double2decimal
// truncates to the price grid, and routing a large integer through a double would also lose precision
// above 2^53. 9007199254740993 is 2^53+1, the smallest integer a double cannot represent.

struct DecimalIntegerDeserializationTest : TestWithParam<int64_t>
{
};

TEST_P(DecimalIntegerDeserializationTest, AnIntegerOnTheWireDeserializesExactly)
{
    const int64_t ref = GetParam();
    serialization::BinaryStream stream;
    msgpack::pack(stream, ref);
    msgpack::object_handle oh = msgpack::unpack(stream.data(), stream.size());
    msgpack::object deserialized = oh.get();

    ASSERT_NO_THROW(deserialized.as<decimal_t>())
        << "an integer volume threw during decode, which discards the whole instruction batch";
    EXPECT_EQ(deserialized.as<decimal_t>(), decimal_t{ref});
}

INSTANTIATE_TEST_SUITE_P(
    DecimalSerializationTests,
    DecimalIntegerDeserializationTest,
    Values(
        int64_t{0},
        int64_t{1},
        int64_t{5},
        int64_t{-7},
        int64_t{1000000000},
        int64_t{-1000000000},
        int64_t{9007199254740993}
    ));

//-------------------------------------------------------------------------

TEST(DecimalDeserialization, AnUnsupportedTypeStillThrows)
{
    // The permissive change must not become "accept anything": a string is still not a decimal, and
    // silently coercing one would hide a malformed payload instead of rejecting it.
    serialization::BinaryStream stream;
    msgpack::pack(stream, std::string{"not-a-number"});
    msgpack::object_handle oh = msgpack::unpack(stream.data(), stream.size());
    msgpack::object deserialized = oh.get();
    EXPECT_THROW(deserialized.as<decimal_t>(), taosim::serialization::MsgPackError);
}

//-------------------------------------------------------------------------