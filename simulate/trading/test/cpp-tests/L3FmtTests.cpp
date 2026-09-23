/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "L3Fmt.hpp"

#include <taosim/decimal/decimal.hpp>
#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <json_util.hpp>

#include <gtest/gtest.h>

#include <string>
#include <vector>

//-------------------------------------------------------------------------
// The fmt-based L3 emission (util/L3Fmt.*) replaces the rapidjson path
// (L3Serialize + json2str). Structure, key order, enum spellings, null/bool
// conventions are identical; numbers are printed from the DECIMAL digits
// (fmt::formatter<decimal_t>: quantum digits, trailing zeros trimmed) instead of
// a double round-trip, which drops rapidjson's 8-decimal truncation and its
// scientific notation for tiny values — the new text always round-trips the true
// decimal. Line-level equality against the legacy renderer is asserted on the
// value domain where the two conventions coincide (fractional quanta, <= 8 dp).

using namespace taosim::literals;

namespace
{

template<typename Item>
std::string legacyLine(const Item& item, BookId bookIdCanon, uint64_t eventId)
{
    rapidjson::Document json;
    item.L3Serialize(json);
    if constexpr (!std::same_as<Item, taosim::InstructionLogContext>) {
        json["g"]["b"].SetUint(bookIdCanon);
    }
    json.AddMember("k", rapidjson::Value{eventId}, json.GetAllocator());
    return taosim::json::json2str(json);
}

std::string fmtLine(const auto& item, BookId bookIdCanon, uint64_t eventId)
{
    fmt::memory_buffer buf;
    if constexpr (std::same_as<std::remove_cvref_t<decltype(item)>,
                               taosim::InstructionLogContext>) {
        taosim::l3fmt::formatItem(buf, item, eventId);
    } else {
        taosim::l3fmt::formatItem(buf, item, bookIdCanon, eventId);
    }
    return fmt::to_string(buf);
}

}  // namespace

//-------------------------------------------------------------------------

TEST(L3FmtTests, DecimalRendering)
{
    const std::vector<std::pair<taosim::decimal_t, const char*>> cases{
        {DEC(0.0), "0.0"},                       // zero always renders as 0.0
        {DEC(0.25), "0.25"},
        {DEC(9.4993), "9.4993"},
        {DEC(298.9855), "298.9855"},
        {DEC(300.00), "300.0"},                  // fractional quantum trims to one digit
        {DEC(3000.0000), "3000.0"},
        {DEC(3000.), "3000"},                    // integral quantum stays intact (trim fix)
        {DEC(0.10500), "0.105"},
        {DEC(2.0), "2.0"},
        {DEC(0.0001), "0.0001"},
        {DEC(0.0207015395), "0.0207015395"},     // full 10 dp kept (legacy truncated at 8)
        {DEC(0.0000001), "1e-07"},               // bdldfp natural form for tiny values
        {DEC(0.0000000001), "1e-10"},            // preserved (legacy: clipped to 0.0)
        {-DEC(13.5693), "-13.5693"},
        {-DEC(0.0000000125), "-1.25e-08"},
    };
    for (const auto& [value, expected] : cases) {
        EXPECT_EQ(fmt::format("{}", value), expected) << "expected: " << expected;
    }
}

//-------------------------------------------------------------------------

TEST(L3FmtTests, OrderLinesMatchLegacy)
{
    const auto ctx = std::make_shared<OrderLogContext>(-42, 3);

    const std::vector<Order::Ptr> orders{
        std::make_shared<MarketOrder>(
            7, Timestamp{123456789}, DEC(9.4993), OrderDirection::BUY),
        std::make_shared<MarketOrder>(
            8, Timestamp{1}, DEC(0.0001), OrderDirection::SELL, DEC(1.5),
            STPFlag::DC, SettleFlag{OrderID{55}}, Currency::QUOTE, 0_dec,
            std::make_optional(DEC(290.01)), std::make_optional(DEC(310.99)),
            std::make_optional(DEC(0.125))),
        std::make_shared<LimitOrder>(
            42997, Timestamp{11177778068}, DEC(0.6551), OrderDirection::BUY,
            DEC(299.42)),
        std::make_shared<LimitOrder>(
            9, Timestamp{2}, DEC(0.25), OrderDirection::SELL, DEC(298.9855),
            DEC(2.0), STPFlag::CN, SettleFlag{SettleType::NONE}, true,
            taosim::TimeInForce::GTT, std::make_optional(Timestamp{45'000'000'000}),
            Currency::BASE, std::nullopt, std::make_optional(DEC(310.55)),
            std::nullopt),
    };

    for (const auto& order : orders) {
        const OrderWithLogContext item{order, ctx};
        EXPECT_EQ(fmtLine(item, 35, 7235), legacyLine(item, 35, 7235));
    }
}

//-------------------------------------------------------------------------

TEST(L3FmtTests, TradeLinesMatchLegacy)
{
    const std::vector<std::shared_ptr<TradeLogContext>> contexts{
        std::make_shared<TradeLogContext>(
            -35, -147, 2, taosim::matching::Fees{.maker = 0_dec, .taker = DEC(0.9331155)}),
        std::make_shared<TradeLogContext>(
            25, -92, 0, taosim::matching::Fees{
                .maker = -DEC(0.05970731), .taker = DEC(0.53783318)}),
    };
    for (const auto& logCtx : contexts) {
        const TradeWithLogContext item{
            Trade::create(
                10, Timestamp{2276407}, OrderDirection::SELL, 3553, 3007,
                DEC(13.5693), DEC(298.9855)),
            logCtx};
        EXPECT_EQ(fmtLine(item, 32, 1769), legacyLine(item, 32, 1769));
    }
}

//-------------------------------------------------------------------------

TEST(L3FmtTests, CancellationLinesMatchLegacy)
{
    for (const auto& volume :
         {std::optional<taosim::decimal_t>{}, std::make_optional(DEC(0.5316))}) {
        const CancellationWithLogContext item{
            taosim::event::Cancellation{11542, volume},
            std::make_shared<CancellationLogContext>(-110, 1, Timestamp{505556166122})};
        EXPECT_EQ(fmtLine(item, 17, 24882), legacyLine(item, 17, 24882));
    }
}

//-------------------------------------------------------------------------

TEST(L3FmtTests, AgentResetLinesMatchLegacy)
{
    // Zero cancellations is the case the record exists for: an agent reset with nothing
    // resting on the book leaves no cancellation lines for a reader to attribute.
    for (const auto cancelled : {0u, 1u, 47u}) {
        const AgentResetLogContext item{-110, 1, Timestamp{505556166122}, cancelled};
        EXPECT_EQ(fmtLine(item, 17, 24882), legacyLine(item, 17, 24882));
    }
}

//-------------------------------------------------------------------------

TEST(L3FmtTests, InstructionLinesMatchLegacy)
{
    const auto marketPld = MessagePayload::create<PlaceOrderMarketPayload>(
        OrderDirection::BUY, DEC(0.3427), 0_dec, BookId{97});
    const auto limitPld = MessagePayload::create<PlaceOrderLimitPayload>(
        OrderDirection::SELL, DEC(0.8321), DEC(299.42), DEC(1.25), BookId{16});
    limitPld->clientOrderId = 12345;
    limitPld->timeInForce = taosim::TimeInForce::IOC;

    const taosim::InstructionLogContext marketItem{17, 29591, marketPld};
    const taosim::InstructionLogContext limitItem{-398, 9542, limitPld};

    EXPECT_EQ(fmtLine(marketItem, 0, 1), legacyLine(marketItem, 0, 1));
    EXPECT_EQ(fmtLine(limitItem, 0, 2), legacyLine(limitItem, 0, 2));
}

//-------------------------------------------------------------------------
// Beyond-legacy precision: values past rapidjson's 8-decimal clamp keep their
// digits and still parse back to the exact decimal (the legacy line lost them).

TEST(L3FmtTests, HighPrecisionValuesRoundTrip)
{
    const auto fee = DEC(0.0207015395);
    const TradeWithLogContext item{
        Trade::create(1, Timestamp{1}, OrderDirection::BUY, 2, 3, DEC(0.0001), DEC(300.01)),
        std::make_shared<TradeLogContext>(1, 2, 0, taosim::matching::Fees{
            .maker = 0_dec, .taker = fee})};

    const auto line = fmtLine(item, 0, 1);
    EXPECT_NE(line.find(fmt::format(R"("tk":{})", fee)), std::string::npos) << line;
    // Still valid JSON; the parsed double is the closest representation of all 10 digits.
    const auto parsed = taosim::json::str2json(line);
    EXPECT_DOUBLE_EQ(parsed["g"]["fs"]["tk"].GetDouble(), taosim::util::decimal2double(fee));
}

//-------------------------------------------------------------------------
