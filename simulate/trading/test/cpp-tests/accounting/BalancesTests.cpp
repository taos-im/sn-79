/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "taosim/accounting/Balances.hpp"
#include "taosim/accounting/margin_utils.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

//-------------------------------------------------------------------------

using namespace taosim;
using namespace taosim::accounting;
using namespace taosim::literals;

using namespace testing;

//-------------------------------------------------------------------------

static constexpr RoundParams s_roundParams{
    .baseDecimals = 4,
    .quoteDecimals = 8
};

struct BalancesCreationDesc
{
    decimal_t base;
    decimal_t quote;
};

[[nodiscard]] static Balances makeBalances(const BalancesCreationDesc& desc)
{
    return Balances({
        .base = Balance(desc.base, "", s_roundParams.baseDecimals),
        .quote = std::make_shared<Balance>(desc.quote, "", s_roundParams.quoteDecimals),
        .roundParams = s_roundParams
    });
}

//-------------------------------------------------------------------------

struct CanBorrowTestParams
{
    decimal_t baseHeld;
    decimal_t quoteHeld;
    decimal_t collateralAmount;
    decimal_t price;
    OrderDirection direction;
    bool refValue;
};

void PrintTo(const CanBorrowTestParams& params, std::ostream* os)
{
    *os << fmt::format(
        "{{.baseHeld = {}, .quoteHeld = {}, .collateralAmount = {}, .price = {}, "
        ".direction = {}, .refValue = {}}}",
        params.baseHeld,
        params.quoteHeld,
        params.collateralAmount,
        params.price,
        params.direction,
        params.refValue);
}

struct CanBorrowTest : TestWithParam<CanBorrowTestParams>
{
    virtual void SetUp() override
    {
        params = GetParam();
        balances = makeBalances({.base = params.baseHeld, .quote = params.quoteHeld});
    }

    CanBorrowTestParams params;
    Balances balances;
};

TEST_P(CanBorrowTest, WorksCorrectly)
{
    EXPECT_EQ(
        balances.canBorrow(params.collateralAmount, params.price, params.direction),
        params.refValue);
}

INSTANTIATE_TEST_SUITE_P(
    BalancesTests,
    CanBorrowTest,
    Values(
        CanBorrowTestParams{
            .baseHeld = DEC(5.5),
            .quoteHeld = DEC(150.97),
            .collateralAmount = 450_dec,
            .price = DEC(54.04),
            .direction = OrderDirection::BUY,
            .refValue = false
        },
        CanBorrowTestParams{
            .baseHeld = DEC(80.6504),
            .quoteHeld = DEC(0.0054),
            .collateralAmount = DEC(491.85),
            .price = DEC(6.0987),
            .direction = OrderDirection::BUY,
            .refValue = true
        },
        CanBorrowTestParams{
            .baseHeld = DEC(5487.0187),
            .quoteHeld = DEC(1911.204145),
            .collateralAmount = DEC(8700711.96),
            .price = DEC(0.0002198),
            .direction = OrderDirection::SELL,
            .refValue = false
        },
        CanBorrowTestParams{
            .baseHeld = DEC(42.322),
            .quoteHeld = 420_dec,
            .collateralAmount = DEC(28042.3),
            .price = DEC(0.015),
            .direction = OrderDirection::SELL,
            .refValue = true
        }));

//-------------------------------------------------------------------------

struct FreeReservationTestParams
{
    decimal_t baseHeld;
    decimal_t quoteHeld;
    OrderID orderId;
    decimal_t reservationPrice;
    decimal_t reservationAmount;
    decimal_t leverage;
    OrderDirection direction;
    std::optional<decimal_t> freeAmount;
    decimal_t freePrice;
    decimal_t refFreedAmountBase;
    decimal_t refFreedAmountQuote;
    decimal_t refBaseReservedAfterFree;
    decimal_t refQuoteReservedAfterFree;
};

void PrintTo(const FreeReservationTestParams& params, std::ostream* os)
{
    *os << fmt::format(
        "{{.baseHeld = {}, .quoteHeld = {}, .orderId = {}, .reservationPrice = {}, "
        ".reservationAmount = {}, .leverage = {}, .direction = {}, .freeAmount = {}, "
        ".freePrice = {}, .refFreedAmountBase = {}, .refFreedAmountQuote = {}, "
        ".refBaseReservedAfterFree = {}, .refQuoteReservedAfterFree = {}}}",
        params.baseHeld,
        params.quoteHeld,
        params.orderId,
        params.reservationPrice,
        params.reservationAmount,
        params.leverage,
        params.direction,
        params.freeAmount ? fmt::format("{}", *params.freeAmount) : "nullopt",
        params.freePrice,
        params.refFreedAmountBase,
        params.refFreedAmountQuote,
        params.refBaseReservedAfterFree,
        params.refQuoteReservedAfterFree);
}

struct FreeReservationTest : TestWithParam<FreeReservationTestParams>
{
    virtual void SetUp() override
    {
        params = GetParam();
        balances = makeBalances({.base = params.baseHeld, .quote = params.quoteHeld});
        balances.makeReservation(
            params.orderId,
            params.reservationPrice,
            DEC(299.38), // bestBid
            300_dec, // bestAsk
            params.reservationAmount,
            params.leverage,
            params.direction,
            0
        );
    }

    FreeReservationTestParams params;
    Balances balances;
};

TEST_P(FreeReservationTest, WorksCorrectly)
{
    fmt::println("*****Reservations #{}: {} | {} | b:{} q:{} | price:{} | pr:{}  am:{}", 
        params.orderId,
        balances.getReservationInBase(params.orderId, params.reservationPrice),
        balances.getReservationInQuote(params.orderId, params.reservationPrice),
        balances.base.getReservation(params.orderId).value_or(0_dec),
        balances.quote->getReservation(params.orderId).value_or(0_dec),
        params.reservationPrice,
        params.freePrice,
        params.freeAmount.value_or(0_dec)
    );
    const auto freedAmount = balances.freeReservation(
        params.orderId, params.freePrice,
        params.reservationPrice, // bestBid
        params.reservationPrice + 1_dec, // bestAsk
        params.direction, 
        0,
        params.freeAmount);
    
    fmt::println("*****Reservations #{}: {} | {} | b:{} q:{} | price:{} | pr:{}  am:{}", 
        params.orderId,
        balances.getReservationInBase(params.orderId, params.reservationPrice),
        balances.getReservationInQuote(params.orderId, params.reservationPrice),
        balances.base.getReservation(params.orderId).value_or(0_dec),
        balances.quote->getReservation(params.orderId).value_or(0_dec),
        params.reservationPrice,
        params.freePrice,
        params.freeAmount.value_or(0_dec)
    );
    
    EXPECT_EQ(freedAmount.base, params.refFreedAmountBase);
    EXPECT_EQ(freedAmount.quote, params.refFreedAmountQuote);
    EXPECT_EQ(balances.base.getReserved(), params.refBaseReservedAfterFree);
    EXPECT_EQ(balances.quote->getReserved(), params.refQuoteReservedAfterFree);
}

INSTANTIATE_TEST_SUITE_P(
    BalancesTests,
    FreeReservationTest,
    Values(
        FreeReservationTestParams{
            .baseHeld = 0_dec,
            .quoteHeld = 4_dec,
            .orderId = 7,
            .reservationPrice = DEC(1.45917245),
            .reservationAmount = DEC(3.5461),
            .leverage = 0_dec,
            .direction = OrderDirection::BUY,
            .freeAmount = std::nullopt,
            .freePrice = 3_dec,
            .refFreedAmountBase = 0_dec,
            .refFreedAmountQuote = DEC(3.5461),
            .refBaseReservedAfterFree = 0_dec,
            .refQuoteReservedAfterFree = 0_dec
        },
        FreeReservationTestParams{
            .baseHeld = 2_dec,
            .quoteHeld = DEC(6.783156),
            .orderId = 11,
            .reservationPrice = DEC(1.45917245),
            .reservationAmount = DEC(1.9999),
            .leverage = 0_dec,
            .direction = OrderDirection::SELL,
            .freeAmount = DEC(1.9998),
            .freePrice = 2_dec,
            .refFreedAmountBase = DEC(1.9998),
            .refFreedAmountQuote = DEC(0.0),
            .refBaseReservedAfterFree = DEC(0.0001),
            .refQuoteReservedAfterFree = 0_dec
        },
        FreeReservationTestParams{
            .baseHeld = DEC(30.9598),
            .quoteHeld = DEC(59.20595134),
            .orderId = 13,
            .reservationPrice = DEC(0.86570800),
            .reservationAmount = 70_dec,
            .leverage = DEC(0.1),
            .direction = OrderDirection::BUY,
            .freeAmount = std::nullopt,
            .freePrice = DEC(1.34097000),
            .refFreedAmountBase = DEC(12.4685),
            .refFreedAmountQuote = DEC(59.20595134),
            .refBaseReservedAfterFree = 0_dec,
            .refQuoteReservedAfterFree = 0_dec
        },
        FreeReservationTestParams{
            .baseHeld = DEC(0.0795),
            .quoteHeld = DEC(110.42010001),
            .orderId = 17,
            .reservationPrice = DEC(4.20),
            .reservationAmount = DEC(3.22),
            .leverage = DEC(0.2),
            .direction = OrderDirection::SELL,
            .freeAmount = DEC(2.2508),
            .freePrice = DEC(5.98120094),
            .refFreedAmountBase = DEC(0.0456),
            .refFreedAmountQuote = DEC(13.190100),
            .refBaseReservedAfterFree = DEC(0.0339),
            .refQuoteReservedAfterFree = 0_dec
        },
        FreeReservationTestParams{
            .baseHeld = DEC(0.2404),
            .quoteHeld = DEC(66.342608),
            .orderId = 19,
            .reservationPrice = DEC(299.38),
            .reservationAmount = DEC(.462),
            .leverage = DEC(1.79),
            .direction = OrderDirection::SELL,
            .freeAmount = DEC(.46209),
            .freePrice = DEC(299.38),
            .refFreedAmountBase = DEC(0.2404),
            .refFreedAmountQuote = DEC(66.342608),
            .refBaseReservedAfterFree = DEC(0.0),
            .refQuoteReservedAfterFree = 0_dec
        }
    ));

//-------------------------------------------------------------------------

struct MakeReservationTestParams
{
    decimal_t baseHeld;
    decimal_t quoteHeld;
    OrderID orderId;
    decimal_t price;
    decimal_t amount;
    decimal_t leverage;
    OrderDirection direction;
    std::optional<decimal_t> refBaseReservation;
    std::optional<decimal_t> refQuoteReservation;
};

void PrintTo(const MakeReservationTestParams& params, std::ostream* os)
{
    *os << fmt::format(
        "{{.baseHeld = {}, .quoteHeld = {}, .orderId = {}, .price = {}, "
        ".amount = {}, .leverage = {}, .direction = {}, "
        ".refBaseReservation = {}, .refQuoteReseration = {}}}",
        params.baseHeld,
        params.quoteHeld,
        params.orderId,
        params.price,
        params.amount,
        params.leverage,
        params.direction,
        params.refBaseReservation ? fmt::format("{}", *params.refBaseReservation) : "nullopt",
        params.refQuoteReservation ? fmt::format("{}", *params.refQuoteReservation) : "nullopt");
}

struct MakeReservationTest : TestWithParam<MakeReservationTestParams>
{
    virtual void SetUp() override
    {
        params = GetParam();
        balances = makeBalances({.base = params.baseHeld, .quote = params.quoteHeld});
    }

    MakeReservationTestParams params;
    Balances balances;
};

TEST_P(MakeReservationTest, WorksCorrectly)
{
    balances.makeReservation(
        params.orderId, params.price,
        299_dec, // bestBid
        300_dec, // bestAsk
        params.amount, params.leverage, params.direction, 0
    );
    EXPECT_EQ(balances.base.getReservation(params.orderId), params.refBaseReservation);
    EXPECT_EQ(balances.quote->getReservation(params.orderId), params.refQuoteReservation);
    if (params.amount > 0_dec){
        EXPECT_EQ(balances.getLeverage(params.orderId, params.direction), params.leverage);
    } else {
        EXPECT_EQ(balances.getLeverage(params.orderId, params.direction), 0_dec);
    }
}

INSTANTIATE_TEST_SUITE_P(
    BalancesTests,
    MakeReservationTest,
    Values(
        MakeReservationTestParams{
            .baseHeld = 1_dec,
            .quoteHeld = 5_dec,
            .orderId = 3,
            .price = DEC(2.5),
            .amount = 5_dec,
            .leverage = 0_dec,
            .direction = OrderDirection::BUY,
            .refBaseReservation = std::nullopt,
            .refQuoteReservation = 5_dec
        },
        MakeReservationTestParams{
            .baseHeld = 2_dec,
            .quoteHeld = 10_dec,
            .orderId = 5,
            .price = DEC(2.5),
            .amount = DEC(0.5),
            .leverage = 0_dec,
            .direction = OrderDirection::SELL,
            .refBaseReservation = DEC(0.5),
            .refQuoteReservation = std::nullopt
        },
        MakeReservationTestParams{
            .baseHeld = DEC(101.0540),
            .quoteHeld = DEC(598.19490040),
            .orderId = 7,
            .price = DEC(23.95),
            .amount = DEC(650.58957610),
            .leverage = DEC(1.5),
            .direction = OrderDirection::BUY,
            .refBaseReservation = DEC(2.1877),
            .refQuoteReservation = DEC(598.19490040)
        },
        MakeReservationTestParams{
            .baseHeld = DEC(5420.9151),
            .quoteHeld = DEC(10380.75176410),
            .orderId = 11,
            .price = DEC(671.98187777),
            .amount = DEC(5425.0),
            .leverage = DEC(0.87),
            .direction = OrderDirection::SELL,
            .refBaseReservation = DEC(5420.9151),
            .refQuoteReservation = DEC(2744.97877251)
        },
        MakeReservationTestParams{
            .baseHeld = 1_dec,
            .quoteHeld = 5_dec,
            .orderId = 79,
            .price = DEC(2.5),
            .amount = 0_dec,
            .leverage = 1_dec,
            .direction = OrderDirection::BUY,
            .refBaseReservation = std::nullopt,
            .refQuoteReservation = std::nullopt
        },
        MakeReservationTestParams{
            .baseHeld = 3_dec,
            .quoteHeld = 15_dec,
            .orderId = 71,
            .price = DEC(3.5),
            .amount = 0_dec,
            .leverage = 2_dec,
            .direction = OrderDirection::SELL,
            .refBaseReservation = std::nullopt,
            .refQuoteReservation = std::nullopt
        }
    ));
