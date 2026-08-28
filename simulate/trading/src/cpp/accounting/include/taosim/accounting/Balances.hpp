/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "CheckpointSerializable.hpp"
#include "FreeInfo.hpp"
#include "JsonSerializable.hpp"
#include "Loan.hpp"
#include "Order.hpp"
#include "common.hpp"
#include <taosim/accounting/common.hpp>
#include "Balance.hpp"
#include "Collateral.hpp"

#include <random>

//-------------------------------------------------------------------------

namespace taosim::accounting
{

//-------------------------------------------------------------------------

struct BalancesDesc
{
    Balance base;
    std::shared_ptr<Balance> quote;
    RoundParams roundParams;
};

struct ReservationAmounts
{
    decimal_t base{};
    decimal_t quote{};
};

//-------------------------------------------------------------------------

class Balances : public JsonSerializable
{
public:
    Balance base{};
    std::shared_ptr<Balance> quote{};
    std::map<OrderID, decimal_t> m_buyLeverages;
    std::map<OrderID, decimal_t> m_sellLeverages;
    std::map<OrderID, Loan> m_loans;
    decimal_t m_quoteLoan{};
    decimal_t m_baseLoan{};
    decimal_t m_quoteCollateral{};
    decimal_t m_baseCollateral{};
    uint32_t m_baseDecimals;
    uint32_t m_quoteDecimals;
    RoundParams m_roundParams;

    Balances() noexcept = default;
    explicit Balances(const BalancesDesc& desc) noexcept;
    Balances(
        Balance base,
        std::shared_ptr<Balance> quote,
        uint32_t baseDecimals,
        uint32_t quoteDecimals) noexcept;

    [[nodiscard]] bool canBorrow(
        decimal_t collateralAmount, decimal_t price, OrderDirection direction) const noexcept;
    [[nodiscard]] bool canFree(OrderID id) const noexcept;
    void releaseReservation(OrderID id, BookId bookId);
    ReservationAmounts freeReservation(OrderID id, decimal_t price, decimal_t bestBid, decimal_t bestAsk,
        OrderDirection direction, BookId bookId, std::optional<decimal_t> amount = {});
    ReservationAmounts makeReservation(OrderID id, decimal_t price, decimal_t bestBid, decimal_t bestAsk,
        decimal_t amount, decimal_t leverage, OrderDirection direction, BookId bookId);
    [[nodiscard]] std::vector<std::pair<OrderID, decimal_t>> commit(
        OrderID orderId, OrderDirection direction, decimal_t amount, decimal_t counterAmount, decimal_t feeBase, decimal_t feeQuote,
        decimal_t bestBid, decimal_t bestAsk, decimal_t marginCallPrice, BookId bookId, SettleFlag settleFlag = SettleType::FIFO);

    [[nodiscard]] decimal_t getLeverage(OrderID id, OrderDirection direction) const noexcept;
    [[nodiscard]] decimal_t getWealth(decimal_t price) const noexcept;
    [[nodiscard]] decimal_t getReservationInQuote(OrderID id, decimal_t price) const noexcept;
    [[nodiscard]] decimal_t getReservationInBase(OrderID id, decimal_t price) const noexcept;
    [[nodiscard]] std::optional<std::reference_wrapper<const Loan>> getLoan(OrderID id) const noexcept;
    [[nodiscard]] decimal_t totalLoanInQuote(decimal_t price) const noexcept;

    [[nodiscard]] auto&& loans(this auto&& self) noexcept { return self.m_loans; }

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static Balances fromJson(const rapidjson::Value& json);
    // rng: when supplied, the pareto wealth draw uses THIS stream instead of a local
    // std::random_device one, which is what makes a seeded run reproducible. Passing nullptr keeps
    // the historical device-seeded behaviour, so runs without an rngSeed are unchanged.
    [[nodiscard]] static Balances fromXML(
        pugi::xml_node node, const RoundParams& roundParams, std::mt19937* rng = nullptr);

private:
    [[nodiscard]] std::vector<std::pair<OrderID, decimal_t>> settleLoan(
        OrderDirection direction, decimal_t amount, decimal_t price, BookId bookId, std::optional<OrderID> marginOrderId = {});

    void borrow(
        OrderID id,
        OrderDirection direction,
        decimal_t amount,
        decimal_t leverage,
        decimal_t bestBid,
        decimal_t bestAsk,
        decimal_t marginCallPrice,
        BookId bookId);

    [[nodiscard]] decimal_t roundAmount(decimal_t amount, OrderDirection direction) const noexcept;
    [[nodiscard]] std::optional<decimal_t> roundAmount(
        std::optional<decimal_t> amount, OrderDirection direction) const noexcept;
    [[nodiscard]] decimal_t roundBase(decimal_t amount) const noexcept;
    [[nodiscard]] decimal_t roundQuote(decimal_t amount) const noexcept;
    [[nodiscard]] decimal_t roundUpAmount(decimal_t amount, OrderDirection direction) const noexcept;
    [[nodiscard]] decimal_t roundUpBase(decimal_t amount) const noexcept;
    [[nodiscard]] decimal_t roundUpQuote(decimal_t amount) const noexcept;
    void checkNegative(std::source_location sl, OrderID id, BookId bookId);
};

//-------------------------------------------------------------------------

}  // namespace taosim::accounting

//-------------------------------------------------------------------------

template<>
struct fmt::formatter<taosim::accounting::Balances>
{
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const taosim::accounting::Balances& bals, FormatContext& ctx) const
    {
        return fmt::format_to(ctx.out(), "Balances{{base: {}, quote: {}}}", bals.base, *bals.quote);
    }
};

//-------------------------------------------------------------------------