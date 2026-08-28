/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/accounting/Account.hpp>
#include <taosim/book/Book.hpp>
#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include "Order.hpp"
#include <taosim/matching/FeePolicyWrapper.hpp>

#include <expected>

//-------------------------------------------------------------------------

class MultiBookExchangeAgent;

//-------------------------------------------------------------------------

namespace taosim::matching
{

//-------------------------------------------------------------------------

class OrderPlacementValidator
{
public:
    struct Result
    {
        OrderDirection direction;
        decimal_t amount;
        decimal_t leverage;
        decimal_t orderSize;
        bool instantTrade;
    };

    struct Parameters
    {
        uint32_t volumeIncrementDecimals = 8;
        uint32_t priceIncrementDecimals = 4;
        uint32_t baseIncrementDecimals = 8;
        uint32_t quoteIncrementDecimals = 10;
    };

    using ExpectedResult = std::expected<Result, OrderErrorCode>;

    OrderPlacementValidator(const Parameters& params, MultiBookExchangeAgent* exchange) noexcept;

    [[nodiscard]] auto&& parameters(this auto&& self) noexcept { return self.m_params; }

    [[nodiscard]] ExpectedResult validateMarketOrderPlacement(
        const accounting::Account& account,
        taosim::book::Book::Ptr book,
        PlaceOrderMarketPayload::Ptr payload,
        FeePolicyWrapper& feePolicy,
        decimal_t maxLeverage,
        decimal_t maxLoan,
        AgentId agentId) const;

    [[nodiscard]] ExpectedResult validateLimitOrderPlacement(
        const accounting::Account& account,
        taosim::book::Book::Ptr book,
        PlaceOrderLimitPayload::Ptr payload,
        FeePolicyWrapper& feePolicy,
        decimal_t maxLeverage,
        decimal_t maxLoan,
        AgentId agentId) const;

private:
    [[nodiscard]] bool checkTimeInForce(
        taosim::book::Book::Ptr book, PlaceOrderLimitPayload::Ptr payload, AgentId agentId, decimal_t takerFeeRate) const noexcept;
    [[nodiscard]] bool checkIOC(
        taosim::book::Book::Ptr book, PlaceOrderLimitPayload::Ptr payload, AgentId agentId, decimal_t takerFeeRate) const noexcept;
    [[nodiscard]] bool checkFOK(
        taosim::book::Book::Ptr book, PlaceOrderLimitPayload::Ptr payload, AgentId agentId, decimal_t takerFeeRate) const noexcept;
    [[nodiscard]] bool checkPostOnly(
        taosim::book::Book::Ptr book, PlaceOrderLimitPayload::Ptr payload, AgentId agentId, decimal_t takerFeeRate) const noexcept;
    [[nodiscard]] bool checkMinOrderSizeLimit(PlaceOrderLimitPayload::Ptr payload) const noexcept;
    // The chain settles in the quote currency and refuses amounts below its minimum stake, so an order
    // whose notional sits under that floor could rest, be hit, and then fail settlement. Inert when the
    // floor is undeclared, which is the case in simulation mode, and inert for local agents, whose
    // fills never reach the chain.
    [[nodiscard]] bool checkMinQuoteOrderSize(AgentId agentId, decimal_t quoteNotional) const noexcept;

    Parameters m_params;
    MultiBookExchangeAgent* m_exchange;
};

//-------------------------------------------------------------------------

}  // namespace taosim::matching

//-------------------------------------------------------------------------
