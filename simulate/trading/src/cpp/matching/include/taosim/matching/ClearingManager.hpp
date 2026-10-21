/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/accounting/AccountRegistry.hpp>
#include <taosim/accounting/Loan.hpp>
#include <taosim/matching/FeePolicyWrapper.hpp>
#include <taosim/matching/OrderPlacementValidator.hpp>

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <variant>

//-------------------------------------------------------------------------

class MultiBookExchangeAgent;

//-------------------------------------------------------------------------

namespace taosim::matching
{

//-------------------------------------------------------------------------

struct MarketOrderDesc
{
    std::variant<AgentId, LocalAgentId> agentId;
    PlaceOrderMarketPayload::Ptr payload;
};

struct LimitOrderDesc
{
    std::variant<AgentId, LocalAgentId> agentId;
    PlaceOrderLimitPayload::Ptr payload;
};

using OrderDesc = std::variant<
    MarketOrderDesc,
    LimitOrderDesc
>;

struct CancelOrderDesc
{
    BookId bookId;
    LimitOrder::Ptr order;
    decimal_t volumeToCancel;
};

struct ClosePositionDesc
{
    BookId bookId;
    AgentId agentId;
    OrderID orderId;
    std::optional<decimal_t> volumeToClose;
};

struct OrderResult
{
    OrderErrorCode ec;
    decimal_t orderSize;
};

//-------------------------------------------------------------------------

class ClearingManager
{
public:
    struct MarginCallContext
    {
        OrderID orderId;
        AgentId agentId;
    };

    using MarginCallContainer = std::vector<std::map<decimal_t, std::vector<MarginCallContext>>>;

    explicit ClearingManager(
        MultiBookExchangeAgent* exchange,
        size_t bookCount,
        std::unique_ptr<FeePolicyWrapper> feePolicy,
        OrderPlacementValidator::Parameters validatorParams) noexcept;

    [[nodiscard]] MultiBookExchangeAgent* exchange() noexcept;
    [[nodiscard]] accounting::AccountRegistry& accounts() noexcept;

    [[nodiscard]] OrderResult handleOrder(const OrderDesc& orderDesc);
    void handleCancelOrder(const CancelOrderDesc& cancelDesc);
    bool handleClosePosition(const ClosePositionDesc& closeDesc);
    // The size of the market order that closes `loan`. A partial close sends `volumeToClose`. A full close,
    // no volume or one that covers the loan, sends the loan grossed up by the agent's taker rate, since the
    // loan is settled from the order's proceeds net of the taker fee. The sell that closes a long is put on
    // the base grid at the best bid, so its proceeds there cover the amount; nothing without a bid.
    [[nodiscard]] std::optional<decimal_t> closeOrderSize(
        BookId bookId,
        AgentId agentId,
        const accounting::Loan& loan,
        std::optional<decimal_t> volumeToClose = {}) const;
    Fees handleTrade(const TradeDesc& tradeDesc);
    // A called position's agent is told once that its margin call could not be placed. True for the first
    // refusal since the position was called, its call last filled or its entry went; false after that.
    [[nodiscard]] bool firstMarginCallRefusal(AgentId agentId, BookId bookId, OrderID loanId);
    void forgetMarginCallRefusal(AgentId agentId, BookId bookId, OrderID loanId) noexcept;
    void forgetMarginCallRefusals(AgentId agentId) noexcept;
    void updateFeeTiers(Timestamp time) noexcept;

    [[nodiscard]] auto&& marginBuys(this auto&& self) noexcept { return self.m_marginBuy; }
    [[nodiscard]] auto&& marginSells(this auto&& self) noexcept { return self.m_marginSell; }
    [[nodiscard]] FeePolicyWrapper* feePolicy() const noexcept { return m_feePolicy.get(); };

private:
    MultiBookExchangeAgent* m_exchange;
    std::unique_ptr<FeePolicyWrapper> m_feePolicy;
    MarginCallContainer m_marginBuy;
    MarginCallContainer m_marginSell;
    std::set<std::tuple<AgentId, BookId, OrderID>> m_refusedMarginCalls;
    OrderPlacementValidator m_orderPlacementValidator;

    void removeMarginOrders(
        BookId bookId, OrderDirection direction, std::span<std::pair<OrderID, taosim::decimal_t>> ids);
};

//-------------------------------------------------------------------------

}  // namespace taosim::matching

//-------------------------------------------------------------------------