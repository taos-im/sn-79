/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <deque>
#include "CSVPrintable.hpp"
#include "IHumanPrintable.hpp"

#include <taosim/book/BookSignals.hpp>
#include <taosim/book/OrderContainer.hpp>
#include <taosim/book/TickContainer.hpp>
#include "common.hpp"

//-------------------------------------------------------------------------

class Simulation;

//-------------------------------------------------------------------------

namespace taosim::book
{

//-------------------------------------------------------------------------

class Book : public CSVPrintable, public JsonSerializable
{
public:
    using Ptr = std::shared_ptr<Book>;
    using OptLevelRef = std::optional<std::reference_wrapper<TickContainer>>;
    using OptConstLevelRef = std::optional<std::reference_wrapper<const TickContainer>>;

    Book(
        Simulation* simulation,
        BookId id,
        size_t maxDepth,
        size_t detailedDepth,
        std::shared_ptr<OrderID> orderIdCounter = {},
        std::shared_ptr<TradeID> tradeIdCounter = {});

    [[nodiscard]] BookId id() const noexcept { return m_id; }
    [[nodiscard]] size_t maxDepth() const noexcept { return m_maxDepth; }
    [[nodiscard]] size_t detailedDepth() const noexcept { return m_detailedDepth; }
    [[nodiscard]] BookSignals& signals() noexcept { return m_signals; }
    [[nodiscard]] auto&& buyQueue(this auto&& self) noexcept { return self.m_buyQueue; }
    [[nodiscard]] auto&& sellQueue(this auto&& self) noexcept { return self.m_sellQueue; }
    [[nodiscard]] auto&& orderIdCounter(this auto&& self) noexcept { return self.m_orderIdCounter; }
    [[nodiscard]] auto&& tradeIdCounter(this auto&& self) noexcept { return self.m_tradeIdCounter; }
    [[nodiscard]] auto&& orderToClientInfo(this auto&& self) noexcept { return self.m_order2clientCtx; }
    [[nodiscard]] auto&& orderIdMap(this auto&& self) noexcept { return self.m_orderIdMap; }
    [[nodiscard]] taosim::decimal_t midPrice() const noexcept;
    [[nodiscard]] taosim::decimal_t bestBid() const noexcept;
    [[nodiscard]] taosim::decimal_t bestAsk() const noexcept;
    [[nodiscard]] OptLevelRef bestBuyLevel() const noexcept;
    [[nodiscard]] OptLevelRef bestSellLevel() const noexcept;

    template<typename... Args>
    [[nodiscard]] MarketOrder::Ptr placeMarketOrder(OrderClientContext ctx, Args&&... args);

    template<typename... Args>
    [[nodiscard]] LimitOrder::Ptr placeLimitOrder(OrderClientContext ctx, Args&&... args);

    template<typename... Args>
    void logTrade(Args&&... args);

    void placeOrder(const MarketOrder::Ptr& order);
    void placeOrder(const LimitOrder::Ptr& order);
    void placeLimitBuy(const LimitOrder::Ptr& order);
    void placeLimitSell(const LimitOrder::Ptr& order);
    bool cancelOrder(OrderID orderId, std::optional<taosim::decimal_t> volumeToCancel = {});
    // Restore un-executed volume onto a still-resting order (partial-fill reconciliation).
    // Updates BOTH the order and its price-level aggregate so the L2 level stays consistent
    // (a raw setVolume would desync the level and later underflow it on cancel/match).
    bool restoreRestingOrderVolume(OrderID orderId, taosim::decimal_t deltaVolume);
    [[nodiscard]] std::optional<LimitOrder::Ptr> getOrder(OrderID orderId) const;
    void registerLimitOrder(const LimitOrder::Ptr& order);
    void unregisterLimitOrder(const LimitOrder::Ptr& order);
    taosim::decimal_t processAgainstTheBuyQueue(const Order::Ptr& order, taosim::decimal_t minPrice);
    taosim::decimal_t processAgainstTheSellQueue(const Order::Ptr& order, taosim::decimal_t maxPrice);
    [[nodiscard]] taosim::book::TickContainer* preventSelfTrade(
        taosim::book::TickContainer* queue, const LimitOrder::Ptr& iop, const Order::Ptr& order, AgentId agentId);
    void clearFilledOrders() noexcept;
    void printCSV(uint32_t depth) const;

    virtual void printCSV() const override { printCSV(5); }
    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    void invalidateTopOfBook() noexcept { m_topOfBook.invalidate(); }

    [[nodiscard]] taosim::decimal_t bandRef() const noexcept;
    void recordBandTrade(Timestamp ts, taosim::decimal_t price, AgentId maker, AgentId taker) noexcept;
    void sampleBandRef(Timestamp ts) noexcept;
    [[nodiscard]] taosim::decimal_t bandLimit(bool isBuy) const noexcept;

private:
    struct TopOfBook
    {
        std::optional<std::reference_wrapper<TickContainer>> bestBuyLevel;
        std::optional<std::reference_wrapper<TickContainer>> bestSellLevel;
        bool isDirty{true};

        void invalidate() noexcept { isDirty = true; }
    };

    void refreshTopOfBook() const noexcept;

    void dumpCSVLOB(auto begin, auto end, uint32_t depth) const;

    void emitL2Signal([[maybe_unused]] auto&&... args) const { m_signals.L2(this); }
    void setupL2Signal();

    Simulation* m_simulation;
    BookId m_id;
    // Slow trailing reference for the price band. Deliberately NOT bestBid/bestAsk: a band referenced to
    // top-of-book ratchets, because a match consumes the level and the next order gets fresh headroom.
    // This EMA moves only a little per trade, so walking it demands sustained volume rather than a burst.
    // Band reference: the MEDIAN of prices SAMPLED once per bandRefInterval of sim time.
    // Sampling (not per-trade averaging) is deliberate and closes two attacks: a burst of orders inside
    // one interval contributes at most ONE sample, so trade COUNT cannot drag the reference; and a
    // minimum-size "dust" print can only ever be one sample, not a vote proportional to its frequency.
    // The median then requires >50% of the window's samples to be captured, i.e. sustained control over
    // most of bandRefWindow, which is real capital and real time rather than a single batch.
    std::deque<taosim::decimal_t> m_bandSamples;
    taosim::decimal_t m_bandLastPrice{};      // prevailing price; survives quiet periods (never resets)
    taosim::decimal_t m_bandRefCached{};      // median, recomputed only when a sample is pushed
    Timestamp m_bandLastSampleTs{};
    bool m_bandSeeded{false};
    size_t m_maxDepth;
    size_t m_detailedDepth;
    BookSignals m_signals;
    std::shared_ptr<OrderID> m_orderIdCounter;
    std::shared_ptr<TradeID> m_tradeIdCounter;
    std::map<OrderID, LimitOrder::Ptr> m_orderIdMap;
    std::map<OrderID, OrderClientContext> m_order2clientCtx;
    taosim::book::OrderContainer m_buyQueue;
    taosim::book::OrderContainer m_sellQueue;
    mutable TopOfBook m_topOfBook;
    bool m_initMode = false;

    friend class ::Simulation;
};

//-------------------------------------------------------------------------

template<typename... Args>
MarketOrder::Ptr Book::placeMarketOrder(OrderClientContext clientCtx, Args&&... args)
{
    static_assert(std::constructible_from<MarketOrder, OrderID, Args...>);
    const auto marketOrder =
        std::make_shared<MarketOrder>((*m_orderIdCounter)++, std::forward<Args>(args)...);
    m_order2clientCtx.insert({marketOrder->id(), clientCtx});
    m_signals.orderCreated(
        marketOrder, OrderContext{clientCtx.agentId, m_id, clientCtx.clientOrderId});
    placeOrder(marketOrder);
    m_order2clientCtx.erase(marketOrder->id());
    m_signals.orderLog(
        marketOrder, OrderContext{clientCtx.agentId, m_id, clientCtx.clientOrderId});
    return marketOrder;
}

//-------------------------------------------------------------------------

template<typename... Args>
LimitOrder::Ptr Book::placeLimitOrder(OrderClientContext clientCtx, Args&&... args)
{
    static_assert(std::constructible_from<LimitOrder, OrderID, Args...>);
    const auto limitOrder =
        std::make_shared<LimitOrder>((*m_orderIdCounter)++, std::forward<Args>(args)...);
    m_order2clientCtx.insert({limitOrder->id(), clientCtx});
    m_signals.orderCreated(
        limitOrder, OrderContext{clientCtx.agentId, m_id, clientCtx.clientOrderId});
    placeOrder(limitOrder);
    m_signals.orderLog(
        limitOrder, OrderContext{clientCtx.agentId, m_id, clientCtx.clientOrderId});
    return limitOrder;
}

//-------------------------------------------------------------------------

template<typename... Args>
void Book::logTrade(Args&&... args)
{
    static_assert(std::constructible_from<Trade, TradeID, Args...>);
    // Assigned through the one shared entry point, so pool fills and matched fills
    // cannot drift into separate id spaces. A book always has a counter here.
    const auto trade = Trade::create(
        assignTradeId(m_tradeIdCounter).value(), std::forward<Args>(args)...);
    m_signals.trade(trade, m_id);
}

//-------------------------------------------------------------------------

void Book::dumpCSVLOB(auto begin, auto end, uint32_t depth) const
{
    while (depth > 0 && begin != end) {
        const taosim::decimal_t totalVolume = [&] {
            // INITIALISE THE ACCUMULATOR. decimal_t is bdldfp::Decimal128, whose default constructor leaves
            // the value INDETERMINATE, so this summed each level starting from garbage. Two symptoms, one
            // cause, both visible in SelfTradePreventionTest.LimitOrderBuyCO:
            //   * level totals off by ~1e-10 (0.9999999999, 6.9999999999, 4.9999999998 where 1, 7, 5 belong)
            //     when the garbage happened to be a tiny negative;
            //   * whole levels MISSING from the dump, because `totalVolume > 0_dec` below tested a garbage
            //     sum and skipped a level that genuinely had volume.
            // Display-only: dumpCSVLOB feeds printCSV, which is test and debug output. Stored order volumes,
            // reservations, fills and accounting never pass through here, which is why the acceptance run's
            // invariants and notional identities all held while this test failed.
            taosim::decimal_t totalVolume = 0_dec;
            for (auto it = begin->begin(); it != begin->end(); ++it) {
                const auto& order = *it;
                totalVolume += order->totalVolume();
            }
            return totalVolume;
        }();

        
        if (totalVolume > 0_dec) {
            std::cout
                << ","
                << begin->price()
                << ","
                << totalVolume;
        }

        --depth;
        ++begin;
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::book

//-------------------------------------------------------------------------
