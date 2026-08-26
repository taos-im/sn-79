/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/accounting/AccountRegistry.hpp>
#include "Agent.hpp"
#include <taosim/accounting/BalanceLogger.hpp>
#include <taosim/book/AcdClockRegistry.hpp>
#include <taosim/book/Book.hpp>
#include <taosim/book/BookProcessManager.hpp>
#include "CheckpointSerializable.hpp"
#include "ExchangeAgentConfig.hpp"
#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/matching/ExchangeSignals.hpp>
#include <taosim/matching/SLTPContainer.hpp>
#include "JsonSerializable.hpp"
#include <taosim/book/L2Logger.hpp>
#include <taosim/book/L3EventLogger.hpp>
#include <taosim/book/FeeLogger.hpp>
#include <taosim/message/MessageQueue.hpp>
#include <taosim/message/MultiBookMessagePayloads.hpp>
#include "Order.hpp"
#include <taosim/matching/ClearingManager.hpp>
#include <taosim/util/SubscriptionRegistry.hpp>
#include <taosim/event/L3RecordContainer.hpp>
#include <taosim/event/serialization/CancellationEvent.hpp>
#include <taosim/event/serialization/OrderEvent.hpp>
#include <taosim/event/serialization/TradeEvent.hpp>
#include <taosim/exchange/ExchangeConfig.hpp>
#include <taosim/matching/ReplayEventLogger.hpp>
#include <taosim/net/net.hpp>

#include <boost/asio.hpp>

#include <set>
#include <span>
#include <map>
#include <tuple>

//-------------------------------------------------------------------------

class MultiBookExchangeAgent
    : public Agent,
      public JsonSerializable
{
public:
    MultiBookExchangeAgent(Simulation* simulation) noexcept;

    [[nodiscard]] taosim::accounting::Account& account(const LocalAgentId& agentId);
    [[nodiscard]] taosim::process::Process* process(const std::string& name, BookId bookId);
    [[nodiscard]] taosim::decimal_t getMaintenanceMargin() const noexcept { return m_config2.maintenanceMargin; }
    [[nodiscard]] taosim::decimal_t getMaxLeverage() const noexcept { return m_config2.maxLeverage; }
    [[nodiscard]] taosim::decimal_t getMaxLoan() const noexcept { return m_config2.maxLoan; }
    [[nodiscard]] const taosim::exchange::ExchangeConfig& config2() const noexcept { return m_config2; }
    [[nodiscard]] auto&& retainRecord(this auto&& self) noexcept { return self.m_retainRecord; }
    [[nodiscard]] bool sharedQuoteBalances() const noexcept { return m_orderIdCounter && m_tradeIdCounter; }

    // The shared trade id counter. The exchange service reads it to repair
    // continuity after a checkpoint restore; simulation never calls this.
    [[nodiscard]] auto&& tradeIdCounter(this auto&& self) noexcept { return self.m_tradeIdCounter; }

    // Mint a trade id for a fill the books never matched (an AMM/pool swap settled
    // straight off the reserves).  Those fills used to leave the engine with no id at
    // all, which forced every downstream consumer to invent one, so a single fill
    // ended up with a different identity on every surface.  Drawing from the same
    // counter Book::logTrade uses keeps engine-matched and pool fills in one id space.
    // Returns nullopt when books own private counters (no shared counter configured),
    // since an id from a per-book sequence would collide across books.
    [[nodiscard]] std::optional<TradeID> mintPoolTradeId() noexcept
    {
        return assignTradeId(m_tradeIdCounter);
    }

    // Placement details of the LIMIT instruction that caused a pool swap, kept against the
    // trade id minted for it. A swept marketable order never rests in the book, so when the
    // chain reports a PARTIAL fill there is no order event to rebuild its remainder from; this
    // is the only record of the price, time-in-force and flags the agent actually asked for.
    // Nothing here is derived: it is the submitted order.
    struct PoolPlacement
    {
        AgentId agentId{};
        PlaceOrderLimitPayload::Ptr order;
    };

    // `order` may be NULL, and that is meaningful rather than an error: a MARKET order can
    // never rest a remainder, so recording the id with no payload is how applyCorrections tells
    // "explained, nothing to restore" from "a remainder was owed and lost". An earlier version
    // guarded with `if (!order) return;`, which silently discarded exactly those entries and made
    // the market-order fix inert while looking correct.
    void recordPoolPlacement(
        TradeID id, AgentId agentId, PlaceOrderLimitPayload::Ptr order) noexcept
    {
        m_poolPlacements.insert_or_assign(id, PoolPlacement{.agentId = agentId, .order = order});
        m_poolPlacementOrder.push_back(id);
        // Bounded: a correction arrives within a batch or two, so history beyond that is dead
        // weight on a live exchange carrying 129 books.
        while (m_poolPlacementOrder.size() > s_poolPlacementCapacity) {
            m_poolPlacements.erase(m_poolPlacementOrder.front());
            m_poolPlacementOrder.pop_front();
        }
    }

    [[nodiscard]] const PoolPlacement* poolPlacement(TradeID id) const noexcept
    {
        const auto it = m_poolPlacements.find(id);
        return it != m_poolPlacements.end() ? &it->second : nullptr;
    }

    void forgetPoolPlacement(TradeID id) noexcept { m_poolPlacements.erase(id); }

    [[nodiscard]] auto&& accounts(this auto&& self) noexcept { return self.m_accounts; }
    [[nodiscard]] auto&& books(this auto&& self) noexcept { return self.m_books; }
    [[nodiscard]] auto&& signals(this auto&& self) noexcept { return self.m_signals; }
    [[nodiscard]] auto&& bookProcessManager(this auto&& self) noexcept { return *self.m_bookProcessManager; }
    [[nodiscard]] auto&& clearingManager(this auto&& self) noexcept { return *self.m_clearingManager; }
    [[nodiscard]] auto&& L3Record(this auto&& self) noexcept { return self.m_L3Record; }
    [[nodiscard]] auto&& marginCallCounter(this auto&& self) noexcept { return self.m_marginCallCounter; }
    [[nodiscard]] auto&& localMarketOrderSubs(this auto&& self) noexcept { return self.m_localMarketOrderSubscribers; }
    [[nodiscard]] auto&& localLimitOrderSubs(this auto&& self) noexcept { return self.m_localLimitOrderSubscribers; }
    [[nodiscard]] auto&& localTradeSubs(this auto&& self) noexcept { return self.m_localTradeSubscribers; }
    [[nodiscard]] auto&& localTradeByOrderSubs(this auto&& self) noexcept { return self.m_localTradeByOrderSubscribers; }
    [[nodiscard]] auto&& localOwnTradeSubs(this auto&& self) noexcept { return self.m_localOwnTradeSubscribers; }
    [[nodiscard]] auto&& bookTradeStats(this auto&& self) noexcept { return self.m_bookTradeStats; }
    // Per-book ACD wakeup chains. Exchange-side shared state, reached like bookTradeStats,
    // but never copied into a response payload — see AcdClockRegistry.hpp.
    [[nodiscard]] auto&& acdClocks(this auto&& self) noexcept { return self.m_acdClocks; }
    [[nodiscard]] auto&& sltpContainer(this auto&& self) noexcept { return self.m_sltpContainer; }
    [[nodiscard]] auto&& L2Loggers(this auto&& self) noexcept { return self.m_L2Loggers; }
    [[nodiscard]] auto&& L3EventLoggers(this auto&& self) noexcept { return self.m_L3EventLoggers; }

    void checkMarginCall() noexcept;

    void instructionLogCallback(const taosim::matching::OrderDesc& orderDesc, OrderID orderId);

    virtual void configure(const pugi::xml_node& node) override;
    virtual void receiveMessage(Message::Ptr msg) override;
    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] const taosim::config::ExchangeAgentConfig& config() const noexcept { return m_config; }

private:
    void handleException();

    void handleDistributedMessage(const Message::Ptr&  msg);
    void handleDistributedAgentReset(const Message::Ptr&  msg);
    void handleDistributedPlaceMarketOrder(const Message::Ptr&  msg);
    void handleDistributedPlaceLimitOrder(const Message::Ptr&  msg);
    void handleDistributedRetrieveOrders(const Message::Ptr&  msg);
    void handleDistributedCancelOrders(const Message::Ptr&  msg);
    void handleDistributedClosePositions(const Message::Ptr&  msg);
    void handleDistributedUnknownMessage(const Message::Ptr&  msg);

    void handleLocalMessage(const Message::Ptr&  msg);
    void handleLocalPlaceMarketOrder(const Message::Ptr&  msg);
    void handleLocalPlaceLimitOrder(const Message::Ptr&  msg);
    void handleLocalRetrieveOrders(const Message::Ptr&  msg);
    void handleLocalCancelOrders(const Message::Ptr&  msg);
    void handleLocalClosePositions(const Message::Ptr&  msg);
    void handleLocalRetrieveL1(const Message::Ptr&  msg);
    void handleLocalRetrieveL2(const Message::Ptr&  msg);
    void handleLocalMarketOrderSubscription(const Message::Ptr&  msg);
    void handleLocalLimitOrderSubscription(const Message::Ptr&  msg);
    void handleMinerLimitOrderSubscription(const Message::Ptr&  msg);
    void handleLocalTradeSubscription(const Message::Ptr&  msg);
    void handleLocalRetrieveL1Ext(const Message::Ptr&  msg);
    void handleLocalTradeByOrderSubscription(const Message::Ptr&  msg);
    void handleLocalOwnTradeSubscription(const Message::Ptr&  msg);
    void handleLocalUnknownMessage(const Message::Ptr&  msg);

    void notifyMarketOrderSubscribers(const MarketOrder::Ptr& marketOrder, BookId bookId, AgentId agentId);
    void notifyLimitOrderSubscribers(
        const LimitOrder::Ptr& limitOrder, BookId bookId, AgentId agentId, bool isRemote);
    void notifyTradeSubscribers(const TradeWithLogContext::Ptr& tradeWithCtx);
    void notifyTradeSubscribersByOrderID(const TradeWithLogContext::Ptr& tradeWithCtx, OrderID orderId);

    void orderCallback(Order::Ptr order, OrderContext ctx);
    void orderLogCallback(Order::Ptr order, OrderContext ctx);
    void tradeCallback(Trade::Ptr trade, BookId bookId);
    void unregisterLimitOrderCallback(LimitOrder::Ptr limitOrder, BookId bookId);
    void marketOrderProcessedCallback(MarketOrder::Ptr marketOrder, OrderContext ctx);

    // Parameters.
    taosim::config::ExchangeAgentConfig m_config;
    taosim::exchange::ExchangeConfig m_config2;
    bool m_retainRecord{};
    bool m_replayLog{};
    std::vector<std::unique_ptr<taosim::book::L2Logger>> m_L2Loggers;
    std::vector<std::unique_ptr<taosim::book::L3EventLogger>> m_L3EventLoggers;
    std::vector<std::unique_ptr<taosim::book::FeeLogger>> m_feeLoggers;
    std::vector<std::unique_ptr<taosim::accounting::BalanceLogger>> m_balanceLoggers;
    std::vector<std::unique_ptr<taosim::matching::ReplayEventLogger>> m_replayEventLoggers;
    
    // State.
    taosim::accounting::AccountRegistry m_accounts;
    std::vector<taosim::book::Book::Ptr> m_books;
    std::map<BookId, std::unique_ptr<ExchangeSignals>> m_signals;
    std::unique_ptr<taosim::book::BookProcessManager> m_bookProcessManager;
    std::unique_ptr<taosim::matching::ClearingManager> m_clearingManager;
    taosim::event::L3RecordContainer m_L3Record;
    uint64_t m_marginCallCounter{};
    taosim::util::SubscriptionRegistry<LocalAgentId> m_localMarketOrderSubscribers;
    taosim::util::SubscriptionRegistry<LocalAgentId> m_localLimitOrderSubscribers;
    // Miner-only scope: dispatched EVENT_ORDER_LIMIT only for remote (distributed/miner)
    // orders, so cost-sensitive consumers (e.g. ArbitrageTraderAgent) are never woken by
    // the background-order flood.
    taosim::util::SubscriptionRegistry<LocalAgentId> m_minerLimitOrderSubscribers;
    taosim::util::SubscriptionRegistry<LocalAgentId> m_localTradeSubscribers;
    std::map<OrderID, taosim::util::SubscriptionRegistry<LocalAgentId>> m_localTradeByOrderSubscribers;
    // Own-fill feed: recipients get only the trades they were a party to, on either
    // side, unlike m_localTradeSubscribers which is the whole public tape.
    taosim::util::SubscriptionRegistry<LocalAgentId> m_localOwnTradeSubscribers;
    // Cumulative tape statistics per book, served by RETRIEVE_L1_EXT. Pure
    // observation: nothing here is read back by the matching engine or consumes
    // rng, so accumulating it leaves simulation output bit-identical.
    std::vector<taosim::book::BookTradeStats> m_bookTradeStats;
    // ACD wakeup chains per book, keyed by agent class base name. Held here rather than
    // inside the MagneticField process so that an agent with a clock does not thereby
    // depend on the herding field.
    std::vector<taosim::book::AcdClockRegistry> m_acdClocks;
    std::shared_ptr<OrderID> m_orderIdCounter;
    std::shared_ptr<TradeID> m_tradeIdCounter;
    static constexpr size_t s_poolPlacementCapacity = 4096;
    std::map<TradeID, PoolPlacement> m_poolPlacements;
    std::deque<TradeID> m_poolPlacementOrder;
    taosim::matching::SLTPContainer m_sltpContainer;

    friend class Simulation;
};

//-------------------------------------------------------------------------
