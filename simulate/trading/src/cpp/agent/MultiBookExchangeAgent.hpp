/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/accounting/AccountRegistry.hpp>
#include "Agent.hpp"
#include <taosim/accounting/BalanceLogger.hpp>
#include <taosim/book/AcdClockRegistry.hpp>
#include <taosim/book/WakeupChainRegistry.hpp>
#include <taosim/book/Book.hpp>
#include <taosim/book/BookProcessManager.hpp>
#include "CheckpointSerializable.hpp"
#include "ExchangeAgentConfig.hpp"
#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/matching/ExchangeSignals.hpp>
#include <taosim/matching/SLTPContainer.hpp>
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
#include <taosim/statshub/StatsHub.hpp>

#include <boost/asio.hpp>

#include <set>
#include <span>
#include <map>
#include <tuple>

//-------------------------------------------------------------------------

class MultiBookExchangeAgent : public Agent
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
    // Publish full account info only for the (account, book) entries changed since the last delivered publish.
    [[nodiscard]] auto&& dirtyAccountsOnly(this auto&& self) noexcept { return self.m_dirtyAccountsOnly; }
    // Under dirtyAccountsOnly, every N-th delivered publish lists an agent in full whatever its marks, a different
    // slice of agents each time (0: never), so a validator that missed a state recovers within N publishes.
    [[nodiscard]] auto&& fullAccountsEvery(this auto&& self) noexcept { return self.m_fullAccountsEvery; }
    // N from the exchange's config; on unless the config turns it off, so no validator copy can drift for good.
    static constexpr uint32_t kFullAccountsEveryDefault = 30;
    [[nodiscard]] static uint32_t fullAccountsEveryOf(pugi::xml_node node) noexcept
    {
        return node.attribute("fullAccountsEvery").as_uint(kFullAccountsEveryDefault);
    }
    // This publish lists the agent in full on every book: dirtyAccountsOnly is off, the validator asked, or the
    // rotation's turn.
    [[nodiscard]] bool accountInFull(AgentId agentId) const noexcept;
    // A change to the account on the book. Under sharedQuoteBalances the quote is one balance for all the
    // agent's books, so every book is marked.
    void markDirty(taosim::accounting::Account& account, BookId bookId) noexcept;
    // The validator answered the published state: marks are cleared, the rotation moves on, a request is served.
    void accountsDelivered();
    // The validator asked for every account in full in the next publish (after a restart, say).
    void requestFullAccounts() noexcept { m_fullAccountsRequested = true; }
    [[nodiscard]] bool sharedQuoteBalances() const noexcept { return m_orderIdCounter && m_tradeIdCounter; }

    // The shared trade id counter. The exchange service reads it to repair
    // continuity after a checkpoint restore; simulation never calls this.
    [[nodiscard]] auto&& tradeIdCounter(this auto&& self) noexcept { return self.m_tradeIdCounter; }
    [[nodiscard]] auto&& orderIdCounter(this auto&& self) noexcept { return self.m_orderIdCounter; }

    // Mint a trade id for a fill the books never matched (an AMM/pool swap settled
    // straight off the reserves).  Without one, such a fill leaves the engine with no id
    // at all, every downstream consumer invents its own, and a single fill ends up
    // with a different identity on every surface.  Drawing from the same
    // counter Book::logTrade uses keeps engine-matched and pool fills in one id space.
    // Returns nullopt when books own private counters (no shared counter configured),
    // since an id from a per-book sequence would collide across books.
    [[nodiscard]] std::optional<TradeID> mintPoolTradeId() noexcept
    {
        return assignTradeId(m_tradeIdCounter);
    }

    // Order id for an instruction the pool executed. Such an order never enters a book, so nothing
    // numbered it. Drawn from the shared counter the books use, so pool and book orders are one id
    // space; nullopt when books own private counters, for the same reason as mintPoolTradeId.
    [[nodiscard]] std::optional<OrderID> mintPoolOrderId() noexcept
    {
        if (!m_orderIdCounter) return std::nullopt;
        return (*m_orderIdCounter)++;
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
        // The id the placement was acknowledged under, so a restored remainder keeps it.
        std::optional<OrderID> orderId;
        // A GTT order's absolute deadline in the sweep's seconds, from the block it was placed in: a remainder
        // restored blocks later continues this order and dies when it would have.
        std::optional<Timestamp> gttDeadline;
    };

    // `order` may be NULL, and that is meaningful rather than an error: a MARKET order can
    // never rest a remainder, so recording the id with no payload is how applyCorrections tells
    // "explained, nothing to restore" from "a remainder was owed and lost". An earlier version
    // guarded with `if (!order) return;`, which silently discarded exactly those entries and made
    // the market-order fix inert while looking correct.
    void recordPoolPlacement(
        TradeID id,
        AgentId agentId,
        PlaceOrderLimitPayload::Ptr order,
        std::optional<OrderID> orderId = std::nullopt) noexcept;

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
    // Lives in the StatsHub, reached the same way l1() is; this stays as the name every
    // existing reader and the checkpoint sections already use.
    [[nodiscard]] auto&& bookTradeStats(this auto&& self) noexcept
    {
        return self.m_statsHub->tradeStats();
    }
    [[nodiscard]] auto&& statsHub(this auto&& self) noexcept { return self.m_statsHub; }
    // Per-book ACD wakeup chains. Exchange-side shared state, reached like bookTradeStats,
    // but never copied into a response payload — see AcdClockRegistry.hpp.
    [[nodiscard]] auto&& acdClocks(this auto&& self) noexcept { return self.m_acdClocks; }
    // Per-book wakeup chain watchdogs. Same placement and the same reason as acdClocks:
    // scheduling state stays exchange-side. See WakeupChainRegistry.hpp.
    [[nodiscard]] auto&& wakeupChains(this auto&& self) noexcept { return self.m_wakeupChains; }
    [[nodiscard]] auto&& sltpContainer(this auto&& self) noexcept { return self.m_sltpContainer; }
    [[nodiscard]] auto&& L2Loggers(this auto&& self) noexcept { return self.m_L2Loggers; }
    [[nodiscard]] auto&& L3EventLoggers(this auto&& self) noexcept { return self.m_L3EventLoggers; }

    void checkMarginCall() noexcept;

    void instructionLogCallback(const taosim::matching::OrderDesc& orderDesc, OrderID orderId);

    virtual void configure(const pugi::xml_node& node) override;
    virtual void receiveMessage(Message::Ptr msg) override;

    [[nodiscard]] const taosim::config::ExchangeAgentConfig& config() const noexcept { return m_config; }

    // OFF-GRID ORDERS, under offGridOrders="refuse": why a miner's order is refused for a price or a
    // size off the book's grid, if it is. Public so every caller that places a miner's order applies the same rule.
    [[nodiscard]] std::optional<OrderErrorCode> offGrid(const PlaceOrderLimitPayload& order) const noexcept;
    [[nodiscard]] std::optional<OrderErrorCode> offGrid(const PlaceOrderMarketPayload& order) const noexcept;

    // The validator's (agent, book) pairs at or over its volume cap, local book ids, replaced with each reply
    // batch that carries them. A placement from a pair in the set is refused when it reaches the exchange.
    void setCappedPairs(std::vector<std::pair<AgentId, BookId>> pairs);
    [[nodiscard]] const auto& cappedPairs() const noexcept { return m_cappedPairs; }

private:
    void handleException();

    // Per-step sweep for token-passing wakeup chains that missed their deadline.
    void sweepWakeupChains();

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

    // A background agent's order rounded to the nearest grid point under offGridOrders="refuse".
    void roundOntoGrid(PlaceOrderLimitPayload& order) const noexcept;
    void roundOntoGrid(PlaceOrderMarketPayload& order) const noexcept;
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
    bool m_dirtyAccountsOnly{};
    uint32_t m_fullAccountsEvery{};
    uint64_t m_accountsDeliveries{};
    bool m_fullAccountsRequested{};
    std::vector<std::unique_ptr<taosim::book::L2Logger>> m_L2Loggers;
    std::vector<std::unique_ptr<taosim::book::L3EventLogger>> m_L3EventLoggers;
    std::vector<std::unique_ptr<taosim::book::FeeLogger>> m_feeLoggers;
    std::vector<std::unique_ptr<taosim::accounting::BalanceLogger>> m_balanceLoggers;
    std::vector<std::unique_ptr<taosim::matching::ReplayEventLogger>> m_replayEventLoggers;
    
    // State.
    taosim::accounting::AccountRegistry m_accounts;
    std::vector<taosim::book::Book::Ptr> m_books;
    std::map<BookId, std::unique_ptr<taosim::matching::ExchangeSignals>> m_signals;
    std::unique_ptr<taosim::stats::StatsHub> m_statsHub;
    std::unique_ptr<taosim::book::BookProcessManager> m_bookProcessManager;
    std::unique_ptr<taosim::matching::ClearingManager> m_clearingManager;
    std::set<std::pair<AgentId, BookId>> m_cappedPairs;
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
    // ACD wakeup chains per book, keyed by agent class base name. Held here rather than
    // inside the MagneticField process so that an agent with a clock does not thereby
    // depend on the herding field.
    std::vector<taosim::book::AcdClockRegistry> m_acdClocks;
    std::vector<taosim::book::WakeupChainRegistry> m_wakeupChains;
    std::shared_ptr<OrderID> m_orderIdCounter;
    std::shared_ptr<TradeID> m_tradeIdCounter;
    static constexpr size_t s_poolPlacementCapacity = 4096;
    std::map<TradeID, PoolPlacement> m_poolPlacements;
    std::deque<TradeID> m_poolPlacementOrder;
    taosim::matching::SLTPContainer m_sltpContainer;

    friend class Simulation;
};

//-------------------------------------------------------------------------
