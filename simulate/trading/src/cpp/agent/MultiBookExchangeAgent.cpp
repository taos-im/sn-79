/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "MultiBookExchangeAgent.hpp"

#include "Simulation.hpp"
#include <json_util.hpp>
#include <taosim/net/net.hpp>
#include <taosim/matching/FeePolicy.hpp>
#include <taosim/util/SLTPDebug.hpp>
#include <taosim/matching/TieredFeePolicy.hpp>
#include <taosim/matching/DynamicFeePolicy.hpp>
#include <taosim/matching/ZeroFeePolicy.hpp>
#include <taosim/book/FeeLogger.hpp>
#include "util.hpp"
#include "InstructionLogger.hpp"
#include <taosim/replay/helpers.hpp>

#include <boost/algorithm/string/erase.hpp>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <date/date.h>
#include <fmt/core.h>
#include <unordered_map>

#include <algorithm>
#include <chrono>
#include <memory>
#include <source_location>
#include <sstream>
#include <string_view>
#include <utility>

//-------------------------------------------------------------------------

MultiBookExchangeAgent::MultiBookExchangeAgent(Simulation* simulation) noexcept
    : Agent{simulation, "EXCHANGE"}
{}

//-------------------------------------------------------------------------

taosim::accounting::Account& MultiBookExchangeAgent::account(const LocalAgentId& agentId)
{
    return m_clearingManager->accounts()[agentId];
}

//-------------------------------------------------------------------------

taosim::process::Process* MultiBookExchangeAgent::process(const std::string& name, BookId bookId)
{
    return m_bookProcessManager->at(name).at(bookId).get();
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::checkMarginCall() noexcept
{
    taosim::decimal_t bestAsk{}, bestBid{};

    for (const auto book : m_books) {
        const auto bookId = book->id();

        //-------------------------- Margin Buying orders ------------------------
        if (!book->buyQueue().empty()){
            
            auto bestBuyDeque = &book->buyQueue().back();
            bestBid = bestBuyDeque->price();
            auto& m_marginBuy = m_clearingManager->marginBuys();
            
            if (!m_marginBuy.empty()){
                auto& marginBuyOrders = m_marginBuy.at(bookId);

                auto it = marginBuyOrders.upper_bound(bestBid);

                for (auto tempIt = it; tempIt != marginBuyOrders.end(); ++tempIt) {
                    const auto& ids = tempIt->second;
                    for (auto idIt = ids.begin(); idIt != ids.end(); ++idIt) {

                        const auto& loan = accounts()[idIt->agentId][bookId].getLoan(idIt->orderId);
                        if (loan.has_value()){
                            taosim::decimal_t remainingVolume = loan->get().amount();

                            simulation()->logDebug("Margin Call for BUY order #{} of agent {} at price {} (marginCall:{}) in Book {} for volume {}x{}",
                                idIt->orderId,
                                idIt->agentId,
                                bestBid,
                                loan->get().marginCallPrice(),
                                simulation()->bookIdCanon(bookId),
                                taosim::util::dec1p(loan->get().leverage()),
                                remainingVolume
                            );

                            taosim::util::SLTPDebugger::log(
                                "margin-call cleanup (BUY): order#{} agent#{} book={}",
                                idIt->orderId, idIt->agentId, bookId);
                            m_sltpContainer.removeOrder(bookId, idIt->orderId);

                            if (idIt->agentId < 0){

                                simulation()->dispatchMessageWithPriority(
                                    simulation()->currentTimestamp(),
                                    0,
                                    accounts().idBimap().right.at(idIt->agentId),
                                    name(),
                                    "PLACE_ORDER_MARKET_MC",
                                    MessagePayload::create<PlaceOrderMarketPayload>(
                                        OrderDirection::SELL,
                                        remainingVolume,
                                        bookId,
                                        Currency::QUOTE,
                                        std::nullopt,
                                        STPFlag::CO,
                                        idIt->orderId
                                    ),
                                    m_marginCallCounter++
                                );

                            } else {

                                simulation()->dispatchMessageWithPriority(
                                    simulation()->currentTimestamp(),
                                    0,
                                    "DISTRIBUTED_PROXY_AGENT",
                                    name(),
                                    "DISTRIBUTED_PLACE_ORDER_MARKET_MC",
                                    MessagePayload::create<DistributedAgentResponsePayload>(
                                        idIt->agentId,
                                        MessagePayload::create<PlaceOrderMarketPayload>(
                                            OrderDirection::SELL,
                                            remainingVolume,
                                            bookId,
                                            Currency::QUOTE,
                                            std::nullopt,
                                            STPFlag::CO,
                                            idIt->orderId
                                        )
                                    ),
                                    m_marginCallCounter++
                                );

                            }

                            

                            // throw std::runtime_error("\n#################    CHECK THE RESULT     ##################\n");
                        }
                    }
                }
                ///##/ This is important to check whether the order successfully happens
                // if it is not then it should be resend
                // if it succeeds it should remove the id from m_marginBuy
                // but if we don't remove it here it might happen several times?
                // this should be checked and the same for m_marginSell ///##/
                marginBuyOrders.erase(it, marginBuyOrders.end());   
            }
        }

        //-------------------------- Short Selling orders ------------------------
        if (!book->sellQueue().empty()){
            auto bestSellDeque = &book->sellQueue().front();
            bestAsk = bestSellDeque->price();
            auto& m_marginSell = m_clearingManager->marginSells();
            
            if (!m_marginSell.empty()){
                auto& marginSellOrders = m_marginSell.at(bookId);

                auto it = marginSellOrders.lower_bound(bestAsk);

                for (auto tempIt = marginSellOrders.begin(); tempIt != it; ++tempIt) {

                    const auto& ids = tempIt->second;
                    
                    for (auto idIt = ids.begin(); idIt != ids.end(); ++idIt) {

                        const auto& loan = accounts()[idIt->agentId][bookId].getLoan(idIt->orderId);
                        if (loan.has_value()){
                            taosim::decimal_t remainingVolume = loan->get().amount();
                            

                            simulation()->logDebug("Margin Call for SELL order #{} of agent {} at price {} (marginCall:{}) in Book {} for volume {}x{}",
                                idIt->orderId,
                                idIt->agentId,
                                bestAsk,
                                loan->get().marginCallPrice(),
                                simulation()->bookIdCanon(bookId),
                                taosim::util::dec1p(loan->get().leverage()),
                                remainingVolume
                            );

                            taosim::util::SLTPDebugger::log(
                                "margin-call cleanup (SELL): order#{} agent#{} book={}",
                                idIt->orderId, idIt->agentId, bookId);
                            m_sltpContainer.removeOrder(bookId, idIt->orderId);

                            if (idIt->agentId < 0){

                                simulation()->dispatchMessageWithPriority(
                                    simulation()->currentTimestamp(),
                                    0,
                                    accounts().idBimap().right.at(idIt->agentId),
                                    name(),
                                    "PLACE_ORDER_MARKET_MC",
                                    MessagePayload::create<PlaceOrderMarketPayload>(
                                        OrderDirection::BUY,
                                        remainingVolume,
                                        bookId,
                                        Currency::BASE,
                                        std::nullopt,
                                        STPFlag::CO,
                                        idIt->orderId
                                    ),
                                    m_marginCallCounter++
                                );

                            } else {

                                simulation()->dispatchMessageWithPriority(
                                    simulation()->currentTimestamp(),
                                    0,
                                    "DISTRIBUTED_PROXY_AGENT",
                                    name(),
                                    "DISTRIBUTED_PLACE_ORDER_MARKET_MC",
                                    MessagePayload::create<DistributedAgentResponsePayload>(
                                        idIt->agentId,
                                        MessagePayload::create<PlaceOrderMarketPayload>(
                                            OrderDirection::BUY,
                                            remainingVolume,
                                            bookId,
                                            Currency::BASE,
                                            std::nullopt,
                                            STPFlag::CO,
                                            idIt->orderId
                                        )
                                    ),
                                    m_marginCallCounter++
                                );
                            }
                        }   
                    }   
                }
                marginSellOrders.erase(marginSellOrders.begin(), it);
            }
        }
    }
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::configure(const pugi::xml_node& node)
{
    static constexpr auto ctx = std::source_location::current().function_name();

    Agent::configure(node);

    m_config.configure(node);
    if (simulation()->blockIdx() == 0) {
        const auto& p = m_config.parameters();
        fmt::println(
            "Exchange precision: price={} volume={} base={} quote={} decimals",
            p.priceIncrementDecimals, p.volumeIncrementDecimals,
            p.baseIncrementDecimals, p.quoteIncrementDecimals);
    }

    // TODO: split this function up.
    try {
        m_config2 = taosim::exchange::makeExchangeConfig(node);

        // Stated at startup so a run's own log is evidence of which floors were active, rather than
        // that having to be inferred from whichever config file is on disk when someone later looks.
        // Must come AFTER m_config2 is assigned: printed before it, this reported 0.0 for both floors
        // on a correctly configured exchange.
        if (simulation()->blockIdx() == 0) {
            fmt::println(
                "Exchange order floors: minOrderSize={} alpha, minQuoteOrderSize={} quote{}",
                m_config2.minOrderSize, m_config2.minQuoteOrderSize,
                m_config2.minQuoteOrderSize <= 0_dec ? " (quote floor DISABLED)" : "");
        }

        const auto booksNode = node.child("Books");
        const uint32_t bookCount = booksNode.attribute("instanceCount").as_uint();
        const size_t maxDepth = booksNode.attribute("maxDepth").as_ullong(21);
        const size_t detailedDepth = booksNode.attribute("detailedDepth").as_ullong(5);

        m_bookProcessManager = taosim::book::BookProcessManager::fromXML(
            booksNode,
            const_cast<Simulation*>(simulation()),
            &m_config2,
            simulation()->sharedResources());

        if (auto feePolicy = std::string_view{node.child("FeePolicy").attribute("type").as_string()};
            feePolicy == "dynamic")
        {
            simulation()->logDebug("DYNAMIC FEE POLICY");
            m_clearingManager = std::make_unique<taosim::matching::ClearingManager>(
                this,
                bookCount,
                std::make_unique<taosim::matching::FeePolicyWrapper>(
                    taosim::matching::DynamicFeePolicy::fromXML(
                        node.child("FeePolicy"), const_cast<Simulation*>(simulation())),
                    &accounts()),
                taosim::matching::OrderPlacementValidator::Parameters{
                    .volumeIncrementDecimals = m_config.parameters().volumeIncrementDecimals,
                    .priceIncrementDecimals = m_config.parameters().priceIncrementDecimals,
                    .baseIncrementDecimals = m_config.parameters().baseIncrementDecimals,
                    .quoteIncrementDecimals = m_config.parameters().quoteIncrementDecimals
                });
        }
        else if (feePolicy == "tiered") {
            m_clearingManager = std::make_unique<taosim::matching::ClearingManager>(
                this,
                bookCount,
                std::make_unique<taosim::matching::FeePolicyWrapper>(
                    taosim::matching::TieredFeePolicy::fromXML(
                        node.child("FeePolicy"), const_cast<Simulation*>(simulation())),
                    &accounts()),
                taosim::matching::OrderPlacementValidator::Parameters{
                    .volumeIncrementDecimals = m_config.parameters().volumeIncrementDecimals,
                    .priceIncrementDecimals = m_config.parameters().priceIncrementDecimals,
                    .baseIncrementDecimals = m_config.parameters().baseIncrementDecimals,
                    .quoteIncrementDecimals = m_config.parameters().quoteIncrementDecimals
                });

            simulation()->logDebug("TIERED FEE POLICY");
            int c = 0;
            if (auto* tiered = dynamic_cast<TieredFeePolicy*>(m_clearingManager->feePolicy()->defaultPolicy())) {
                for (auto& tier : tiered->tiers()) {
                    simulation()->logDebug("TIER {} : VOL >= {} | MAKER {} TAKER {}", c, 
                        tier.volumeRequired, 
                        tier.makerFeeRate, tier.takerFeeRate
                    );
                    c++;
                }
            }
        }
        else {
            simulation()->logDebug("ZERO FEE POLICY");
            m_clearingManager = std::make_unique<taosim::matching::ClearingManager>(
                this,
                bookCount,
                std::make_unique<taosim::matching::FeePolicyWrapper>(
                    std::make_unique<taosim::matching::ZeroFeePolicy>(),
                    &accounts()
                ),
                taosim::matching::OrderPlacementValidator::Parameters{
                    .volumeIncrementDecimals = m_config.parameters().volumeIncrementDecimals,
                    .priceIncrementDecimals = m_config.parameters().priceIncrementDecimals,
                    .baseIncrementDecimals = m_config.parameters().baseIncrementDecimals,
                    .quoteIncrementDecimals = m_config.parameters().quoteIncrementDecimals
                });
        }

        const auto balancesNode = node.child("Balances");
        const auto baseNode = balancesNode.child("Base");
        const auto quoteNode = balancesNode.child("Quote");

        std::chrono::system_clock::time_point startTimePoint;
        std::string L2LogTag;
        int L2Depth;
        std::string L3LogTag;
        std::string feeLogTag;
        pugi::xml_node loggingNode;
        pugi::xml_node L2Node;
        pugi::xml_node L3Node;
        pugi::xml_node feeLogNode;
        pugi::xml_node replayNode;
        if (loggingNode = node.child("Logging")) {
            std::istringstream in{loggingNode.attribute("startDate").as_string()};
            // TODO: Handle the timezone
            date::from_stream(in, "%Y-%m-%d %H:%M:%S", startTimePoint);
            if (L2Node = loggingNode.child("L2")) {
                L2LogTag = L2Node.attribute("tag").as_string();
                L2Depth = L2Node.attribute("depth").as_int(21);
            }
            if (L3Node = loggingNode.child("L3")) {
                L3LogTag = L3Node.attribute("tag").as_string();
            }
            if (feeLogNode = loggingNode.child("FeeLog")) {
                feeLogTag = feeLogNode.attribute("tag").as_string();
            }
            replayNode = loggingNode.child("Replay");
        }

        m_L3Record = taosim::event::L3RecordContainer{bookCount};
        m_bookTradeStats.assign(bookCount, taosim::book::BookTradeStats{});
        m_acdClocks.assign(bookCount, taosim::book::AcdClockRegistry{});

        m_L2Loggers.resize(bookCount);
        m_L3EventLoggers.resize(bookCount);
        m_feeLoggers.resize(bookCount);
        m_sltpContainer.resize(bookCount);

        // SL/TP plumbing: closing market orders go through the agent's
        // normal local message path, so accounting, validation and
        // matching all stay on the standard flow.
        m_sltpContainer.setDispatch(
            [this](const taosim::matching::SLTPEntry& entry, taosim::decimal_t observedPrice) {
                fmt::println("{} | SLTP DISPATCH {} | AGENT #{} BOOK {} : "
                    "closingSide={} volume={} currency={} observedPrice={}",
                    simulation()->currentTimestamp(),
                    entry.isSL ? "SL" : "TP",
                    entry.agentId, simulation()->bookIdCanon(entry.bookId),
                    std::to_underlying(entry.closingSide),
                    entry.volume,
                    std::to_underlying(entry.currency),
                    observedPrice);
                auto pld = MessagePayload::create<PlaceOrderMarketPayload>();
                pld->direction = entry.closingSide;
                // SLTPEntry.volume = trade->volume() (pre-fee BASE). For a close SELL
                // (BUY position), the agent only received volume*(1-takerFee) BASE, so
                // canReserve(volume) can fail by exactly one fee quantum when the agent's
                // free BASE is nearly exhausted.  Cap to actual free BASE so the close
                // always executes and produces a fill/badge even in that edge case.
                pld->volume = entry.volume;
                if (entry.closingSide == OrderDirection::SELL
                        && entry.currency == Currency::BASE
                        && entry.agentId >= 0) {
                    const taosim::decimal_t freeBase =
                        accounts()[entry.agentId][entry.bookId].base.getFree();
                    if (freeBase <= 0_dec) {
                        fmt::println("{} | SLTP CLOSE {} | AGENT #{} BOOK {} : "
                            "no free BASE ({}) — skipping close order",
                            simulation()->currentTimestamp(),
                            entry.isSL ? "SL" : "TP",
                            entry.agentId, simulation()->bookIdCanon(entry.bookId),
                            freeBase);
                        return;
                    }
                    if (pld->volume > freeBase) {
                        fmt::println("{} | SLTP CLOSE {} | AGENT #{} BOOK {} : "
                            "volume {} > freeBase {} — capping to freeBase",
                            simulation()->currentTimestamp(),
                            entry.isSL ? "SL" : "TP",
                            entry.agentId, simulation()->bookIdCanon(entry.bookId),
                            pld->volume, freeBase);
                        pld->volume = freeBase;
                    }
                }
                pld->leverage = entry.leverage;
                pld->bookId = entry.bookId;
                // entry.volume is always in BASE units (trade->volume()).  Using the
                // original order's currency (e.g. QUOTE) would mis-interpret the BASE
                // volume as a QUOTE amount and compute a micro BASE order that fails
                // min-size.  Always close in BASE so the validator uses volume directly.
                pld->currency = Currency::BASE;
                pld->clientOrderId = entry.clientCtx.clientOrderId;
                pld->delegate = entry.clientCtx.delegate;
                pld->skipMinSizeCheck = true;
                pld->closeReason = entry.isSL ? 1 : 2;
                pld->originatingOrderId = entry.originatingOrderId;
                if (entry.agentId < 0) {
                    simulation()->dispatchMessage(
                        simulation()->currentTimestamp(),
                        Timestamp{},
                        accounts().idBimap().right.at(entry.agentId),
                        name(),
                        "PLACE_ORDER_MARKET",
                        pld);
                } else {
                    // Self-dispatch: send to the exchange itself so the close is
                    // processed directly by handleDistributedPlaceMarketOrder without
                    // a validator round-trip.  In sim mode execute() is a no-op, so
                    // routing via DISTRIBUTED_PROXY_AGENT would silently drop the order.
                    simulation()->dispatchMessage(
                        simulation()->currentTimestamp(),
                        Timestamp{},
                        name(),
                        name(),
                        "DISTRIBUTED_PLACE_ORDER_MARKET",
                        MessagePayload::create<DistributedAgentResponsePayload>(
                            entry.agentId,
                            pld));
                }
            });

        using IdCounters = std::tuple<decltype(m_orderIdCounter), decltype(m_tradeIdCounter)>;
        std::tie(m_orderIdCounter, m_tradeIdCounter) = [&] -> IdCounters {
            if (!node.attribute("sharedQuoteBalances").as_bool()) {
                return {nullptr, nullptr};
            }
            return {std::make_shared<OrderID>(), std::make_shared<TradeID>()};
        }();

        for (BookId bookId{}; bookId < bookCount; ++bookId) {
            auto book = std::make_shared<taosim::book::Book>(
                simulation(), bookId, maxDepth, detailedDepth, m_orderIdCounter, m_tradeIdCounter);
            book->signals().orderCreated.connect(
                [this](Order::Ptr order, OrderContext ctx) { orderCallback(order, ctx); });
            book->signals().orderLog.connect(
                [this](Order::Ptr order, OrderContext ctx) { orderLogCallback(order, ctx); });
            book->signals().trade.connect(
                [this](Trade::Ptr trade, BookId bookId) { tradeCallback(trade, bookId); });
            book->signals().unregister.connect(
                [this](LimitOrder::Ptr order, BookId bookId) {
                    unregisterLimitOrderCallback(order, bookId);
                });
            book->signals().cancelOrderDetails.connect(
                [this](LimitOrder::Ptr order, taosim::decimal_t volumeToCancel, BookId bookId) {
                    m_clearingManager->handleCancelOrder({
                        .bookId = bookId,
                        .order = order,
                        .volumeToCancel = volumeToCancel
                    });
                    m_L3Record.at(bookId).push(taosim::event::CancellationEvent(
                        taosim::event::Cancellation(order->id(), volumeToCancel),
                        simulation()->currentTimestamp(),
                        order->price()
                    ));
                    // Debit canceled volume from the SL/TP side store; no-op
                    // for unflagged orders. Without this, fully-canceled
                    // SLTP orders leak their orderInfo template forever.
                    m_sltpContainer.onOrderCanceled(bookId, order->id(), volumeToCancel);
                });
            book->signals().marketOrderProcessed.connect(
                [this](MarketOrder::Ptr marketOrder, OrderContext ctx) {
                    marketOrderProcessedCallback(marketOrder, ctx);
                });
            // SL/TP price feed: project Book trade events to
            // (bookId, latest price) and forward to the container.
            m_sltpContainer.priceFeed(bookId) =
                book->signals().trade.connect(
                    [this, bookId](Trade::Ptr trade, BookId) {
                        m_sltpContainer.onPriceUpdate(bookId, trade->price());
                    });
            m_books.push_back(book);
            m_signals[bookId] = std::make_unique<ExchangeSignals>();
            const BookId bookIdCanon = simulation()->m_blockIdx * bookCount + bookId;
            if (loggingNode) {
                if (L2Node) {
                    const fs::path logPath =
                        simulation()->logDir() / fmt::format(
                            "{}L2-{}.log",
                            !L2LogTag.empty() ? L2LogTag + "-" : "",
                            bookIdCanon);
                    m_L2Loggers[bookId] = std::make_unique<taosim::book::L2Logger>(
                        logPath,
                        L2Depth,
                        startTimePoint,
                        book->signals().L2,
                        simulation());
                }
                if (L3Node) {
                    const fs::path logPath =
                        simulation()->logDir() / fmt::format(
                            "{}L3-{}.log",
                            !L3LogTag.empty() ? L3LogTag + "-" : "",
                            bookIdCanon);
                    m_L3EventLoggers[bookId] = std::make_unique<taosim::book::L3EventLogger>(
                        logPath,
                        startTimePoint,
                        m_signals.at(bookId)->L3,
                        simulation());
                }
                if (feeLogNode) {
                    const fs::path logPath =
                        simulation()->logDir() / fmt::format(
                            "{}fees-{}.log",
                            !feeLogTag.empty() ? feeLogTag + "-" : "",
                            bookIdCanon);
                    m_feeLoggers[bookId] = std::make_unique<taosim::book::FeeLogger>(
                        logPath,
                        startTimePoint,
                        m_signals.at(bookId)->feeLog,
                        simulation());
                }
                if (replayNode) {
                    m_replayLog = true;
                    const fs::path path =
                        simulation()->logDir() / fmt::format("Replay-{}.log", bookIdCanon);
                    m_replayEventLoggers.push_back(
                        std::make_unique<taosim::matching::ReplayEventLogger>(
                            path, startTimePoint, simulation()));
                }
            }
        }

        auto doc = std::make_shared<pugi::xml_document>();
        doc->append_copy(balancesNode);
        m_clearingManager->accounts().setAccountTemplate([this, doc, bookCount] {
            const pugi::xml_node balancesNode = doc->child("Balances");
            taosim::accounting::Account accountTemplate;
            for (auto bookId : views::iota(BookId{}, bookCount)) {
                accountTemplate.holdings().push_back(
                    taosim::accounting::Balances::fromXML(
                        balancesNode,
                        taosim::accounting::RoundParams{
                            .baseDecimals = m_config.parameters().baseIncrementDecimals,
                            .quoteDecimals = m_config.parameters().quoteIncrementDecimals},
                        &simulation()->rng()));
                accountTemplate.activeOrders().emplace_back();
            }
            return accountTemplate;
        });

        if (const uint32_t remoteAgentCount = node.attribute("remoteAgentCount").as_uint()) {
            for (AgentId agentId{}; agentId < remoteAgentCount; ++agentId) {
                m_clearingManager->accounts().registerRemote();
            }
        }

        simulation()->signals().agentsCreated.connect([=, this] {
            if (!balancesNode.attribute("log").as_bool()) return;
            for (BookId bookId = 0; bookId < bookCount; ++bookId) {
                auto balanceLogger = std::make_unique<taosim::accounting::BalanceLogger>(
                    simulation()->logDir() / fmt::format("bals-{}.log", bookId),
                    m_signals.at(bookId)->L3,
                    &accounts());
                m_balanceLoggers.push_back(std::move(balanceLogger));
            }
        });
        simulation()->signals().agentsCreated.connect([this] {
            if (!m_replayLog) return;
            rapidjson::Document json{rapidjson::kObjectType};
            auto& allocator = json.GetAllocator();
            for (const auto& [agentId, acct] : accounts()) {
                json.AddMember(
                    rapidjson::Value{std::to_string(agentId).c_str(), allocator},
                    [&] {
                        rapidjson::Document balancesJson{rapidjson::kArrayType, &allocator};
                        for (const auto& bals : acct) {
                            rapidjson::Document balanceJson{rapidjson::kObjectType, &allocator};
                            balanceJson.AddMember(
                                "base",
                                taosim::json::packedDecimal2json(bals.base.getTotal(), allocator),
                                allocator);
                            balanceJson.AddMember(
                                "quote",
                                taosim::json::packedDecimal2json(bals.quote->getTotal(), allocator),
                                allocator);
                            balancesJson.PushBack(balanceJson, allocator);
                        }
                        return balancesJson;
                    }().Move(),
                    allocator);
            }
            std::ofstream ofs{
                simulation()->logDir() /
                fmt::format(
                    "Replay-Balances-{}-{}.json",
                    simulation()->blockIdx() * m_books.size(),
                    (simulation()->blockIdx() + 1) * m_books.size() - 1)};
            taosim::json::dumpJson(
                json, ofs, taosim::json::FormatOptions{.indent = taosim::json::IndentOptions{}});
        });
    }
    catch (...) {
        handleException();
    }
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::receiveMessage(Message::Ptr msg)
{
    try {
        if (msg->type.starts_with("DISTRIBUTED")) {
            return handleDistributedMessage(msg);
        }
        handleLocalMessage(msg);
    }
    catch (...) {
        handleException();
    }
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::jsonSerialize(
    rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("logDir", rapidjson::Value{simulation()->logDir().c_str(), allocator}, allocator);
        auto serializeBooks = [this](rapidjson::Document& json) {
            json.SetArray();
            auto& allocator = json.GetAllocator();
            for (const auto book : m_books) {
                const BookId bookId = book->id();
                rapidjson::Document bookJson{rapidjson::kObjectType, &allocator};
                bookJson.AddMember("bookId", rapidjson::Value{bookId}, allocator);
                m_L3Record.at(bookId).jsonSerialize(bookJson, "record");
                rapidjson::Document bidAskJson{&allocator};
                book->jsonSerialize(bidAskJson);
                bookJson.AddMember("bid", bidAskJson["bid"], allocator);
                bookJson.AddMember("ask", bidAskJson["ask"], allocator);
                json.PushBack(bookJson, allocator);
            }
        };
        taosim::json::serializeHelper(json, "books", serializeBooks);
        auto serializeAccounts = [this](rapidjson::Document& json) {
            auto& allocator = json.GetAllocator();
            m_clearingManager->accounts().jsonSerialize(json);
            const auto feePolicy = m_clearingManager->feePolicy();
            const auto bookCount = m_books.size();

            // Cache per-agent and per-book stringified IDs once — they were being
            // recomputed per (agent, book) pair in the original code.
            std::vector<AgentId> agentIds;
            for (AgentId agentId : views::keys(accounts())) agentIds.push_back(agentId);
            std::vector<std::string> agentIdStrs;
            agentIdStrs.reserve(agentIds.size());
            for (AgentId a : agentIds) agentIdStrs.emplace_back(std::to_string(a));
            std::vector<std::string> bookIdStrs;
            bookIdStrs.reserve(bookCount);
            for (size_t b = 0; b < bookCount; ++b) bookIdStrs.emplace_back(std::to_string(b));

            for (size_t i = 0; i < agentIds.size(); ++i) {
                const AgentId agentId = agentIds[i];
                const char* agentIdCStr = agentIdStrs[i].c_str();
                // Use rapidjson::Value (shared allocator) instead of Document (owns own).
                rapidjson::Value ordersJson{rapidjson::kArrayType};
                json[agentIdCStr].AddMember("orders", ordersJson, allocator);
                rapidjson::Value feesJson{rapidjson::kObjectType};
                for (size_t bookId = 0; bookId < bookCount; ++bookId) {
                    rapidjson::Value feeJson{rapidjson::kObjectType};
                    feeJson.AddMember(
                        "volume",
                        rapidjson::Value{taosim::util::decimal2double(
                            feePolicy->agentVolume(bookId, agentId))},
                        allocator);
                    const auto rates = feePolicy->getRates(bookId, agentId);
                    feeJson.AddMember(
                        "makerFeeRate",
                        rapidjson::Value{taosim::util::decimal2double(rates.maker)},
                        allocator);
                    feeJson.AddMember(
                        "takerFeeRate",
                        rapidjson::Value{taosim::util::decimal2double(rates.taker)},
                        allocator);
                    feesJson.AddMember(
                        rapidjson::Value{bookIdStrs[bookId].c_str(), allocator},
                        feeJson,
                        allocator);
                }
                json[agentIdCStr].AddMember("fees", feesJson, allocator);
            }
            // Index agentIds → idx, built ONCE outside the per-book loop so we
            // can resolve agentIdStrs without recomputing std::to_string in the
            // inner order loops.
            std::unordered_map<AgentId, size_t> agentIdx;
            agentIdx.reserve(agentIds.size());
            for (size_t i = 0; i < agentIds.size(); ++i) agentIdx[agentIds[i]] = i;

            for (const auto book : m_books) {
                const BookId bookId = book->id();
                for (size_t i = 0; i < agentIds.size(); ++i) {
                    const char* agentIdCStr = agentIdStrs[i].c_str();
                    json[agentIdCStr]["orders"].PushBack(
                        rapidjson::Value{rapidjson::kArrayType},
                        allocator);
                }
                for (const taosim::book::TickContainer& bidLevel : book->buyQueue()) {
                    for (const auto& bid : bidLevel) {
                        const auto& ctx = m_books[bookId]->orderToClientInfo().at(bid->id());
                        const auto it = agentIdx.find(ctx.agentId);
                        if (it == agentIdx.end()) continue;
                        const char* agentIdCStr = agentIdStrs[it->second].c_str();
                        rapidjson::Value orderJson{rapidjson::kObjectType};
                        // bid->jsonSerialize expects Document; wrap via shared allocator below.
                        rapidjson::Document orderDoc{&allocator};
                        bid->jsonSerialize(orderDoc);
                        taosim::json::setOptionalMember(orderDoc, "clientOrderId", ctx.clientOrderId);
                        json[agentIdCStr]["orders"][bookId].PushBack(orderDoc, allocator);
                    }
                }
                for (const taosim::book::TickContainer& askLevel : book->sellQueue()) {
                    for (const auto& ask : askLevel) {
                        const auto& ctx = m_books[bookId]->orderToClientInfo().at(ask->id());
                        const auto it = agentIdx.find(ctx.agentId);
                        if (it == agentIdx.end()) continue;
                        const char* agentIdCStr = agentIdStrs[it->second].c_str();
                        rapidjson::Document orderDoc{&allocator};
                        ask->jsonSerialize(orderDoc);
                        taosim::json::setOptionalMember(orderDoc, "clientOrderId", ctx.clientOrderId);
                        json[agentIdCStr]["orders"][bookId].PushBack(orderDoc, allocator);
                    }
                }
            }
        };
        taosim::json::serializeHelper(json, "accounts", serializeAccounts);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleException()
{
    try {
        throw;
    }
    catch (const std::exception& e) {
        fmt::println("{}", e.what());
        throw;
    }
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleDistributedMessage(const Message::Ptr&  msg)
{
    if (msg->type.contains("PLACE_ORDER_MARKET")) {
        handleDistributedPlaceMarketOrder(msg);
    }
    else if (msg->type.ends_with("PLACE_ORDER_LIMIT")) {
        handleDistributedPlaceLimitOrder(msg);
    }
    else if (msg->type.ends_with("RETRIEVE_ORDERS")) {
        handleDistributedRetrieveOrders(msg);
    }
    else if (msg->type.ends_with("CANCEL_ORDERS")) {
        handleDistributedCancelOrders(msg);
    }
    else if (msg->type.ends_with("CLOSE_POSITIONS")) {
        handleDistributedClosePositions(msg);
    }
    else if (msg->type.ends_with("RESET_AGENT")) {
        handleDistributedAgentReset(msg);
    }
    else {
        handleDistributedUnknownMessage(msg);
    }
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleDistributedAgentReset(const Message::Ptr&  msg)
{
    const auto payload = std::static_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
    const auto subPayload = std::static_pointer_cast<ResetAgentsPayload>(payload->payload);

    if (m_replayLog) {
        for (const auto& logger : m_replayEventLoggers) {
            logger->log(msg);
        }
    }

    std::vector<AgentId> valid = {};
    for (AgentId agentId : subPayload->agentIds) {
        if (accounts().contains(agentId)) {
            valid.push_back(agentId);
        } else {
            simulation()->logDebug("{} | RESET AGENTS : AGENT #{} NOT FOUND IN ACCOUNTS.", simulation()->currentTimestamp(), agentId);
        }
    }

    if (valid.empty()) {
        return;
    }

    if (simulation()->m_replayMode) {
        for (auto agentId : valid) {
            auto& acct = accounts().at(agentId);
            for (auto book : m_books) {
                const auto activeOrders = acct.activeOrders().at(book->id());
                for (Order::Ptr order : activeOrders) {
                    auto limitOrder = std::dynamic_pointer_cast<LimitOrder>(order);
                    if (limitOrder == nullptr) {
                        throw taosim::replay::helpers::ReplayError{};
                    }
                    auto cancelRes = book->cancelOrder(limitOrder->id());
                    if (!cancelRes) {
                        throw taosim::replay::helpers::ReplayError{};
                    }
                    const taosim::event::Cancellation cancellation{limitOrder->id()};
                    m_signals.at(book->id())->cancelLog(CancellationWithLogContext(
                        cancellation,
                        std::make_shared<CancellationLogContext>(
                            agentId,
                            book->id(),
                            simulation()->currentTimestamp())));
                }
            }
        }
        rapidjson::Document json;
        const auto bookIdRange = std::pair{
            simulation()->blockIdx() * m_books.size(),
            (simulation()->blockIdx() + 1) * m_books.size() - 1
        };
        const auto balsPath =
            fs::path{boost::erase_last_copy(simulation()->logDir().string(), "-replay")}
                / fmt::format(
                    "Replay-Balances-{}-{}-{}.json",
                    bookIdRange.first,
                    bookIdRange.second,
                    simulation()->currentTimestamp());
        json = taosim::json::loadJson(balsPath);
        for (const auto& member : json.GetObject()) {
            const auto agentId = std::stoi(member.name.GetString());
            auto& acct = accounts().at(agentId);
            const auto& balsJson = member.value;
            for (const auto& balsMember : balsJson.GetObject()) {
                const auto bookIdCanon = std::stoi(balsMember.name.GetString());
                if (!(bookIdRange.first <= bookIdCanon && bookIdCanon <= bookIdRange.second)) {
                    continue;
                }
                const auto bookId = bookIdCanon % m_books.size();
                acct.at(bookId) = taosim::accounting::Balances(taosim::accounting::BalancesDesc{
                    .base = taosim::accounting::Balance{taosim::json::getDecimal(balsMember.value["base"])},
                    .quote = std::make_shared<taosim::accounting::Balance>(
                        taosim::json::getDecimal(balsMember.value["quote"])
                    ),
                    .roundParams = acct.at(bookId).m_roundParams
                });
            }
        }
        const std::unordered_set<AgentId> resetAgentIds{valid.begin(), valid.end()};
        m_clearingManager->feePolicy()->resetHistory(resetAgentIds);
        return;
    }

    std::vector<std::vector<taosim::event::Cancellation>> cancellations;
    for (AgentId agentId : valid) {
        simulation()->logDebug("{} | AGENT #{} : RESET-CANCELS", simulation()->currentTimestamp(), agentId);
        for (BookId bookId = 0; bookId < m_books.size(); ++bookId) {
            simulation()->logDebug("{} | AGENT #{} BOOK {} : RESET-CANCELS", simulation()->currentTimestamp(), agentId, simulation()->bookIdCanon(bookId));
            std::vector<taosim::event::Cancellation> bookCancellations;
            const auto orders = accounts()[agentId].activeOrders()[bookId];
            const auto book = m_books.at(bookId);
            for (Order::Ptr order : orders) {
                if (auto limitOrder = std::dynamic_pointer_cast<LimitOrder>(order)) {
                    simulation()->logDebug("{} | AGENT #{} BOOK {} : START RESET-CANCEL OF ORDER {}", simulation()->currentTimestamp(), agentId, simulation()->bookIdCanon(bookId), limitOrder->id());
                    if (book->cancelOrder(limitOrder->id())) {
                        const taosim::event::Cancellation cancellation{limitOrder->id()};
                        bookCancellations.push_back(cancellation);
                        m_signals.at(bookId)->cancelLog(CancellationWithLogContext(
                            cancellation,
                            std::make_shared<CancellationLogContext>(
                                agentId,
                                bookId,
                                simulation()->currentTimestamp())));
                        simulation()->logDebug("{} | AGENT #{} BOOK {} : END RESET-CANCEL OF ORDER {}", simulation()->currentTimestamp(), agentId, simulation()->bookIdCanon(bookId), limitOrder->id());
                    } else {
                        simulation()->logDebug("{} | AGENT #{} BOOK {} : RESET-CANCEL OF ORDER {} FAILED", simulation()->currentTimestamp(), agentId, simulation()->bookIdCanon(bookId), limitOrder->id());
                    }
                }
            }
            cancellations.push_back(std::move(bookCancellations));
        }
        accounts().reset(agentId);
        simulation()->logDebug("{} | AGENT #{} : RESET-CANCELS DONE", simulation()->currentTimestamp(), agentId);
    }
    simulation()->logDebug("{} | ALL RESET-CANCELS DONE", simulation()->currentTimestamp());

    const std::unordered_set<AgentId> resetAgentIds{valid.begin(), valid.end()};

    m_clearingManager->feePolicy()->resetHistory(resetAgentIds);

    if (simulation()->m_replayMode) return;

    if (m_replayLog) {
        rapidjson::Document json{rapidjson::kObjectType};
        auto& allocator = json.GetAllocator();
        for (auto agentId : valid) {
            const auto& acct = accounts().at(agentId);
            json.AddMember(
                rapidjson::Value{std::to_string(agentId).c_str(), allocator},
                [&] {
                    rapidjson::Value balancesJson{rapidjson::kObjectType};
                    for (const auto& [bookId, bals] : views::enumerate(acct)) {
                        rapidjson::Value balanceJson{rapidjson::kObjectType};
                        balanceJson.AddMember(
                            "base",
                            taosim::json::packedDecimal2json(bals.base.getTotal(), allocator),
                            allocator);
                        balanceJson.AddMember(
                            "quote",
                            taosim::json::packedDecimal2json(bals.quote->getTotal(), allocator),
                            allocator);
                        balancesJson.AddMember(
                            rapidjson::Value{
                                std::to_string(simulation()->bookIdCanon(bookId)).c_str(), allocator},
                            balanceJson,
                            allocator);
                    }
                    return balancesJson;
                }().Move(),
                allocator);
        }
        std::ofstream ofs{
            simulation()->logDir() / fmt::format(
                "Replay-Balances-{}-{}-{}.json",
                simulation()->blockIdx() * m_books.size(),
                (simulation()->blockIdx() + 1) * m_books.size() - 1,
                simulation()->currentTimestamp())};
        taosim::json::dumpJson(
            json, ofs, taosim::json::FormatOptions{.indent = taosim::json::IndentOptions{}});
    }

    simulation()->m_messageQueue = taosim::message::MessageQueue{
        simulation()->m_messageQueue.queue().underlying()
            | views::filter([&](const auto& prioMsgWithId) {
                const auto distributedPayload =
                    std::static_pointer_cast<DistributedAgentResponsePayload>(
                        prioMsgWithId.pmsg.msg->payload);
                return !(distributedPayload && resetAgentIds.contains(distributedPayload->agentId));
            })
            | ranges::to<std::vector>};
    simulation()->logDebug("{} | MESSAGE QUEUE CLEARED", simulation()->currentTimestamp());

    simulation()->fastRespondToMessage(
        msg,
        MessagePayload::create<DistributedAgentResponsePayload>(
            payload->agentId,
            MessagePayload::create<ResetAgentsResponsePayload>(valid, subPayload)));
    simulation()->logDebug("{} | RESET COMPLETE", simulation()->currentTimestamp());
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleDistributedPlaceMarketOrder(const Message::Ptr&  msg)
{
    const auto payload = std::static_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
    const auto subPayload = std::static_pointer_cast<PlaceOrderMarketPayload>(payload->payload);

    const bool isMarginCall = msg->type.ends_with("_MC");
    if (isMarginCall) {
        msg->type = "DISTRIBUTED_PLACE_MARKET_ORDER";
    }
    if (m_replayLog && !isMarginCall) {
        m_replayEventLoggers.at(subPayload->bookId)->log(msg);
    }

    if (simulation()->debug()) {
        const auto& balances = simulation()->exchange()->accounts()[payload->agentId][subPayload->bookId];
        simulation()->logDebug("{} | AGENT #{} BOOK {} : QUOTE : {}  BASE : {}", simulation()->currentTimestamp(), payload->agentId, simulation()->bookIdCanon(subPayload->bookId), *balances.quote, balances.base);
    }
    const auto orderResult = m_clearingManager->handleOrder(
        taosim::matching::MarketOrderDesc{
            .agentId = payload->agentId,
            .payload = subPayload
        });
    if (simulation()->debug()) {
        const auto& balances = simulation()->exchange()->accounts()[payload->agentId][subPayload->bookId];
        simulation()->logDebug("{} | AGENT #{} BOOK {} : QUOTE : {}  BASE : {}", simulation()->currentTimestamp(), payload->agentId, simulation()->bookIdCanon(subPayload->bookId), *balances.quote, balances.base);
    }

    if (orderResult.ec != OrderErrorCode::VALID) {
        if (subPayload->closeReason != 0) {
            const auto& bals =
                simulation()->exchange()->accounts()[payload->agentId][subPayload->bookId];
            fmt::println("{} | SLTP CLOSE {} | AGENT #{} BOOK {} : "
                "placement FAILED ({}) | volume={} freeBase={} freeQuote={}",
                simulation()->currentTimestamp(),
                subPayload->closeReason == 1 ? "SL" : "TP",
                payload->agentId, simulation()->bookIdCanon(subPayload->bookId),
                OrderErrorCode2StrView(orderResult.ec),
                subPayload->volume, bals.base.getFree(), bals.quote->getFree());
        }
        simulation()->logDebug(
            "Invalid Market Order Placement by Distributed Agent - {} : {}",
            orderResult.ec,
            taosim::json::jsonSerializable2str(payload));
        if (simulation()->m_replayMode && !simulation()->isReplacedAgent(msg->source)) return;
        return fastRespondToMessage(
            msg,
            "ERROR",
            MessagePayload::create<DistributedAgentResponsePayload>(
                payload->agentId,
                MessagePayload::create<PlaceOrderMarketErrorResponsePayload>(
                    subPayload,
                    MessagePayload::create<ErrorResponsePayload>(
                        OrderErrorCode2StrView(orderResult.ec).data()))));
    }

    auto clientCtx = OrderClientContext(
        payload->agentId, subPayload->clientOrderId, subPayload->delegate, subPayload->currency);
    clientCtx.closeReason        = subPayload->closeReason;
    clientCtx.originatingOrderId = subPayload->originatingOrderId;
    const auto order = m_books[subPayload->bookId]->placeMarketOrder(
        std::move(clientCtx),
        msg->arrival,
        orderResult.orderSize,
        subPayload->direction,
        subPayload->leverage,
        subPayload->stpFlag,
        subPayload->settleFlag,
        subPayload->currency,
        subPayload->maxSlippage,
        subPayload->stopLoss,
        subPayload->takeProfit,
        subPayload->placeholder);

    // Gap fix: miner (distributed) market orders fire the subscriber event too.
    notifyMarketOrderSubscribers(order, subPayload->bookId, payload->agentId);

    if (simulation()->m_replayMode && !simulation()->isReplacedAgent(msg->source)) return;

    const auto retSubPayload =
        MessagePayload::create<PlaceOrderMarketResponsePayload>(order->id(), subPayload);

    respondToMessage(
        msg,
        MessagePayload::create<DistributedAgentResponsePayload>(
            payload->agentId,
            MessagePayload::create<PlaceOrderMarketResponsePayload>(order->id(), subPayload)),
        0);
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleDistributedPlaceLimitOrder(const Message::Ptr&  msg)
{
    const auto payload = std::static_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
    const auto subPayload = std::static_pointer_cast<PlaceOrderLimitPayload>(payload->payload);

    if (m_replayLog) {
        m_replayEventLoggers.at(subPayload->bookId)->log(msg);
    }
    if (simulation()->shouldAdjustLimitPrice(msg)) {
        const auto now = simulation()->currentTimestamp();
        auto it = simulation()->timestampToMidPrice().find(now);
        const auto prevailingMidPrice = m_books[subPayload->bookId]->midPrice();
        if (it != simulation()->timestampToMidPrice().end() && it->second > 0_dec && prevailingMidPrice > 0_dec) {
            subPayload->price = subPayload->price / it->second * prevailingMidPrice;
        }
    }

    if (simulation()->debug()) {
        const auto& balances = simulation()->exchange()->accounts()[payload->agentId][subPayload->bookId];
        simulation()->logDebug("{} | AGENT #{} BOOK {} : QUOTE : {}  BASE : {}", simulation()->currentTimestamp(), payload->agentId, simulation()->bookIdCanon(subPayload->bookId), *balances.quote, balances.base);
    }
    const auto orderResult = m_clearingManager->handleOrder(
        taosim::matching::LimitOrderDesc{
            .agentId = payload->agentId,
            .payload = subPayload
        });
    if (simulation()->debug()) {
        const auto& balances = simulation()->exchange()->accounts()[payload->agentId][subPayload->bookId];
        simulation()->logDebug("{} | AGENT #{} BOOK {} : QUOTE : {}  BASE : {}", simulation()->currentTimestamp(), payload->agentId, simulation()->bookIdCanon(subPayload->bookId), *balances.quote, balances.base);
    }

    if (orderResult.ec != OrderErrorCode::VALID) {
        simulation()->logDebug(
            "Invalid Limit Order Placement by Distributed Agent - {} : {}",
            orderResult.ec,
            taosim::json::jsonSerializable2str(payload));
        if (simulation()->m_replayMode  && !simulation()->isReplacedAgent(msg->source)) return;
        return fastRespondToMessage(
            msg,
            "ERROR",
            MessagePayload::create<DistributedAgentResponsePayload>(
                payload->agentId,
                MessagePayload::create<PlaceOrderLimitErrorResponsePayload>(
                    subPayload,
                    MessagePayload::create<ErrorResponsePayload>(
                        OrderErrorCode2StrView(orderResult.ec).data()))));
    }

    const auto order = m_books[subPayload->bookId]->placeLimitOrder(
        OrderClientContext(
            payload->agentId, subPayload->clientOrderId, subPayload->delegate, subPayload->currency),
        msg->arrival,
        orderResult.orderSize,
        subPayload->direction,
        subPayload->price,
        subPayload->leverage,
        subPayload->stpFlag,
        subPayload->settleFlag,
        subPayload->postOnly,
        subPayload->timeInForce,
        subPayload->expiryPeriod,
        subPayload->currency,
        subPayload->stopLoss,
        subPayload->takeProfit,
        subPayload->placeholder);

    // Miner (distributed) limit orders fire the same subscriber event as local ones so
    // subscribers see both origins; isRemote=true also reaches miner-only subscribers.
    notifyLimitOrderSubscribers(order, subPayload->bookId, payload->agentId, true);

    if (simulation()->m_replayMode && !simulation()->isReplacedAgent(msg->source)) return;

    const auto retSubPayload =
        MessagePayload::create<PlaceOrderLimitResponsePayload>(order->id(), subPayload);

    respondToMessage(
        msg,
        MessagePayload::create<DistributedAgentResponsePayload>(
            payload->agentId,
            MessagePayload::create<PlaceOrderLimitResponsePayload>(order->id(), subPayload)),
        0);

    // In exchange-service mode the sim clock (m_time.current) is set per instruction to
    // instr.delay and never advances monotonically, so a time-scheduled expiry queued here
    // would never fire (and would leak in the queue). GTT expiry is instead driven per batch
    // off the block clock by Exchange::expireGTTOrders. Keep the scheduler for the stepped
    // distributed sim, where the message queue is drained by Simulation::step().
    if (subPayload->timeInForce == taosim::TimeInForce::GTT && subPayload->expiryPeriod.has_value()
        && !simulation()->proxy()->exchangeServiceMode()) {
        simulation()->dispatchMessage(
            simulation()->currentTimestamp(),
            subPayload->expiryPeriod.value(),
            msg->source,
            name(),
            "DISTRIBUTED_CANCEL_ORDERS",
            MessagePayload::create<DistributedAgentResponsePayload>(
                payload->agentId,
                MessagePayload::create<CancelOrdersPayload>(
                    std::vector{taosim::event::Cancellation{order->id()}}, subPayload->bookId)));
    }
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleDistributedRetrieveOrders(const Message::Ptr&  msg)
{
    const auto payload = std::static_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
    const auto subPayload = std::static_pointer_cast<RetrieveOrdersPayload>(payload->payload);

    const auto book = m_books[subPayload->bookId];

    auto retSubPayload = MessagePayload::create<RetrieveOrdersResponsePayload>();
    for (OrderID id : subPayload->ids) {
        if (auto order = book->getOrder(id)) {
            retSubPayload->orders.push_back(**order);
        }
    }

    respondToMessage(
        msg,
        MessagePayload::create<DistributedAgentResponsePayload>(payload->agentId, retSubPayload),
        0);
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleDistributedCancelOrders(const Message::Ptr&  msg)
{
    const auto payload = std::static_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
    const auto subPayload = std::static_pointer_cast<CancelOrdersPayload>(payload->payload);

    const auto bookId = subPayload->bookId;

    std::vector<taosim::event::Cancellation> cancellations;
    std::vector<taosim::event::Cancellation> failures;

    if (bookId >= m_books.size()) {
        for (const auto& cancellation : subPayload->cancellations) {
            failures.push_back(cancellation);
        }
    }
    else {
        if (m_replayLog) {
            m_replayEventLoggers.at(bookId)->log(msg);
        }

        const auto book = m_books[bookId];
        const auto& order2ctx = book->orderToClientInfo();
        for (const auto& cancellation : subPayload->cancellations) {
            auto ctxIt = order2ctx.find(cancellation.id);
            if (ctxIt == order2ctx.end() || ctxIt->second.agentId != payload->agentId) {
                failures.push_back(cancellation);
                continue;
            }
            if (book->cancelOrder(cancellation.id, cancellation.volume)) {
                cancellations.push_back(cancellation);
                m_signals[bookId]->cancelLog(CancellationWithLogContext(
                    cancellation,
                    std::make_shared<CancellationLogContext>(
                        payload->agentId, bookId, simulation()->currentTimestamp())));
            }
            else {
                failures.push_back(cancellation);
            }
        }
    }

    if (simulation()->m_replayMode && !simulation()->isReplacedAgent(msg->source)) return;

    if (!cancellations.empty()) {        
        std::vector<OrderID> orderIds;
        for (const auto& canc : cancellations) {
            orderIds.push_back(canc.id);
        }
        respondToMessage(
            msg,
            MessagePayload::create<DistributedAgentResponsePayload>(
                payload->agentId,
                MessagePayload::create<CancelOrdersResponsePayload>(
                    std::move(orderIds),
                    MessagePayload::create<CancelOrdersPayload>(
                        std::move(cancellations), bookId)))
            );
    }

    if (!failures.empty()) {
        std::vector<OrderID> orderIds = failures
            | views::transform([](const auto& c) { return c.id; })
            | ranges::to<std::vector>();
        auto errorMsg = fmt::format("Order IDs {} do not exist.", fmt::join(orderIds, ", "));
        auto retSubPayload = MessagePayload::create<CancelOrdersErrorResponsePayload>(
            std::move(orderIds),
            MessagePayload::create<CancelOrdersPayload>(std::move(failures), bookId), 
            MessagePayload::create<ErrorResponsePayload>(std::move(errorMsg))
        );
        respondToMessage(
            msg,
            "ERROR",
            MessagePayload::create<DistributedAgentResponsePayload>(
                payload->agentId,
                retSubPayload),
            0);
    }
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleDistributedClosePositions(const Message::Ptr&  msg)
{
    const auto payload = std::static_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
    const auto subPayload = std::static_pointer_cast<ClosePositionsPayload>(payload->payload);

    if (m_replayLog) {
        m_replayEventLoggers.at(subPayload->bookId)->log(msg);
    }

    const auto bookId = subPayload->bookId;
    const auto book = m_books[bookId];

    std::vector<ClosePosition> closes;
    std::vector<ClosePosition> failures;
    for (const auto& close : subPayload->closePositions) {        
        if (m_clearingManager->handleClosePosition(ClosePositionDesc{
                .bookId = bookId,
                .agentId = payload->agentId,
                .orderId = close.id,
                .volumeToClose = close.volume
            })
        ) {
            closes.push_back(close);
        }
        else {
            failures.push_back(close);
        }
    }

    if (simulation()->m_replayMode && !simulation()->isReplacedAgent(msg->source)) return;

    if (!closes.empty()) {        
        std::vector<OrderID> orderIds;
        for (const ClosePosition& close : closes) {
            orderIds.push_back(close.id);
        }
        respondToMessage(
            msg,
            MessagePayload::create<DistributedAgentResponsePayload>(
                payload->agentId,
                MessagePayload::create<ClosePositionsResponsePayload>(
                    std::move(orderIds),
                    MessagePayload::create<ClosePositionsPayload>(
                        std::move(closes), bookId)))
            );
    }

    if (!failures.empty()) {
        std::vector<OrderID> orderIds = failures
            | views::transform([](const ClosePosition& c) { return c.id; })
            | ranges::to<std::vector>();
        auto errorMsg = fmt::format("Order IDs {} do not exist.", fmt::join(orderIds, ", "));
        auto retSubPayload = MessagePayload::create<ClosePositionsErrorResponsePayload>(
            std::move(orderIds),
            MessagePayload::create<ClosePositionsPayload>(std::move(failures), bookId), 
            MessagePayload::create<ErrorResponsePayload>(std::move(errorMsg))
        );
        respondToMessage(
            msg,
            "ERROR",
            MessagePayload::create<DistributedAgentResponsePayload>(
                payload->agentId,
                retSubPayload),
            0);
    }
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleDistributedUnknownMessage(const Message::Ptr&  msg)
{
    const auto payload = std::static_pointer_cast<DistributedAgentResponsePayload>(msg->payload);

    auto retSubPayload =
        MessagePayload::create<ErrorResponsePayload>(
            fmt::format("Unknown message type: {}", msg->type));
    fastRespondToMessage(
        msg,
        "ERROR",
        MessagePayload::create<DistributedAgentResponsePayload>(
            payload->agentId,
            retSubPayload));
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleLocalMessage(const Message::Ptr&  msg)
{
    if (msg->type.starts_with("PLACE_ORDER_MARKET")) {
        handleLocalPlaceMarketOrder(msg);
    }
    else if (msg->type == "PLACE_ORDER_LIMIT") {
        handleLocalPlaceLimitOrder(msg);
    }
    else if (msg->type == "RETRIEVE_ORDERS") {
        handleLocalRetrieveOrders(msg);
    }
    else if (msg->type == "CANCEL_ORDERS") {
        handleLocalCancelOrders(msg);
    }
    else if (msg->type == "CLOSE_POSITIONS") {
        handleLocalClosePositions(msg);
    }
    else if (msg->type == "RETRIEVE_L1") {
        handleLocalRetrieveL1(msg);
    }
    else if (msg->type == "RETRIEVE_L1_EXT") {
        handleLocalRetrieveL1Ext(msg);
    }
    else if (msg->type == "RETRIEVE_L2") {
        handleLocalRetrieveL2(msg);
    }
    else if (msg->type == "SUBSCRIBE_EVENT_ORDER_MARKET") {
        handleLocalMarketOrderSubscription(msg);
    }
    else if (msg->type == "SUBSCRIBE_EVENT_ORDER_LIMIT") {
        handleLocalLimitOrderSubscription(msg);
    }
    else if (msg->type == "SUBSCRIBE_EVENT_ORDER_LIMIT_MINER") {
        handleMinerLimitOrderSubscription(msg);
    }
    else if (msg->type == "SUBSCRIBE_EVENT_TRADE") {
        handleLocalTradeSubscription(msg);
    }
    else if (msg->type == "SUBSCRIBE_EVENT_ORDER_TRADE") {
        handleLocalTradeByOrderSubscription(msg);
    }
    else if (msg->type == "SUBSCRIBE_EVENT_TRADE_OWN") {
        handleLocalOwnTradeSubscription(msg);
    }
    else {
        handleLocalUnknownMessage(msg);
    }
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleLocalPlaceMarketOrder(const Message::Ptr&  msg)
{
    const auto& payload = std::static_pointer_cast<PlaceOrderMarketPayload>(msg->payload);

    const bool isMarginCall = msg->type.ends_with("_MC");
    if (isMarginCall) {
        msg->type = "PLACE_MARKET_ORDER";
    }
    if (m_replayLog && !isMarginCall) {
        m_replayEventLoggers.at(payload->bookId)->log(msg);
    }

    if (simulation()->debug()) {
        auto agentId = accounts().lookupLocalAgentId(msg->source);
        const auto& balances = simulation()->exchange()->accounts()[agentId][payload->bookId];
        simulation()->logDebug("{} | AGENT #{} BOOK {} : QUOTE : {}  BASE : {}", simulation()->currentTimestamp(), agentId, simulation()->bookIdCanon(payload->bookId), *balances.quote, balances.base);
    }
    const auto orderResult = m_clearingManager->handleOrder(
        taosim::matching::MarketOrderDesc{
            .agentId = msg->source,
            .payload = payload
        });
    if (simulation()->debug()) {
        auto agentId = accounts().lookupLocalAgentId(msg->source);
        const auto& balances = simulation()->exchange()->accounts()[agentId][payload->bookId];
        simulation()->logDebug("{} | AGENT #{} BOOK {} : QUOTE : {}  BASE : {}", simulation()->currentTimestamp(), agentId, simulation()->bookIdCanon(payload->bookId), *balances.quote, balances.base);
    }
    
    if (orderResult.ec != OrderErrorCode::VALID) {
        simulation()->logDebug(
            "Invalid Market Order Placement by Local Agent - {} : {}",
            orderResult.ec,
            taosim::json::jsonSerializable2str(payload));
        if (simulation()->m_replayMode && !simulation()->isReplacedAgent(msg->source)) return;
        return fastRespondToMessage(
            msg,
            "ERROR",
            MessagePayload::create<PlaceOrderMarketErrorResponsePayload>(
                payload,
                MessagePayload::create<ErrorResponsePayload>(
                    OrderErrorCode2StrView(orderResult.ec).data())));
    }

    // The sweep path arrives here: the order belongs to EXCHANGE, but the payload may name the miner
    // whose instruction caused it. Carry that through so the self-trade guard and the counterparty on
    // the resulting fill can see who actually acted.
    auto sweepCtx = OrderClientContext(
        accounts().lookupLocalAgentId(msg->source), payload->clientOrderId, payload->delegate,
        payload->currency);
    sweepCtx.initiatorAgentId = payload->initiatorAgentId;
    const auto order = m_books[payload->bookId]->placeMarketOrder(
        std::move(sweepCtx),
        msg->arrival,
        orderResult.orderSize,
        payload->direction,
        payload->leverage,
        payload->stpFlag,
        payload->settleFlag,
        payload->currency,
        payload->maxSlippage,
        payload->stopLoss,
        payload->takeProfit,
        payload->placeholder);

    notifyMarketOrderSubscribers(order, payload->bookId, accounts().lookupLocalAgentId(msg->source));

    if (simulation()->m_replayMode && !simulation()->isReplacedAgent(msg->source)) return;

    respondToMessage(
        msg,
        MessagePayload::create<PlaceOrderMarketResponsePayload>(order->id(), payload),
        1);
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleLocalPlaceLimitOrder(const Message::Ptr&  msg)
{
    const auto& payload = std::static_pointer_cast<PlaceOrderLimitPayload>(msg->payload);

    if (m_replayLog) {
        m_replayEventLoggers.at(payload->bookId)->log(msg);
    }
    if (simulation()->shouldAdjustLimitPrice(msg)) {
        const auto now = simulation()->currentTimestamp();
        auto it = simulation()->timestampToMidPrice().find(now);
        const auto prevailingMidPrice = m_books[payload->bookId]->midPrice();
        if (it != simulation()->timestampToMidPrice().end() && it->second > 0_dec && prevailingMidPrice > 0_dec) {
            payload->price = payload->price / it->second * prevailingMidPrice;
        }
    }

    if (simulation()->debug()) {
        auto agentId = accounts().lookupLocalAgentId(msg->source);
        const auto& balances = simulation()->exchange()->accounts()[agentId][payload->bookId];
        simulation()->logDebug("{} | AGENT #{} BOOK {} : QUOTE : {}  BASE : {}", simulation()->currentTimestamp(), agentId, simulation()->bookIdCanon(payload->bookId), *balances.quote, balances.base);
    }
    const auto orderResult = m_clearingManager->handleOrder(
        taosim::matching::LimitOrderDesc{
            .agentId = msg->source,
            .payload = payload
        });
    if (simulation()->debug()) {
        auto agentId = accounts().lookupLocalAgentId(msg->source);
        const auto& balances = simulation()->exchange()->accounts()[agentId][payload->bookId];
        simulation()->logDebug("{} | AGENT #{} BOOK {} : QUOTE : {}  BASE : {}", simulation()->currentTimestamp(), agentId, simulation()->bookIdCanon(payload->bookId), *balances.quote, balances.base);
    }

    if (orderResult.ec != OrderErrorCode::VALID) {
        simulation()->logDebug(
            "Invalid Limit Order Placement by Local Agent - {} : {}",
            orderResult.ec,
            taosim::json::jsonSerializable2str(payload));
        if (simulation()->m_replayMode && !simulation()->isReplacedAgent(msg->source)) return;
        return fastRespondToMessage(
            msg,
            "ERROR",
            MessagePayload::create<PlaceOrderLimitErrorResponsePayload>(
                payload,
                MessagePayload::create<ErrorResponsePayload>(
                    OrderErrorCode2StrView(orderResult.ec).data())));
    }

    const auto order = m_books[payload->bookId]->placeLimitOrder(
        OrderClientContext(
            accounts().lookupLocalAgentId(msg->source), payload->clientOrderId, payload->delegate, payload->currency),
        msg->arrival,
        orderResult.orderSize,
        payload->direction,
        payload->price,
        payload->leverage,
        payload->stpFlag,
        payload->settleFlag,
        payload->postOnly,
        payload->timeInForce,
        payload->expiryPeriod,
        payload->currency,
        payload->stopLoss,
        payload->takeProfit,
        payload->placeholder);

    notifyLimitOrderSubscribers(order, payload->bookId, accounts().lookupLocalAgentId(msg->source), false);

    if (simulation()->m_replayMode && !simulation()->isReplacedAgent(msg->source)) return;

    respondToMessage(
        msg,
        MessagePayload::create<PlaceOrderLimitResponsePayload>(order->id(), payload),
        1);

    if (payload->timeInForce == taosim::TimeInForce::GTT && payload->expiryPeriod.has_value()) {
        simulation()->dispatchMessage(
            simulation()->currentTimestamp(),
            payload->expiryPeriod.value(),
            msg->source,
            name(),
            "CANCEL_ORDERS",
            MessagePayload::create<CancelOrdersPayload>(
                std::vector{taosim::event::Cancellation{order->id()}}, payload->bookId));
    }
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleLocalRetrieveOrders(const Message::Ptr&  msg)
{
    const auto payload = std::static_pointer_cast<RetrieveOrdersPayload>(msg->payload);

    const auto book = m_books[payload->bookId];

    respondToMessage(
        msg,
        MessagePayload::create<RetrieveOrdersResponsePayload>(
            payload->ids
                | views::transform([&](OrderID id) { return book->getOrder(id); })
                | views::filter([](std::optional<LimitOrder::Ptr> order) { return order.has_value(); })
                | views::transform([](std::optional<LimitOrder::Ptr> order) { return **order; })
                | ranges::to<std::vector>(),
            payload->bookId));
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleLocalCancelOrders(const Message::Ptr&  msg)
{
    const auto& payload = std::static_pointer_cast<CancelOrdersPayload>(msg->payload);

    if (m_replayLog) {
        m_replayEventLoggers.at(payload->bookId)->log(msg);
    }

    const auto bookId = payload->bookId;
    const auto book = m_books[bookId];
    const auto agentId = accounts().lookupLocalAgentId(msg->source);
    const auto now = simulation()->currentTimestamp();
    const auto& signal = m_signals[bookId];
    const auto volumeIncrementDecimals = m_config.parameters().volumeIncrementDecimals;

    std::vector<taosim::event::Cancellation> cancellations;
    std::vector<taosim::event::Cancellation> failures;
    for (auto& cancellation : payload->cancellations) {
        if (cancellation.volume) {
            cancellation.volume = taosim::util::round(
                cancellation.volume.value(), volumeIncrementDecimals);
        }
        if (book->cancelOrder(cancellation.id, cancellation.volume)) {
            cancellations.push_back(cancellation);
            signal->cancelLog(CancellationWithLogContext(
                cancellation,
                std::make_shared<CancellationLogContext>(agentId, bookId, now)));
        }
        else {
            failures.push_back(cancellation);
        }
    }

    if (simulation()->m_replayMode && !simulation()->isReplacedAgent(msg->source)) return;

    if (!cancellations.empty()) {
        respondToMessage(
            msg,
            MessagePayload::create<CancelOrdersResponsePayload>(
                cancellations
                    | views::transform([](const auto& c) { return c.id; })
                    | ranges::to<std::vector>(),
                MessagePayload::create<CancelOrdersPayload>(
                    std::move(cancellations), payload->bookId)),
            0);
    }

    if (!failures.empty()) {
        std::vector<OrderID> orderIds = failures
            | views::transform([](const auto& c) { return c.id; })
            | ranges::to<std::vector>();
        auto errorMsg = fmt::format("Order IDs {} do not exist.", fmt::join(orderIds, ", "));
        auto retSubPayload = MessagePayload::create<CancelOrdersErrorResponsePayload>(
            std::move(orderIds),
            payload,
            MessagePayload::create<ErrorResponsePayload>(std::move(errorMsg))
        );
        respondToMessage(msg, "ERROR", retSubPayload);
    }
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleLocalClosePositions(const Message::Ptr&  msg)
{
    const auto& payload = std::static_pointer_cast<ClosePositionsPayload>(msg->payload);

    if (m_replayLog) {
        m_replayEventLoggers.at(payload->bookId)->log(msg);
    }

    const auto bookId = payload->bookId;
    const auto book = m_books[bookId];
    const auto agentId = accounts().lookupLocalAgentId(msg->source);

    std::vector<ClosePosition> closes;
    std::vector<ClosePosition> failures;
    for (const auto& close : payload->closePositions) {
        const auto res = m_clearingManager->handleClosePosition(ClosePositionDesc{
            .bookId = bookId,
            .agentId = agentId,
            .orderId = close.id,
            .volumeToClose = close.volume
        });
        if (res) {
            closes.push_back(close);
        } else {
            failures.push_back(close);
        }
    }

    if (simulation()->m_replayMode && !simulation()->isReplacedAgent(msg->source)) return;

    if (!closes.empty()) {
        respondToMessage(
            msg,
            MessagePayload::create<ClosePositionsResponsePayload>(
                closes
                    | views::transform([](const ClosePosition& c) { return c.id; })
                    | ranges::to<std::vector>(),
                MessagePayload::create<ClosePositionsPayload>(
                    std::move(closes), bookId)),
            0);
    }

    if (!failures.empty()) {
        std::vector<OrderID> orderIds = failures
            | views::transform([](const ClosePosition& c) { return c.id; })
            | ranges::to<std::vector>();
        auto errorMsg = fmt::format("Order IDs {} do not exist.", fmt::join(orderIds, ", "));
        auto retSubPayload = MessagePayload::create<ClosePositionsErrorResponsePayload>(
            std::move(orderIds),
            payload,
            MessagePayload::create<ErrorResponsePayload>(std::move(errorMsg))
        );
        respondToMessage(msg, "ERROR", retSubPayload);
    }
}

//-------------------------------------------------------------------------

namespace
{

// Shared by both L1 handlers so the two responses can never disagree on what the
// top of book is.
struct TopOfBook
{
    taosim::decimal_t bestAskPrice{};
    taosim::decimal_t bestAskVolume{};
    taosim::decimal_t askTotalVolume{};
    taosim::decimal_t bestBidPrice{};
    taosim::decimal_t bestBidVolume{};
    taosim::decimal_t bidTotalVolume{};
};

[[nodiscard]] TopOfBook topOfBook(const taosim::book::Book::Ptr& book)
{
    TopOfBook res{};

    if (!book->sellQueue().empty()) {
        const auto& bestSellLevel = book->sellQueue().front();
        res.bestAskPrice = bestSellLevel.price();
        res.bestAskVolume = bestSellLevel.volume();
        res.askTotalVolume = book->sellQueue().volume();
    }

    if (!book->buyQueue().empty()) {
        const auto& bestBuyLevel = book->buyQueue().back();
        res.bestBidPrice = bestBuyLevel.price();
        res.bestBidVolume = bestBuyLevel.volume();
        res.bidTotalVolume = book->buyQueue().volume();
    }

    return res;
}

}  // namespace

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleLocalRetrieveL1(const Message::Ptr&  msg)
{
    const auto payload = std::static_pointer_cast<RetrieveL1Payload>(msg->payload);

    const auto top = topOfBook(m_books[payload->bookId]);

    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        1,
        name(),
        msg->source,
        "RESPONSE_RETRIEVE_L1",
        MessagePayload::create<RetrieveL1ResponsePayload>(
            simulation()->currentTimestamp(),
            top.bestAskPrice,
            top.bestAskVolume,
            top.askTotalVolume,
            top.bestBidPrice,
            top.bestBidVolume,
            top.bidTotalVolume,
            payload->bookId));
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleLocalRetrieveL1Ext(const Message::Ptr&  msg)
{
    const auto payload = std::dynamic_pointer_cast<RetrieveL1ExtPayload>(msg->payload);

    const auto top = topOfBook(m_books[payload->bookId]);

    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        1,
        name(),
        msg->source,
        "RESPONSE_RETRIEVE_L1_EXT",
        MessagePayload::create<RetrieveL1ExtResponsePayload>(
            simulation()->currentTimestamp(),
            top.bestAskPrice,
            top.bestAskVolume,
            top.askTotalVolume,
            top.bestBidPrice,
            top.bestBidVolume,
            top.bidTotalVolume,
            m_bookTradeStats.at(payload->bookId),
            payload->bookId));
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleLocalRetrieveL2(const Message::Ptr&  msg)
{
    const auto payload = std::static_pointer_cast<RetrieveL2Payload>(msg->payload);

    const auto book = m_books[payload->bookId];

    respondToMessage(
        msg,
        MessagePayload::create<RetrieveL2ResponsePayload>(
            simulation()->currentTimestamp(),
            book->buyQueue()
            | views::reverse
            | views::take(payload->depth)
            | views::transform([](const auto& level) -> BookLevel {
                return {
                    .price = level.price(),
                    .quantity = level.volume()
                };
            })
            | ranges::to<std::vector>,
            book->sellQueue()
            | views::take(payload->depth)
            | views::transform([](const auto& level) -> BookLevel {
                return {
                    .price = level.price(),
                    .quantity = level.volume()
                };
            })
            | ranges::to<std::vector>,
            book->id()));
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleLocalMarketOrderSubscription(const Message::Ptr&  msg)
{
    const auto& sub = msg->source;

    if (!m_localMarketOrderSubscribers.add(sub)) {
        return fastRespondToMessage(
            msg,
            "ERROR",
            MessagePayload::create<ErrorResponsePayload>(
                fmt::format("Agent {} is already subscribed to market order events", sub)));
    }

    fastRespondToMessage(
        msg,
        MessagePayload::create<SuccessResponsePayload>(
            fmt::format("Agent {} subscribed successfully to market order events", sub)));
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleLocalLimitOrderSubscription(const Message::Ptr&  msg)
{
    const auto& sub = msg->source;

    if (!m_localLimitOrderSubscribers.add(sub)) {
        return fastRespondToMessage(
            msg,
            "ERROR",
            MessagePayload::create<ErrorResponsePayload>(
                fmt::format("Agent {} is already subscribed to limit order events", sub)));
    }

    fastRespondToMessage(
        msg,
        MessagePayload::create<SuccessResponsePayload>(
            fmt::format("Agent {} subscribed successfully to limit order events", sub)));
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleMinerLimitOrderSubscription(const Message::Ptr&  msg)
{
    const auto& sub = msg->source;

    if (!m_minerLimitOrderSubscribers.add(sub)) {
        return fastRespondToMessage(
            msg,
            "ERROR",
            MessagePayload::create<ErrorResponsePayload>(
                fmt::format("Agent {} is already subscribed to miner limit order events", sub)));
    }

    fastRespondToMessage(
        msg,
        MessagePayload::create<SuccessResponsePayload>(
            fmt::format("Agent {} subscribed successfully to miner limit order events", sub)));
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleLocalTradeSubscription(const Message::Ptr&  msg)
{
    const auto& sub = msg->source;

    if (!m_localTradeSubscribers.add(sub)) {
        return fastRespondToMessage(
            msg,
            "ERROR",
            MessagePayload::create<ErrorResponsePayload>(
                fmt::format("Agent {} is already subscribed to trade events", sub)));
    }

    fastRespondToMessage(
        msg,
        MessagePayload::create<SuccessResponsePayload>(
            fmt::format("Agent {} subscribed successfully to trade events", sub)));
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleLocalTradeByOrderSubscription(const Message::Ptr&  msg)
{
    const auto& sub = msg->source;
    auto pptr = std::static_pointer_cast<SubscribeEventTradeByOrderPayload>(msg->payload);
    const auto orderId = pptr->id;

    if (!m_localTradeByOrderSubscribers[orderId].add(sub)) {
        return fastRespondToMessage(
            msg,
            "ERROR",
            MessagePayload::create<ErrorResponsePayload>(fmt::format(
                "Agent {} is already subscribed to trade events for order {}", sub, orderId)));
    }

    fastRespondToMessage(
        msg,
        MessagePayload::create<SuccessResponsePayload>(fmt::format(
            "Agent {} subscribed successfully to trade events for order {}", sub, orderId)));
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleLocalOwnTradeSubscription(const Message::Ptr&  msg)
{
    const auto& sub = msg->source;

    if (!m_localOwnTradeSubscribers.add(sub)) {
        return fastRespondToMessage(
            msg,
            "ERROR",
            MessagePayload::create<ErrorResponsePayload>(
                fmt::format("Agent {} is already subscribed to own trade events", sub)));
    }

    fastRespondToMessage(
        msg,
        MessagePayload::create<SuccessResponsePayload>(
            fmt::format("Agent {} subscribed successfully to own trade events", sub)));
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::handleLocalUnknownMessage(const Message::Ptr&  msg)
{
    if (msg->source == name()) { return; }
    fastRespondToMessage(
        msg,
        "ERROR",
        MessagePayload::create<ErrorResponsePayload>(fmt::format(
            "Unknown message type: {}",
            msg->type)));
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::notifyMarketOrderSubscribers(
    const MarketOrder::Ptr& marketOrder, BookId bookId, AgentId agentId)
{
    const Timestamp now = simulation()->currentTimestamp();

    auto subs = m_localMarketOrderSubscribers
        | views::filter([&](auto&& sub) {
            if (!simulation()->m_replayMode) return true;
            return simulation()->isReplacedAgent(sub);
        });

    for (const auto& sub : subs) {
        simulation()->dispatchMessage(
            now,
            1,
            name(),
            sub,
            "EVENT_ORDER_MARKET",
            MessagePayload::create<EventOrderMarketPayload>(*marketOrder, bookId, agentId));
    }
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::notifyLimitOrderSubscribers(
    const LimitOrder::Ptr& limitOrder, BookId bookId, AgentId agentId, bool isRemote)
{
    const Timestamp now = simulation()->currentTimestamp();

    auto dispatch = [&](auto&& registry) {
        auto subs = registry
            | views::filter([&](auto&& sub) {
                if (!simulation()->m_replayMode) return true;
                return simulation()->isReplacedAgent(sub);
            });
        for (const auto& sub : subs) {
            simulation()->dispatchMessage(
                now,
                0,  // deliver at the order's rest timestamp so a reactive subscriber can
                    // act before any later-delay order in the same batch (intra-round intercept)
                name(),
                sub,
                "EVENT_ORDER_LIMIT",
                MessagePayload::create<EventOrderLimitPayload>(*limitOrder, bookId, agentId));
        }
    };

    // "all" subscribers see every order (local + miner) and distinguish origin via
    // agentId; miner-only subscribers are dispatched solely for remote orders so they
    // never pay the per-background-order dispatch cost.
    dispatch(m_localLimitOrderSubscribers);
    if (isRemote) dispatch(m_minerLimitOrderSubscribers);
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::notifyTradeSubscribers(const TradeWithLogContext::Ptr& tradeWithCtx)
{
    const Timestamp now = simulation()->currentTimestamp();
    // The trade happens exactly on the receipt of the aggressing order, no processing
    // delay there; the processing delay only kicks in sending out a response and events
    // related to the matching.
    tradeWithCtx->trade->setTimestamp(now);

    auto subs = m_localTradeSubscribers
        | views::filter([&](auto&& sub) {
            if (!simulation()->m_replayMode) return true;
            return simulation()->isReplacedAgent(sub);
        });

    for (const auto& sub : subs) {
        simulation()->dispatchMessage(
            now,
            Timestamp{},
            name(),
            sub,
            "EVENT_TRADE",
            MessagePayload::create<EventTradePayload>(
                *(tradeWithCtx->trade),
                *(tradeWithCtx->logContext),
                tradeWithCtx->logContext->bookId));
    }

    // Own-fill delivery. A party to the trade is notified whether it was the aggressor
    // (the fill precedes its own placement response) or the resting side (the fill
    // arrives whenever a counterparty does), which the by-order route below cannot do
    // since that subscription is only established after matching. Remote agents are
    // served by the DISTRIBUTED_PROXY_AGENT fan-out in tradeCallback instead.
    const auto& tradeCtx = *tradeWithCtx->logContext;
    auto notifyOwn = [&](AgentId agentId) {
        if (agentId >= AgentId{}) return;
        const auto& idMap = accounts().idBimap().right;
        const auto it = idMap.find(agentId);
        if (it == idMap.end()) return;
        const LocalAgentId& sub = it->second;
        if (!m_localOwnTradeSubscribers.contains(sub)) return;
        // Already served above; a subscriber of both feeds must not net the fill twice.
        if (m_localTradeSubscribers.contains(sub)) return;
        if (simulation()->m_replayMode && !simulation()->isReplacedAgent(sub)) return;
        simulation()->dispatchMessage(
            now,
            Timestamp{},
            name(),
            sub,
            "EVENT_TRADE",
            MessagePayload::create<EventTradePayload>(
                *(tradeWithCtx->trade), tradeCtx, tradeCtx.bookId));
    };
    notifyOwn(tradeCtx.aggressingAgentId);
    // Self-match: deliver once. Consumers net both legs out of a single message, so a
    // second delivery would double-count.
    if (tradeCtx.restingAgentId != tradeCtx.aggressingAgentId) {
        notifyOwn(tradeCtx.restingAgentId);
    }

    notifyTradeSubscribersByOrderID(tradeWithCtx, tradeWithCtx->trade->aggressingOrderID());
    notifyTradeSubscribersByOrderID(tradeWithCtx, tradeWithCtx->trade->restingOrderID());
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::notifyTradeSubscribersByOrderID(
    const TradeWithLogContext::Ptr& tradeWithCtx, OrderID orderId)
{
    auto it = m_localTradeByOrderSubscribers.find(orderId);
    if (it == m_localTradeByOrderSubscribers.end()) return;

    auto subs = it->second
        | views::filter([&](auto&& sub) {
            if (!simulation()->m_replayMode) return true;
            return simulation()->isReplacedAgent(sub);
        });

    const Timestamp now = simulation()->currentTimestamp();

    for (const auto& sub : subs) {
        simulation()->dispatchMessage(
            now,
            1,
            name(),
            sub,
            "EVENT_TRADE",
            MessagePayload::create<EventTradePayload>(
                *(tradeWithCtx->trade),
                *(tradeWithCtx->logContext),
                tradeWithCtx->logContext->bookId));
    }
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::orderCallback(Order::Ptr order, OrderContext ctx)
{
    accounts()[ctx.agentId].activeOrders()[ctx.bookId].insert(order);

    // Snapshot SL/TP info into the container's side store at creation
    // time. This covers aggressing market / marketable-limit orders that
    // the book never retains, as well as resting limit orders. Looked up
    // by orderId on each subsequent fill in tradeCallback.
    if (order->hasSLTP()) {
        m_sltpContainer.onOrderCreated({
            .orderId = order->id(),
            .bookId = ctx.bookId,
            .agentId = ctx.agentId,
            .clientCtx = OrderClientContext{
                ctx.agentId, ctx.clientOrderId, {}, order->currency()},
            .originatingSide = order->direction(),
            .volume = order->volume(),
            .leverage = order->leverage(),
            .currency = order->currency(),
            .stopLoss = order->stopLoss(),
            .takeProfit = order->takeProfit(),
            .placeholder = order->placeholder()
        });
    }
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::orderLogCallback(Order::Ptr order, OrderContext ctx)
{
    if (order->totalVolume() == 0_dec) return;
    m_L3Record.at(ctx.bookId).push(taosim::event::OrderEvent(order, ctx));
    // The probe: the push above demonstrably lands (retainFrom counts it every batch) while the relay
    // slot inside ExchangeSignals never fires, so the emit below is either not reached or reaches a
    // signal with no slots. This reports which, once, including how many slots the signal actually has.
    m_signals.at(ctx.bookId)->orderLog(OrderWithLogContext(
        order, std::make_shared<OrderLogContext>(ctx.agentId, ctx.bookId)));
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::instructionLogCallback(
    const taosim::matching::OrderDesc& orderDesc, OrderID orderId)
{
    std::visit([&](auto&& desc) {
        const auto& agentId = desc.agentId;
        const auto& payload = desc.payload;

        m_signals[payload->bookId]->instructionLog(
            taosim::InstructionLogContext(
                std::visit([this](auto&& agId) -> AgentId {
                    if constexpr (std::same_as<std::remove_cvref_t<decltype(agId)>, LocalAgentId>) {
                        return accounts().lookupLocalAgentId(agId);
                    } else {
                        return agId;
                    }
                }, agentId),
                orderId,
                payload));

    }, orderDesc);
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::tradeCallback(Trade::Ptr trade, BookId bookId)
{
    const auto restingOrderId = trade->restingOrderID();
    const auto aggressingOrderId = trade->aggressingOrderID();

    const auto& book = m_books.at(bookId);

    const auto restingClientInfo = book->orderToClientInfo().at(restingOrderId);
    const auto aggressiveClientInfo = book->orderToClientInfo().at(aggressingOrderId);

    const auto& fees = m_clearingManager->handleTrade(taosim::matching::TradeDesc{
        .bookId = bookId,
        .restingAgentId = restingClientInfo.agentId,
        .aggressingAgentId = aggressiveClientInfo.agentId,
        .trade = trade
    });

    m_bookTradeStats.at(bookId).record(
        trade->price(), trade->volume(), simulation()->currentTimestamp());

    auto tradeCtx = TradeContext(bookId, aggressiveClientInfo.agentId, restingClientInfo.agentId, fees);
    tradeCtx.initiatorAgentId             = aggressiveClientInfo.initiatorAgentId;
    tradeCtx.aggressingCloseReason        = aggressiveClientInfo.closeReason;
    tradeCtx.aggressingOriginatingOrderId = aggressiveClientInfo.originatingOrderId;
    m_L3Record.at(bookId).push(taosim::event::TradeEvent(trade, std::move(tradeCtx)));

    auto logCtx = std::make_shared<TradeLogContext>(aggressiveClientInfo.agentId, restingClientInfo.agentId, bookId, fees);
    logCtx->initiatorAgentId = aggressiveClientInfo.initiatorAgentId;
    logCtx->aggressingCloseReason        = aggressiveClientInfo.closeReason;
    logCtx->aggressingOriginatingOrderId = aggressiveClientInfo.originatingOrderId;
    auto tradeWithCtx = std::make_shared<TradeWithLogContext>(trade, logCtx);

    if (!simulation()->m_replayMode) {
        const Timestamp now = simulation()->currentTimestamp();
        struct ClientInfoWithOrderId { OrderClientContext ctx; OrderID orderId{}; };
        const std::array<ClientInfoWithOrderId, 2> clientInfosWithOrderId{{
            {.ctx = restingClientInfo, .orderId = restingOrderId},
            {.ctx = aggressiveClientInfo, .orderId = aggressingOrderId}
        }};
        for (const auto& [ctx, orderId] : clientInfosWithOrderId) {
            const bool isLocalAgent = ctx.agentId < AgentId{};
            if (isLocalAgent) { continue; }
            simulation()->dispatchMessage(
                now,
                Timestamp{},
                name(),
                "DISTRIBUTED_PROXY_AGENT",
                "EVENT_TRADE",
                MessagePayload::create<DistributedAgentResponsePayload>(
                    ctx.agentId,
                    MessagePayload::create<EventTradePayload>(
                        *trade,
                        *tradeWithCtx->logContext,
                        bookId,
                        ctx.clientOrderId,
                        ctx.delegate,
                        ctx.currency,
                        ctx.agentId == restingClientInfo.agentId
                    )));
        }
    }

    // Forward both sides of the fill to the SL/TP container. The container
    // walks each agent's per-side FIFO unconditionally (so unflagged
    // counter-trades still drain prior coverage), and only produces new
    // triggers for the portion that grows the position on a flagged order.
    const auto aggressingSide = trade->direction();
    const auto restingSide = aggressingSide == OrderDirection::BUY
        ? OrderDirection::SELL
        : OrderDirection::BUY;
    struct Side { OrderID orderId; AgentId agentId; OrderDirection side; };
    const std::array<Side, 2> sides{{
        {.orderId = restingOrderId, .agentId = restingClientInfo.agentId, .side = restingSide},
        {.orderId = aggressingOrderId, .agentId = aggressiveClientInfo.agentId, .side = aggressingSide}
    }};
    for (const auto& s : sides) {
        m_sltpContainer.onOrderTrade({
            .bookId = bookId,
            .originatingOrderId = s.orderId,
            .agentId = s.agentId,
            .side = s.side,
            .fillPrice = trade->price(),
            .filledVolume = trade->volume()
        });
    }

    m_signals[bookId]->tradeLog(*tradeWithCtx);
    m_signals[bookId]->feeLog(
        m_clearingManager->feePolicy(), 
        taosim::FeeLogEvent{
            .bookId = bookId,
            .restingAgentId = restingClientInfo.agentId,
            .aggressingAgentId = aggressiveClientInfo.agentId,
            .fees = fees,
            .price = trade->price(),
            .volume = trade->volume(),
            .restingRatio = m_clearingManager->feePolicy()->makerTakerRatio(bookId, restingClientInfo.agentId),
            .aggressingRatio = m_clearingManager->feePolicy()->makerTakerRatio(bookId, aggressiveClientInfo.agentId)
        }   
    );

    notifyTradeSubscribers(tradeWithCtx);
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::unregisterLimitOrderCallback(LimitOrder::Ptr limitOrder, BookId bookId)
{
    const OrderID orderId = limitOrder->id();
    const AgentId agentId = m_books[bookId]->orderToClientInfo().at(orderId).agentId;

    taosim::accounting::Balances& balances = accounts().at(agentId).at(bookId);
    const auto freed = [&] -> taosim::accounting::ReservationAmounts {
        if (balances.canFree(orderId)){
            if (limitOrder->direction() == OrderDirection::BUY) {
                simulation()->logDebug("FREEING RESERVATION OF {} BASE + {} QUOTE for BUY order #{}", 
                    balances.base.getReservation(orderId).value_or(0_dec), balances.quote->getReservation(orderId).value_or(0_dec), orderId);
            } else {
                simulation()->logDebug("FREEING RESERVATION OF {} BASE + {} QUOTE for SELL order #{}", 
                    balances.base.getReservation(orderId).value_or(0_dec), balances.quote->getReservation(orderId).value_or(0_dec), orderId);
            }
            return balances.freeReservation(orderId, limitOrder->price(),
                    m_books[bookId]->bestBid(), m_books[bookId]->bestAsk(), limitOrder->direction(), 
                    simulation()->bookIdCanon(bookId));
        }
        return {};
    }();

    accounts()[agentId].activeOrders()[bookId].erase(limitOrder);

    if (limitOrder->volume() > 0_dec) {
        simulation()->logDebug(
            "{} | AGENT #{} BOOK {} : UNREGISTERED {} ORDER #{} ({}@{}) (FREED {} BASE + {} QUOTE) | RESERVED_QUOTE={} | RESERVED_BASE={}",
            simulation()->currentTimestamp(),
            agentId,
            simulation()->bookIdCanon(bookId),
            limitOrder->direction() == OrderDirection::BUY ? "BUY" : "SELL",
            orderId,
            limitOrder->leverage() > 0_dec ? fmt::format("{}x{}",1_dec + limitOrder->leverage(),limitOrder->volume()) : fmt::format("{}",limitOrder->volume()),
            limitOrder->price(),
            freed.base, freed.quote,
            balances.quote->getReserved(),
            balances.base.getReserved());
    }

    if (balances.quote->getReserved() < 0_dec) {
        throw std::runtime_error(fmt::format(
            "{} | AGENT #{} BOOK {} | {}: Reserved quote balance {} < 0 after unregistering order #{}", 
            simulation()->currentTimestamp(),
            agentId,
            simulation()->bookIdCanon(bookId), std::source_location::current().function_name(),
            balances.quote->getReserved(), agentId, orderId));
    }
    if (accounts()[agentId].activeOrders()[bookId].empty()) {

        if (balances.quote->getReserved() > 0_dec){
            for (const auto& res : balances.quote->getReservations()){
                fmt::println("unregisterLimitOrderCallback | Releasing Quote residual reservation {} with no corresponding active order #{} in book #{}", 
                    res.second, res.first, simulation()->bookIdCanon(bookId));
                // balances.releaseReservation(res.first, simulation()->bookIdCanon(bookId));
            }
        }

        if (balances.base.getReserved() > 0_dec){
            for (const auto& res : balances.base.getReservations()){
                fmt::println("unregisterLimitOrderCallback | Releasing Base residual reservation {} with no corresponding active order #{} in book #{}", 
                    res.second, res.first, simulation()->bookIdCanon(bookId));
                // balances.releaseReservation(res.first, simulation()->bookIdCanon(bookId));
            }
        }
    }
}

//-------------------------------------------------------------------------

void MultiBookExchangeAgent::marketOrderProcessedCallback(
    MarketOrder::Ptr marketOrder, OrderContext ctx)
{
    accounts()[ctx.agentId].activeOrders()[ctx.bookId].erase(marketOrder);
    taosim::accounting::Balances& balances = accounts()[ctx.agentId][ctx.bookId];

    if (balances.canFree(marketOrder->id())){
        balances.freeReservation(marketOrder->id(), m_books[ctx.bookId]->bestAsk(),
            m_books[ctx.bookId]->bestBid(), m_books[ctx.bookId]->bestAsk(), marketOrder->direction(), 
            simulation()->bookIdCanon(ctx.bookId));
    }
}

//-------------------------------------------------------------------------
