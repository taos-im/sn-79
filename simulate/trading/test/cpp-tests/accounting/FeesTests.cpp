/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

#include "taosim/decimal/decimal.hpp"
#include "taosim/matching/TieredFeePolicy.hpp"
#include "taosim/matching/DynamicFeePolicy.hpp"
#include <taosim/checkpoint/serialization/matching/DynamicFeePolicy.hpp>
#include "taosim/matching/FeePolicyWrapper.hpp"
#include "taosim/message/PayloadFactory.hpp"
#include "formatting.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>


#include "MultiBookExchangeAgent.hpp"
#include "Order.hpp"
#include "Simulation.hpp"
#include <taosim/net/server.hpp>
#include "util.hpp"
#include <taosim/matching/ClearingManager.hpp>

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <pugixml.hpp>

#include <typeinfo>
#include <regex>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <functional>
#include <latch>
#include <memory>
#include <string>
#include <thread>
#include <utility>



//-------------------------------------------------------------------------

using namespace taosim;
using namespace taosim::accounting;
using namespace taosim::literals;

using namespace testing;

//-------------------------------------------------------------------------



//-------------------------------------------------------------------------

using namespace taosim::literals;
using namespace taosim::matching;

using testing::StrEq;
using testing::Values;

namespace fs = std::filesystem;


//-------------------------------------------------------------------------

namespace
{

const auto kTestDataPath = fs::path{__FILE__}.parent_path().parent_path() / "data";

std::string normalizeOutput(const std::string& input) {
    std::string result = std::regex_replace(input, std::regex(R"((\.\d*?[1-9])0+|\.(0+))"), "$1");
    result = std::regex_replace(result, std::regex(R"(\s{2,})"), " ");
    return result;

}

void printBalances(const taosim::accounting::Balances& balances, const AgentId agentId){
    std::string baseString = normalizeOutput(fmt::format("{}", balances.base));
    std::string quoteString = normalizeOutput(fmt::format("{}", *balances.quote));
    fmt::println("Agent {} => \tBase: {} \n\t\tQuote: {}", agentId, baseString, quoteString);
    for (auto it = balances.m_loans.begin(); it != balances.m_loans.end(); it++){
        if (it == balances.m_loans.begin())
            fmt::println("----------------------------");

        const auto& loan = it->second;
        fmt::println("Loan id:{}  amount:{}  lev:{}  dir:{}  col:(B:{}|Q:{})  margin:{}", 
            it->first, loan.amount(), loan.leverage(), 
            loan.direction() == OrderDirection::BUY ? "BUY" : "SELL",
            loan.collateral().base(), loan.collateral().quote(), loan.marginCallPrice()
        );
    }
    fmt::println("======================================================");
}

//-------------------------------------------------------------------------

template<typename... Args>
requires std::constructible_from<PlaceOrderMarketPayload, Args..., BookId>
std::pair<MarketOrder::Ptr, OrderErrorCode> placeMarketOrder(
    MultiBookExchangeAgent* exchange, AgentId agentId, BookId bookId, Args&&... args)
{
    const auto payload =
        MessagePayload::create<PlaceOrderMarketPayload>(std::forward<Args>(args)..., bookId);
    const auto orderResult = exchange->clearingManager().handleOrder(MarketOrderDesc{.agentId = agentId, .payload = payload});
    auto marketOrderPtr = exchange->books()[bookId]->placeMarketOrder(
        OrderClientContext{agentId},
        Timestamp{},
        orderResult.orderSize,
        payload->direction,
        payload->leverage);
    return {marketOrderPtr, orderResult.ec};
}

template<typename... Args>
requires std::constructible_from<PlaceOrderLimitPayload, Args..., BookId>
std::pair<LimitOrder::Ptr, OrderErrorCode> placeLimitOrder(
    MultiBookExchangeAgent* exchange, AgentId agentId, BookId bookId, Args&&... args)
{
    const auto payload =
        MessagePayload::create<PlaceOrderLimitPayload>(std::forward<Args>(args)..., bookId);
    const auto orderResult = exchange->clearingManager().handleOrder(LimitOrderDesc{.agentId = agentId, .payload = payload});
    auto limitOrderPtr = exchange->books()[bookId]->placeLimitOrder(
        OrderClientContext{agentId},
        Timestamp{},
        orderResult.orderSize,
        payload->direction,
        payload->price,
        payload->leverage);
    return {limitOrderPtr, orderResult.ec};
}

//-------------------------------------------------------------------------

class FeePolicyPublic: public TieredFeePolicy
{
public:
    friend class TieredFeePolicyTest;

    using TieredFeePolicy::findTierForAgent;
    using TieredFeePolicy::findTierForVolume;
};

//-------------------------------------------------------------------------

class TieredFeePolicyTest
    : public testing::TestWithParam<std::pair<Timestamp, fs::path>>
{
public:
    friend class FeePolicyPublic;

    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange;
    FeePolicyWrapper* feePolicyWrapper;
    FeePolicyPublic* feePolicy;

protected:
    void SetUp() override
    {
        nodes = taosim::util::parseSimulationFile(kTestDataPath / "MultiAgentFees.xml");
        simulation = std::make_unique<Simulation>();
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        feePolicyWrapper = exchange->clearingManager().feePolicy();
        auto feePolicyPrivate = feePolicyWrapper->defaultPolicy();
        feePolicy = reinterpret_cast<FeePolicyPublic*>(feePolicyPrivate);
        simulation->setDebug(true);
    }

};


//-------------------------------------------------------------------------

// class DyFeePolicyPublic: public DynamicFeePolicy
// {
// public:
//     friend class TieredFeePolicyTest;

//     using TieredFeePolicy::findTierForAgent;
//     using TieredFeePolicy::findTierForVolume;
// };

//-------------------------------------------------------------------------


class DynamicFeePolicyTest
    : public testing::TestWithParam<std::pair<Timestamp, fs::path>>
{
public:

    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange;
    FeePolicyWrapper* feePolicyWrapper;
    DynamicFeePolicy* feePolicy;

protected:
    void SetUp() override
    {
        nodes = taosim::util::parseSimulationFile(kTestDataPath / "MultiAgentDynamicFees.xml");
        simulation = std::make_unique<Simulation>();
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        feePolicyWrapper = exchange->clearingManager().feePolicy();
        auto feePolicyPrivate = feePolicyWrapper->defaultPolicy();
        feePolicy = reinterpret_cast<DynamicFeePolicy*>(feePolicyPrivate);
        simulation->setDebug(true);
    }

};

//-------------------------------------------------------------------------


class NegativeTieredFeePolicyTest
    : public TieredFeePolicyTest
{

protected:
    void SetUp() override
    {
        nodes = taosim::util::parseSimulationFile(kTestDataPath / "MultiAgentFeesNegative.xml");
        simulation = std::make_unique<Simulation>();
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        feePolicyWrapper = exchange->clearingManager().feePolicy();
        auto feePolicyPrivate = feePolicyWrapper->defaultPolicy();
        feePolicy = reinterpret_cast<FeePolicyPublic*>(feePolicyPrivate);
        simulation->setDebug(true);
    }

};

//-------------------------------------------------------------------------


}  // namespace



TEST_F(TieredFeePolicyTest, trackVolumes)
{
    const AgentId agent1 = -2, agent2 = -3, agent3 = -4;
    const auto agentIdBegin = agent1;
    const auto agentIdEnd = agent3;
    const BookId bookId{};
    auto book = exchange->books()[bookId];
    
    exchange->accounts().registerLocal("agent1");
    exchange->accounts().registerLocal("agent2");
    exchange->accounts().registerLocal("agent3");

    feePolicy->updateAgentsTiers();

    const auto tiers = feePolicy->tiers();

    for (auto tier: tiers){
        fmt::println("TIER  vol:{}  mkr:{}  tkr:{}", 
            tier.volumeRequired, tier.makerFeeRate, tier.takerFeeRate);
    }

    for (const auto& tier: feePolicy->agentTiers())
        fmt::println("Agent #{} is in tier {} now", tier.first, tier.second.at(bookId));

    for (AgentId agId = agentIdBegin; agId >= agentIdEnd; agId--){
        printBalances(exchange->accounts()[agId][bookId], agId);
    }

    for (AgentId agId = agentIdBegin; agId >= agentIdEnd; agId--){
        EXPECT_EQ(feePolicy->findTierForAgent(bookId, agId).volumeRequired, tiers[0].volumeRequired);
        fmt::println("FeeRates for agent#{} is ({} | {})",
            agId,
            feePolicyWrapper->getRates(bookId, agId).maker,
            feePolicyWrapper->getRates(bookId, agId).taker);
    }
    
    placeLimitOrder(exchange, agent1, bookId, OrderDirection::SELL, 61_dec, 10_dec, DEC(0.));
    placeLimitOrder(exchange, agent2, bookId, OrderDirection::BUY, 6_dec, 10_dec, DEC(1.));
    placeLimitOrder(exchange, agent3, bookId, OrderDirection::BUY, 25_dec, 10_dec, DEC(1.));

    EXPECT_EQ(feePolicy->agentVolumes().at(agent1).at(bookId).back(), 610_dec);
    EXPECT_EQ(feePolicy->agentVolumes().at(agent2).at(bookId).back(), 120_dec);
    EXPECT_EQ(feePolicy->agentVolumes().at(agent3).at(bookId).back(), 490_dec);

    for (AgentId agId = agentIdBegin; agId >= agentIdEnd; agId--){
        EXPECT_EQ(feePolicy->findTierForAgent(bookId, agId).volumeRequired, tiers[0].volumeRequired);
    }

    feePolicy->updateAgentsTiers();

    for (AgentId agId = agentIdBegin; agId >= agentIdEnd; agId--){
        EXPECT_EQ(feePolicy->findTierForAgent(bookId, agId).volumeRequired, 
            feePolicy->findTierForVolume(feePolicy->agentVolumes().at(agId).at(bookId)[feePolicy->historySlots()-2]).volumeRequired);
        fmt::println("FeeRates for agent#{} is ({} | {})",
            agId,
            feePolicyWrapper->getRates(bookId, agId).maker,
            feePolicyWrapper->getRates(bookId, agId).taker);
    }

    EXPECT_EQ(feePolicy->findTierForAgent(bookId, agent1).volumeRequired, tiers[2].volumeRequired);
    EXPECT_EQ(feePolicy->findTierForAgent(bookId, agent2).volumeRequired, tiers[1].volumeRequired);
    EXPECT_EQ(feePolicy->findTierForAgent(bookId, agent3).volumeRequired, tiers[1].volumeRequired);

    for (AgentId agId = agentIdBegin; agId >= agentIdEnd; agId--){
        printBalances(exchange->accounts()[agId][bookId], agId);
    }

    for (const auto& tier: feePolicy->agentTiers())
        fmt::println("Agent #{} is in tier {} now", tier.first, tier.second.at(bookId));
        

    placeLimitOrder(exchange, agent1, bookId, OrderDirection::SELL, 30_dec, 10_dec, DEC(0.));
    placeLimitOrder(exchange, agent2, bookId, OrderDirection::BUY, 10_dec, 10_dec, DEC(1.));
    placeLimitOrder(exchange, agent3, bookId, OrderDirection::BUY, 5_dec, 10_dec, DEC(1.));
    
    feePolicy->resetHistory();
    
    for (AgentId agId = agentIdBegin; agId >= agentIdEnd; agId--){
        EXPECT_EQ(feePolicy->findTierForAgent(bookId, agId).volumeRequired, tiers[0].volumeRequired);
        fmt::println("FeeRates for agent#{} is ({} | {})",
            agId,
            feePolicyWrapper->getRates(bookId, agId).maker,
            feePolicyWrapper->getRates(bookId, agId).taker);
    }    

}



//-------------------------------------------------------------------------



TEST_F(NegativeTieredFeePolicyTest, trackVolumes)
{
    const AgentId agent1 = -2, agent2 = -3, agent3 = -4;
    const auto agentIdBegin = agent1;
    const auto agentIdEnd = agent3;
    const BookId bookId{};
    auto book = exchange->books()[bookId];
    
    exchange->accounts().registerLocal("agent1");
    exchange->accounts().registerLocal("agent2");
    exchange->accounts().registerLocal("agent3");

    feePolicy->updateAgentsTiers();

    const auto tiers = feePolicy->tiers();

    for (auto tier: tiers){
        fmt::println("TIER  vol:{}  mkr:{}  tkr:{}", 
            tier.volumeRequired, tier.makerFeeRate, tier.takerFeeRate);
    }

    for (const auto& tier: feePolicy->agentTiers())
        fmt::println("Agent #{} is in tier {} now", tier.first, tier.second.at(bookId));

    for (AgentId agId = agentIdBegin; agId >= agentIdEnd; agId--){
        printBalances(exchange->accounts()[agId][bookId], agId);
    }

    for (AgentId agId = agentIdBegin; agId >= agentIdEnd; agId--){
        EXPECT_EQ(feePolicy->findTierForAgent(bookId, agId).volumeRequired, tiers[0].volumeRequired);
        fmt::println("FeeRates for agent#{} is ({} | {})",
            agId,
            feePolicyWrapper->getRates(bookId, agId).maker,
            feePolicyWrapper->getRates(bookId, agId).taker);
    }
    
    placeLimitOrder(exchange, agent1, bookId, OrderDirection::SELL, 61_dec, 10_dec, DEC(0.));
    placeLimitOrder(exchange, agent2, bookId, OrderDirection::BUY, 6_dec, 10_dec, DEC(1.));
    placeLimitOrder(exchange, agent3, bookId, OrderDirection::BUY, 25_dec, 10_dec, DEC(1.));

    EXPECT_EQ(feePolicy->agentVolumes().at(agent1).at(bookId).back(), 610_dec);
    EXPECT_EQ(feePolicy->agentVolumes().at(agent2).at(bookId).back(), 120_dec);
    EXPECT_EQ(feePolicy->agentVolumes().at(agent3).at(bookId).back(), 490_dec);

    for (AgentId agId = agentIdBegin; agId >= agentIdEnd; agId--){
        EXPECT_EQ(feePolicy->findTierForAgent(bookId, agId).volumeRequired, tiers[0].volumeRequired);
    }

    feePolicy->updateAgentsTiers();

    for (AgentId agId = agentIdBegin; agId >= agentIdEnd; agId--){
        EXPECT_EQ(feePolicy->findTierForAgent(bookId, agId).volumeRequired, 
            feePolicy->findTierForVolume(feePolicy->agentVolumes().at(agId).at(bookId)[feePolicy->historySlots()-2]).volumeRequired);
        fmt::println("FeeRates for agent#{} is ({} | {})",
            agId,
            feePolicyWrapper->getRates(bookId, agId).maker,
            feePolicyWrapper->getRates(bookId, agId).taker);
    }

    EXPECT_EQ(feePolicy->findTierForAgent(bookId, agent1).volumeRequired, tiers[2].volumeRequired);
    EXPECT_EQ(feePolicy->findTierForAgent(bookId, agent2).volumeRequired, tiers[1].volumeRequired);
    EXPECT_EQ(feePolicy->findTierForAgent(bookId, agent3).volumeRequired, tiers[1].volumeRequired);

    for (AgentId agId = agentIdBegin; agId >= agentIdEnd; agId--){
        printBalances(exchange->accounts()[agId][bookId], agId);
    }

    for (const auto& tier: feePolicy->agentTiers())
        fmt::println("Agent #{} is in tier {} now", tier.first, tier.second.at(bookId));
        

    placeLimitOrder(exchange, agent1, bookId, OrderDirection::SELL, 30_dec, 10_dec, DEC(0.));
    placeLimitOrder(exchange, agent2, bookId, OrderDirection::BUY, 10_dec, 10_dec, DEC(1.));
    placeLimitOrder(exchange, agent3, bookId, OrderDirection::BUY, 5_dec, 10_dec, DEC(1.));
    
    feePolicy->resetHistory();
    
    for (AgentId agId = agentIdBegin; agId >= agentIdEnd; agId--){
        EXPECT_EQ(feePolicy->findTierForAgent(bookId, agId).volumeRequired, tiers[0].volumeRequired);
        fmt::println("FeeRates for agent#{} is ({} | {})",
            agId,
            feePolicyWrapper->getRates(bookId, agId).maker,
            feePolicyWrapper->getRates(bookId, agId).taker);
    }    

}



//-------------------------------------------------------------------------


TEST_F(DynamicFeePolicyTest, trackVolumes)
{
    const AgentId agent1 = -4, agent2 = -5;
    const AgentId agent_1 = -2, agent_2 = -3;
    const auto agentIdBegin = agent2;
    const auto agentIdEnd = agent_1;
    const BookId bookId{};
    auto book = exchange->books()[bookId];
    
    // exchange->accounts().registerRemote();
    // exchange->accounts().registerRemote();
    exchange->accounts().registerLocal("agent_1");
    exchange->accounts().registerLocal("agent_2");
    exchange->accounts().registerLocal("stylized_3");
    exchange->accounts().registerLocal("stylized_4");

    for (AgentId agId = agentIdBegin; agId <= agentIdEnd; agId++){
        if (agId == 0) continue;
        printBalances(exchange->accounts()[agId][bookId], agId);
    }


    for (AgentId agId = agentIdBegin; agId <= agentIdEnd; agId++){
        // EXPECT_EQ(feePolicy->findTierForAgent(bookId, agId).volumeRequired, tiers[0].volumeRequired);
        if (agId == 0) continue;
        fmt::println("FeeRates for agent#{} is ({} | {})",
            agId,
            feePolicyWrapper->getRates(bookId, agId).maker,
            feePolicyWrapper->getRates(bookId, agId).taker);
    }
    
    placeLimitOrder(exchange, agent1, bookId, OrderDirection::SELL, 61_dec, 10_dec, DEC(0.));

    exchange->simulation()->step();

    placeLimitOrder(exchange, agent2, bookId, OrderDirection::BUY, 6_dec, 10_dec, DEC(1.));
    placeLimitOrder(exchange, agent_1, bookId, OrderDirection::BUY, 25_dec, 10_dec, DEC(1.));
    exchange->simulation()->step();

    placeLimitOrder(exchange, agent1, bookId, OrderDirection::SELL, 61_dec, 10_dec, DEC(0.));

    exchange->simulation()->step();

    for (AgentId agId = agentIdBegin; agId <= agentIdEnd; agId++){
        // EXPECT_EQ(feePolicy->findTierForAgent(bookId, agId).volumeRequired, tiers[0].volumeRequired);
        if (agId == 0) continue;
        fmt::println("FeeRates for agent#{} is ({} | {})",
            agId,
            feePolicyWrapper->getRates(bookId, agId).maker,
            feePolicyWrapper->getRates(bookId, agId).taker);
    }

    placeLimitOrder(exchange, agent2, bookId, OrderDirection::BUY, 6_dec, 10_dec, DEC(1.));
    placeLimitOrder(exchange, agent_1, bookId, OrderDirection::BUY, 25_dec, 10_dec, DEC(1.));
    exchange->simulation()->step();
    placeLimitOrder(exchange, agent_2, bookId, OrderDirection::BUY, 25_dec, 10_dec, DEC(1.));
    placeLimitOrder(exchange, agent1, bookId, OrderDirection::SELL, 25_dec, 10_dec, DEC(1.));

    for (AgentId agId = agentIdBegin; agId <= agentIdEnd; agId++){
        // EXPECT_EQ(feePolicy->findTierForAgent(bookId, agId).volumeRequired, tiers[0].volumeRequired);
        if (agId == 0) continue;
        fmt::println("FeeRates for agent#{} is ({} | {})",
            agId,
            feePolicyWrapper->getRates(bookId, agId).maker,
            feePolicyWrapper->getRates(bookId, agId).taker);
    }
}



//-------------------------------------------------------------------------









//-------------------------------------------------------------------------
// The dynamic fee policy keeps one history per book. A book's slots, running total, previous-step total and
// last-update time are its own, whichever other books of the same policy trade at the same time.

namespace
{

std::unique_ptr<DynamicFeePolicy> makeDynamicPolicy(int historySlots)
{
    return std::make_unique<DynamicFeePolicy>(DFPolicyDesc{
        .simulation = nullptr,
        .historySlots = historySlots,
        .makerFee = 0_dec,
        .takerFee = 0_dec,
        .maxMakerRate = DEC(0.015),
        .maxTakerRate = DEC(0.015),
        .targetMTR = DEC(0.4),
        .shapeMakerFee = 2_dec,
        .shapeMakerRebate = 2_dec});
}

struct FeeTrade
{
    Timestamp t;
    BookId book;
    decimal_t volume;
    bool aggressive;
};

struct BookHistory
{
    std::vector<Volumes> slots;
    Volumes total{};
    std::optional<Volumes> prev;
};

// What each book must hold after `trades`, recomputed from that book's trades alone: a new slot whenever the
// book's trade time changes, the newest `capacity` slots, their sum, and the sum when its newest slot opened.
std::map<BookId, BookHistory> recomputeHistory(const std::vector<FeeTrade>& trades, size_t capacity)
{
    std::map<BookId, std::vector<std::pair<Timestamp, Volumes>>> perBook;
    for (const auto& tr : trades) {
        auto& slots = perBook[tr.book];
        if (slots.empty() || slots.back().first != tr.t) {
            slots.push_back({tr.t, Volumes{}});
        }
        (tr.aggressive ? slots.back().second.aggressive : slots.back().second.passive) += tr.volume;
    }
    std::map<BookId, BookHistory> out;
    for (const auto& [book, slots] : perBook) {
        BookHistory h;
        const size_t newest = slots.size() - 1;
        for (size_t i = slots.size() > capacity ? slots.size() - capacity : 0; i < slots.size(); ++i) {
            h.slots.push_back(slots[i].second);
            h.total.aggressive += slots[i].second.aggressive;
            h.total.passive += slots[i].second.passive;
        }
        if (newest >= 1) {
            Volumes prev{};
            for (size_t i = newest > capacity ? newest - capacity : 0; i < newest; ++i) {
                prev.aggressive += slots[i].second.aggressive;
                prev.passive += slots[i].second.passive;
            }
            h.prev = prev;
        }
        out[book] = h;
    }
    return out;
}

void expectHistory(const DynamicFeePolicy& policy, const std::map<BookId, BookHistory>& want)
{
    for (const auto& [book, h] : want) {
        ASSERT_TRUE(policy.volumes().contains(book)) << "book " << book;
        const auto& slots = policy.volumes().at(book);
        ASSERT_EQ(slots.size(), h.slots.size()) << "book " << book;
        for (size_t i = 0; i < slots.size(); ++i) {
            EXPECT_TRUE(slots[i].aggressive == h.slots[i].aggressive) << "book " << book << " slot " << i;
            EXPECT_TRUE(slots[i].passive == h.slots[i].passive) << "book " << book << " slot " << i;
        }
        EXPECT_TRUE(policy.totalVolumes().at(book).aggressive == h.total.aggressive) << "book " << book;
        EXPECT_TRUE(policy.totalVolumes().at(book).passive == h.total.passive) << "book " << book;
        if (h.prev) {
            ASSERT_TRUE(policy.totalVolumesPrev().contains(book)) << "book " << book;
            EXPECT_TRUE(policy.totalVolumesPrev().at(book).aggressive == h.prev->aggressive) << "book " << book;
            EXPECT_TRUE(policy.totalVolumesPrev().at(book).passive == h.prev->passive) << "book " << book;
        }
    }
}

}  // namespace

TEST(DynamicFeePolicyHistory, aBooksFirstTradeAtAnotherBooksTimeOpensItsOwnSlot)
{
    const std::vector<FeeTrade> trades{{10, 0, 5_dec, true}, {10, 1, 7_dec, false}};
    auto policy = makeDynamicPolicy(3);
    for (const auto& tr : trades) policy->updateHistory(tr.t, tr.book, 1, tr.volume, tr.aggressive);
    ASSERT_EQ(policy->volumes().at(1).size(), 1u);
    EXPECT_TRUE(policy->volumes().at(1).back().passive == 7_dec);
    expectHistory(*policy, recomputeHistory(trades, 3));
}

TEST(DynamicFeePolicyHistory, interleavedBooksKeepTheirOwnSlotsTotalsAndEviction)
{
    std::vector<FeeTrade> trades;
    Timestamp t = 100;
    for (int round = 0; round < 12; ++round) {
        t += 10;
        trades.push_back({t, 0, util::double2decimal(round + 1), round % 2 == 0});
        trades.push_back({t, 1, util::double2decimal(2 * (round + 1)), round % 3 == 0});
        if (round % 4 == 0) trades.push_back({t, 0, 1_dec, false});
        if (round % 3 == 1) trades.push_back({t + 5, 1, 3_dec, true});
    }
    auto policy = makeDynamicPolicy(4);
    for (const auto& tr : trades) policy->updateHistory(tr.t, tr.book, 1, tr.volume, tr.aggressive);
    expectHistory(*policy, recomputeHistory(trades, 4));
}

TEST(DynamicFeePolicyHistory, aResetClearsTheTotalsWithTheSlots)
{
    auto policy = makeDynamicPolicy(3);
    policy->updateHistory(10, 0, 1, 5_dec, true);
    policy->updateHistory(20, 0, 1, 6_dec, false);
    policy->resetHistory();
    policy->updateHistory(20, 0, 1, 2_dec, true);
    ASSERT_EQ(policy->volumes().at(0).size(), 1u);
    EXPECT_TRUE(policy->totalVolumes().at(0).aggressive == 2_dec);
    EXPECT_TRUE(policy->totalVolumes().at(0).passive == 0_dec);
}

TEST(DynamicFeePolicyHistory, aCheckpointKeepsEachBooksTimeAndAPolicyWideOneStillResumes)
{
    auto policy = makeDynamicPolicy(3);
    policy->updateHistory(10, 0, 1, 5_dec, true);
    policy->updateHistory(10, 1, 1, 7_dec, false);
    policy->updateHistory(20, 0, 1, 1_dec, false);

    // Packed through BinaryStream, as a checkpoint is: the decimal adaptor accepts no other stream.
    taosim::serialization::BinaryStream current;
    msgpack::packer<taosim::serialization::BinaryStream>{current}.pack(*policy);
    auto restored = makeDynamicPolicy(3);
    msgpack::unpack(current.data(), current.size()).get().convert(*restored);
    EXPECT_EQ(restored->lastUpdate(), policy->lastUpdate());
    restored->updateHistory(20, 1, 1, 4_dec, true);
    EXPECT_EQ(restored->volumes().at(1).size(), 2u) << "book 1 last traded at 10, so 20 opens its own slot";

    taosim::serialization::BinaryStream policyWide;
    msgpack::packer<taosim::serialization::BinaryStream> pk{policyWide};
    pk.pack_map(5);
    pk.pack("type");
    pk.pack("dynamic");
    pk.pack("lastUpdate");
    pk.pack(Timestamp{20});
    pk.pack("totalVolumes");
    pk.pack(policy->totalVolumes());
    pk.pack("totalVolumesPrev");
    pk.pack(policy->totalVolumesPrev());
    pk.pack("volumes");
    pk.pack(policy->volumes());
    auto resumed = makeDynamicPolicy(3);
    msgpack::unpack(policyWide.data(), policyWide.size()).get().convert(*resumed);
    EXPECT_EQ(resumed->lastUpdate().at(0), Timestamp{20});
    EXPECT_EQ(resumed->lastUpdate().at(1), Timestamp{20});
    resumed->updateHistory(30, 1, 1, 4_dec, true);
    EXPECT_EQ(resumed->volumes().at(1).size(), 2u);
    EXPECT_TRUE(resumed->totalVolumes().at(1).aggressive == 4_dec);
}

//-------------------------------------------------------------------------
// A REBATE NEVER EXCEEDS THE FEE THAT PAYS IT. The exchange pays a maker's rebate out of the taker's fee,
// so a configuration whose rates could sum below zero is refused, and a fill whose two agents' policies would still
// pair such rates has its rebate cut to the taker's fee.

namespace
{

[[nodiscard]] pugi::xml_node feeNode(pugi::xml_document& doc, const char* xml)
{
    EXPECT_TRUE(doc.load_string(xml));
    return doc.first_child();
}

// Rates by agent: agent 1 rests with a large rebate, agent 2 takes with a small fee.
struct PairedRatesPolicy : FeePolicy
{
    Fees calculateFees(const TradeDesc&) const override { return {}; }
    void updateHistory(Timestamp, BookId, AgentId, decimal_t, std::optional<bool>) override {}
    void resetHistory() noexcept override {}
    void resetHistory(const std::unordered_set<AgentId>&) noexcept override {}
    Fees getRates(BookId, AgentId agentId) const override
    {
        return agentId == 1 ? Fees{.maker = -DEC(0.003), .taker = DEC(0.003)} : Fees{.maker = DEC(0.0), .taker = DEC(0.001)};
    }
};

}  // namespace

TEST(FeeSumGuard, ADynamicPolicyWhoseRatesSumBelowZeroIsRefused)
{
    pugi::xml_document doc;
    const auto node = feeNode(doc,
        R"(<FeePolicy type="dynamic" historySlots="2" makerFee="-0.0003" takerFee="0.0002" maxMakerRate="0.0001" )"
        R"(maxTakerRate="0.0001" targetMTR="0.5" shapeMakerFee="1" shapeMakerRebate="1"/>)");
    EXPECT_THROW({ [[maybe_unused]] auto p = DynamicFeePolicy::fromXML(node, nullptr); }, std::invalid_argument);
}

TEST(FeeSumGuard, ADynamicPolicyWhoseRatesSumToZeroOrMoreIsRead)
{
    pugi::xml_document doc;
    const auto node = feeNode(doc,
        R"(<FeePolicy type="dynamic" historySlots="2" makerFee="-0.0002" takerFee="0.0002" maxMakerRate="0.0001" )"
        R"(maxTakerRate="0.0001" targetMTR="0.5" shapeMakerFee="1" shapeMakerRebate="1"/>)");
    EXPECT_NO_THROW({ [[maybe_unused]] auto p = DynamicFeePolicy::fromXML(node, nullptr); });
}

TEST(FeeSumGuard, TiersWhoseLowestMakerAndTakerRatesSumBelowZeroAreRefused)
{
    pugi::xml_document doc;
    const auto node = feeNode(doc,
        R"(<FeePolicy type="tiered" historySlots="2" slotPeriod="10">)"
        R"(<Tier volumeRequired="0" makerFee="0.001" takerFee="0.001"/>)"
        R"(<Tier volumeRequired="100" makerFee="-0.002" takerFee="0.003"/>)"
        R"(</FeePolicy>)");
    EXPECT_THROW({ [[maybe_unused]] auto p = TieredFeePolicy::fromXML(node, nullptr); }, std::invalid_argument)
        << "a maker in the second tier (-0.002) can trade with a taker in the first (0.001)";
}

TEST(FeeSumGuard, AFillWhoseTwoAgentsRatesSumBelowZeroHasItsRebateCutToTheFee)
{
    accounting::AccountRegistry registry;
    FeePolicyWrapper wrapper{std::make_unique<PairedRatesPolicy>(), &registry};
    const auto trade = Trade::create(1, Timestamp{1}, OrderDirection::BUY, 2, 3, 2_dec, 100_dec);
    const auto fees = wrapper.calculateFees(TradeDesc{.bookId = 0, .restingAgentId = 1, .aggressingAgentId = 2, .trade = trade});
    EXPECT_EQ(fees.taker, DEC(0.2)) << "the taker pays its own rate";
    EXPECT_EQ(fees.maker, -DEC(0.2)) << "the rebate is cut to the fee that pays it, so the fill sums to zero";
    const auto even = wrapper.calculateFees(TradeDesc{.bookId = 0, .restingAgentId = 2, .aggressingAgentId = 1, .trade = trade});
    EXPECT_EQ(even.maker, DEC(0.0));
    EXPECT_EQ(even.taker, DEC(0.6)) << "a fill whose rates sum to zero or more is unchanged";
}

