/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

#include "taosim/decimal/decimal.hpp"
#include "taosim/matching/TieredFeePolicy.hpp"
#include "taosim/matching/DynamicFeePolicy.hpp"
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








