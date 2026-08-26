/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

#include "taosim/matching/FeePolicy.hpp"
#include "taosim/matching/FeePolicyWrapper.hpp"
#include "taosim/decimal/decimal.hpp"
#include "formatting.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "MultiBookExchangeAgent.hpp"
#include "Order.hpp"
#include "taosim/message/PayloadFactory.hpp"
#include "Simulation.hpp"
#include <taosim/net/server.hpp>
#include "util.hpp"
#include <taosim/matching/ClearingManager.hpp>

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <pugixml.hpp>

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
using namespace taosim::book;
using namespace taosim::matching;
using namespace taosim::literals;

using namespace testing;

using testing::StrEq;
using testing::Values;

namespace fs = std::filesystem;

//-------------------------------------------------------------------------

static constexpr bool postOnly = false;
static constexpr taosim::TimeInForce timeInForce = taosim::TimeInForce::GTC;

//-------------------------------------------------------------------------

namespace
{

const auto kTestDataPath = fs::path{__FILE__}.parent_path() / "data";

std::string normalizeOutput(const std::string& input) {
    std::string result = std::regex_replace(input, std::regex(R"((\.\d*?[1-9])0+|\.(0+))"), "$1");
    result = std::regex_replace(result, std::regex(R"(\s{2,})"), " ");
    return result;

}

void printOrderbook(Book::Ptr book){
    const auto orderbookState = normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); }));
    fmt::println("{}", orderbookState);
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
    MultiBookExchangeAgent* exchange, AgentId agentId, BookId bookId, STPFlag stpFlag, Args&&... args)
{
    const auto payload = MessagePayload::create<PlaceOrderMarketPayload>(
        std::forward<Args>(args)..., bookId, Currency::BASE, std::nullopt, stpFlag);
    const auto orderResult = exchange->clearingManager().handleOrder(MarketOrderDesc{.agentId = agentId, .payload = payload});
    auto marketOrderPtr = exchange->books()[bookId]->placeMarketOrder(
        OrderClientContext{agentId},
        Timestamp{},
        orderResult.orderSize,
        payload->direction,
        payload->leverage,
        payload->stpFlag);
    return {marketOrderPtr, orderResult.ec};
}

// A miner's whole batch of orders carries ONE timestamp, so two orders on the same book in the
// same batch are separate instructions that share one. Pinning the clock reproduces that.
std::optional<Timestamp> g_pinnedOrderTimestamp{};

struct PinnedOrderTimestamp
{
    explicit PinnedOrderTimestamp(Timestamp ts) { g_pinnedOrderTimestamp = ts; }
    ~PinnedOrderTimestamp() { g_pinnedOrderTimestamp.reset(); }
};

template<typename... Args>
requires std::constructible_from<PlaceOrderLimitPayload, Args..., BookId>
std::pair<LimitOrder::Ptr, OrderErrorCode> placeLimitOrder(
    MultiBookExchangeAgent* exchange,
    AgentId agentId,
    BookId bookId,
    bool postOnly,
    taosim::TimeInForce timeInForce,
    std::optional<Timestamp> expiryPeriod,
    STPFlag stpFlag,
    Args&&... args)
{
    const auto payload = MessagePayload::create<PlaceOrderLimitPayload>(
        std::forward<Args>(args)..., bookId, Currency::BASE, std::nullopt, postOnly, timeInForce, std::nullopt, stpFlag);
    const auto orderResult = exchange->clearingManager().handleOrder(LimitOrderDesc{.agentId = agentId, .payload = payload});
    // DISTINCT TIMESTAMPS, because equal ones mean "same instruction" to the engine. Book.cpp stands STP
    // down when iop->timestamp() == order->timestamp(): that is the sweep-and-its-target case, where
    // cancelling the resting side would destroy the very order the sweep exists to fill. Passing
    // Timestamp{} for every order therefore told the engine these were all one instruction, STP correctly
    // never fired, and the test asserted that it had. Separate instructions carry separate timestamps in
    // production, so a monotonic counter is what this test actually means.
    static Timestamp s_testOrderClock{};
    ++s_testOrderClock;
    const Timestamp orderTimestamp = g_pinnedOrderTimestamp.value_or(s_testOrderClock);
    auto limitOrderPtr = exchange->books()[bookId]->placeLimitOrder(
        OrderClientContext{agentId},
        orderTimestamp,
        orderResult.orderSize,
        payload->direction,
        payload->price,
        payload->leverage,
        payload->stpFlag);
    return {limitOrderPtr, orderResult.ec};
}

template<typename... Args>
requires std::constructible_from<PlaceOrderLimitPayload, Args..., BookId>
std::pair<LimitOrder::Ptr, OrderErrorCode> placeLimitOrder(
    MultiBookExchangeAgent* exchange, AgentId agentId, BookId bookId, Args&&... args)
{
    return placeLimitOrder(
        exchange,
        agentId,
        bookId,
        false,
        taosim::TimeInForce::GTC,
        std::nullopt,
        STPFlag::CO,
        std::forward<Args>(args)...);
}

//-------------------------------------------------------------------------

class SelfTradePreventionTest
    : public testing::TestWithParam<std::pair<Timestamp, fs::path>>
{
public:
    const AgentId agent1 = -1, agent2 = -2, agent3 = -3, agent4 = -4;
    const BookId bookId{};

    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange;
    Book::Ptr book;

protected:
    void SetUp() override
    {
        static constexpr Timestamp kStepSize = 10;
        nodes = taosim::util::parseSimulationFile(kTestDataPath / "MultiAgentFees.xml");
        simulation = std::make_unique<Simulation>();
        simulation->setDebug(false);
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        book = exchange->books()[bookId];
    
        exchange->accounts().registerLocal("agent1");
        exchange->accounts().registerLocal("agent2");
        exchange->accounts().registerLocal("agent3");
        exchange->accounts().registerLocal("agent4");

        // filling the book
        placeLimitOrder(exchange, agent4, bookId, OrderDirection::BUY, 3_dec, 291_dec, DEC(0.));
        placeLimitOrder(exchange, agent4, bookId, OrderDirection::BUY, 1_dec, 297_dec, DEC(0.));
        placeLimitOrder(exchange, agent4, bookId, OrderDirection::SELL, 2_dec, 303_dec, DEC(0.));
        placeLimitOrder(exchange, agent4, bookId, OrderDirection::SELL, 8_dec, 307_dec, DEC(0.));
    }

};

//-------------------------------------------------------------------------

}  // namespace

//-------------------------------------------------------------------------

TEST_F(SelfTradePreventionTest, LimitOrderBuyCO)
{

    EXPECT_THAT(
        normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
        StrEq("ask,303,2,307,8\n"
              "bid,297,1,291,3\n"));
              
    
    //---------------------- No prevention trades
    placeLimitOrder(exchange, agent1, bookId, OrderDirection::BUY, 5_dec, 301_dec, DEC(0.));

    EXPECT_THAT(
        normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
        StrEq("ask,303,2,307,8\n"
              "bid,301,5,297,1,291,3\n"));

    placeLimitOrder(exchange, agent2, bookId, OrderDirection::SELL, 4_dec, 301_dec, DEC(1.));

    EXPECT_THAT(
        normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
        StrEq("ask,301,3,303,2,307,8\n"
              "bid,297,1,291,3\n"));

    placeLimitOrder(exchange, agent3, bookId, OrderDirection::SELL, 2_dec, 301_dec, DEC(.5));

    EXPECT_THAT(
        normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
        StrEq("ask,301,6,303,2,307,8\n"
              "bid,297,1,291,3\n"));
    //-------------------------------------------------------------------------

    
    //---------------------- Buy STP | Normal
    placeLimitOrder(exchange, agent2, bookId, OrderDirection::BUY, 5_dec, 301_dec, DEC(0.));

    EXPECT_THAT(
        normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
        StrEq("ask,303,2,307,8\n"
              "bid,301,2,297,1,291,3\n"));
    //-------------------------------------------------------------------------


    //---------------------- Buy STP | Margin
    placeLimitOrder(exchange, agent3, bookId, OrderDirection::SELL, 3_dec, 301_dec, DEC(1.));

    EXPECT_THAT(
        normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
        StrEq("ask,301,4,303,2,307,8\n"
              "bid,297,1,291,3\n"));

    placeLimitOrder(exchange, agent3, bookId, OrderDirection::BUY, 2_dec, 301_dec, DEC(0.));

    EXPECT_THAT(
        normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
        StrEq("ask,303,2,307,8\n"
              "bid,301,2,297,1,291,3\n"));
    //-------------------------------------------------------------------------

}

//-------------------------------------------------------------------------

// SEPARATE INSTRUCTIONS CAN SHARE A TIMESTAMP, so STP must not key off timestamp equality.
//
// Simulation::m_time.current is set per delivered message to instr.delay, a per-batch offset, so a
// miner's whole batch carries one timestamp. Measured on the running exchange: agent 4 registered
// 126 orders stamped 12893798, one per book, plus groups of 122 and 118. Today's traffic happens to
// place at most one order per book per batch, which is the only reason Book.cpp's
// `iop->timestamp() != order->timestamp()` guard does not misfire. Two orders on the SAME book in
// one batch -- a bid and an ask, or two ladder levels -- are separate instructions sharing a
// timestamp, and the guard reads them as one instruction and stands STP down.
TEST_F(SelfTradePreventionTest, SameBatchTimestampStillPrevents)
{
    PinnedOrderTimestamp pinned{Timestamp{4242}};

    placeLimitOrder(exchange, agent1, bookId, OrderDirection::SELL, 3_dec, 301_dec, DEC(0.));

    EXPECT_THAT(
        normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
        StrEq("ask,301,3,303,2,307,8\n"
              "bid,297,1,291,3\n"));

    // Same agent, same book, same batch timestamp: CO must cancel the resting sell and rest the buy.
    placeLimitOrder(exchange, agent1, bookId, OrderDirection::BUY, 2_dec, 301_dec, DEC(0.));

    EXPECT_THAT(
        normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
        StrEq("ask,303,2,307,8\n"
              "bid,301,2,297,1,291,3\n"));
}

// //-------------------------------------------------------------------------

// TEST_F(SelfTradePreventionTest, LimitOrderSellCO)
// {
//     //---------------------- No prevention trades
//     placeLimitOrder(exchange, agent1, bookId, OrderDirection::SELL, 5_dec, 299_dec, DEC(0.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,299,5,303,2,307,8\n"
//               "bid,297,1,291,3\n"));

//     placeLimitOrder(exchange, agent2, bookId, OrderDirection::BUY, 4_dec, 299_dec, DEC(1.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,303,2,307,8\n"
//               "bid,299,3,297,1,291,3\n"));

//     placeLimitOrder(exchange, agent3, bookId, OrderDirection::BUY, 2_dec, 299_dec, DEC(.5));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,303,2,307,8\n"
//               "bid,299,6,297,1,291,3\n"));
//     //-------------------------------------------------------------------------

    
//     //---------------------- SELL STP | Normal
//     placeLimitOrder(exchange, agent2, bookId, OrderDirection::SELL, 5_dec, 299_dec, DEC(0.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,299,2,303,2,307,8\n"
//               "bid,297,1,291,3\n"));
//     //-------------------------------------------------------------------------


//     //---------------------- SELL STP | Margin
//     placeLimitOrder(exchange, agent3, bookId, OrderDirection::BUY, 3_dec, 299_dec, DEC(1.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,303,2,307,8\n"
//               "bid,299,4,297,1,291,3\n"));

//     placeLimitOrder(exchange, agent3, bookId, OrderDirection::SELL, 1_dec, 299_dec, DEC(1.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,299,2,303,2,307,8\n"
//               "bid,297,1,291,3\n"));
//     //-------------------------------------------------------------------------

// }

// //-------------------------------------------------------------------------

// TEST_F(SelfTradePreventionTest, LimitOrderBuyNone)
// {
//     //---------------------- No prevention trades
//     placeLimitOrder(exchange, agent1, bookId, OrderDirection::BUY, 5_dec, 301_dec, DEC(0.));
//     placeLimitOrder(exchange, agent2, bookId, OrderDirection::SELL, 4_dec, 301_dec, DEC(1.));
//     placeLimitOrder(exchange, agent3, bookId, OrderDirection::SELL, 2_dec, 301_dec, DEC(.5));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,301,6,303,2,307,8\n"
//               "bid,297,1,291,3\n"));
//     //-------------------------------------------------------------------------

//     const STPFlag stpFlag = STPFlag::NONE;

//     //---------------------- Buy STP | Normal
//     placeLimitOrder(exchange, agent2, bookId, postOnly, timeInForce, std::nullopt, stpFlag, OrderDirection::BUY, 2_dec, 301_dec, DEC(0.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,301,4,303,2,307,8\n"
//               "bid,297,1,291,3\n"));
//     //-------------------------------------------------------------------------


//     //---------------------- Buy STP | Margin
//     placeLimitOrder(exchange, agent3, bookId, postOnly, timeInForce, std::nullopt, stpFlag, OrderDirection::BUY, 2_dec, 301_dec, DEC(1.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,303,2,307,8\n"
//               "bid,297,1,291,3\n"));
//     //-------------------------------------------------------------------------

// }

// //-------------------------------------------------------------------------

// TEST_F(SelfTradePreventionTest, LimitOrderSellNone)
// {
//     //---------------------- No prevention trades
//     placeLimitOrder(exchange, agent1, bookId, OrderDirection::SELL, 5_dec, 299_dec, DEC(0.));
//     placeLimitOrder(exchange, agent2, bookId, OrderDirection::BUY, 4_dec, 299_dec, DEC(1.));
//     placeLimitOrder(exchange, agent3, bookId, OrderDirection::BUY, 2_dec, 299_dec, DEC(.5));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,303,2,307,8\n"
//               "bid,299,6,297,1,291,3\n"));
//     //-------------------------------------------------------------------------

//     const STPFlag stpFlag = STPFlag::NONE;
    
//     //---------------------- SELL STP | Normal
//     placeLimitOrder(exchange, agent2, bookId, postOnly, timeInForce, std::nullopt, stpFlag, OrderDirection::SELL, 2_dec, 299_dec, DEC(0.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,303,2,307,8\n"
//               "bid,299,4,297,1,291,3\n"));
//     //-------------------------------------------------------------------------


//     //---------------------- SELL STP | Margin
//     placeLimitOrder(exchange, agent3, bookId, postOnly, timeInForce, std::nullopt, stpFlag, OrderDirection::SELL, 2_dec, 299_dec, DEC(1.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,303,2,307,8\n"
//               "bid,297,1,291,3\n"));
//     //-------------------------------------------------------------------------

// }

// //-------------------------------------------------------------------------

// TEST_F(SelfTradePreventionTest, LimitOrderBuyCN)
// {
                 
//     //---------------------- No prevention trades
//     placeLimitOrder(exchange, agent1, bookId, OrderDirection::BUY, 5_dec, 301_dec, DEC(0.));
//     placeLimitOrder(exchange, agent2, bookId, OrderDirection::SELL, 4_dec, 301_dec, DEC(1.));
//     placeLimitOrder(exchange, agent3, bookId, OrderDirection::SELL, 2_dec, 301_dec, DEC(.5));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,301,6,303,2,307,8\n"
//               "bid,297,1,291,3\n"));
//     //-------------------------------------------------------------------------

    
//     const STPFlag stpFlag = STPFlag::CN;

//     //---------------------- Buy STP | Normal
//     placeLimitOrder(exchange, agent2, bookId, postOnly, timeInForce, std::nullopt, stpFlag, OrderDirection::BUY, 5_dec, 301_dec, DEC(0.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,301,6,303,2,307,8\n"
//               "bid,297,1,291,3\n"));
//     //-------------------------------------------------------------------------


//     //---------------------- Buy STP | Margin
//     placeLimitOrder(exchange, agent3, bookId, postOnly, timeInForce, std::nullopt, stpFlag, OrderDirection::BUY, 1_dec, 301_dec, DEC(1.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,301,4,303,2,307,8\n"
//               "bid,297,1,291,3\n"));

//     placeLimitOrder(exchange, agent3, bookId, postOnly, timeInForce, std::nullopt, stpFlag, OrderDirection::BUY, 2_dec, 301_dec, DEC(1.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,301,3,303,2,307,8\n"
//               "bid,297,1,291,3\n"));
//     //-------------------------------------------------------------------------

// }

// //-------------------------------------------------------------------------

// TEST_F(SelfTradePreventionTest, LimitOrderSellCN)
// {
//     //---------------------- No prevention trades
//     placeLimitOrder(exchange, agent1, bookId, OrderDirection::SELL, 5_dec, 299_dec, DEC(0.));
//     placeLimitOrder(exchange, agent2, bookId, OrderDirection::BUY, 4_dec, 299_dec, DEC(1.));
//     placeLimitOrder(exchange, agent3, bookId, OrderDirection::BUY, 2_dec, 299_dec, DEC(.5));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,303,2,307,8\n"
//               "bid,299,6,297,1,291,3\n"));
//     //-------------------------------------------------------------------------

//     const STPFlag stpFlag = STPFlag::CN;
    
//     //---------------------- SELL STP | Normal
//     placeLimitOrder(exchange, agent2, bookId, postOnly, timeInForce, std::nullopt, stpFlag, OrderDirection::SELL, 5_dec, 299_dec, DEC(0.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,303,2,307,8\n"
//               "bid,299,6,297,1,291,3\n"));
//     //-------------------------------------------------------------------------


//     //---------------------- SELL STP | Margin
//     placeLimitOrder(exchange, agent3, bookId, postOnly, timeInForce, std::nullopt, stpFlag, OrderDirection::SELL, 1_dec, 299_dec, DEC(1.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,303,2,307,8\n"
//               "bid,299,4,297,1,291,3\n"));

//     placeLimitOrder(exchange, agent3, bookId, postOnly, timeInForce, std::nullopt, stpFlag, OrderDirection::SELL, 2_dec, 299_dec, DEC(1.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,303,2,307,8\n"
//               "bid,299,3,297,1,291,3\n"));
//     //-------------------------------------------------------------------------

// }

// //-------------------------------------------------------------------------

// TEST_F(SelfTradePreventionTest, LimitOrderBuyCB)
// {
                 
//     //---------------------- No prevention trades
//     placeLimitOrder(exchange, agent1, bookId, OrderDirection::BUY, 5_dec, 301_dec, DEC(0.));
//     placeLimitOrder(exchange, agent2, bookId, OrderDirection::SELL, 4_dec, 301_dec, DEC(1.));
//     placeLimitOrder(exchange, agent4, bookId, OrderDirection::SELL, 2_dec, 301_dec, DEC(0.));
//     placeLimitOrder(exchange, agent3, bookId, OrderDirection::SELL, 2_dec, 301_dec, DEC(.5));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,301,8,303,2,307,8\n"
//               "bid,297,1,291,3\n"));
//     //-------------------------------------------------------------------------

    
//     const STPFlag stpFlag = STPFlag::CB;

//     //---------------------- Buy STP | Normal
//     placeLimitOrder(exchange, agent2, bookId, postOnly, timeInForce, std::nullopt, stpFlag, OrderDirection::BUY, 2_dec, 301_dec, DEC(0.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,301,5,303,2,307,8\n"
//               "bid,297,1,291,3\n"));
//     //-------------------------------------------------------------------------


//     //---------------------- Buy STP | Margin
//     placeLimitOrder(exchange, agent3, bookId, postOnly, timeInForce, std::nullopt, stpFlag, OrderDirection::BUY, 1_dec, 301_dec, DEC(.0));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,301,4,303,2,307,8\n"
//               "bid,297,1,291,3\n"));

//     placeLimitOrder(exchange, agent3, bookId, postOnly, timeInForce, std::nullopt, stpFlag, OrderDirection::BUY, 1_dec, 301_dec, DEC(1.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,303,2,307,8\n"
//               "bid,297,1,291,3\n"));
//     //-------------------------------------------------------------------------

// }

// //-------------------------------------------------------------------------

// TEST_F(SelfTradePreventionTest, LimitOrderSellCB)
// {
//     //---------------------- No prevention trades
//     placeLimitOrder(exchange, agent1, bookId, OrderDirection::SELL, 5_dec, 299_dec, DEC(0.));
//     placeLimitOrder(exchange, agent2, bookId, OrderDirection::BUY, 4_dec, 299_dec, DEC(1.));
//     placeLimitOrder(exchange, agent4, bookId, OrderDirection::BUY, 2_dec, 299_dec, DEC(0.));
//     placeLimitOrder(exchange, agent3, bookId, OrderDirection::BUY, 2_dec, 299_dec, DEC(.5));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,303,2,307,8\n"
//               "bid,299,8,297,1,291,3\n"));
//     //-------------------------------------------------------------------------

//     const STPFlag stpFlag = STPFlag::CB;
    
//     //---------------------- SELL STP | Normal
//     placeLimitOrder(exchange, agent2, bookId, postOnly, timeInForce, std::nullopt, stpFlag, OrderDirection::SELL, 2_dec, 299_dec, DEC(0.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,303,2,307,8\n"
//               "bid,299,5,297,1,291,3\n"));
//     //-------------------------------------------------------------------------


//     //---------------------- SELL STP | Margin
//     placeLimitOrder(exchange, agent3, bookId, postOnly, timeInForce, std::nullopt, stpFlag, OrderDirection::SELL, 1_dec, 299_dec, DEC(0.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,303,2,307,8\n"
//               "bid,299,4,297,1,291,3\n"));

//     placeLimitOrder(exchange, agent3, bookId, postOnly, timeInForce, std::nullopt, stpFlag, OrderDirection::SELL, 2_dec, 299_dec, DEC(1.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,303,2,307,8\n"
//               "bid,297,1,291,3\n"));
//     //-------------------------------------------------------------------------

// }

// //-------------------------------------------------------------------------

// TEST_F(SelfTradePreventionTest, LimitOrderBuyDC)
// {
                 
//     //---------------------- No prevention trades
//     placeLimitOrder(exchange, agent1, bookId, OrderDirection::BUY, 5_dec, 301_dec, DEC(0.));
//     placeLimitOrder(exchange, agent2, bookId, OrderDirection::SELL, 4_dec, 301_dec, DEC(1.));
//     placeLimitOrder(exchange, agent3, bookId, OrderDirection::SELL, 2_dec, 301_dec, DEC(.5));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,301,6,303,2,307,8\n"
//               "bid,297,1,291,3\n"));
//     //-------------------------------------------------------------------------

    
//     const STPFlag stpFlag = STPFlag::DC;

//     //---------------------- Buy STP | Normal
//     placeLimitOrder(exchange, agent2, bookId, postOnly, timeInForce, std::nullopt, stpFlag, OrderDirection::BUY, 2_dec, 301_dec, DEC(0.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,301,4,303,2,307,8\n"
//               "bid,297,1,291,3\n"));
//     //-------------------------------------------------------------------------


//     //---------------------- Buy STP | Margin
//     placeLimitOrder(exchange, agent2, bookId, postOnly, timeInForce, std::nullopt, stpFlag, OrderDirection::BUY, 2_dec, 301_dec, DEC(1.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,303,2,307,8\n"
//               "bid,297,1,291,3\n"));
//     //-------------------------------------------------------------------------

// }

// //-------------------------------------------------------------------------

// TEST_F(SelfTradePreventionTest, LimitOrderSellDC)
// {
//     //---------------------- No prevention trades
//     placeLimitOrder(exchange, agent1, bookId, OrderDirection::SELL, 5_dec, 299_dec, DEC(0.));
//     placeLimitOrder(exchange, agent2, bookId, OrderDirection::BUY, 4_dec, 299_dec, DEC(1.));
//     placeLimitOrder(exchange, agent3, bookId, OrderDirection::BUY, 2_dec, 299_dec, DEC(.5));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,303,2,307,8\n"
//               "bid,299,6,297,1,291,3\n"));
//     //-------------------------------------------------------------------------

//     const STPFlag stpFlag = STPFlag::DC;
    
//     //---------------------- SELL STP | Normal
//     placeLimitOrder(exchange, agent2, bookId, postOnly, timeInForce, std::nullopt, stpFlag, OrderDirection::SELL, 2_dec, 299_dec, DEC(0.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,303,2,307,8\n"
//               "bid,299,4,297,1,291,3\n"));
//     //-------------------------------------------------------------------------


//     //---------------------- SELL STP | Margin
//     placeLimitOrder(exchange, agent2, bookId, postOnly, timeInForce, std::nullopt, stpFlag, OrderDirection::SELL, 2_dec, 299_dec, DEC(1.));

//     EXPECT_THAT(
//         normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
//         StrEq("ask,303,2,307,8\n"
//               "bid,297,1,291,3\n"));
//     //-------------------------------------------------------------------------

// }