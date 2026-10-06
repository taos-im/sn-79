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
    fmt::println("#################\n\n{}\n\n################", orderbookState);
}

void printBalances(const taosim::accounting::Balances& balances, const AgentId agentId){
    std::string baseString = normalizeOutput(fmt::format("{}", balances.base));
    std::string quoteString = normalizeOutput(fmt::format("{}", *balances.quote));
    fmt::println("Agent {} => \tBase: {} \n\t\tQuote: {}", agentId, baseString, quoteString);
    for (auto it = balances.m_loans.begin(); it != balances.m_loans.end(); it++){
        if (it == balances.m_loans.begin())
            fmt::println("-------------- baseLoan: {}  |  quoteLoan: {} --------------", balances.m_baseLoan, balances.m_quoteLoan);

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

bool closePositionReq(MultiBookExchangeAgent* exchange, 
    AgentId agentId, BookId bookId, OrderID orderId, std::optional<decimal_t> volume)
{
    const bool close = exchange->clearingManager().handleClosePosition(
        ClosePositionDesc{
            .bookId = bookId,
            .agentId = agentId,
            .orderId = orderId,
            .volumeToClose = volume
        }
    );

    exchange->simulation()->step();

    return close;
}

//-------------------------------------------------------------------------

template<typename... Args>
requires std::constructible_from<PlaceOrderMarketPayload, Args..., BookId>
std::pair<MarketOrder::Ptr, OrderErrorCode> placeMarketOrder(
    MultiBookExchangeAgent* exchange, AgentId agentId, BookId bookId, Currency currency, STPFlag stpFlag, SettleFlag settleFlag, Args&&... args)
{
    const auto payload = MessagePayload::create<PlaceOrderMarketPayload>(
        std::forward<Args>(args)..., bookId, currency, std::nullopt, stpFlag, settleFlag);
    const auto orderResult = exchange->clearingManager().handleOrder(MarketOrderDesc{.agentId = agentId, .payload = payload});
    auto marketOrderPtr = exchange->books()[bookId]->placeMarketOrder(
        OrderClientContext{agentId},
        Timestamp{},
        orderResult.orderSize,
        payload->direction,
        payload->leverage,
        payload->stpFlag,
        payload->settleFlag);
    return {marketOrderPtr, orderResult.ec};
}


template<typename... Args>
requires std::constructible_from<PlaceOrderLimitPayload, Args..., BookId>
std::pair<LimitOrder::Ptr, OrderErrorCode> placeLimitOrder(
    MultiBookExchangeAgent* exchange,
    AgentId agentId,
    BookId bookId,
    Currency currency,
    bool postOnly,
    taosim::TimeInForce timeInForce,
    STPFlag stpFlag,
    SettleFlag settleFlag,
    Args&&... args)
{
    const auto payload = MessagePayload::create<PlaceOrderLimitPayload>(
        std::forward<Args>(args)..., bookId, currency, std::nullopt, postOnly, timeInForce, std::nullopt, stpFlag, settleFlag);
    const auto orderResult = exchange->clearingManager().handleOrder(LimitOrderDesc{.agentId = agentId, .payload = payload});
    auto limitOrderPtr = exchange->books()[bookId]->placeLimitOrder(
        OrderClientContext{agentId},
        Timestamp{},
        orderResult.orderSize,
        payload->direction,
        payload->price,
        payload->leverage,
        payload->stpFlag,
        payload->settleFlag);
    return {limitOrderPtr, orderResult.ec};
}

template<typename... Args>
requires std::constructible_from<PlaceOrderLimitPayload, Args..., BookId>
std::pair<LimitOrder::Ptr, OrderErrorCode> placeLimitOrder(
    MultiBookExchangeAgent* exchange, AgentId agentId, BookId bookId, Currency currency, SettleFlag settleFlag, Args&&... args)
{
    return placeLimitOrder(
        exchange,
        agentId,
        bookId,
        currency,
        false,
        taosim::TimeInForce::GTC,
        STPFlag::CO,
        settleFlag,
        std::forward<Args>(args)...);
}

template<typename... Args>
requires std::constructible_from<PlaceOrderMarketPayload, Args..., BookId>
std::pair<MarketOrder::Ptr, OrderErrorCode> placeMarketOrder(
    MultiBookExchangeAgent* exchange, AgentId agentId, BookId bookId, Currency currency, SettleFlag settleFlag, Args&&... args)
{
    return placeMarketOrder(
        exchange,
        agentId,
        bookId,
        currency,
        STPFlag::CO,
        settleFlag,
        std::forward<Args>(args)...);
}



//-------------------------------------------------------------------------

struct OrderParams
{
    OrderDirection direction;
    decimal_t price;
    decimal_t volume;
    decimal_t leverage;
};

struct TestParams
{
    std::vector<OrderParams> initOrders;
    OrderParams testOrder;
};

void PrintTo(const TestParams& params, std::ostream* os)
{
    *os << fmt::format(
        "{{Order {}x{}@{} in {} direction}}",
        params.testOrder.leverage + 1_dec,
        params.testOrder.volume,
        params.testOrder.price,
        params.testOrder.direction == OrderDirection::BUY ? "BUY" : "SELL");
}

class LoanSettlementTest : public testing::TestWithParam<TestParams>
{

public:

    TestParams params;
    const AgentId agent1 = -1, agent2 = -2, agent3 = -3, agent4 = -4;
    const BookId bookId{};

    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange;
    Book::Ptr book;

    void fill(){
        placeLimitOrder(exchange, agent4, bookId, Currency::BASE, SettleType::NONE, OrderDirection::BUY, 3_dec, 291_dec, DEC(1.));
        placeLimitOrder(exchange, agent4, bookId, Currency::BASE, SettleType::NONE, OrderDirection::BUY, 1_dec, 297_dec, DEC(1.));
        placeLimitOrder(exchange, agent4, bookId, Currency::BASE, SettleType::NONE, OrderDirection::SELL, 2_dec, 303_dec, DEC(0.));
        placeLimitOrder(exchange, agent4, bookId, Currency::BASE, SettleType::NONE, OrderDirection::SELL, 8_dec, 307_dec, DEC(0.));
    }

    void fillOrderBook(std::vector<OrderParams> orders, AgentId agentId)
    {
        for (const OrderParams order : orders){
            printOrderbook(book);
            // placeLimitOrder(exchange, agentId, bookId, Currency::BASE, SettleType::NONE,
            //     order.direction, order.volume, order.price, order.leverage);    
            placeMarketOrder(exchange, agentId, bookId, Currency::BASE, SettleType::NONE,
                order.direction, order.volume, order.leverage);    
        }
    }

    void cancelAll()
    {
        for (AgentId agentId = -1; agentId > -5; agentId--){
            const auto orders = exchange->accounts()[agentId].activeOrders()[bookId];
            for (Order::Ptr order : orders) {
                if (auto limitOrder = std::dynamic_pointer_cast<LimitOrder>(order)) {
                    book->cancelOrder(limitOrder->id());
                }
            }
        }
    }


protected:
    void SetUp() override
    {
        params = GetParam();
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
    }

};

//-------------------------------------------------------------------------

}  // namespace

//-------------------------------------------------------------------------



TEST_P(LoanSettlementTest, NoneFlag)
{
    const auto [initOrders, testOrder] = params;

    fill();
    fillOrderBook(initOrders, agent1); 
    
    printOrderbook(book);

    for (AgentId agId = -1; agId > -5; agId--){
        printBalances(exchange->accounts()[agId][bookId], agId);
    }

    // placeLimitOrder(exchange, agent1, bookId, Currency::BASE, SettleType::NONE,
    //     testOrder.direction, testOrder.volume, testOrder.price, testOrder.leverage);
    placeMarketOrder(exchange, agent1, bookId, Currency::BASE, SettleType::NONE,
        testOrder.direction, testOrder.volume, testOrder.leverage);
    
    printOrderbook(book);
    for (AgentId agId = -1; agId > -5; agId--){
        printBalances(exchange->accounts()[agId][bookId], agId);
    }
    
}

INSTANTIATE_TEST_SUITE_P(
    QuoteVsVolumeLimitOrders,
    LoanSettlementTest,
    Values(
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(301.0), .volume = DEC(3.2), .leverage = DEC(0.5)},
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.0), .volume = DEC(2.), .leverage = DEC(1.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::SELL, .price = DEC(299.5), .volume = DEC(20.2), .leverage = DEC(0.)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.0), .volume = DEC(4.2), .leverage = DEC(0.3)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(302.0), .volume = DEC(2.), .leverage = DEC(0.7)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::BUY, .price = DEC(299.0), .volume = DEC(10.2), .leverage = DEC(0.)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.0), .volume = DEC(6.2), .leverage = DEC(1.)}
                // ,
                // OrderParams{.direction = OrderDirection::BUY, .price = DEC(299.1), .volume = DEC(3.5), .leverage = DEC(1.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::BUY, .price = DEC(301.), .volume = DEC(10.2), .leverage = DEC(0.)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(301.0), .volume = DEC(10.2), .leverage = DEC(0.3)}
                // ,
                // OrderParams{.direction = OrderDirection::BUY, .price = DEC(299.1), .volume = DEC(3.5), .leverage = DEC(0.2)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.0), .volume = DEC(10.2), .leverage = DEC(0.)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(299.95), .volume = DEC(44.5583), .leverage = DEC(1.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::SELL, .price = DEC(299.95), .volume = DEC(22.27915), .leverage = DEC(0.)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(299.71), .volume = DEC(0.01), .leverage = DEC(1.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::SELL, .price = DEC(299.68), .volume = DEC(0.02), .leverage = DEC(0.)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(299.71), .volume = DEC(0.01), .leverage = DEC(1.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::SELL, .price = DEC(299.68), .volume = DEC(0.03), .leverage = DEC(0.)}
        }
        )
    );

//-------------------------------------------------------------------------


TEST_P(LoanSettlementTest, FIFOFlag)
{
    const auto [initOrders, testOrder] = params;

    fill();
    fillOrderBook(initOrders, agent1); 
    
    printOrderbook(book);

    for (AgentId agId = -1; agId > -5; agId--){
        printBalances(exchange->accounts()[agId][bookId], agId);
    }

    // placeLimitOrder(exchange, agent1, bookId, Currency::BASE, SettleType::NONE,
    //     testOrder.direction, testOrder.volume, testOrder.price, testOrder.leverage);
    placeMarketOrder(exchange, agent1, bookId, Currency::BASE, SettleType::FIFO,
        testOrder.direction, testOrder.volume, testOrder.leverage);
    
    printOrderbook(book);
    for (AgentId agId = -1; agId > -5; agId--){
        printBalances(exchange->accounts()[agId][bookId], agId);
    }
    
}

//-------------------------------------------------------------------------


TEST_P(LoanSettlementTest, OrderIdFlag)
{
    const auto [initOrders, testOrder] = params;

    fill();
    fillOrderBook(initOrders, agent1); 
    
    printOrderbook(book);

    for (AgentId agId = -1; agId > -5; agId--){
        printBalances(exchange->accounts()[agId][bookId], agId);
    }

    OrderID orderId = 4;
    // placeLimitOrder(exchange, agent1, bookId, Currency::BASE, SettleType::NONE,
    //     testOrder.direction, testOrder.volume, testOrder.price, testOrder.leverage);
    placeMarketOrder(exchange, agent1, bookId, Currency::BASE, orderId,
        testOrder.direction, testOrder.volume, testOrder.leverage);
    
    for (AgentId agId = -1; agId > -5; agId--){
        printBalances(exchange->accounts()[agId][bookId], agId);
    }
    
}


//-------------------------------------------------------------------------


TEST_P(LoanSettlementTest, ClosePosition)
{
    const auto [initOrders, testOrder] = params;

    fill();
    fillOrderBook(initOrders, agent1); 
    
    printOrderbook(book);

    for (AgentId agId = -1; agId > -5; agId--){
        printBalances(exchange->accounts()[agId][bookId], agId);
    }

    OrderID orderId = 4;

    fmt::println("CLOSE POSITION .. {}  b#:{}  o#:{}  v:{}  l:{}  {}", agent1, bookId, orderId, testOrder.volume, testOrder.leverage,
        testOrder.volume * util::dec1p(testOrder.leverage) * testOrder.price);

    const bool done = closePositionReq(exchange, agent1, bookId, orderId, std::nullopt);
    // const bool done = closePositionReq(exchange, agent1, bookId, orderId, testOrder.volume * util::dec1p(testOrder.leverage) * testOrder.price);

    if (done){
        fmt::println("CLOSE POSITION SUCCEED");
    } else {
        fmt::println("CLOSE POSITION FAILED");
    }

    
    for (AgentId agId = -1; agId > -5; agId--){
        printBalances(exchange->accounts()[agId][bookId], agId);
    }
    
}


//-------------------------------------------------------------------------