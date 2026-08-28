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
using namespace taosim::literals;

using namespace testing;

using testing::StrEq;
using testing::Values;

namespace fs = std::filesystem;


//-------------------------------------------------------------------------

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
    MultiBookExchangeAgent* exchange, AgentId agentId, BookId bookId, Currency currency, STPFlag stpFlag, Args&&... args)
{
    const auto payload = MessagePayload::create<PlaceOrderMarketPayload>(
        std::forward<Args>(args)..., bookId, currency, std::nullopt, stpFlag);
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
    Args&&... args)
{
    const auto payload = MessagePayload::create<PlaceOrderLimitPayload>(
        std::forward<Args>(args)..., bookId, currency, std::nullopt, postOnly, timeInForce, std::nullopt, stpFlag);
    const auto orderResult = exchange->clearingManager().handleOrder(LimitOrderDesc{.agentId = agentId, .payload = payload});
    auto limitOrderPtr = exchange->books()[bookId]->placeLimitOrder(
        OrderClientContext{agentId},
        Timestamp{},
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
    MultiBookExchangeAgent* exchange, AgentId agentId, BookId bookId, Currency currency, Args&&... args)
{
    return placeLimitOrder(
        exchange,
        agentId,
        bookId,
        currency,
        false,
        taosim::TimeInForce::GTC,
        STPFlag::CO,
        std::forward<Args>(args)...);
}

template<typename... Args>
requires std::constructible_from<PlaceOrderMarketPayload, Args..., BookId>
std::pair<MarketOrder::Ptr, OrderErrorCode> placeMarketOrder(
    MultiBookExchangeAgent* exchange, AgentId agentId, BookId bookId, Currency currency, Args&&... args)
{
    return placeMarketOrder(
        exchange,
        agentId,
        bookId,
        currency,
        STPFlag::CO,
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

class QuoteOrderTest : public testing::TestWithParam<TestParams>
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
        placeLimitOrder(exchange, agent4, bookId, Currency::BASE, OrderDirection::BUY, 3_dec, 291_dec, DEC(0.));
        placeLimitOrder(exchange, agent4, bookId, Currency::BASE, OrderDirection::BUY, 1_dec, 297_dec, DEC(0.));
        placeLimitOrder(exchange, agent4, bookId, Currency::BASE, OrderDirection::SELL, 2_dec, 303_dec, DEC(0.));
        placeLimitOrder(exchange, agent4, bookId, Currency::BASE, OrderDirection::SELL, 8_dec, 307_dec, DEC(0.));
    }

    void fillOrderBook(std::vector<OrderParams> orders)
    {
        for (const OrderParams order : orders){
            printOrderbook(book);
            placeLimitOrder(exchange, agent3, bookId, Currency::BASE, order.direction, 
                order.volume, order.price, order.leverage);    
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
        simulation->setDebug(false);///####
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        book = exchange->books()[bookId];
    
        exchange->accounts().registerLocal("agent1");
        exchange->accounts().registerLocal("agent2");
        exchange->accounts().registerLocal("agent3");
        exchange->accounts().registerLocal("agent4");
    }

};


TEST_P(QuoteOrderTest, LimitOrders)
{
    const auto [initOrders, testOrder] = params;

    fill();
    fillOrderBook(initOrders);
    printOrderbook(book);
    placeLimitOrder(exchange, agent1, bookId, Currency::BASE, testOrder.direction,
        testOrder.volume, testOrder.price, testOrder.leverage);
    // fillOrderBook(initOrders);
    printOrderbook(book);
    const auto bookStateVolume = normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); }));

    cancelAll();
    EXPECT_THAT(
        normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
        StrEq("ask\n"
              "bid\n"));

    fill();
    fillOrderBook(initOrders);
    printOrderbook(book);
    placeLimitOrder(exchange, agent1, bookId, Currency::QUOTE, testOrder.direction,
        testOrder.volume * testOrder.price, testOrder.price, testOrder.leverage);
    const auto bookStateQuote = normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); }));

    EXPECT_THAT(bookStateVolume, bookStateQuote);

    cancelAll();
    EXPECT_THAT(
        normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
        StrEq("ask\n"
              "bid\n"));
    
}


INSTANTIATE_TEST_SUITE_P(
    QuoteVsVolumeLimitOrders,
    QuoteOrderTest,
    Values(
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.0), .volume = DEC(6.2), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(299.5), .volume = DEC(4.2), .leverage = DEC(0.)}},
            .testOrder = OrderParams{.direction = OrderDirection::SELL, .price = DEC(299.5), .volume = DEC(1.2), .leverage = DEC(0.)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.0), .volume = DEC(6.2), .leverage = DEC(0.3)},
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(299.), .volume = DEC(3.5), .leverage = DEC(0.)}},
            .testOrder = OrderParams{.direction = OrderDirection::SELL, .price = DEC(299.0), .volume = DEC(1.2), .leverage = DEC(0.49)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.0), .volume = DEC(6.2), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(299.1), .volume = DEC(3.5), .leverage = DEC(0.)}},
            .testOrder = OrderParams{.direction = OrderDirection::BUY, .price = DEC(301.), .volume = DEC(1.2), .leverage = DEC(0.)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.0), .volume = DEC(10.2), .leverage = DEC(0.3)},
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(299.1), .volume = DEC(4.2), .leverage = DEC(0.)}},
            .testOrder = OrderParams{.direction = OrderDirection::BUY, .price = DEC(301.0), .volume = DEC(1.2), .leverage = DEC(0.5)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(299.95), .volume = DEC(44.54), .leverage = DEC(0.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::SELL, .price = DEC(299.95), .volume = DEC(22.27), .leverage = DEC(0.)}
        }
        )
    );

//-------------------------------------------------------------------------


TEST_P(QuoteOrderTest, MarketOrders)
{
    const auto [initOrders, testOrder] = params;

    fill();
    fillOrderBook(initOrders);
    printOrderbook(book);
    placeMarketOrder(exchange, agent1, bookId, Currency::BASE, testOrder.direction,
        testOrder.volume, testOrder.leverage);
    const auto bookStateVolume = normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); }));

    fmt::println("{}",bookStateVolume);

    cancelAll();
    EXPECT_THAT(
        normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
        StrEq("ask\n"
              "bid\n"));

    fill();
    fillOrderBook(initOrders);
    printOrderbook(book);
    placeMarketOrder(exchange, agent1, bookId, Currency::QUOTE, testOrder.direction,
        testOrder.volume * testOrder.price, testOrder.leverage);
    const auto bookStateQuote = normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); }));

    EXPECT_THAT(bookStateVolume, bookStateQuote);

    cancelAll();
    EXPECT_THAT(
        normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
        StrEq("ask\n"
              "bid\n"));
    
}

//-------------------------------------------------------------------------


class ResidualTest : public testing::TestWithParam<TestParams>
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
        placeLimitOrder(exchange, agent4, bookId, Currency::BASE, OrderDirection::BUY, 3_dec, 291_dec, DEC(0.1));
        placeLimitOrder(exchange, agent4, bookId, Currency::BASE, OrderDirection::BUY, 1_dec, 297_dec, DEC(0.1));
        placeLimitOrder(exchange, agent4, bookId, Currency::BASE, OrderDirection::SELL, 2_dec, 303_dec, DEC(0.1));
        placeLimitOrder(exchange, agent4, bookId, Currency::BASE, OrderDirection::SELL, 8_dec, 307_dec, DEC(0.1));
    }

    void fillOrderBook(std::vector<OrderParams> orders)
    {
        for (const OrderParams order : orders){
            printOrderbook(book);
            placeLimitOrder(exchange, agent3, bookId, Currency::BASE, order.direction, 
                order.volume, order.price, order.leverage);    
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
        simulation->setDebug(false);///####
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        book = exchange->books()[bookId];
    
        exchange->accounts().registerLocal("agent1");
        exchange->accounts().registerLocal("agent2");
        exchange->accounts().registerLocal("agent3");
        exchange->accounts().registerLocal("agent4");
    }

};


TEST_P(ResidualTest, LimitOrders)
{
    const auto [initOrders, testOrder] = params;

    fill();
    placeLimitOrder(exchange, agent1, bookId, Currency::BASE, testOrder.direction,
        testOrder.volume, testOrder.price, testOrder.leverage);
    printOrderbook(book);
    fillOrderBook(initOrders);
    printOrderbook(book);
    // const auto bookStateVolume = normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); }));

    // // fmt::println("{}",bookStateVolume);

    // cancelAll();
    // EXPECT_THAT(
    //     normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
    //     StrEq("ask\n"
    //           "bid\n"));

    // fill();
    // fillOrderBook(initOrders);
    // printOrderbook(book);
    // placeLimitOrder(exchange, agent1, bookId, Currency::QUOTE, testOrder.direction,
    //     testOrder.volume * testOrder.price, testOrder.price, testOrder.leverage);
    // const auto bookStateQuote = normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); }));

    // cancelAll();
    // EXPECT_THAT(
    //     normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
    //     StrEq("ask\n"
    //           "bid\n"));
    
}


INSTANTIATE_TEST_SUITE_P(
    QuoteVsVolumeLimitOrders,
    ResidualTest,
    Values(
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.028), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.048), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.048), .leverage = DEC(0.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.0919), .leverage = DEC(0.35)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.028), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.048), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.048), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.01), .leverage = DEC(0.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.0919), .leverage = DEC(0.35)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(4.8273), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.3), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.0253), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.0594), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.0594), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.0594), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.01), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.01), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.01), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.01), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.01), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.01), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.01), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.01), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.01), .leverage = DEC(0.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(3.), .leverage = DEC(0.8)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.2142), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.7479), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.92), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.0079), .leverage = DEC(0.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(1.4), .leverage = DEC(0.35)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.2142), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.7479), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.92), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.0079), .leverage = DEC(0.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(1.89), .leverage = DEC(0.0)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.1014), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.11), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.00429), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.00013), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.0001), .leverage = DEC(0.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.1308), .leverage = DEC(0.65)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.1014), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.11), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.00429), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.00013), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.0001), .leverage = DEC(0.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.21582), .leverage = DEC(0.0)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.43), .volume = DEC(3.149396), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.43), .volume = DEC(0.3), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.43), .volume = DEC(0.12), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.43), .volume = DEC(0.048), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.43), .volume = DEC(0.01), .leverage = DEC(0.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.65), .volume = DEC(2.8), .leverage = DEC(0.3)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.43), .volume = DEC(3.149396), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.43), .volume = DEC(0.3), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.43), .volume = DEC(0.12), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.43), .volume = DEC(0.048), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.43), .volume = DEC(0.01), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.43), .volume = DEC(0.01), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.43), .volume = DEC(0.01), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.43), .volume = DEC(0.01), .leverage = DEC(0.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.65), .volume = DEC(2.8), .leverage = DEC(0.3)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.43), .volume = DEC(3.149396), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.43), .volume = DEC(0.3), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.43), .volume = DEC(0.12), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.43), .volume = DEC(0.048), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.43), .volume = DEC(0.01), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.2), .leverage = DEC(0.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.65), .volume = DEC(2.8), .leverage = DEC(0.3)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.85), .volume = DEC(0.069), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.85), .volume = DEC(2.0), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.85), .volume = DEC(1.57099999), .leverage = DEC(0.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.65), .volume = DEC(2.8), .leverage = DEC(0.3)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.85), .volume = DEC(0.069), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.85), .volume = DEC(2.0), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.85), .volume = DEC(1.57099999), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.85), .volume = DEC(0.01), .leverage = DEC(0.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.65), .volume = DEC(2.8), .leverage = DEC(0.3)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.85), .volume = DEC(0.069), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.85), .volume = DEC(2.0), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.85), .volume = DEC(1.8), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.85), .volume = DEC(0.2), .leverage = DEC(0.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.65), .volume = DEC(2.8), .leverage = DEC(0.3)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.2142), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.7479), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.92), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.0079), .leverage = DEC(0.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(1.4), .leverage = DEC(0.35)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.2142), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.7479), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.92), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.0079), .leverage = DEC(0.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(1.89), .leverage = DEC(0.0)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.1014), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.11), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.00429), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.00013), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.0001), .leverage = DEC(0.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.1308), .leverage = DEC(0.65)}
        },
        TestParams{
            .initOrders = {
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.1014), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.11), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.00429), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.00013), .leverage = DEC(0.)},
                OrderParams{.direction = OrderDirection::BUY, .price = DEC(302.71), .volume = DEC(0.0001), .leverage = DEC(0.)}
            },
            .testOrder = OrderParams{.direction = OrderDirection::SELL, .price = DEC(301.53), .volume = DEC(0.21582), .leverage = DEC(0.0)}
        }
        )
    );

//-------------------------------------------------------------------------


TEST_P(ResidualTest, MarketOrders)
{
    const auto [initOrders, testOrder] = params;

    fill();
    fillOrderBook(initOrders);
    printOrderbook(book);
    placeMarketOrder(exchange, agent1, bookId, Currency::BASE, testOrder.direction,
        testOrder.volume, testOrder.leverage);
    printOrderbook(book);
    // const auto bookStateVolume = normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); }));

    // cancelAll();
    // EXPECT_THAT(
    //     normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
    //     StrEq("ask\n"
    //           "bid\n"));

    // fill();
    // fillOrderBook(initOrders);
    // printOrderbook(book);
    // placeMarketOrder(exchange, agent1, bookId, Currency::QUOTE, testOrder.direction,
    //     testOrder.volume * testOrder.price, testOrder.leverage);
    // const auto bookStateQuote = normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); }));

    // cancelAll();
    // EXPECT_THAT(
    //     normalizeOutput(taosim::util::captureOutput([&] { book->printCSV(); })),
    //     StrEq("ask\n"
    //           "bid\n"));
    
}

//-------------------------------------------------------------------------