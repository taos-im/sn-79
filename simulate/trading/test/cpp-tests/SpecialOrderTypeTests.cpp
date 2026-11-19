/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

#include "formatting.hpp"
#include "MultiBookExchangeAgent.hpp"
#include "Order.hpp"
#include "Simulation.hpp"
#include <taosim/net/server.hpp>
#include "util.hpp"
#include <taosim/decimal/decimal.hpp>
#include <taosim/matching/ClearingManager.hpp>
#include <taosim/matching/FeePolicy.hpp>
#include <taosim/matching/FeePolicyWrapper.hpp>
#include <taosim/agent/DistributedProxyAgent.hpp>
#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/message/MultiBookMessagePayloads.hpp>
#include <taosim/message/PayloadFactory.hpp>

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <boost/signals2.hpp>
#include <filesystem>
#include <memory>
#include <optional>

//-------------------------------------------------------------------------

using namespace taosim;
using namespace taosim::accounting;
using namespace taosim::book;
using namespace taosim::matching;
using namespace taosim::literals;

using namespace testing;

namespace fs = std::filesystem;

//-------------------------------------------------------------------------

namespace
{

const auto kTestDataPath = fs::path{__FILE__}.parent_path() / "data";

//-------------------------------------------------------------------------
// Helpers
//-------------------------------------------------------------------------

void setupLimitOrder(
    MultiBookExchangeAgent* exchange,
    AgentId agentId,
    BookId bookId,
    OrderDirection direction,
    decimal_t volume,
    decimal_t price,
    decimal_t leverage)
{
    const auto payload = MessagePayload::create<PlaceOrderLimitPayload>(
        direction, volume, price, leverage, bookId, Currency::BASE,
        std::nullopt, false, TimeInForce::GTC, std::nullopt, STPFlag::CO);
    const auto orderResult = exchange->clearingManager().handleOrder(
        LimitOrderDesc{.agentId = agentId, .payload = payload});
    [[maybe_unused]] auto _ = exchange->books()[bookId]->placeLimitOrder(
        OrderClientContext{agentId},
        Timestamp{},
        orderResult.orderSize,
        payload->direction,
        payload->price,
        payload->leverage,
        payload->stpFlag);
}

OrderErrorCode submitTestOrder(
    MultiBookExchangeAgent* exchange,
    AgentId agentId,
    BookId bookId,
    OrderDirection direction,
    decimal_t volume,
    decimal_t price,
    decimal_t leverage,
    bool postOnly,
    TimeInForce timeInForce,
    STPFlag stpFlag,
    Currency currency = Currency::BASE)
{
    const auto payload = MessagePayload::create<PlaceOrderLimitPayload>(
        direction, volume, price, leverage, bookId, currency,
        std::nullopt, postOnly, timeInForce, std::nullopt, stpFlag);
    return exchange->clearingManager().handleOrder(
        LimitOrderDesc{.agentId = agentId, .payload = payload}).ec;
}

//-------------------------------------------------------------------------
// Common simulation setup
//
// Book state after setup:
//   asks: 2@303, 8@307
//   bids: 1@297, 3@291
//-------------------------------------------------------------------------

struct SimSetup
{
    static constexpr AgentId testAgent = -1;
    static constexpr AgentId bookAgent = -2;
    static constexpr BookId bookId{};

    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange{};
    Book::Ptr book;

    void setUp(double maxPriceBand = 0.0)
    {
        nodes = taosim::util::parseSimulationFile(kTestDataPath / "MultiAgentFees.xml");
        if (maxPriceBand > 0.0) {
            nodes.simulation.child("Agents").child("MultiBookExchangeAgent").append_attribute("maxPriceBand") =
                maxPriceBand;
        }
        simulation = std::make_unique<Simulation>();
        simulation->setDebug(false);
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        book = exchange->books()[bookId];

        exchange->accounts().registerLocal("testAgent");
        exchange->accounts().registerLocal("bookAgent");

        setupLimitOrder(exchange, bookAgent, bookId, OrderDirection::BUY, 3_dec, 291_dec, 0_dec);
        setupLimitOrder(exchange, bookAgent, bookId, OrderDirection::BUY, 1_dec, 297_dec, 0_dec);
        setupLimitOrder(exchange, bookAgent, bookId, OrderDirection::SELL, 2_dec, 303_dec, 0_dec);
        setupLimitOrder(exchange, bookAgent, bookId, OrderDirection::SELL, 8_dec, 307_dec, 0_dec);
    }
};

//-------------------------------------------------------------------------
// Ghost-order simulation setup
//
// Book state after setup (ghost levels from filled orders):
//   asks: ghost@300, 2@303, 8@307
//   bids: 3@291, 1@297, ghost@298
//
// bestAsk() = 303,  sellQueue().front().price() = 300 (ghost)
// bestBid() = 297,  buyQueue().back().price()   = 298 (ghost)
//-------------------------------------------------------------------------

struct GhostSimSetup
{
    static constexpr AgentId testAgent = -1;
    static constexpr AgentId bookAgent = -2;
    static constexpr AgentId fillerAgent = -3;
    static constexpr BookId bookId{};

    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange{};
    Book::Ptr book;

    void setUp()
    {
        nodes = taosim::util::parseSimulationFile(kTestDataPath / "MultiAgentFees.xml");
        simulation = std::make_unique<Simulation>();
        simulation->setDebug(false);
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        book = exchange->books()[bookId];

        exchange->accounts().registerLocal("testAgent");
        exchange->accounts().registerLocal("bookAgent");
        exchange->accounts().registerLocal("fillerAgent");

        // Place orders that will become ghosts
        setupLimitOrder(exchange, bookAgent, bookId, OrderDirection::SELL, 1_dec, 300_dec, 0_dec);
        setupLimitOrder(exchange, bookAgent, bookId, OrderDirection::BUY, 1_dec, 298_dec, 0_dec);

        // Place normal (surviving) orders
        setupLimitOrder(exchange, bookAgent, bookId, OrderDirection::BUY, 3_dec, 291_dec, 0_dec);
        setupLimitOrder(exchange, bookAgent, bookId, OrderDirection::BUY, 1_dec, 297_dec, 0_dec);
        setupLimitOrder(exchange, bookAgent, bookId, OrderDirection::SELL, 2_dec, 303_dec, 0_dec);
        setupLimitOrder(exchange, bookAgent, bookId, OrderDirection::SELL, 8_dec, 307_dec, 0_dec);

        // Fill the ghost-to-be orders with crossing orders
        setupLimitOrder(exchange, fillerAgent, bookId, OrderDirection::BUY, 1_dec, 300_dec, 0_dec);
        setupLimitOrder(exchange, fillerAgent, bookId, OrderDirection::SELL, 1_dec, 298_dec, 0_dec);
    }
};

//-------------------------------------------------------------------------
// IOC (Immediate or Cancel) Tests
//-------------------------------------------------------------------------

struct IOCTestParams
{
    OrderDirection direction{OrderDirection::BUY};
    decimal_t volume{};
    decimal_t price{};
    bool postOnly{false};
    OrderErrorCode expectedEc{OrderErrorCode::VALID};
};

void PrintTo(const IOCTestParams& params, std::ostream* os)
{
    *os << fmt::format(
        "{{direction = {}, volume = {}, price = {}, postOnly = {}, expectedEc = {}}}",
        params.direction == OrderDirection::BUY ? "BUY" : "SELL",
        params.volume,
        params.price,
        params.postOnly,
        static_cast<uint32_t>(params.expectedEc));
}

struct IOCTest : TestWithParam<IOCTestParams>
{
    void SetUp() override
    {
        sim.setUp();
        params = GetParam();
    }

    SimSetup sim;
    IOCTestParams params;
};

TEST_P(IOCTest, WorksCorrectly)
{
    const auto ec = submitTestOrder(
        sim.exchange, SimSetup::testAgent, SimSetup::bookId,
        params.direction, params.volume, params.price, 0_dec,
        params.postOnly, TimeInForce::IOC, STPFlag::NONE);
    EXPECT_EQ(ec, params.expectedEc);
}

INSTANTIATE_TEST_SUITE_P(
    SpecialOrderTypes,
    IOCTest,
    Values(
        // BUY IOC at 303 — 2 available at 303, fills 1
        IOCTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 303_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // BUY IOC at 307 — crosses both ask levels, fills 5
        IOCTestParams{
            .direction = OrderDirection::BUY,
            .volume = 5_dec,
            .price = 307_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // BUY IOC at 290 — no asks at or below 290
        IOCTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 290_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // SELL IOC at 297 — 1 available at 297, fills 1
        IOCTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 297_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // SELL IOC at 310 — no bids at or above 310
        IOCTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 310_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // IOC + postOnly — mutually exclusive, always rejected
        IOCTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 303_dec,
            .postOnly = true,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        }));

//-------------------------------------------------------------------------
// IOC with ghost orders — nonZeroLevelsView must skip zero-volume levels
//-------------------------------------------------------------------------

struct GhostIOCTest : TestWithParam<IOCTestParams>
{
    void SetUp() override
    {
        sim.setUp();
        params = GetParam();
    }

    GhostSimSetup sim;
    IOCTestParams params;
};

TEST_P(GhostIOCTest, WorksCorrectly)
{
    const auto ec = submitTestOrder(
        sim.exchange, GhostSimSetup::testAgent, GhostSimSetup::bookId,
        params.direction, params.volume, params.price, 0_dec,
        params.postOnly, TimeInForce::IOC, STPFlag::NONE);
    EXPECT_EQ(ec, params.expectedEc);
}

INSTANTIATE_TEST_SUITE_P(
    GhostOrders,
    GhostIOCTest,
    Values(
        // BUY IOC at 301 — ghost@300 skipped, no real asks at ≤301
        IOCTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 301_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // BUY IOC at 303 — ghost@300 skipped, fills from 2@303
        IOCTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 303_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // SELL IOC at 298 — ghost@298 skipped, no real bids at ≥298
        IOCTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 298_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // SELL IOC at 297 — ghost@298 skipped, fills from 1@297
        IOCTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 297_dec,
            .expectedEc = OrderErrorCode::VALID
        }));

//-------------------------------------------------------------------------
// FOK (Fill or Kill) Tests
//-------------------------------------------------------------------------

struct FOKTestParams
{
    OrderDirection direction{OrderDirection::BUY};
    decimal_t volume{};
    decimal_t price{};
    bool postOnly{false};
    OrderErrorCode expectedEc{OrderErrorCode::VALID};
};

void PrintTo(const FOKTestParams& params, std::ostream* os)
{
    *os << fmt::format(
        "{{direction = {}, volume = {}, price = {}, postOnly = {}, expectedEc = {}}}",
        params.direction == OrderDirection::BUY ? "BUY" : "SELL",
        params.volume,
        params.price,
        params.postOnly,
        static_cast<uint32_t>(params.expectedEc));
}

struct FOKTest : TestWithParam<FOKTestParams>
{
    void SetUp() override
    {
        sim.setUp();
        params = GetParam();
    }

    SimSetup sim;
    FOKTestParams params;
};

TEST_P(FOKTest, WorksCorrectly)
{
    const auto ec = submitTestOrder(
        sim.exchange, SimSetup::testAgent, SimSetup::bookId,
        params.direction, params.volume, params.price, 0_dec,
        params.postOnly, TimeInForce::FOK, STPFlag::CO);
    EXPECT_EQ(ec, params.expectedEc);
}

INSTANTIATE_TEST_SUITE_P(
    SpecialOrderTypes,
    FOKTest,
    Values(
        // BUY FOK 2@303 — exactly 2 available at 303
        FOKTestParams{
            .direction = OrderDirection::BUY,
            .volume = 2_dec,
            .price = 303_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // BUY FOK 5@303 — only 2 at 303, next level 307 > limit 303
        FOKTestParams{
            .direction = OrderDirection::BUY,
            .volume = 5_dec,
            .price = 303_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // BUY FOK 5@307 — 2@303 + 8@307 = 10, enough to fill 5
        FOKTestParams{
            .direction = OrderDirection::BUY,
            .volume = 5_dec,
            .price = 307_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // SELL FOK 4@291 — 1@297 + 3@291 = 4, exact fill
        FOKTestParams{
            .direction = OrderDirection::SELL,
            .volume = 4_dec,
            .price = 291_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // SELL FOK 3@297 — only 1@297, next level 291 < limit 297
        FOKTestParams{
            .direction = OrderDirection::SELL,
            .volume = 3_dec,
            .price = 297_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // FOK + postOnly — mutually exclusive, always rejected
        FOKTestParams{
            .direction = OrderDirection::BUY,
            .volume = 2_dec,
            .price = 307_dec,
            .postOnly = true,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        }));

//-------------------------------------------------------------------------
// FOK with ghost orders — nonZeroLevelsView must skip zero-volume levels
//-------------------------------------------------------------------------

struct GhostFOKTest : TestWithParam<FOKTestParams>
{
    void SetUp() override
    {
        sim.setUp();
        params = GetParam();
    }

    GhostSimSetup sim;
    FOKTestParams params;
};

TEST_P(GhostFOKTest, WorksCorrectly)
{
    const auto ec = submitTestOrder(
        sim.exchange, GhostSimSetup::testAgent, GhostSimSetup::bookId,
        params.direction, params.volume, params.price, 0_dec,
        params.postOnly, TimeInForce::FOK, STPFlag::CO);
    EXPECT_EQ(ec, params.expectedEc);
}

INSTANTIATE_TEST_SUITE_P(
    GhostOrders,
    GhostFOKTest,
    Values(
        // BUY FOK 2@303 — ghost@300 skipped, 2@303 fills
        FOKTestParams{
            .direction = OrderDirection::BUY,
            .volume = 2_dec,
            .price = 303_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // BUY FOK 5@303 — ghost@300 skipped, only 2@303, 307>303
        FOKTestParams{
            .direction = OrderDirection::BUY,
            .volume = 5_dec,
            .price = 303_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // SELL FOK 1@297 — ghost@298 skipped, 1@297 fills
        FOKTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 297_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // SELL FOK 3@297 — ghost@298 skipped, only 1@297, 291<297
        FOKTestParams{
            .direction = OrderDirection::SELL,
            .volume = 3_dec,
            .price = 297_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        }));

//-------------------------------------------------------------------------
// Post-Only Tests
//-------------------------------------------------------------------------

struct PostOnlyTestParams
{
    OrderDirection direction{OrderDirection::BUY};
    decimal_t volume{};
    decimal_t price{};
    TimeInForce timeInForce{TimeInForce::GTC};
    STPFlag stpFlag{STPFlag::NONE};
    OrderErrorCode expectedEc{OrderErrorCode::VALID};
};

void PrintTo(const PostOnlyTestParams& params, std::ostream* os)
{
    *os << fmt::format(
        "{{direction = {}, volume = {}, price = {}, timeInForce = {}, stpFlag = {}, expectedEc = {}}}",
        params.direction == OrderDirection::BUY ? "BUY" : "SELL",
        params.volume,
        params.price,
        static_cast<uint32_t>(params.timeInForce),
        static_cast<uint32_t>(params.stpFlag),
        static_cast<uint32_t>(params.expectedEc));
}

struct PostOnlyTest : TestWithParam<PostOnlyTestParams>
{
    void SetUp() override
    {
        sim.setUp();
        params = GetParam();
    }

    SimSetup sim;
    PostOnlyTestParams params;
};

TEST_P(PostOnlyTest, WorksCorrectly)
{
    const auto ec = submitTestOrder(
        sim.exchange, SimSetup::testAgent, SimSetup::bookId,
        params.direction, params.volume, params.price, 0_dec,
        true, params.timeInForce, params.stpFlag);
    EXPECT_EQ(ec, params.expectedEc);
}

INSTANTIATE_TEST_SUITE_P(
    SpecialOrderTypes,
    PostOnlyTest,
    Values(
        // BUY post-only at 302 — below best ask 303, does not cross
        PostOnlyTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 302_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // BUY post-only at 303 — equals best ask, would cross
        PostOnlyTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 303_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // BUY post-only at 310 — above best ask, would cross
        PostOnlyTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 310_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // SELL post-only at 298 — above best bid 297, does not cross
        PostOnlyTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 298_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // SELL post-only at 297 — equals best bid, would cross
        PostOnlyTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 297_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // SELL post-only at 290 — below best bid, would cross
        PostOnlyTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 290_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // post-only + IOC — mutually exclusive, rejected at timeInForce check
        PostOnlyTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 302_dec,
            .timeInForce = TimeInForce::IOC,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // post-only + FOK — mutually exclusive, rejected at timeInForce check
        PostOnlyTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 302_dec,
            .timeInForce = TimeInForce::FOK,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        }));

//-------------------------------------------------------------------------
// Post-Only with ghost orders
//
// Ghost levels: ghost@300 on sell side, ghost@298 on buy side.
// checkPostOnly with CO uses nonZeroLevelsView and skips ghosts correctly.
// checkPostOnly NONE/CN else-branch uses sellQueue().front() / buyQueue().back()
// directly which may include ghost levels.
//-------------------------------------------------------------------------

struct GhostPostOnlyTest : TestWithParam<PostOnlyTestParams>
{
    void SetUp() override
    {
        sim.setUp();
        params = GetParam();
    }

    GhostSimSetup sim;
    PostOnlyTestParams params;
};

TEST_P(GhostPostOnlyTest, WorksCorrectly)
{
    const auto ec = submitTestOrder(
        sim.exchange, GhostSimSetup::testAgent, GhostSimSetup::bookId,
        params.direction, params.volume, params.price, 0_dec,
        true, params.timeInForce, params.stpFlag);
    EXPECT_EQ(ec, params.expectedEc);
}

INSTANTIATE_TEST_SUITE_P(
    GhostOrders,
    GhostPostOnlyTest,
    Values(
        // BUY post-only at 301 (CO) — ghost@300 skipped via nonZeroLevelsView,
        // no real asks cross 301 (best real ask is 303)
        PostOnlyTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 301_dec,
            .stpFlag = STPFlag::CO,
            .expectedEc = OrderErrorCode::VALID
        },
        // BUY post-only at 303 (CO) — ghost@300 skipped, real ask at 303 crosses
        PostOnlyTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 303_dec,
            .stpFlag = STPFlag::CO,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // SELL post-only at 298 (CO) — ghost@298 skipped via nonZeroLevelsView,
        // no real bids cross 298 (best real bid is 297)
        PostOnlyTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 298_dec,
            .stpFlag = STPFlag::CO,
            .expectedEc = OrderErrorCode::VALID
        },
        // SELL post-only at 297 (CO) — ghost@298 skipped, real bid at 297 crosses
        PostOnlyTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 297_dec,
            .stpFlag = STPFlag::CO,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // BUY post-only at 301 (NONE) — ghost@300 present at sellQueue().front(),
        // real best ask is 303 so 301 should not cross
        PostOnlyTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 301_dec,
            .stpFlag = STPFlag::NONE,
            .expectedEc = OrderErrorCode::VALID
        },
        // SELL post-only at 298 (NONE) — ghost@298 present at buyQueue().back(),
        // real best bid is 297 so 298 should not cross
        PostOnlyTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 298_dec,
            .stpFlag = STPFlag::NONE,
            .expectedEc = OrderErrorCode::VALID
        }));

//-------------------------------------------------------------------------
// Minimum Order Size Tests
//
// Config: volumeDecimals=4 → minOrderSize = 10^(-4) = 0.0001
//-------------------------------------------------------------------------

struct MinOrderSizeTestParams
{
    OrderDirection direction{OrderDirection::BUY};
    decimal_t volume{};
    decimal_t price{};
    Currency currency{Currency::BASE};
    OrderErrorCode expectedEc{OrderErrorCode::VALID};
};

void PrintTo(const MinOrderSizeTestParams& params, std::ostream* os)
{
    *os << fmt::format(
        "{{direction = {}, volume = {}, price = {}, currency = {}, expectedEc = {}}}",
        params.direction == OrderDirection::BUY ? "BUY" : "SELL",
        params.volume,
        params.price,
        params.currency == Currency::BASE ? "BASE" : "QUOTE",
        static_cast<uint32_t>(params.expectedEc));
}

struct MinOrderSizeTest : TestWithParam<MinOrderSizeTestParams>
{
    void SetUp() override
    {
        sim.setUp();
        params = GetParam();
    }

    SimSetup sim;
    MinOrderSizeTestParams params;
};

TEST_P(MinOrderSizeTest, WorksCorrectly)
{
    const auto ec = submitTestOrder(
        sim.exchange, SimSetup::testAgent, SimSetup::bookId,
        params.direction, params.volume, params.price, 0_dec,
        false, TimeInForce::GTC, STPFlag::CO, params.currency);
    EXPECT_EQ(ec, params.expectedEc);
}

INSTANTIATE_TEST_SUITE_P(
    SpecialOrderTypes,
    MinOrderSizeTest,
    Values(
        // BASE vol=1 — well above minOrderSize 0.0001
        MinOrderSizeTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 302_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // BASE vol=0.0001 — exactly at minOrderSize boundary
        MinOrderSizeTestParams{
            .direction = OrderDirection::BUY,
            .volume = DEC(0.0001),
            .price = 302_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // SELL BASE vol=1 — above minOrderSize
        MinOrderSizeTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 298_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // QUOTE vol=0.01 at price 302 — base equivalent rounds to 0.0000 < 0.0001
        MinOrderSizeTestParams{
            .direction = OrderDirection::BUY,
            .volume = DEC(0.01),
            .price = 302_dec,
            .currency = Currency::QUOTE,
            .expectedEc = OrderErrorCode::MINIMUM_ORDER_SIZE_VIOLATION
        },
        // QUOTE vol=1 at price 302 — base equivalent ≈ 0.0033, above minOrderSize
        MinOrderSizeTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 302_dec,
            .currency = Currency::QUOTE,
            .expectedEc = OrderErrorCode::VALID
        }));

//-------------------------------------------------------------------------
// Time in force after matching
//
// The up-front check sizes an IOC or FOK to what the book holds when the order is validated, and
// that size funds its reservation. The book can hold less by the time the order matches: whatever
// matching leaves of an IOC or FOK is cancelled and its owner told, never rested. A GTC's leftover
// rests.
//-------------------------------------------------------------------------

struct TimeInForceAfterMatchingTest : Test
{
    struct Validated
    {
        PlaceOrderLimitPayload::Ptr payload;
        OrderResult result;
    };

    void SetUp() override
    {
        sim.setUp();
        proxy = sim.simulation->proxy();
        ASSERT_NE(proxy, nullptr);
        proxy->clearMessages();
        miner = sim.exchange->accounts().registerRemote();
    }

    // The exchange's check of a placement, which sizes the order and reserves for it.
    [[nodiscard]] Validated validate(
        OrderDirection direction, decimal_t volume, decimal_t price, TimeInForce timeInForce)
    {
        const auto payload = MessagePayload::create<PlaceOrderLimitPayload>(
            direction, volume, price, 0_dec, SimSetup::bookId, Currency::BASE,
            std::nullopt, false, timeInForce, std::nullopt, STPFlag::CO);
        const auto result = sim.exchange->clearingManager().handleOrder(
            LimitOrderDesc{.agentId = miner, .payload = payload});
        return {payload, result};
    }

    // The placement of a validated order, at the size the check gave it.
    [[nodiscard]] LimitOrder::Ptr place(const Validated& validated)
    {
        const auto& payload = validated.payload;
        return sim.book->placeLimitOrder(
            OrderClientContext{miner},
            Timestamp{},
            validated.result.orderSize,
            payload->direction,
            payload->price,
            payload->leverage,
            payload->stpFlag,
            payload->settleFlag,
            payload->postOnly,
            payload->timeInForce);
    }

    // Another agent's order at the touch loses `volume` between the check and the match.
    void shrinkTouch(OrderDirection side, decimal_t volume)
    {
        const auto level =
            side == OrderDirection::BUY ? sim.book->bestBuyLevel() : sim.book->bestSellLevel();
        ASSERT_TRUE(level.has_value());
        const auto order = level->get().getFirstActiveOrder();
        ASSERT_NE(order, nullptr);
        ASSERT_TRUE(sim.book->cancelOrder(order->id(), volume));
    }

    // The volume the owner was told was cancelled from the order, if it was told.
    [[nodiscard]] std::optional<decimal_t> toldCancelled(OrderID orderId) const
    {
        for (const auto& msg : proxy->messages()) {
            if (msg->type != "RESPONSE_DISTRIBUTED_CANCEL_ORDERS") continue;
            const auto pld = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
            if (pld == nullptr || pld->agentId != miner) continue;
            const auto sub = std::dynamic_pointer_cast<CancelOrdersResponsePayload>(pld->payload);
            if (sub == nullptr || sub->requestPayload == nullptr) continue;
            for (const auto& cancellation : sub->requestPayload->cancellations) {
                if (cancellation.id == orderId) return cancellation.volume.value_or(0_dec);
            }
        }
        return std::nullopt;
    }

    // The volume the L3 record carries as cancelled from the order, if it carries a cancellation: the tape and
    // the venue's view of the book read the order's end from it.
    [[nodiscard]] std::optional<decimal_t> recordedCancelled(OrderID orderId) const
    {
        for (const auto& entry : sim.exchange->L3Record().at(SimSetup::bookId)) {
            if (const auto* cancel = std::get_if<taosim::event::CancellationEvent>(&entry)) {
                if (cancel->cancellation.id == orderId) return cancel->cancellation.volume.value_or(0_dec);
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] auto& balances() { return sim.exchange->accounts().at(miner).at(SimSetup::bookId); }

    [[nodiscard]] bool hasActiveOrders()
    {
        return !sim.exchange->accounts().at(miner).activeOrders().at(SimSetup::bookId).empty();
    }

    SimSetup sim;
    taosim::agent::DistributedProxyAgent* proxy{};
    AgentId miner{};
};

TEST_F(TimeInForceAfterMatchingTest, AnIocLeftoverIsCancelledAndItsOwnerTold)
{
    const auto ioc = validate(OrderDirection::BUY, 2_dec, 303_dec, TimeInForce::IOC);
    ASSERT_EQ(ioc.result.ec, OrderErrorCode::VALID);
    ASSERT_EQ(ioc.result.orderSize, 2_dec) << "the up-front cut sizes the IOC to the 2 resting at 303";
    shrinkTouch(OrderDirection::SELL, 1_dec);

    const auto order = place(ioc);

    EXPECT_EQ(sim.book->bestAsk(), 307_dec) << "the IOC did not take the 1 left at its limit";
    EXPECT_FALSE(sim.book->getOrder(order->id()).has_value()) << "the IOC's leftover rests";
    EXPECT_EQ(sim.book->bestBid(), 297_dec) << "the IOC's leftover rests as the best bid";
    EXPECT_EQ(order->volume(), 0_dec) << "the IOC's leftover was not cancelled";
    EXPECT_FALSE(hasActiveOrders()) << "the IOC is still one of its owner's active orders";
    EXPECT_EQ(balances().quote->getReserved(), 0_dec) << "the leftover's reservation was kept";
    const auto told = toldCancelled(order->id());
    ASSERT_TRUE(told.has_value()) << "the owner was not told its IOC's leftover was cancelled";
    EXPECT_EQ(*told, 1_dec);
    EXPECT_EQ(recordedCancelled(order->id()), std::optional<decimal_t>{1_dec})
        << "the L3 record does not carry the IOC's leftover as cancelled, so the tape reads it as resting";
}

TEST_F(TimeInForceAfterMatchingTest, ASellIocLeftoverIsCancelledAndItsOwnerTold)
{
    const auto ioc = validate(OrderDirection::SELL, 4_dec, 291_dec, TimeInForce::IOC);
    ASSERT_EQ(ioc.result.ec, OrderErrorCode::VALID);
    ASSERT_EQ(ioc.result.orderSize, 4_dec)
        << "the up-front cut sizes the IOC to the 1 resting at 297 and the 3 at 291";
    shrinkTouch(OrderDirection::BUY, 1_dec);

    const auto order = place(ioc);

    EXPECT_EQ(sim.book->bestBid(), 0_dec) << "the IOC did not take the 3 left at its limit";
    EXPECT_FALSE(sim.book->getOrder(order->id()).has_value()) << "the IOC's leftover rests";
    EXPECT_EQ(sim.book->bestAsk(), 303_dec) << "the IOC's leftover rests as the best ask";
    EXPECT_EQ(order->volume(), 0_dec) << "the IOC's leftover was not cancelled";
    EXPECT_FALSE(hasActiveOrders()) << "the IOC is still one of its owner's active orders";
    EXPECT_EQ(balances().base.getReserved(), 0_dec) << "the leftover's reservation was kept";
    const auto told = toldCancelled(order->id());
    ASSERT_TRUE(told.has_value()) << "the owner was not told its IOC's leftover was cancelled";
    EXPECT_EQ(*told, 1_dec);
    EXPECT_EQ(recordedCancelled(order->id()), std::optional<decimal_t>{1_dec})
        << "the L3 record does not carry the IOC's leftover as cancelled, so the tape reads it as resting";
}

TEST_F(TimeInForceAfterMatchingTest, AGtcLeftoverStillRests)
{
    const auto gtc = validate(OrderDirection::BUY, 2_dec, 303_dec, TimeInForce::GTC);
    ASSERT_EQ(gtc.result.ec, OrderErrorCode::VALID);
    shrinkTouch(OrderDirection::SELL, 1_dec);

    const auto order = place(gtc);

    const auto rest = sim.book->getOrder(order->id());
    ASSERT_TRUE(rest.has_value()) << "a GTC's leftover must rest";
    EXPECT_EQ((*rest)->volume(), 1_dec);
    EXPECT_EQ(sim.book->bestBid(), 303_dec);
    EXPECT_TRUE(hasActiveOrders());
    EXPECT_FALSE(toldCancelled(order->id()).has_value()) << "a resting GTC was announced as cancelled";
    EXPECT_FALSE(recordedCancelled(order->id()).has_value()) << "a resting GTC was recorded as cancelled";
}

TEST_F(TimeInForceAfterMatchingTest, AFokThatPassesTheUpFrontCheckFillsWhole)
{
    const auto fok = validate(OrderDirection::BUY, 5_dec, 307_dec, TimeInForce::FOK);
    ASSERT_EQ(fok.result.ec, OrderErrorCode::VALID);
    ASSERT_EQ(fok.result.orderSize, 5_dec);

    const auto order = place(fok);

    EXPECT_EQ(order->volume(), 0_dec) << "the FOK did not fill whole";
    EXPECT_FALSE(sim.book->getOrder(order->id()).has_value());
    const auto touch = sim.book->bestSellLevel();
    ASSERT_TRUE(touch.has_value());
    EXPECT_EQ(touch->get().price(), 307_dec) << "the FOK did not take the 2 at 303";
    ASSERT_NE(touch->get().getFirstActiveOrder(), nullptr);
    EXPECT_EQ(touch->get().getFirstActiveOrder()->volume(), 5_dec) << "the FOK did not take 3 at 307";
    EXPECT_FALSE(hasActiveOrders());
    EXPECT_FALSE(toldCancelled(order->id()).has_value())
        << "a FOK that filled whole was announced as cancelled";
}

TEST_F(TimeInForceAfterMatchingTest, AFokLeftoverIsCancelledNeverRested)
{
    const auto fok = validate(OrderDirection::BUY, 2_dec, 303_dec, TimeInForce::FOK);
    ASSERT_EQ(fok.result.ec, OrderErrorCode::VALID);
    shrinkTouch(OrderDirection::SELL, 1_dec);

    const auto order = place(fok);

    EXPECT_FALSE(sim.book->getOrder(order->id()).has_value()) << "the FOK's leftover rests";
    EXPECT_EQ(sim.book->bestBid(), 297_dec) << "the FOK's leftover rests as the best bid";
    EXPECT_FALSE(hasActiveOrders()) << "the FOK is still one of its owner's active orders";
    const auto told = toldCancelled(order->id());
    ASSERT_TRUE(told.has_value()) << "the owner was not told its FOK's leftover was cancelled";
    EXPECT_EQ(*told, 1_dec);
    EXPECT_EQ(recordedCancelled(order->id()), std::optional<decimal_t>{1_dec})
        << "the L3 record does not carry the FOK's leftover as cancelled, so the tape reads it as resting";
}

//-------------------------------------------------------------------------

}  // namespace

//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
// THE INSTRUCTION LOG CARRIES EVERY ORDER THAT CAN TAKE LIQUIDITY ON ARRIVAL. Under an active band a buy
// predicts no sweep, so the placement check's instantTrade is false for a buy that crosses the asks; the instruction
// line comes from the order crossing the opposite best price, not from that prediction.

namespace
{

struct InstructionLogUnderTheBand
{
    SimSetup sim;
    AgentId miner{};
    std::vector<OrderID> logged;
    boost::signals2::scoped_connection feed;

    explicit InstructionLogUnderTheBand(double band)
    {
        sim.setUp(band);
        if (band > 0.0) {
            sim.book->bandRefCached() = 300_dec;
            sim.book->bandSeeded() = true;
        }
        miner = sim.exchange->accounts().registerRemote();
        feed = sim.exchange->signals().at(SimSetup::bookId)->instructionLog.connect(
            [this](taosim::InstructionLogContext ctx) { logged.push_back(ctx.orderId); });
    }

    // The check and the placement, as the exchange runs them for a miner's limit.
    LimitOrder::Ptr buy(decimal_t volume, decimal_t price)
    {
        const auto payload = MessagePayload::create<PlaceOrderLimitPayload>(
            OrderDirection::BUY, volume, price, 0_dec, SimSetup::bookId, Currency::BASE,
            std::nullopt, false, TimeInForce::GTC, std::nullopt, STPFlag::CO);
        const auto result = sim.exchange->clearingManager().handleOrder(
            LimitOrderDesc{.agentId = miner, .payload = payload});
        EXPECT_EQ(result.ec, OrderErrorCode::VALID);
        return sim.book->placeLimitOrder(
            OrderClientContext{miner}, Timestamp{}, result.orderSize, payload->direction, payload->price,
            payload->leverage, payload->stpFlag, payload->settleFlag, payload->postOnly, payload->timeInForce);
    }

    [[nodiscard]] bool wasLogged(OrderID id) const { return std::ranges::find(logged, id) != logged.end(); }
};

}  // namespace

TEST(InstructionLogUnderTheBand, ABandedBuyThatTakesLiquidityOnArrivalIsLogged)
{
    InstructionLogUnderTheBand t{0.10};
    ASSERT_EQ(t.sim.book->bandLimit(true), 330_dec) << "the band is active, 10 per cent around 300";
    const auto order = t.buy(1_dec, 305_dec);
    ASSERT_NE(order, nullptr);
    EXPECT_EQ(order->volume(), 0_dec) << "the buy took 1 of the 2 resting at 303";
    EXPECT_TRUE(t.wasLogged(order->id())) << "a banded buy that took liquidity on arrival has no instruction line";
}

TEST(InstructionLogUnderTheBand, ABandedBuyThatRestsBelowTheAsksIsNotAnInstruction)
{
    InstructionLogUnderTheBand t{0.10};
    const auto order = t.buy(1_dec, 300_dec);
    ASSERT_NE(order, nullptr);
    EXPECT_EQ(order->volume(), 1_dec) << "nothing at or below 300 to take";
    EXPECT_FALSE(t.wasLogged(order->id())) << "an order that rests is the order log's, not an instruction";
}

TEST(InstructionLogUnderTheBand, WithTheBandOffATakingBuyIsLoggedAsBefore)
{
    InstructionLogUnderTheBand t{0.0};
    const auto order = t.buy(1_dec, 305_dec);
    ASSERT_NE(order, nullptr);
    EXPECT_TRUE(t.wasLogged(order->id()));
}

