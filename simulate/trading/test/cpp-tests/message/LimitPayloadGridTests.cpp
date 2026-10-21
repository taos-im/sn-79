// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT
//
// A limit price and size sent as floats rest where they were written when they are on the grid, and a
// price with more places than the grid is still truncated onto it. The payload is decoded as the
// validator's message is, then placed through the exchange's own validation, which truncates the price
// and the size onto their grids.

#include "MultiBookExchangeAgent.hpp"
#include "Order.hpp"
#include "Simulation.hpp"
#include "util.hpp"

#include <taosim/matching/ClearingManager.hpp>
#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/message/PayloadFactory.hpp>

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <msgpack.hpp>

#include <filesystem>
#include <memory>
#include <sstream>
#include <string>

//-------------------------------------------------------------------------

using namespace taosim;
using namespace taosim::literals;
using namespace taosim::matching;

using namespace testing;

namespace fs = std::filesystem;

//-------------------------------------------------------------------------

namespace
{

const auto kTestDataPath = fs::path{__FILE__}.parent_path().parent_path() / "data";

constexpr AgentId kRemoteAgent = 0;
constexpr BookId kBook = 0;
constexpr uint32_t kPriceDecimals = 2;

struct Placed
{
    decimal_t price;
    decimal_t volume;
};

class LimitPayloadGridTest : public Test
{
protected:
    void SetUp() override
    {
        // A 2dp price grid, as on the simulation, beside the 4dp volume grid.
        m_nodes = util::parseSimulationFile(kTestDataPath / "MinQuoteOrderSize.xml");
        m_nodes.exchange.attribute("priceDecimals").set_value(kPriceDecimals);
        m_simulation = std::make_unique<Simulation>();
        m_simulation->setDebug(false);
        m_simulation->configure(m_nodes.simulation);
    }

    // The payload as the validator sends it: a msgpack map whose price and volume are floats.
    [[nodiscard]] static PlaceOrderLimitPayload::Ptr decode(
        OrderDirection direction, double volume, double price)
    {
        std::stringstream ss;
        msgpack::packer<std::stringstream> pk{&ss};
        pk.pack_map(8);
        pk.pack(std::string{"direction"});
        pk.pack(direction);
        pk.pack(std::string{"volume"});
        pk.pack(volume);
        pk.pack(std::string{"price"});
        pk.pack(price);
        pk.pack(std::string{"leverage"});
        pk.pack(0.0);
        pk.pack(std::string{"bookId"});
        pk.pack(kBook);
        pk.pack(std::string{"currency"});
        pk.pack(Currency::BASE);
        pk.pack(std::string{"timeInForce"});
        pk.pack(TimeInForce::GTC);
        pk.pack(std::string{"stp"});
        pk.pack(STPFlag::CO);
        const auto buf = ss.str();
        const auto oh = msgpack::unpack(buf.data(), buf.size());
        return std::dynamic_pointer_cast<PlaceOrderLimitPayload>(
            PayloadFactory::createFromMessagePack(oh.get(), "PLACE_ORDER_LIMIT"));
    }

    // Where the order rests once the exchange has validated it and put it on its grids.
    [[nodiscard]] Placed place(OrderDirection direction, double volume, double price)
    {
        const auto payload = decode(direction, volume, price);
        EXPECT_NE(payload, nullptr);
        if (!payload) return {};
        auto* exchange = m_simulation->exchange();
        const auto result = exchange->clearingManager().handleOrder(
            LimitOrderDesc{.agentId = kRemoteAgent, .payload = payload});
        EXPECT_EQ(result.ec, OrderErrorCode::VALID);
        if (result.ec != OrderErrorCode::VALID) return {};
        const auto order = exchange->books()[kBook]->placeLimitOrder(
            OrderClientContext{kRemoteAgent},
            Timestamp{},
            result.orderSize,
            payload->direction,
            payload->price,
            payload->leverage,
            payload->stpFlag);
        return {.price = order->price(), .volume = order->volume()};
    }

    util::Nodes m_nodes;
    std::unique_ptr<Simulation> m_simulation;
};

}  // namespace

//-------------------------------------------------------------------------

TEST_F(LimitPayloadGridTest, AnOnGridFloatPriceRestsWhereItWasWritten)
{
    const auto placed = place(OrderDirection::SELL, 3.0, 29.56);
    EXPECT_EQ(placed.price, DEC(29.56));
    EXPECT_EQ(placed.volume, DEC(3.0));
}

TEST_F(LimitPayloadGridTest, AnOnGridFloatSizeRestsWhereItWasWritten)
{
    const auto placed = place(OrderDirection::BUY, 0.3, 29.5);
    EXPECT_EQ(placed.volume, DEC(0.3));
    EXPECT_EQ(placed.price, DEC(29.5));
}

TEST_F(LimitPayloadGridTest, AFloatPriceBeyondTheGridIsStillTruncated)
{
    const auto placed = place(OrderDirection::BUY, 1.0, 29.567);
    EXPECT_EQ(placed.price, DEC(29.56));
}

TEST_F(LimitPayloadGridTest, AFloatTheDoubleHoldsExactlyIsUnchanged)
{
    const auto placed = place(OrderDirection::BUY, 0.25, 29.75);
    EXPECT_EQ(placed.price, DEC(29.75));
    EXPECT_EQ(placed.volume, DEC(0.25));
}

//-------------------------------------------------------------------------
