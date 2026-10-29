/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

// A local agent's cancel response names the orders it cancelled.
//
// handleLocalCancelOrders built the response from two arguments of one call: the ids, read from the cancellations, and
// the request payload, which moved them. The order the two are evaluated in is unspecified, and the move came first, so
// the response named no order. handleLocalClosePositions had the same shape; the distributed handlers build the ids
// first and were right.

#include "MultiBookExchangeAgent.hpp"
#include "Simulation.hpp"
#include "util.hpp"
#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/message/MultiBookMessagePayloads.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>

//-------------------------------------------------------------------------

using namespace taosim::literals;

namespace fs = std::filesystem;

namespace
{

const auto kTestDataPath = fs::path{__FILE__}.parent_path().parent_path() / "data";
constexpr std::string_view kAgent = "LOCAL_RESPONSE_TEST_AGENT";

template<typename T>
std::shared_ptr<T> queued(Simulation& sim, std::string_view type)
{
    for (const auto& entry : sim.messageQueue().queue().underlying()) {
        if (entry.pmsg.msg->type != type) continue;
        if (auto pld = std::dynamic_pointer_cast<T>(entry.pmsg.msg->payload)) return pld;
    }
    return nullptr;
}

}  // namespace

//-------------------------------------------------------------------------

TEST(LocalResponses, ALocalCancelNamesTheOrdersItCancelled)
{
    auto nodes = taosim::util::parseSimulationFile(kTestDataPath / "MultiAgentFees.xml");
    auto sim = std::make_unique<Simulation>();
    sim->setDebug(false);
    sim->configure(nodes.simulation);
    auto* exchange = sim->exchange();
    exchange->accounts().registerLocal(std::string{kAgent});
    const auto now = sim->currentTimestamp();

    exchange->receiveMessage(Message::create(now, now, std::string{kAgent}, exchange->name(), "PLACE_ORDER_LIMIT",
        MessagePayload::create<PlaceOrderLimitPayload>(OrderDirection::BUY, 1_dec, 290_dec, BookId{0})));
    const auto placed = queued<PlaceOrderLimitResponsePayload>(*sim, "RESPONSE_PLACE_ORDER_LIMIT");
    ASSERT_NE(placed, nullptr) << "the local placement was not answered";

    exchange->receiveMessage(Message::create(now, now, std::string{kAgent}, exchange->name(), "CANCEL_ORDERS",
        MessagePayload::create<CancelOrdersPayload>(std::vector{taosim::event::Cancellation{placed->id}}, BookId{0})));
    const auto cancelled = queued<CancelOrdersResponsePayload>(*sim, "RESPONSE_CANCEL_ORDERS");
    ASSERT_NE(cancelled, nullptr) << "the local cancel was not answered";
    EXPECT_EQ(cancelled->orderIds, std::vector<OrderID>{placed->id})
        << "the cancel response names no order: its ids were read after the cancellations were moved";
}

//-------------------------------------------------------------------------
