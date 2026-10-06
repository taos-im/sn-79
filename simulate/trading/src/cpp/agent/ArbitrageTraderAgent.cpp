/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/agent/ArbitrageTraderAgent.hpp>

#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/message/MessagePayload.hpp>
#include <taosim/book/Book.hpp>
#include <Simulation.hpp>

#include <stdexcept>

//-------------------------------------------------------------------------

namespace taosim::agent
{

//-------------------------------------------------------------------------

ArbitrageTraderAgent::ArbitrageTraderAgent(Simulation* simulation) noexcept
    : Agent{simulation}
{}

//-------------------------------------------------------------------------

void ArbitrageTraderAgent::configure(const pugi::xml_node& node)
{
    Agent::configure(node);

    pugi::xml_attribute attr;

    if (attr = node.attribute("exchange"); attr.empty()) {
        throw std::invalid_argument(
            "ArbitrageTraderAgent::configure: missing required attribute 'exchange'");
    }
    m_exchange = attr.as_string();

    m_edge = node.attribute("edge").as_double(8e-4);
    m_cross = node.attribute("cross").as_double(2e-4);
    m_alpha = node.attribute("alpha").as_double(0.02);
    m_latency = node.attribute("latency").as_ullong(1);
}

//-------------------------------------------------------------------------

void ArbitrageTraderAgent::receiveMessage(Message::Ptr msg)
{
    if (msg->type == "EVENT_SIMULATION_START") {
        handleSimulationStart();
    }
    else if (msg->type == "EVENT_ORDER_LIMIT") {
        handleMinerLimitOrder(msg);
    }
    else if (msg->type == "EVENT_TRADE") {
        handleTrade(msg);
    }
    else if (msg->type == "EVENT_SIMULATION_END") {
        fmt::println(
            "{}: ARB_SUMMARY miner_limit_events={} takes={} fills={}",
            name(), m_eventsSeen, m_takes, m_fills);
    }
    // RESPONSE_PLACE_ORDER_LIMIT / ERROR_RESPONSE_PLACE_ORDER_LIMIT: fire-and-forget IOC.
}

//-------------------------------------------------------------------------

void ArbitrageTraderAgent::handleSimulationStart()
{
    m_id = simulation()->exchange()->accounts().lookupLocalAgentId(name());
    const Timestamp now = simulation()->currentTimestamp();
    // Miner-only limit-order stream (cheap: dispatched solely for distributed/miner
    // orders) + trades to maintain a fair reference.
    simulation()->dispatchMessage(
        now, 0, name(), m_exchange,
        "SUBSCRIBE_EVENT_ORDER_LIMIT_MINER", std::make_shared<EmptyPayload>());
    simulation()->dispatchMessage(
        now, 0, name(), m_exchange,
        "SUBSCRIBE_EVENT_TRADE", std::make_shared<EmptyPayload>());
}

//-------------------------------------------------------------------------

void ArbitrageTraderAgent::handleTrade(Message::Ptr msg)
{
    // Sample the BOOK MID (not the trade price) on each trade: a wash trade prints at the
    // off-market gift price, but the mid reflects the real market level (the transient
    // resting gift barely moves it and snaps back once taken), so fair stays clean without
    // needing to identify miner/wash trades.
    const auto payload = std::static_pointer_cast<EventTradePayload>(msg->payload);
    if (payload->context.aggressingAgentId == m_id || payload->context.restingAgentId == m_id) {
        ++m_fills;   // this arb was a party to the trade => its take actually executed
    }
    const BookId bookId = payload->bookId;
    const auto& books = simulation()->exchange()->books();
    if (bookId >= books.size()) return;
    const double mid = taosim::util::decimal2double(books[bookId]->midPrice());
    if (mid <= 0.0) return;
    auto it = m_fair.find(bookId);
    if (it == m_fair.end()) {
        m_fair[bookId] = mid;
    } else {
        it->second = (1.0 - m_alpha) * it->second + m_alpha * mid;
    }
}

//-------------------------------------------------------------------------

void ArbitrageTraderAgent::handleMinerLimitOrder(Message::Ptr msg)
{
    ++m_eventsSeen;
    const auto payload = std::static_pointer_cast<EventOrderLimitPayload>(msg->payload);
    const BookId bookId = payload->bookId;

    auto it = m_fair.find(bookId);
    if (it == m_fair.end()) return;   // no fair reference yet (warmup)
    const double fair = it->second;
    if (fair <= 0.0) return;

    const double price = taosim::util::decimal2double(payload->order.price());
    const double volume = taosim::util::decimal2double(payload->order.volume());
    if (price <= 0.0 || volume <= 0.0) return;

    const OrderDirection restingDir = payload->order.direction();
    OrderDirection takeDir;
    if (restingDir == OrderDirection::SELL && price < fair * (1.0 - m_edge)) {
        takeDir = OrderDirection::BUY;   // a resting sell below fair: buy it cheap
    } else if (restingDir == OrderDirection::BUY && price > fair * (1.0 + m_edge)) {
        takeDir = OrderDirection::SELL;  // a resting buy above fair: sell into it
    } else {
        return;                          // not a through-fair gift
    }

    ++m_takes;
    // Take-only IOC at the resting order's price, bumped by a tiny `cross` toward the
    // aggressor side so the marketable limit reliably crosses despite decimal rounding.
    // The bump stays well inside `edge` (cross < edge), so the price limit still stops at
    // the mispriced level and never walks into normal book liquidity.
    const double takePrice = (takeDir == OrderDirection::BUY)
        ? price * (1.0 + m_cross)
        : price * (1.0 - m_cross);
    const auto& params = simulation()->exchange()->config().parameters();
    auto orderPayload = MessagePayload::create<PlaceOrderLimitPayload>(
        takeDir,
        taosim::util::double2decimal(volume, params.volumeIncrementDecimals),
        taosim::util::double2decimal(takePrice, params.priceIncrementDecimals),
        taosim::util::double2decimal(0.0),
        bookId);
    orderPayload->timeInForce = taosim::TimeInForce::IOC;
    orderPayload->postOnly = false;
    simulation()->dispatchMessage(
        simulation()->currentTimestamp(), m_latency,
        name(), m_exchange, "PLACE_ORDER_LIMIT", orderPayload);
}

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------
