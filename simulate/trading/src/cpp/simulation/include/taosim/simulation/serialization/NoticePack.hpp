/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

// ONE notice serializer, shared by both mechanisms.
//
// Serializes an agent response into the wire notice a miner receives. Both the simulation's
// Both mechanisms' response builders call it, so a notice has one shape
// whichever mechanism produced it. `logDir` is read only for EVENT_SIMULATION_START; `ctx` names the
// caller in the cast-failure messages.

#include <taosim/event/serialization/CancellationEvent.hpp>
#include <taosim/event/serialization/OrderEvent.hpp>
#include <taosim/event/serialization/TradeEvent.hpp>
#include <common.hpp>

#include <boost/algorithm/string.hpp>
#include <msgpack.hpp>

#include <string>

//-------------------------------------------------------------------------

namespace taosim::simulation::serialization
{

//-------------------------------------------------------------------------

template<typename Packer>
void packNotice(Packer& o, Message::Ptr msg, const std::string& logDir, const std::string& ctx)
{
    using namespace std::string_literals;

            if (msg->type == "EVENT_SIMULATION_START") {
                o.pack_map(4);
            } else if (msg->type == "RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT"
                || msg->type == "ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT") {
                o.pack_map(13);
            } else if (msg->type == "RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET"
                || msg->type == "ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET") {
                o.pack_map(13);
            } else if (msg->type == "EVENT_TRADE") {
                o.pack_map(17);
            } else if (msg->type == "RESPONSE_DISTRIBUTED_CANCEL_ORDERS" 
                || msg->type == "ERROR_RESPONSE_DISTRIBUTED_CANCEL_ORDERS") {
                o.pack_map(5);
            } else if (msg->type == "RESPONSE_DISTRIBUTED_CLOSE_POSITIONS"
                || msg->type == "ERROR_RESPONSE_DISTRIBUTED_CLOSE_POSITIONS") {
                o.pack_map(5);
            } else if (msg->type == "RESPONSE_DISTRIBUTED_RESET_AGENT"
                || msg->type == "ERROR_RESPONSE_DISTRIBUTED_RESET_AGENT") {
                o.pack_map(4);
            } else if (msg->type == "EVENT_SIMULATION_END") {
                o.pack_map(3);
            } else {
                o.pack_map(3);
            }

            auto abbreviate = [](const std::string& str) {
                std::vector<std::string> parts;
                boost::split(parts, str, boost::is_any_of("_"));
                return fmt::format(
                    "{}",
                    fmt::join(
                        parts
                        | views::transform([](auto&& part) {
                            return part.empty() ? ""s : std::string(1, part.front());
                        }),
                        ""));
            };

            o.pack("y"s);
            o.pack(abbreviate(msg->type));

            o.pack("t"s);
            o.pack(msg->occurrence);

            o.pack("a"s);
            [&] {
                if (std::dynamic_pointer_cast<StartSimulationPayload>(msg->payload) != nullptr
                    || std::dynamic_pointer_cast<EmptyPayload>(msg->payload) != nullptr) {
                    o.pack_nil();
                    return;
                }
                const auto pld = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
                if (pld == nullptr) {
                    throw std::runtime_error{fmt::format(
                        "{}: Failed to cast to DistributedAgentResponsePayload in 'packNotice'", ctx)};
                }
                if (pld->agentId > 0) {
                    o.pack(pld->agentId);
                } else {
                    o.pack_nil();
                }
            }();

            if (msg->type == "EVENT_SIMULATION_START") {
                o.pack("l"s);
                o.pack(logDir);
            }
            else if (msg->type == "RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT") {
                const auto pld = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
                const auto subPld = std::dynamic_pointer_cast<PlaceOrderLimitResponsePayload>(pld->payload);
                const auto reqPld = subPld->requestPayload;

                o.pack("b"s);
                o.pack(reqPld->bookId);

                o.pack("o"s);
                o.pack(subPld->id);

                o.pack("c"s);
                o.pack(reqPld->clientOrderId);

                o.pack("s"s);
                o.pack(reqPld->direction);

                o.pack("q"s);
                o.pack(reqPld->volume);

                o.pack("u"s);
                o.pack(true);

                o.pack("m"s);
                o.pack(""s);

                o.pack("l"s);
                o.pack(reqPld->leverage);

                o.pack("f"s);
                o.pack(reqPld->settleFlag);

                o.pack("p"s);
                o.pack(reqPld->price);
            }
            else if (msg->type == "ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT") {
                const auto pld = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
                const auto subPld = std::dynamic_pointer_cast<PlaceOrderLimitErrorResponsePayload>(pld->payload);
                const auto reqPld = subPld->requestPayload;
                const auto errPld = subPld->errorPayload;

                o.pack("b"s);
                o.pack(reqPld->bookId);

                o.pack("o"s);
                o.pack_nil();

                o.pack("c"s);
                o.pack(reqPld->clientOrderId);

                o.pack("s"s);
                o.pack(reqPld->direction);

                o.pack("q"s);
                o.pack(reqPld->volume);

                o.pack("u"s);
                o.pack(false);

                o.pack("m"s);
                o.pack(errPld->message);

                o.pack("l"s);
                o.pack(reqPld->leverage);

                o.pack("f"s);
                o.pack(reqPld->settleFlag);

                o.pack("p"s);
                o.pack(reqPld->price);
            }
            else if (msg->type == "RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET") {
                const auto pld = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
                const auto subPld = std::dynamic_pointer_cast<PlaceOrderMarketResponsePayload>(pld->payload);
                const auto reqPld = subPld->requestPayload;

                o.pack("b"s);
                o.pack(reqPld->bookId);

                o.pack("o"s);
                o.pack(subPld->id);

                o.pack("c"s);
                o.pack(reqPld->clientOrderId);

                o.pack("s"s);
                o.pack(reqPld->direction);

                o.pack("q"s);
                o.pack(reqPld->volume);

                o.pack("u"s);
                o.pack(true);

                o.pack("m"s);
                o.pack(""s);

                o.pack("l"s);
                o.pack(reqPld->leverage);

                o.pack("f"s);
                o.pack(reqPld->settleFlag);

                o.pack("r"s);
                o.pack(reqPld->currency);
            }
            else if (msg->type == "ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET") {
                const auto pld = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
                const auto subPld = std::dynamic_pointer_cast<PlaceOrderMarketErrorResponsePayload>(pld->payload);
                const auto reqPld = subPld->requestPayload;
                const auto errPld = subPld->errorPayload;

                o.pack("b"s);
                o.pack(reqPld->bookId);

                o.pack("o"s);
                o.pack_nil();

                o.pack("c"s);
                o.pack(reqPld->clientOrderId);

                o.pack("s"s);
                o.pack(reqPld->direction);

                o.pack("q"s);
                o.pack(reqPld->volume);

                o.pack("u"s);
                o.pack(false);

                o.pack("m"s);
                o.pack(errPld->message);

                o.pack("l"s);
                o.pack(reqPld->leverage);

                o.pack("f"s);
                o.pack(reqPld->settleFlag);

                o.pack("r"s);
                o.pack(reqPld->currency);
            }
            else if (msg->type == "EVENT_TRADE") {
                const auto pld = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
                const auto subPld = std::dynamic_pointer_cast<EventTradePayload>(pld->payload);

                o.pack("b"s);
                o.pack(subPld->bookId);

                o.pack("i"s);
                o.pack(subPld->trade.m_id);

                o.pack("c"s);
                o.pack(subPld->clientOrderId);

                o.pack("Ta"s);
                // The EFFECTIVE taker. A sweep order is placed under the exchange's own id (-1) once a
                // miner's instruction has moved the pool through a resting price; packing that id
                // leaves the validator with no settleable counterparty, so the fill reports
                // executed=false, corrections unwind it, and the resting miner is never filled nor
                // told. initiatorAgentId names the miner that caused it, and is nullopt otherwise.
                o.pack(subPld->context.initiatorAgentId.value_or(subPld->context.aggressingAgentId));

                o.pack("Ti"s);
                o.pack(subPld->trade.m_aggressingOrderID);

                o.pack("Tf"s);
                o.pack(subPld->context.fees.taker);

                o.pack("Ma"s);
                o.pack(subPld->context.restingAgentId);

                o.pack("Mi"s);
                o.pack(subPld->trade.m_restingOrderID);

                o.pack("Mf"s);
                o.pack(subPld->context.fees.maker);

                o.pack("s"s);
                o.pack(subPld->trade.m_direction);

                o.pack("p"s);
                o.pack(subPld->trade.m_price);

                o.pack("q"s);
                o.pack(subPld->trade.m_volume);

                o.pack("cr"s);
                o.pack(subPld->context.aggressingCloseReason);

                o.pack("Toi"s);
                o.pack(subPld->context.aggressingOriginatingOrderId);
            }
            else if (msg->type == "RESPONSE_DISTRIBUTED_CANCEL_ORDERS") {
                const auto pld = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
                const auto subPld = std::dynamic_pointer_cast<CancelOrdersResponsePayload>(pld->payload);
                const auto reqPld = subPld->requestPayload;

                o.pack("b"s);
                o.pack(reqPld->bookId);

                o.pack("c"s);
                o.pack_array(reqPld->cancellations.size());
                for (const auto& cancellation : reqPld->cancellations) {
                    o.pack_map(6);

                    o.pack("t"s);
                    o.pack(msg->occurrence);

                    o.pack("b"s);
                    o.pack(reqPld->bookId);

                    o.pack("o"s);
                    o.pack(cancellation.id);

                    o.pack("q"s);
                    o.pack(cancellation.volume);

                    o.pack("u"s);
                    o.pack(true);

                    o.pack("m"s);
                    o.pack(""s);
                }
            }
            else if (msg->type == "ERROR_RESPONSE_DISTRIBUTED_CANCEL_ORDERS") {
                const auto pld = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
                const auto subPld = std::dynamic_pointer_cast<CancelOrdersErrorResponsePayload>(pld->payload);
                const auto reqPld = subPld->requestPayload;
                const auto errPld = subPld->errorPayload;

                o.pack("b"s);
                o.pack(reqPld->bookId);

                o.pack("c"s);
                o.pack_array(reqPld->cancellations.size());
                for (const auto& cancellation : reqPld->cancellations) {
                    o.pack_map(6);

                    o.pack("t"s);
                    o.pack(msg->occurrence);

                    o.pack("b"s);
                    o.pack(reqPld->bookId);

                    o.pack("o"s);
                    o.pack(cancellation.id);

                    o.pack("q"s);
                    o.pack(cancellation.volume);

                    o.pack("u"s);
                    o.pack(false);

                    o.pack("m"s);
                    o.pack(errPld->message);
                }
            }
            else if (msg->type == "RESPONSE_DISTRIBUTED_CLOSE_POSITIONS") {
                const auto pld = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
                const auto subPld = std::dynamic_pointer_cast<ClosePositionsResponsePayload>(pld->payload);
                const auto reqPld = subPld->requestPayload;

                o.pack("b"s);
                o.pack(reqPld->bookId);

                o.pack("o"s);
                o.pack_array(reqPld->closePositions.size());
                for (const auto& close : reqPld->closePositions) {
                    o.pack_map(6);

                    o.pack("t"s);
                    o.pack(msg->occurrence);

                    o.pack("b"s);
                    o.pack(reqPld->bookId);

                    o.pack("o"s);
                    o.pack(close.id);

                    o.pack("q"s);
                    o.pack(close.volume);

                    o.pack("u"s);
                    o.pack(true);

                    o.pack("m"s);
                    o.pack(""s);
                }
            }
            else if (msg->type == "ERROR_RESPONSE_DISTRIBUTED_CLOSE_POSITIONS") {
                const auto pld = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
                const auto subPld = std::dynamic_pointer_cast<ClosePositionsErrorResponsePayload>(pld->payload);
                const auto reqPld = subPld->requestPayload;
                const auto errPld = subPld->errorPayload;

                o.pack("b"s);
                o.pack(reqPld->bookId);

                o.pack("o"s);
                o.pack_array(reqPld->closePositions.size());
                for (const auto& close : reqPld->closePositions) {
                    o.pack_map(6);

                    o.pack("t"s);
                    o.pack(msg->occurrence);

                    o.pack("b"s);
                    o.pack(reqPld->bookId);

                    o.pack("o"s);
                    o.pack(close.id);

                    o.pack("q"s);
                    o.pack(close.volume);

                    o.pack("u"s);
                    o.pack(false);

                    o.pack("m"s);
                    o.pack(errPld->message);
                }
            }
            else if (msg->type == "RESPONSE_DISTRIBUTED_RESET_AGENT") {
                const auto pld = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
                const auto subPld = std::dynamic_pointer_cast<ResetAgentsResponsePayload>(pld->payload);
                const auto reqPld = subPld->requestPayload;

                o.pack("r"s);
                o.pack_array(reqPld->agentIds.size());
                for (auto agentId : reqPld->agentIds) {
                    o.pack_map(4);

                    o.pack("a"s);
                    o.pack(agentId);

                    o.pack("t"s);
                    o.pack(msg->occurrence);

                    o.pack("u"s);
                    o.pack(true);

                    o.pack("m"s);
                    o.pack(""s);
                }
            }
            else if (msg->type == "ERROR_RESPONSE_DISTRIBUTED_RESET_AGENT") {
                const auto pld = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
                const auto subPld = std::dynamic_pointer_cast<ResetAgentsErrorResponsePayload>(pld->payload);
                const auto reqPld = subPld->requestPayload;
                const auto errPld = subPld->errorPayload;

                o.pack("r"s);
                o.pack_array(reqPld->agentIds.size());
                for (auto agentId : reqPld->agentIds) {
                    o.pack_map(4);

                    o.pack("a"s);
                    o.pack(agentId);

                    o.pack("t"s);
                    o.pack(msg->occurrence);

                    o.pack("u"s);
                    o.pack(false);

                    o.pack("m"s);
                    o.pack(errPld->message);
                }
            }
}

//-------------------------------------------------------------------------

}  // namespace taosim::simulation::serialization

//-------------------------------------------------------------------------
