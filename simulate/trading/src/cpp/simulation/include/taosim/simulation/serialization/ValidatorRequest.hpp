/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/event/serialization/CancellationEvent.hpp>
#include <taosim/event/serialization/L3RecordContainer.hpp>
#include <taosim/event/serialization/OrderEvent.hpp>
#include <taosim/event/serialization/TradeEvent.hpp>
#include <taosim/simulation/util.hpp>
#include <taosim/simulation/serialization/LimitOrder.hpp>
#include <taosim/simulation/serialization/NoticePack.hpp>
#include <common.hpp>

#include <boost/algorithm/string.hpp>
#include <msgpack.hpp>

#include <source_location>

//-------------------------------------------------------------------------

namespace taosim::simulation::serialization
{

struct ValidatorRequest
{
    SimulationManager* mngr;
};

}  // taosim::simulation::serialization

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct pack<taosim::simulation::serialization::ValidatorRequest>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o,
        const taosim::simulation::serialization::ValidatorRequest& v) const
    {
        using namespace std::string_literals;

        static constexpr auto ctx = std::source_location::current().function_name();

        const auto& representativeSimulation = v.mngr->simulations().front();
        const auto& blockInfo = v.mngr->blockInfo();
        const auto bookCount = blockInfo.count * blockInfo.dimension;
        const auto remoteAgentCount = ranges::count_if(
            views::keys(representativeSimulation->exchange()->accounts()),
            [](AgentId agentId) {
                return agentId >= 0; 
            });

        o.pack_map(6);

        // Log directory.
        o.pack("logDir"s);
        o.pack(v.mngr->logDir().string());

        // Timestamp.
        o.pack("timestamp"s);
        o.pack(representativeSimulation->currentTimestamp());

        // Model.
        o.pack("model"s);
        o.pack("im"s);

        // Books.
        o.pack("books"s);
        o.pack_map(bookCount);
        for (const auto& [blockIdx, simulation] : views::enumerate(v.mngr->simulations())) {
            const auto exchange = simulation->exchange();
            for (const auto& book : exchange->books()) {
                const BookId bookIdCanon = blockIdx * blockInfo.dimension + book->id();

                o.pack(bookIdCanon);

                o.pack_map(6);

                o.pack("i"s);
                o.pack(bookIdCanon);

                o.pack("r"s);
                o.pack(exchange->clearingManager().feePolicy()->makerTakerRatio(book->id(), 0));

                // THE RATES THE ENGINE ACTUALLY APPLIES, alongside the ratio they derive from.
                // The platform used to recompute these in Python from a curve of its own, and not one
                // parameter matched: target 0.50 against the engine's 0.4, max 0.0075 against 0.015,
                // base rates 0.0001/0.0003 against 0.000/0.00023, and no equivalent of getRates'
                // zero-MTR early return. So every rate the platform displayed was a plausible number
                // the venue had never charged. Same 'fs'/'mk'/'tk' shape a trade's fees already use.
                o.pack("fs"s);
                {
                    const auto rates = exchange->clearingManager().feePolicy()->getRates(book->id(), 0);
                    o.pack_map(2);
                    o.pack("mk"s);
                    o.pack(rates.maker);
                    o.pack("tk"s);
                    o.pack(rates.taker);
                }

                o.pack("e"s);
                o.pack(exchange->L3Record().at(book->id()));

                auto packLevel = [&](auto& o, const taosim::book::TickContainer& v) {
                    o.pack_map(3);

                    o.pack("p"s);
                    o.pack(v.price());
                
                    o.pack("q"s);
                    o.pack(v.volume());

                    o.pack("o"s);
                    if (v.empty()) {
                        o.pack_nil();
                    }
                    else {
                        o.pack_array(v.size());
                        for (const auto& order : v) {
                            o.pack_map(8);

                            o.pack("y"s);
                            o.pack("o"s);

                            o.pack("i"s);
                            o.pack(order->m_id);

                            o.pack("c"s);
                            o.pack(book->orderToClientInfo().at(order->m_id).clientOrderId);

                            o.pack("t"s);
                            o.pack(order->m_timestamp);

                            o.pack("q"s);
                            o.pack(order->m_volume);

                            o.pack("s"s);
                            o.pack(order->m_direction);

                            o.pack("p"s);
                            if (auto limitOrder = std::dynamic_pointer_cast<::LimitOrder>(order)) {
                                o.pack(limitOrder->m_price);
                            } else {
                                o.pack_nil();
                            }

                            o.pack("l"s);
                            o.pack(order->m_leverage);
                        }
                    }
                };

                auto packLevelBroad = [&](auto& o, const taosim::book::TickContainer& v) {
                    o.pack_map(2);

                    o.pack("p"s);
                    o.pack(v.price());

                    o.pack("q"s);
                    o.pack(v.volume());
                };

                const auto maxDepth = book->maxDepth();
                const auto detailedDepth = book->detailedDepth();

                const auto& buyQueue = book->buyQueue();
                o.pack("b"s);
                o.pack_array(std::min(buyQueue.size(), maxDepth));
                for (const auto& level : buyQueue | views::reverse | views::take(detailedDepth)) {
                    packLevel(o, level);
                }
                auto broadBuyView = buyQueue
                    | views::reverse
                    | views::drop(detailedDepth)
                    | views::take(maxDepth - detailedDepth);
                for (const auto& level : broadBuyView) {
                    packLevelBroad(o, level);
                }
    
                const auto& sellQueue = book->sellQueue();
                o.pack("a"s);
                o.pack_array(std::min(sellQueue.size(), maxDepth));
                for (const auto& level : sellQueue | views::take(detailedDepth)) {
                    packLevel(o, level);
                }
                auto broadSellView = sellQueue
                    | views::drop(detailedDepth)
                    | views::take(maxDepth - detailedDepth);
                for (const auto& level : broadSellView) {
                    packLevelBroad(o, level);
                }
            }
        }

        // Accounts.
        o.pack("accounts"s);
        o.pack_map(remoteAgentCount);

        for (AgentId agentId : views::keys(representativeSimulation->exchange()->accounts())) {
            if (agentId < 0) continue;

            o.pack(agentId);

            o.pack_map(bookCount);

            for (const auto& [blockIdx, simulation] : views::enumerate(v.mngr->simulations())) {
                const auto exchange = simulation->exchange();
                const auto& account = exchange->accounts().at(agentId);
                const auto feePolicy = exchange->clearingManager().feePolicy();
                for (const auto& book : exchange->books()) {
                    const BookId bookIdCanon = blockIdx * blockInfo.dimension + book->id();

                    o.pack(bookIdCanon);

                    o.pack_map(11);

                    o.pack("i"s);
                    o.pack(agentId);

                    o.pack("b"s);
                    o.pack(bookIdCanon);

                    const auto& balances = account.at(book->id());

                    auto packBalance = [](auto& o, const taosim::accounting::Balance& balance, std::string_view currency) {
                        o.pack_map(5);

                        o.pack("c"s);
                        o.pack(currency);

                        o.pack("t"s);
                        o.pack(balance.getTotal());

                        o.pack("f"s);
                        o.pack(balance.getFree());

                        o.pack("r"s);
                        o.pack(balance.getReserved());

                        o.pack("i"s);
                        o.pack(balance.getInitial());
                    };

                    o.pack("bb"s);
                    packBalance(o, balances.base, "BASE");

                    o.pack("qb"s);
                    packBalance(o, *balances.quote, "QUOTE");

                    o.pack("bl"s);
                    o.pack(balances.m_baseLoan);

                    o.pack("ql"s);
                    o.pack(balances.m_quoteLoan);

                    o.pack("bc"s);
                    o.pack(balances.m_baseCollateral);

                    o.pack("qc"s);
                    o.pack(balances.m_quoteCollateral);

                    o.pack("o"s);
                    const auto limitOrders = [&] {
                        std::vector<taosim::simulation::serialization::LimitOrder> limitOrders;
                        const auto& activeOrders = account.activeOrders().at(book->id());
                        for (const auto& order : activeOrders) {
                            const auto limitOrder =
                                std::dynamic_pointer_cast<::LimitOrder>(order);
                            if (limitOrder == nullptr) continue;
                            limitOrders.push_back(taosim::simulation::serialization::LimitOrder{
                                .limitOrder = limitOrder,
                                .clientOrderId = book->orderToClientInfo().at(order->id()).clientOrderId
                            });
                        }
                        return limitOrders;
                    }();
                    o.pack_array(limitOrders.size());
                    for (const auto& limitOrder : limitOrders) {
                        o.pack(limitOrder);
                    }

                    auto packLoan = [](auto& o, const taosim::accounting::Loan& loan, OrderID id) {
                        o.pack_map(5);

                        o.pack("i"s);
                        o.pack(id);

                        o.pack("a"s);
                        o.pack(loan.amount());

                        o.pack("c"s);
                        o.pack(
                            std::to_underlying(
                                loan.direction() == OrderDirection::BUY ? Currency::QUOTE : Currency::BASE));

                        o.pack("bc"s);
                        o.pack(loan.collateral().base());

                        o.pack("qc"s);
                        o.pack(loan.collateral().quote());
                    };

                    o.pack("l"s);
                    o.pack_map(balances.m_loans.size());
                    for (const auto& [id, loan] : balances.m_loans) {
                        o.pack(id);
                        packLoan(o, loan, id);
                    }

                    o.pack("f"s);

                    o.pack_map(3);

                    o.pack("v"s);
                    if (feePolicy->isTiered()) {
                        o.pack(feePolicy->agentVolume(book->id(), agentId));
                    } else {
                        o.pack_nil();
                    }

                    const auto rates = feePolicy->getRates(book->id(), agentId);

                    o.pack("m"s);
                    o.pack(rates.maker);

                    o.pack("t"s);
                    o.pack(rates.taker);
                }
            }
        }

        // Notices.
        o.pack("notices"s);
        
        auto collectiveRemoteResponses = [&] {
            std::unordered_map<std::string, uint32_t> msgTypeToCount{
                { "RESPONSE_DISTRIBUTED_RESET_AGENT", 0 },
                { "ERROR_RESPONSE_DISTRIBUTED_RESET_AGENT", 0 }
            };
            auto checkGlobalDuplicate = [&](Message::Ptr msg) -> bool {
                const auto payload = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
                if (payload == nullptr) return false;
                auto relevantPayload = [&] {
                    const auto pld = payload->payload;
                    return std::dynamic_pointer_cast<ResetAgentsResponsePayload>(pld) != nullptr
                        || std::dynamic_pointer_cast<ResetAgentsErrorResponsePayload>(pld) != nullptr;
                };
                if (!relevantPayload()) return true;
                auto it = msgTypeToCount.find(msg->type);
                if (it == msgTypeToCount.end()) return true;
                if (it->second > 0) return false;
                it->second++;
                return true;
            };
            std::vector<Message::Ptr> res;
            for (const auto& [blockIdx, simulation] : views::enumerate(v.mngr->simulations())) {
                for (const auto& msg : simulation->proxy()->messages()) {
                    if (!checkGlobalDuplicate(msg)) continue;
                    taosim::simulation::canonize(msg, blockIdx, blockInfo.dimension);
                    res.push_back(msg);
                }
                simulation->proxy()->clearMessages();
            }
            return res;
        }();
        ranges::sort(
            collectiveRemoteResponses,
            [](auto&& lhs, auto&& rhs) {
                if (lhs->occurrence != rhs->occurrence) {
                    return lhs->occurrence < rhs->occurrence;
                }
                return lhs->arrival - lhs->occurrence > rhs->arrival - rhs->occurrence;
            });
        const auto remoteResponsesPerAgent = [&] {
            std::map<AgentId, std::vector<Message::Ptr>> res;
            for (const auto& msg : collectiveRemoteResponses) {
                if (std::dynamic_pointer_cast<StartSimulationPayload>(msg->payload) != nullptr
                    || std::dynamic_pointer_cast<EmptyPayload>(msg->payload) != nullptr) {
                    for (auto agentId : views::keys(representativeSimulation->exchange()->accounts())) {
                        if (agentId < 0) continue;
                        res[agentId].push_back(msg);
                    }
                    continue;
                }
                const auto pld = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
                if (pld == nullptr) {
                    throw std::runtime_error{fmt::format(
                        "{}: Failed to cast to DistributedAgentResponsePayload in 'remoteResponsesPerAgent'", ctx)};
                }
                res[pld->agentId].push_back(msg);
            }
            return res;
        }();

        auto packNotice = [&](auto& o, Message::Ptr msg) {
            // Body lives in NoticePack.hpp so the exchange mechanism serializes notices with the very
            // same code. See that header for why.
            taosim::simulation::serialization::packNotice(
                o, msg, v.mngr->logDir().string(), std::string{ctx});
        };

        o.pack_map(remoteAgentCount);
        for (AgentId agentId{}; agentId < remoteAgentCount; ++agentId) {
            o.pack(agentId);
            auto it = remoteResponsesPerAgent.find(agentId);
            if (it == remoteResponsesPerAgent.end()) {
                o.pack_array(0);
            } else {
                const auto& msgs = it->second;
                o.pack_array(msgs.size());
                for (const auto& msg : msgs) {
                    packNotice(o, msg);
                }
            }
        }

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
