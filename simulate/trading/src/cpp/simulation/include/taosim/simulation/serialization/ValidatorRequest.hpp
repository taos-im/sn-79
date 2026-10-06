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

class Simulation;

//-------------------------------------------------------------------------

namespace taosim::simulation::serialization
{

// A serializable view over a contiguous run of Simulations. Every book id on the wire is
// canonical and each Simulation knows its own numbering (Simulation::bookIdCanon, from its
// bookIdBase), so the view carries no arithmetic of its own: the single-config manager
// passes all of its blocks, the multi-asset request composes one of these per realization
// (a single-element span), and decanonize() on the way back reads the same bases.
struct ValidatorRequest
{
    std::span<const std::unique_ptr<Simulation>> simulations;
    std::filesystem::path logDir;
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

        const auto& representativeSimulation = v.simulations.front();
        const auto bookCount = static_cast<uint32_t>(ranges::accumulate(
            v.simulations
                | views::transform([](const auto& simulation) {
                    return simulation->exchange()->books().size();
                }),
            size_t{}));
        const auto remoteAgentCount = ranges::count_if(
            views::keys(representativeSimulation->exchange()->accounts()),
            [](AgentId agentId) {
                return agentId >= 0; 
            });

        o.pack_map(6);

        // Log directory.
        o.pack("logDir"s);
        o.pack(v.logDir.string());

        // Timestamp.
        o.pack("timestamp"s);
        o.pack(representativeSimulation->currentTimestamp());

        // Model.
        o.pack("model"s);
        o.pack("im"s);

        // Books.
        o.pack("books"s);
        o.pack_map(bookCount);
        for (const auto& simulation : v.simulations) {
            const auto exchange = simulation->exchange();
            for (const auto& book : exchange->books()) {
                const BookId bookIdCanon = simulation->bookIdCanon(book->id());

                o.pack(bookIdCanon);

                o.pack_map(6);

                o.pack("i"s);
                o.pack(bookIdCanon);

                o.pack("r"s);
                o.pack(exchange->clearingManager().feePolicy()->makerTakerRatio(book->id(), 0));

                // THE RATES THE ENGINE ACTUALLY APPLIES, alongside the ratio they derive from.
                // Sent rather than recomputed downstream: a consumer deriving these from a
                // curve of its own has no way to match the engine's parameters or getRates' zero-MTR
                // early return, and would display plausible rates the venue never charged. Same 'fs'/'mk'/'tk' shape a trade's fees already use.
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

            for (const auto& simulation : v.simulations) {
                const auto exchange = simulation->exchange();
                const auto& account = exchange->accounts().at(agentId);
                const auto feePolicy = exchange->clearingManager().feePolicy();
                for (const auto& book : exchange->books()) {
                    const BookId bookIdCanon = simulation->bookIdCanon(book->id());

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

        // Book ids are canonized at write time by handing each notice its source block's
        // offset; the buffered payload objects are never mutated.
        struct NoticeWithOffset
        {
            Message::Ptr msg;
            BookId bookIdOffset;
        };

        auto collectiveRemoteResponses = [&] {
            std::unordered_map<std::string, uint32_t> msgTypeToCount{
                { "RESPONSE_DISTRIBUTED_RESET_AGENT", 0 },
                { "ERROR_RESPONSE_DISTRIBUTED_RESET_AGENT", 0 }
            };
            std::set<std::string> lifecycleSeen;
            auto checkGlobalDuplicate = [&](Message::Ptr msg) -> bool {
                const auto payload = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
                if (payload == nullptr) {
                    // A LIFECYCLE NOTICE IS NOT A RESPONSE, AND DROPPING IT HERE BROKE THE ONE PATH
                    // THAT WAS SUPPOSED TO CARRY IT.
                    //
                    // This guard exists so the reset-dedup below can cast safely. It also silently
                    // removed every notice whose payload is not an agent response -- which is exactly
                    // EVENT_SIMULATION_START (StartSimulationPayload) and EVENT_SIMULATION_END
                    // (EmptyPayload). The consequence is that the fan-out further down, which goes to
                    // the trouble of naming those two payload types and pushing them to EVERY agent,
                    // could never receive one: it has been unreachable since it was written, and so
                    // has NoticePack's EVENT_SIMULATION_END branch.
                    //
                    // Without this, the published state carries an empty notice list for every
                    // agent and the end event reaches only the lifecycle HTTP, so an agent cannot be
                    // told a simulation started or ended by the route every other event takes.
                    //
                    // Let exactly the two payload types the fan-out handles
                    // through -- not everything, because packNotice throws on a payload that is
                    // neither a response nor one of these two.
                    const bool lifecycle =
                        std::dynamic_pointer_cast<StartSimulationPayload>(msg->payload) != nullptr
                        || std::dynamic_pointer_cast<EmptyPayload>(msg->payload) != nullptr;
                    if (!lifecycle) return false;
                    // ONE PER PUBLISH, NOT ONE PER REALIZATION. Every block's proxy queues its own
                    // copy of a global event, so a multi-book wrapper would hand each agent one
                    // identical start and end notice per realization. The agent dispatch calls
                    // onStart inside its notice loop, so that would be one onStart call per block
                    // for a single simulation starting. Deduplicated here, by the same counter this function
                    // already uses for reset notices, which is the thing it is named for.
                    return lifecycleSeen.emplace(msg->type).second;
                }
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
            std::vector<NoticeWithOffset> res;
            for (const auto& simulation : v.simulations) {
                // The notice's source realization offsets its local book ids by its
                // canonical base at pack time (NoticePack.hpp), so the buffered payload
                // objects are never mutated.
                const BookId bookIdOffset = simulation->bookIdBase();
                // A background without a DistributedProxyAgent is a purely-local
                // constituent: it publishes books/accounts but produces no notices.
                if (simulation->proxy() == nullptr) continue;
                for (const auto& msg : simulation->proxy()->messages()) {
                    if (!checkGlobalDuplicate(msg)) continue;
                    res.push_back({msg, bookIdOffset});
                }
                simulation->proxy()->clearMessages();
            }
            return res;
        }();
        // SORT ON WHAT THE AGENT IS TOLD, which for a scheduled lifecycle notice is its arrival
        // rather than the moment it was queued. See NoticePack.hpp: the end event is queued at the
        // simulation's start and arrives at its end, so ordering by occurrence put it first in the
        // final update's notice list. Ordinary notices are unaffected -- their occurrence and arrival
        // are the same timestamp -- so this only moves the two lifecycle events to where they belong.
        auto effectiveTimestamp = [](const Message::Ptr& msg) {
            const bool lifecycle =
                std::dynamic_pointer_cast<StartSimulationPayload>(msg->payload) != nullptr
                || std::dynamic_pointer_cast<EmptyPayload>(msg->payload) != nullptr;
            return lifecycle ? msg->arrival : msg->occurrence;
        };
        ranges::sort(
            collectiveRemoteResponses,
            [&](auto&& lhs, auto&& rhs) {
                const auto lt = effectiveTimestamp(lhs.msg);
                const auto rt = effectiveTimestamp(rhs.msg);
                if (lt != rt) {
                    return lt < rt;
                }
                return lhs.msg->arrival - lhs.msg->occurrence
                    > rhs.msg->arrival - rhs.msg->occurrence;
            });
        const auto remoteResponsesPerAgent = [&] {
            std::map<AgentId, std::vector<NoticeWithOffset>> res;
            for (const auto& notice : collectiveRemoteResponses) {
                if (std::dynamic_pointer_cast<StartSimulationPayload>(notice.msg->payload) != nullptr
                    || std::dynamic_pointer_cast<EmptyPayload>(notice.msg->payload) != nullptr) {
                    for (auto agentId : views::keys(representativeSimulation->exchange()->accounts())) {
                        if (agentId < 0) continue;
                        res[agentId].push_back(notice);
                    }
                    continue;
                }
                const auto pld =
                    std::dynamic_pointer_cast<DistributedAgentResponsePayload>(notice.msg->payload);
                if (pld == nullptr) {
                    throw std::runtime_error{fmt::format(
                        "{}: Failed to cast to DistributedAgentResponsePayload in 'remoteResponsesPerAgent'", ctx)};
                }
                res[pld->agentId].push_back(notice);
            }
            return res;
        }();

        auto packNotice = [&](auto& o, const NoticeWithOffset& notice) {
            // Body lives in NoticePack.hpp so the exchange mechanism serializes notices with the very
            // same code. See that header for why.
            taosim::simulation::serialization::packNotice(
                o, notice.msg, v.logDir.string(), std::string{ctx}, notice.bookIdOffset);
        };

        o.pack_map(remoteAgentCount);
        for (AgentId agentId{}; agentId < remoteAgentCount; ++agentId) {
            o.pack(agentId);
            auto it = remoteResponsesPerAgent.find(agentId);
            if (it == remoteResponsesPerAgent.end()) {
                o.pack_array(0);
            } else {
                const auto& notices = it->second;
                o.pack_array(notices.size());
                for (const auto& notice : notices) {
                    packNotice(o, notice);
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
