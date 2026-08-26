/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/checkpoint/SimulatorState.hpp>
#include <taosim/xml/helpers.hpp>
#include <taosim/checkpoint/serialization/accounting/AccountRegistry.hpp>
#include <taosim/checkpoint/serialization/agent/ALGOTraderAgent.hpp>
#include <taosim/checkpoint/serialization/agent/FuturesTraderAgent.hpp>
#include <taosim/checkpoint/serialization/agent/HighFrequencyTraderAgent.hpp>
#include <taosim/checkpoint/serialization/agent/NoiseTraderAgent.hpp>
#include <taosim/checkpoint/serialization/agent/RandomTraderAgent.hpp>
#include <taosim/checkpoint/serialization/agent/StylizedTraderAgent.hpp>
#include <taosim/checkpoint/serialization/book/Book.hpp>
#include <taosim/checkpoint/serialization/book/BookProcessManager.hpp>
#include <taosim/checkpoint/serialization/matching/ClearingManager.hpp>
#include <taosim/checkpoint/serialization/matching/ExchangeSignals.hpp>
#include <taosim/checkpoint/serialization/matching/SLTPContainer.hpp>
#include <taosim/event/serialization/L3RecordContainer.hpp>
#include <taosim/filesystem/utils.hpp>
#include <taosim/message/MessagePayload.hpp>
#include <taosim/message/MultiBookMessagePayloads.hpp>
#include <taosim/util/serialization/SubscriptionRegistry.hpp>
#include <taosim/serialization/msgpack/common.hpp>

//-------------------------------------------------------------------------

namespace taosim::checkpoint::serialization
{

//-------------------------------------------------------------------------

void packConfig(auto& o, const SimulatorState& v)
{
    const auto& reprSimu = v.mngr->simulations().front();

    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_string(reprSimu->configSv().data());
    pugi::xml_node node = xml::rootNode(doc);

    if (pugi::xml_attribute attr = node.attribute("id")) {
        attr.set_value(reprSimu->id().data());
    } else {
        node.append_attribute("id") = reprSimu->id().data();
    }

    const auto& time = reprSimu->time();

    if (pugi::xml_attribute attr = node.attribute("current")) {
        attr.set_value(time.current);
    } else {
        node.append_attribute("current") = time.current;
    }

    std::ostringstream oss;
    doc.print(oss, "");
    
    o.pack(oss.str());
}

//-------------------------------------------------------------------------

void packAgents(auto& o, const Simulation& simulation)
{
    using namespace taosim::agent;

    auto relevantAgents = simulation.agents()
        | views::filter([](auto&& agent) {
            return agent->name() != "EXCHANGE"
                && agent->name() != "DISTRIBUTED_PROXY_AGENT";
        })
        | views::transform([](auto&& agent) { return agent.get(); })
        | ranges::to<std::vector>;
    
    o.pack_map(relevantAgents.size());

    for (auto&& agent : relevantAgents) {
        o.pack(agent->name());

        if (auto a = dynamic_cast<const ALGOTraderAgent*>(agent)) {
            o.pack(*a);
        }
        else if (auto a = dynamic_cast<const FuturesTraderAgent*>(agent)) {
            o.pack(*a);
        }
        else if (auto a = dynamic_cast<const HighFrequencyTraderAgent*>(agent)) {
            o.pack(*a);
        }
        else if (auto a = dynamic_cast<const NoiseTraderAgent*>(agent)) {
            o.pack(*a);
        }
        else if (auto a = dynamic_cast<const RandomTraderAgent*>(agent)) {
            o.pack(*a);
        }
        else if (auto a = dynamic_cast<const StylizedTraderAgent*>(agent)) {
            o.pack(*a);
        }
        else {
            o.pack_nil();
        }
    }
}

//-------------------------------------------------------------------------

void packExchange(auto& o, const Simulation& simulation)
{
    const auto exch = simulation.exchange();

    // MUST match the number of key/value pairs packed below: an under-count makes
    // the trailing entries bleed into the PARENT block map, displacing its
    // messageQueue key — every checkpoint written that way fails to restore.
    o.pack_map(14);

    o.pack("accounts");
    o.pack(exch->accounts());

    o.pack("books");
    o.pack(exch->books());

    o.pack("signals");
    o.pack(exch->signals());

    o.pack("bookProcessManager");
    o.pack(exch->bookProcessManager());

    o.pack("clearingManager");
    o.pack(exch->clearingManager());

    o.pack("L3Record");
    o.pack(exch->L3Record());

    o.pack("marginCallCounter");
    o.pack(exch->marginCallCounter());

    o.pack("localMarketOrderSubs");
    o.pack(exch->localMarketOrderSubs());

    o.pack("localLimitOrderSubs");
    o.pack(exch->localLimitOrderSubs());

    o.pack("localTradeSubs");
    o.pack(exch->localTradeSubs());

    o.pack("localTradeByOrderSubs");
    o.pack(exch->localTradeByOrderSubs());

    o.pack("localOwnTradeSubs");
    o.pack(exch->localOwnTradeSubs());

    o.pack("bookTradeStats");
    o.pack(exch->bookTradeStats());

    o.pack("sltpContainer");
    o.pack(exch->sltpContainer());
}

//-------------------------------------------------------------------------

void packLogFileSizes(auto& o, const taosim::simulation::SimulationManager& simuMngr)
{
    const auto files = filesystem::collectMatchingPaths(
        simuMngr.logDir(),
        [](auto&& p) {
            return fs::is_regular_file(p)
                && std::regex_match(
                    p.filename().string(),
                    taosim::checkpoint::CheckpointManager::s_relevantLogFilePattern);
        });

    o.pack_map(files.size());

    for (const auto& file : files) {
        o.pack(file.filename().c_str());
        o.pack(fs::file_size(file));
    }
}

//-------------------------------------------------------------------------

void packCommon(auto& o, const taosim::simulation::SimulationManager* simuMngr)
{
    o.pack_map(2);

    o.pack("timestamp");
    o.pack(simuMngr->simulations().front()->currentTimestamp());

    o.pack("logFileSizes");
    packLogFileSizes(o, *simuMngr);
}

//-------------------------------------------------------------------------

void packBlock(auto& o, const Simulation& simulation)
{
    o.pack_map(3);

    o.pack("agents");
    packAgents(o, simulation);

    o.pack("exchange");
    packExchange(o, simulation);

    o.pack("messageQueue");
    o.pack(simulation.messageQueue());
}

//-------------------------------------------------------------------------

}  // namespace taosim::checkpoint::serialization

//-------------------------------------------------------------------------
