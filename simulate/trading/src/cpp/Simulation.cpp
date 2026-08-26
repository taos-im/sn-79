/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "Simulation.hpp"

#include <taosim/simulation/SimulationException.hpp>
#include "GBMValuationModel.hpp"
#include "util.hpp"

#include <boost/algorithm/string/regex.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <date/date.h>
#include <date/tz.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <source_location>
#include <stdexcept>

//-------------------------------------------------------------------------

Simulation::Simulation() noexcept
    : IMessageable{this, "SIMULATION"},
      m_localAgentManager{std::make_unique<LocalAgentManager>(this)}
{}

//-------------------------------------------------------------------------

Simulation::Simulation(
    uint32_t blockIdx,
    uint32_t blockDim,
    const fs::path& baseLogDir,
    const taosim::simulation::SharedResources* sharedResources,
    bool replayMode,
    taosim::replay::ReplayDesc replayDesc)
    : IMessageable{this, "SIMULATION"},
      m_blockIdx{blockIdx},
      m_blockDim{blockDim},
      m_baseLogDir{baseLogDir},
      m_localAgentManager{std::make_unique<LocalAgentManager>(this)},
      m_replayMode{replayMode},
      m_replayDesc{replayDesc},
      m_sharedResources{sharedResources}
{}

//-------------------------------------------------------------------------

void Simulation::dispatchMessage(
    Timestamp occurrence,
    Timestamp delay,
    const std::string& source,
    const std::string& target,
    const std::string& type,
    MessagePayload::Ptr payload) const
{
    queueMessage(
        Message::create(occurrence, occurrence + delay, source, target, type, payload));
}

//-------------------------------------------------------------------------

void Simulation::dispatchGenericMessage(
    Timestamp occurrence,
    Timestamp delay,
    const std::string& source,
    const std::string& target,
    const std::string& type,
    std::map<std::string, std::string> payload) const
{
    queueMessage(Message::create(
        occurrence,
        occurrence + delay,
        source,
        target,
        type,
        MessagePayload::create<GenericPayload>(std::move(payload))));
}

//-------------------------------------------------------------------------

void Simulation::queueMessage(const Message::Ptr& msg) const
{
    m_messageQueue.push(msg);
}

//-------------------------------------------------------------------------

void Simulation::simulate()
{
    if (m_state == taosim::simulation::SimulationState::STOPPED) return;
    else if (m_state == taosim::simulation::SimulationState::INACTIVE) start();

    while (m_time.current < m_time.start + m_time.duration) {
        step();
        m_exchange->L3Record().clear();
    }

    stop();
}

//-------------------------------------------------------------------------

taosim::accounting::Account& Simulation::account(const LocalAgentId& id) const noexcept
{
    return m_exchange->account(id);
}

//-------------------------------------------------------------------------

Timestamp Simulation::currentTimestamp() const noexcept
{
    return m_time.current;
}

//-------------------------------------------------------------------------

Timestamp Simulation::duration() const noexcept
{
    return m_time.duration;
}

//-------------------------------------------------------------------------

taosim::simulation::SimulationSignals& Simulation::signals() const noexcept
{
    return m_signals;
}

//-------------------------------------------------------------------------

std::mt19937& Simulation::rng() const noexcept
{
    return m_rng;
};

//-------------------------------------------------------------------------

void Simulation::receiveMessage(Message::Ptr msg)
{
}

//-------------------------------------------------------------------------

void Simulation::configure(const pugi::xml_node& node)
{
    m_config2 = taosim::simulation::SimulationConfig::fromXML(node);

    pugi::xml_attribute attr;

    if (attr = node.attribute("start"); attr.empty()) {
        throw std::invalid_argument(fmt::format(
            "{}: missing required attribute 'start'",
            std::source_location::current().function_name()));
    }
    m_time.start = attr.as_ullong();

    if (attr = node.attribute("duration"); attr.empty()) {
        throw std::invalid_argument(fmt::format(
            "{}: missing required attribute 'duration'",
            std::source_location::current().function_name()));
    }
    m_time.duration = attr.as_ullong();

    m_time.step = node.attribute("step").as_ullong(1);
    m_time.current = node.attribute("current").as_ullong(m_time.start);

    // An explicit `rngSeed` attribute makes runs reproducible; without one the legacy
    // non-deterministic random_device stands. The block index is mixed
    // in so parallel blocks configured from the same XML don't share a stream
    // (splitmix-style scramble rather than +blockIdx, which would make adjacent
    // seeds collide across blocks).
    if (auto seedAttr = node.attribute("rngSeed"); !seedAttr.empty()) {
        uint64_t s = seedAttr.as_ullong() ^ (0x9E3779B97F4A7C15ULL * (m_blockIdx + 1));
        s ^= s >> 30; s *= 0xBF58476D1CE4E5B9ULL;
        s ^= s >> 27; s *= 0x94D049BB133111EBULL;
        s ^= s >> 31;
        m_rng = std::mt19937{static_cast<std::mt19937::result_type>(s)};
    } else {
        m_rng = std::mt19937{std::random_device{}()};
    }

    m_config = [node] -> std::string {
        std::ostringstream oss;
        node.print(oss);
        return oss.str();
    }();

    m_debug = node.attribute("debug").as_bool();

    m_error = node.attribute("error").as_bool();

    m_logWindow = [&] -> Timestamp {
        static constexpr const char* attrName = "logWindow";
        const auto logWindow = node.attribute(attrName).as_ullong();
        using namespace taosim::simulation;
        if (!(logWindow == 0ul || kLogWindowMin <= logWindow && logWindow <= kLogWindowMax)) {
            throw std::runtime_error{fmt::format(
                "{}: '{}' must be either empty (or 0), or in [{}, {}], was {}",
                std::source_location::current().function_name(),
                attrName,
                kLogWindowMin, kLogWindowMax,
                node.attribute(attrName).as_string())};
        }
        return logWindow;
    }();

    // NOTE: Ordering important!
    configureLogging(node);
    configureAgents(node);
}

//-------------------------------------------------------------------------

void Simulation::configureAgents(pugi::xml_node node)
{
    static constexpr auto ctx = std::source_location::current().function_name();

    static const std::set<std::string> specialAgents{
        "DISTRIBUTED_PROXY_AGENT",
        //"EXCHANGE",
        "LOGGER_TRADES"
    };

    pugi::xml_node agentsNode;

    if (agentsNode = node.child("Agents"); !agentsNode) {
        throw std::invalid_argument{fmt::format(
            "{}: missing required child 'Agents'", ctx)};
    }

    m_localAgentManager->createAgentsInstanced(
        agentsNode,
        [&](pugi::xml_node agentNode) {
            if (specialAgents.contains(agentNode.attribute("name").as_string())) return;
            if (m_exchange == nullptr) {
                throw std::runtime_error{fmt::format("{}: m_exchange == nullptr!", ctx)};
            }
            [&] {
                const std::string agentType = agentNode.name();
                auto& accounts = m_exchange->accounts();
                if (m_exchange->accounts().agentTypeAccountTemplates().contains(agentType)) return;
                if (pugi::xml_node balancesNode = agentNode.child("Balances")) {
                    auto doc = std::make_shared<pugi::xml_document>();
                    doc->append_copy(balancesNode);
                    m_exchange->accounts().setAccountTemplate(
                        agentType,
                        [=, this] -> taosim::accounting::Account {
                            const auto& params = m_exchange->config().parameters();
                            if (m_exchange->sharedQuoteBalances()) {
                                return taosim::accounting::Account{
                                    static_cast<uint32_t>(m_exchange->books().size()),
                                    taosim::accounting::Balances::fromXML(
                                        doc->child("Balances"),
                                        taosim::accounting::RoundParams{
                                            .baseDecimals = params.baseIncrementDecimals,
                                            .quoteDecimals = params.quoteIncrementDecimals
                                        },
                                        &m_rng)
                                    };
                            }
                            else {
                                return taosim::accounting::Account(
                                    ranges::views::iota(0u, m_exchange->books().size())
                                    | ranges::views::transform([&](auto) {
                                        return taosim::accounting::Balances::fromXML(
                                            doc->child("Balances"),
                                            taosim::accounting::RoundParams{
                                                .baseDecimals = params.baseIncrementDecimals,
                                                .quoteDecimals = params.quoteIncrementDecimals
                                            },
                                            &m_rng);
                                    })
                                    | ranges::to<std::vector>
                                );
                            }
                        });
                }
            }();
            [&] {
                if (pugi::xml_node feePolicyNode = agentNode.child("FeePolicy")) {
                    auto feePolicy = m_exchange->clearingManager().feePolicy();
                    const auto agentBaseName = agentNode.attribute("name").as_string();
                    if (feePolicy->contains(agentBaseName)) return;

                    if (std::string(agentNode.child("FeePolicy").attribute("type").as_string()) == "tiered") {
                        if (!feePolicy->isTiered()){
                            throw std::runtime_error(fmt::format(
                                "{}'s fee policy type must be the same as default", std::string(agentBaseName)));
                        }
                        (*feePolicy)[agentBaseName] =
                            taosim::matching::TieredFeePolicy::fromXML(feePolicyNode, this);
                        logDebug("TIERED FEE POLICY - {}", agentBaseName);
                        int c = 0;
                        if (auto* tiered = dynamic_cast<TieredFeePolicy*>((*feePolicy)[agentBaseName].get())) {
                            for (auto& tier : tiered->tiers()) {
                                logDebug("TIER {} : VOL >= {} | MAKER {} TAKER {}", c, 
                                    tier.volumeRequired, 
                                    tier.makerFeeRate, tier.takerFeeRate
                                );
                                c++;
                            }
                        }
                    } else {
                        if (feePolicy->isTiered()) {
                            if (auto* tiered = dynamic_cast<TieredFeePolicy*>(feePolicy->defaultPolicy())) {
                                (*feePolicy)[agentBaseName] = std::make_unique<TieredFeePolicy>(*tiered);
                                logDebug("DEFAULT TIERED FEE POLICY - {}", agentBaseName);
                            } else {
                                throw std::runtime_error("Default policy is not TieredFeePolicy as expected");
                            }
                        } 
                    }
                }
            }();
        });

    static constexpr std::array<std::pair<std::string_view, std::string_view>, 2> kSpecialAgents{{
        {"EXCHANGE", "MultiBookExchangeAgent"},
        {"DISTRIBUTED_PROXY_AGENT", "DistributedProxyAgent"}
    }};
    for (const auto& [name, nodeName] : kSpecialAgents) {
        auto it = ranges::find_if(
            *m_localAgentManager, [&](const auto& agent) { return agent->name() == name; });
        if (it == m_localAgentManager->end()) {
            throw std::invalid_argument{fmt::format(
                "{}: missing required agent node '{}'", ctx, nodeName)};
        }
    }

    for (const auto& agent : m_localAgentManager->agents()) {
        if (specialAgents.contains(agent->name())) continue;
        m_exchange->accounts().registerLocal(agent->name(), agent->type());
    }

    m_signals.agentsCreated();
}

//-------------------------------------------------------------------------

const std::valarray<double>& Simulation::getOrComputeGbmPath(
    uint64_t seed, double S0, double mu, double sigma, uint32_t N)
{
    auto key = std::make_tuple(seed, S0, mu, sigma, N);
    if (auto it = m_gbmPathCache.find(key); it != m_gbmPathCache.end()) {
        return it->second;
    }
    GBMValuationModel<double> gbm{S0, mu, sigma, seed};
    auto [it, _] = m_gbmPathCache.emplace(key, gbm.generatePriceSeries(1, N));
    return it->second;
}

//-------------------------------------------------------------------------

void Simulation::configureLogging(pugi::xml_node node)
{
    m_logDir = m_baseLogDir;
}

//-------------------------------------------------------------------------

void Simulation::deliverMessage(const Message::Ptr& msg)
{
    for (const auto& target : msg->targets) {
        if (target == "*") {
            receiveMessage(msg);
            for (const auto& agent : m_localAgentManager->agents()) {
                agent->receiveMessage(msg);
            }
        }
        else if (target == "EXCHANGE") {
            m_exchange->receiveMessage(msg);
        }
        else if (target == name()) {
            receiveMessage(msg);
        }
        else if (target.back() == '*') {
            const auto prefix = target.substr(0, target.size() - 1);
            auto lb = std::lower_bound(
                m_localAgentManager->begin(),
                m_localAgentManager->end(),
                prefix,
                [](const auto& agent, const auto& needle) {
                    const auto& haystack = agent->name();
                    return haystack.find(needle);
                });
            auto ub = std::upper_bound(
                m_localAgentManager->begin(),
                m_localAgentManager->end(),
                prefix,
                [](const auto& needle, const auto& agent) {
                    const auto& haystack = agent->name();
                    return haystack.find(needle);
                });
            std::for_each(lb, ub, [&msg](const auto& agent) { agent->receiveMessage(msg); });
        }
        else {
            // O(1) hash lookup (was O(log N) lower_bound — hot path with
            // millions of dispatches per tick across 30k+ agents).
            Agent* const agentPtr = m_localAgentManager->findByName(target);
            if (agentPtr == nullptr) {
                // A message addressed to a name no agent answers to is DROPPED. Say so, once per name:
                // this was silent, and a whole class of defect looks like "the component never produced
                // the message" from every other surface. `continue`, not `return`: a miss on one target
                // must not also drop the message for the targets after it.
                static std::set<std::string> reported;
                if (reported.insert(target).second) {
                    fmt::println(
                        "[WARN] message to unknown target '{}' dropped (type {}); no agent answers to "
                        "that name", target, msg->type);
                }
                continue;
            }
            agentPtr->receiveMessage(msg);
        }
    }
}

//-------------------------------------------------------------------------

void Simulation::start()
{
    if (!m_replayMode) {
        dispatchMessage(
            m_time.start,
            0,
            "SIMULATION",
            "*",
            "EVENT_SIMULATION_START",
            MessagePayload::create<StartSimulationPayload>(logDir().generic_string()));
        dispatchMessage(
            m_time.start,
            m_time.duration - 1,
            "SIMULATION",
            "*",
            "EVENT_SIMULATION_END",
            MessagePayload::create<EmptyPayload>());
    }
    else if (!m_replayDesc.replacedAgents.empty()) {
        auto replacedAgentNames = m_localAgentManager->agents()
            | views::filter([this](auto&& agent) {
                const std::regex pat{fmt::format(
                    "^({})_(\\d+)$", fmt::join(m_replayDesc.replacedAgents, "|"))};
                return std::regex_match(agent->name(), pat);
            })
            | views::transform([](auto&& agent) {
                return agent->name();
            });
        dispatchMessage(
            m_time.start,
            0,
            "SIMULATION",
            fmt::format("{}", fmt::join(replacedAgentNames, "|")),
            "EVENT_SIMULATION_START",
            MessagePayload::create<StartSimulationPayload>(logDir().generic_string()));
        dispatchMessage(
            m_time.start,
            m_time.duration - 1,
            "SIMULATION",
            fmt::format("{}", fmt::join(replacedAgentNames, "|")),
            "EVENT_SIMULATION_END",
            MessagePayload::create<EmptyPayload>());
    }

    m_state = taosim::simulation::SimulationState::STARTED;
    m_signals.start();
}

//-------------------------------------------------------------------------

void Simulation::step()
{
    const Timestamp cutoff = m_time.current + m_time.step;

    m_exchange->clearingManager().updateFeeTiers(cutoff);        
    m_exchange->checkMarginCall();

    auto loopCondition = [&] -> bool {
        return !m_messageQueue.empty()
            && m_messageQueue.top()->arrival < cutoff;
    };

    while (loopCondition()) {
        Message::Ptr msg = m_messageQueue.top();
        m_messageQueue.pop();
        updateTime(msg->arrival);
        deliverMessage(msg);
    }

    updateTime(std::max(m_time.current, cutoff));
    m_signals.step();
}

//-------------------------------------------------------------------------

void Simulation::clearFilledOrders() noexcept
{
    for (auto& book : m_exchange->books()) {
        book->clearFilledOrders();
    }
}

//-------------------------------------------------------------------------

void Simulation::stop()
{
    m_state = taosim::simulation::SimulationState::STOPPED;
    m_signals.stop();
}

//-------------------------------------------------------------------------

bool Simulation::isReplacedAgent(const std::string& name) const noexcept
{
    auto name2BaseName = [](auto&& name) {
        std::string res = name;
        boost::algorithm::erase_regex(res, boost::regex("(_\\d+)$"));
        return res;
    };
    return m_replayDesc.replacedAgents.contains(name2BaseName(name));
}

//-------------------------------------------------------------------------

bool Simulation::isInitAgent(const std::string& name) const noexcept
{
    auto name2BaseName = [](auto&& name) {
        std::string res = name;
        boost::algorithm::erase_regex(res, boost::regex("(_\\d+)$"));
        return res;
    };
    return name2BaseName(name) == "INITIALIZATION_AGENT";
}

//-------------------------------------------------------------------------
