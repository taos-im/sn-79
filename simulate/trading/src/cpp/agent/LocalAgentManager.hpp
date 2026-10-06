/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/agent/DistributedProxyAgent.hpp>
#include <taosim/util/TransparentStringHash.hpp>
#include "Agent.hpp"
#include "LocalAgentRoster.hpp"
#include "MultiBookExchangeAgent.hpp"
#include "PythonAgent.hpp"

#include <span>
#include <string_view>
#include <unordered_map>

//-------------------------------------------------------------------------

class Simulation;

//-------------------------------------------------------------------------

class LocalAgentManager
{
public:
    LocalAgentManager(Simulation* simulation) noexcept;

    [[nodiscard]] auto begin() { return m_agents.begin(); }
    [[nodiscard]] auto end() { return m_agents.end(); }

    void createAgentsInstanced(
        pugi::xml_node node,
        std::function<void(pugi::xml_node)> creationCallback = {});

    [[nodiscard]] auto&& agents(this auto&& self) noexcept { return self.m_agents; }
    [[nodiscard]] const std::unique_ptr<LocalAgentRoster>& roster() const noexcept { return m_roster; }

    // O(1) lookup by exact agent name. Returns nullptr on miss.
    // Cache is built once after createAgentsInstanced finishes the sort+populate.
    [[nodiscard]] Agent* findByName(std::string_view name) const noexcept
    {
        // unordered_map<std::string, ...>::find supports heterogeneous lookup with
        // the transparent hash configured below.
        const auto it = m_byName.find(name);
        return it == m_byName.end() ? nullptr : it->second;
    }

private:
    template<std::derived_from<Agent> T>
    void createAgentInstanced(pugi::xml_node node);

    Simulation* m_simulation;
    // Invariant: sorted (kept for the few existing range-based callers).
    std::vector<std::unique_ptr<Agent>> m_agents;
    std::unique_ptr<LocalAgentRoster> m_roster;
    // O(1) by-name index, populated alongside the sort in createAgentsInstanced.
    std::unordered_map<std::string, Agent*, taosim::util::TransparentStringHash, std::equal_to<>>
        m_byName;
};

//-------------------------------------------------------------------------

template<>
void LocalAgentManager::createAgentInstanced<taosim::agent::DistributedProxyAgent>(pugi::xml_node node);

template<>
void LocalAgentManager::createAgentInstanced<MultiBookExchangeAgent>(pugi::xml_node node);

template<>
void LocalAgentManager::createAgentInstanced<PythonAgent>(pugi::xml_node node);

//-------------------------------------------------------------------------
