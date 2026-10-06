/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "CheckpointSerializable.hpp"
#include <taosim/accounting/Account.hpp>
#include <taosim/util/TransparentStringHash.hpp>
#include "JsonSerializable.hpp"
#include "common.hpp"

#include <boost/bimap.hpp>

#include <string_view>
#include <unordered_map>

//-------------------------------------------------------------------------

namespace taosim::accounting
{

//-------------------------------------------------------------------------

class AccountRegistry : public JsonSerializable
{
public:
    using Accounts = std::map<AgentId, Account>;
    using AgentIdBimap = boost::bimap<LocalAgentId, AgentId>;
    using AgentIdToBaseNameMap = std::map<AgentId, std::string>;

    [[nodiscard]] Account& at(const std::variant<AgentId, LocalAgentId>& agentId);
    [[nodiscard]] Account& operator[](const std::variant<AgentId, LocalAgentId>& agentId);

    [[nodiscard]] decltype(auto) begin(this auto&& self) { return self.m_accounts.begin(); }
    [[nodiscard]] decltype(auto) end(this auto&& self) { return self.m_accounts.end(); }

    void registerLocal(const LocalAgentId& agentId, std::optional<Account> account = {}) noexcept;
    void registerLocal(
        const LocalAgentId& agentId,
        const std::string& agentType,
        std::optional<Account> account = {}) noexcept;
    AgentId registerRemote(std::optional<Account> account = {}) noexcept;
    bool registerRemote(AgentId agentId, Account::Holdings holdings) noexcept;
    void registerJson(const rapidjson::Value& json);

    [[nodiscard]] bool contains(const std::variant<AgentId, LocalAgentId>& agentId) const;
    [[nodiscard]] AgentId getAgentId(const std::variant<AgentId, LocalAgentId>& agentId) const;

    // O(1) local-agent name → AgentId lookup. Mirrors m_idBimap.left.at() but on
    // an unordered_map with heterogeneous lookup; replaces ~14 string compares
    // per call (rb-tree at 30k+ agents) on hot order-placement / settlement
    // paths in MultiBookExchangeAgent + ClearingManager. Throws on miss to match
    // the .left.at() semantics callers already rely on.
    [[nodiscard]] AgentId lookupLocalAgentId(std::string_view name) const;
    [[nodiscard]] std::optional<std::reference_wrapper<const std::string>> getAgentBaseName(
        AgentId agentId) const noexcept;

    [[nodiscard]] auto&& agentTypeAccountTemplates(this auto&& self) noexcept
    {
        return self.m_agentTypeAccountTemplates;
    }

    void setAccountTemplate(std::function<Account()> factory) noexcept;
    void setAccountTemplate(const std::string& agentType, std::function<Account()> factory) noexcept;
    void reset(AgentId agentId);

    [[nodiscard]] auto&& localIdCounter(this auto&& self) noexcept { return self.m_localIdCounter; }
    [[nodiscard]] auto&& remoteIdCounter(this auto&& self) noexcept { return self.m_remoteIdCounter; }
    [[nodiscard]] auto&& accounts(this auto&& self) noexcept { return self.m_accounts; }
    [[nodiscard]] auto&& idBimap(this auto&& self) noexcept { return self.m_idBimap; }
    [[nodiscard]] auto&& agentIdToBaseName(this auto&& self) noexcept { return self.m_agentIdToBaseName; }

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

private:
    // Policy: Local agents have ID < 0, remote >= 0.
    AgentId m_localIdCounter{};
    AgentId m_remoteIdCounter{};

    Accounts m_accounts;
    std::function<Account()> m_accountTemplate;
    std::map<std::string, std::function<Account()>> m_agentTypeAccountTemplates;
    AgentIdBimap m_idBimap;
    AgentIdToBaseNameMap m_agentIdToBaseName;
    // Mirrors m_idBimap.left; kept in lock-step in registerLocal + registerJson.
    std::unordered_map<std::string, AgentId, util::TransparentStringHash, std::equal_to<>>
        m_idLookup;

    friend class Simulation;
};

//-------------------------------------------------------------------------

}  // namespace taosim::accounting

//-------------------------------------------------------------------------