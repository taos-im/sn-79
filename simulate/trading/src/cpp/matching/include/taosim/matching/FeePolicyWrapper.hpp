/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/accounting/AccountRegistry.hpp>
#include <taosim/matching/FeePolicy.hpp>
#include <taosim/matching/TieredFeePolicy.hpp>
#include <taosim/matching/DynamicFeePolicy.hpp>
#include <taosim/matching/ZeroFeePolicy.hpp>

//-------------------------------------------------------------------------

namespace taosim::matching
{

//-------------------------------------------------------------------------

class FeePolicyWrapper
{
public:
    FeePolicyWrapper(
        std::unique_ptr<FeePolicy> feePolicy,
        accounting::AccountRegistry* accountRegistry) noexcept;

    [[nodiscard]] auto&& feePolicy(this auto&& self) noexcept { return self.m_feePolicy; }
    [[nodiscard]] auto&& agentBaseNameFeePolicies(this auto&& self) noexcept { return self.m_agentBaseNameFeePolicies; }

    auto&& operator[](this auto&& self, const std::string& agentBaseName)
    { 
        return self.m_agentBaseNameFeePolicies[agentBaseName];
    }

    auto&& operator[](this const auto& self, const std::string& agentBaseName)
    {
        auto it = self.m_agentBaseNameFeePolicies.find(agentBaseName);
        if (it != self.m_agentBaseNameFeePolicies.end()) return it->second;
        return self.m_feePolicy;
    }

    [[nodiscard]] Fees calculateFees(const TradeDesc& trade);
    [[nodiscard]] Fees getRates(BookId bookId, AgentId agentId) const;
    // The agent's base name has a policy of its own, so its rates may differ from the default policy's.
    [[nodiscard]] bool hasOwnPolicy(AgentId agentId) const;
    [[nodiscard]] decimal_t agentVolume(BookId bookId, AgentId agentId) const noexcept;
    [[nodiscard]] decimal_t makerTakerRatio(BookId bookId, AgentId agentId) const noexcept;
    [[nodiscard]] auto&& isTiered(this auto&& self) noexcept { return self.m_isTiered; }

    [[nodiscard]] bool contains(const std::string& agentBaseName) const noexcept;

    FeePolicy* defaultPolicy() noexcept { return m_feePolicy.get(); }
    void updateAgentsTiers(Timestamp time) noexcept;
    void updateHistory(Timestamp timestamp, BookId bookId, AgentId agentId, decimal_t volume, std::optional<bool> isAggressive = {});
    void resetHistory() noexcept;
    void resetHistory(const std::unordered_set<AgentId>& agentIds) noexcept;

private:
    [[nodiscard]] decltype(auto) policiesView() noexcept
    {
        return views::concat(
            views::values(m_agentBaseNameFeePolicies)
            | views::transform([](auto& feePolicy) { return feePolicy.get(); }),
            views::single(m_feePolicy.get()));
    }

    accounting::AccountRegistry* m_accountRegistry;
    std::map<std::string, std::unique_ptr<FeePolicy>> m_agentBaseNameFeePolicies;
    std::unique_ptr<FeePolicy> m_feePolicy;
    bool m_isTiered;
};

//-------------------------------------------------------------------------

}  // namespace taosim::matching

//-------------------------------------------------------------------------
