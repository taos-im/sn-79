/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/matching/FeePolicyWrapper.hpp>

#include <mutex>

//-------------------------------------------------------------------------

namespace taosim::matching
{

//-------------------------------------------------------------------------

FeePolicyWrapper::FeePolicyWrapper(
    std::unique_ptr<FeePolicy> feePolicy,
    accounting::AccountRegistry* accountRegistry) noexcept
    : m_feePolicy{std::move(feePolicy)},
      m_accountRegistry{accountRegistry}
{
    m_isTiered = false;
    if (dynamic_cast<TieredFeePolicy*>(m_feePolicy.get()))
        m_isTiered = true;
}

//-------------------------------------------------------------------------

Fees FeePolicyWrapper::calculateFees(const TradeDesc& tradeDesc)
{
    const auto [bookId, restingAgentId, aggressingAgentId, trade] = tradeDesc;
    const auto volumeWeightedPrice = trade->volume() * trade->price();
    return {
        .maker = getRates(bookId, restingAgentId).maker * volumeWeightedPrice,
        .taker = getRates(bookId, aggressingAgentId).taker * volumeWeightedPrice
    };
}

//-------------------------------------------------------------------------

Fees FeePolicyWrapper::getRates(BookId bookId, AgentId agentId) const
{
    const auto agentBaseName = m_accountRegistry->getAgentBaseName(agentId);
    if (agentBaseName.has_value()) {
        auto it = m_agentBaseNameFeePolicies.find(agentBaseName.value());
        if (it != m_agentBaseNameFeePolicies.end()) {
            return it->second->getRates(bookId, agentId);
        }
    }
    return m_feePolicy->getRates(bookId, agentId);
}

//-------------------------------------------------------------------------

bool FeePolicyWrapper::hasOwnPolicy(AgentId agentId) const
{
    const auto agentBaseName = m_accountRegistry->getAgentBaseName(agentId);
    return agentBaseName.has_value() && m_agentBaseNameFeePolicies.contains(agentBaseName.value());
}

//-------------------------------------------------------------------------

decimal_t FeePolicyWrapper::agentVolume(BookId bookId, AgentId agentId) const noexcept
{
    if (!isTiered()) return {};

    const auto agentBaseName = m_accountRegistry->getAgentBaseName(agentId);
    if (agentBaseName.has_value()) {
        auto agentFeePolicyIt = m_agentBaseNameFeePolicies.find(agentBaseName.value().get());
        if (agentFeePolicyIt != m_agentBaseNameFeePolicies.end()) {
            const auto* agentFeePolicy = dynamic_cast<TieredFeePolicy*>(agentFeePolicyIt->second.get());
            auto agentIdIt = agentFeePolicy->agentVolumes().find(agentId);
            if (agentIdIt != agentFeePolicy->agentVolumes().end()) {
                const auto& bookIdToVolumeHistory = agentIdIt->second;
                auto bookIdIt = bookIdToVolumeHistory.find(bookId);
                if (bookIdIt != bookIdToVolumeHistory.end()) {
                    return ranges::accumulate(bookIdIt->second, 0_dec);
                }
            }
        }
    }
    const auto* tired = dynamic_cast<TieredFeePolicy*>(m_feePolicy.get());
    auto agentIdIt = tired->agentVolumes().find(agentId);
    if (agentIdIt == tired->agentVolumes().end()) return 0_dec;
    const auto& bookIdToVolumeHistory = agentIdIt->second;
    auto bookIdIt = bookIdToVolumeHistory.find(bookId);
    if (bookIdIt == bookIdToVolumeHistory.end()) return 0_dec;
    return ranges::accumulate(bookIdIt->second, 0_dec);
}

//-------------------------------------------------------------------------

decimal_t FeePolicyWrapper::makerTakerRatio(BookId bookId, AgentId agentId) const noexcept
{
    if (isTiered()) { return {}; }
    else if (dynamic_cast<ZeroFeePolicy*>(m_feePolicy.get())) { return {}; }

    const auto* dynamic = dynamic_cast<DynamicFeePolicy*>(m_feePolicy.get());
    return dynamic->makerTakerRatio(bookId);
}

//-------------------------------------------------------------------------

bool FeePolicyWrapper::contains(const std::string& agentBaseName) const noexcept
{
    return m_agentBaseNameFeePolicies.contains(agentBaseName);
}

//-------------------------------------------------------------------------

void FeePolicyWrapper::updateAgentsTiers(Timestamp time) noexcept
{
    if (!isTiered()) return;

    ranges::for_each(
        policiesView()
        | views::filter([time](auto feePolicy) {
            auto* tiered = dynamic_cast<TieredFeePolicy*>(const_cast<FeePolicy*>(feePolicy));
            return tiered && time % tiered->slotPeriod() == 0;
        }),
        [time](auto feePolicy) {
            auto* tiered = dynamic_cast<TieredFeePolicy*>(const_cast<FeePolicy*>(feePolicy));
            if (tiered)
                tiered->updateAgentsTiers();
        });
}

//-------------------------------------------------------------------------

void FeePolicyWrapper::updateHistory(Timestamp timestamp, BookId bookId, AgentId agentId, decimal_t volume, std::optional<bool> isAggressive)
{
    const auto agentBaseName = m_accountRegistry->getAgentBaseName(agentId);
    if (agentBaseName.has_value()) {
        auto it = m_agentBaseNameFeePolicies.find(agentBaseName.value());
        if (it != m_agentBaseNameFeePolicies.end()) {
            return it->second->updateHistory(timestamp, bookId, agentId, volume, isAggressive);
        }
    }
    return m_feePolicy->updateHistory(timestamp, bookId, agentId, volume, isAggressive);
}

//-------------------------------------------------------------------------

void FeePolicyWrapper::resetHistory() noexcept
{
    for (auto feePolicy : policiesView()) {
        feePolicy->resetHistory();
    }
}

//-------------------------------------------------------------------------

void FeePolicyWrapper::resetHistory(const std::unordered_set<AgentId>& agentIds) noexcept
{
    m_feePolicy->resetHistory(agentIds);

    std::unordered_map<std::string, std::unordered_set<AgentId>> categorizedAgents;

    for (const auto& agentId : agentIds) {
        const auto agentBaseName = m_accountRegistry->getAgentBaseName(agentId);
        if (agentBaseName.has_value()) {
            const auto& baseName = agentBaseName->get();
            categorizedAgents[baseName].insert(agentId);
        }
    }

    for (auto& [baseName, idsForBase] : categorizedAgents) {
        auto it = m_agentBaseNameFeePolicies.find(baseName);
        if (it != m_agentBaseNameFeePolicies.end() && it->second) {
            it->second->resetHistory(idsForBase);
        }
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::matching

//-------------------------------------------------------------------------
