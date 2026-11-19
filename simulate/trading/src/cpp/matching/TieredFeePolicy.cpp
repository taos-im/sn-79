/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <algorithm>
#include "Simulation.hpp"
#include <taosim/matching/TieredFeePolicy.hpp>

#include "common.hpp"

//-------------------------------------------------------------------------

namespace taosim::matching
{

//-------------------------------------------------------------------------
    
TieredFeePolicy::TieredFeePolicy(const TFPolicyDesc& desc)
    : m_simulation{desc.simulation},
      m_historySlots{desc.historySlots},
      m_slotPeriod{desc.slotPeriod},
      m_tiers{desc.tiers}
{
    static constexpr auto ctx = std::source_location::current().function_name();
    ranges::sort(
        m_tiers,
        [](const Tier& lhs, const Tier& rhs) {
            if (lhs.volumeRequired == rhs.volumeRequired) {
                throw std::invalid_argument{fmt::format(
                    "{}: Tiers must have different required volumes!", ctx)};
            }
            return lhs.volumeRequired < rhs.volumeRequired;
        });
    updateAgentsTiers();
}

//-------------------------------------------------------------------------

Fees TieredFeePolicy::calculateFees(const TradeDesc& tradeDesc) const
{
    [[maybe_unused]] const auto [bookId, restingAgentId, aggressingAgentId, trade] = tradeDesc;
    const auto volumeWeightedPrice = trade->volume() * trade->price();
    return {
        .maker = getRates(bookId, restingAgentId).maker * volumeWeightedPrice,
        .taker = getRates(bookId, aggressingAgentId).taker * volumeWeightedPrice
    };
}

//-------------------------------------------------------------------------

void TieredFeePolicy::updateAgentsTiers() noexcept
{
    TierIdx idx;
    for (auto& [agentId, bookVolumes]: m_agentVolumes){
        for (auto& [bookId, volumes]: bookVolumes){
            auto initialTier = findTierForAgent(bookId, agentId);
            const decimal_t totalVolume = ranges::accumulate(volumes, 0_dec);
            idx = -1;
            for (const auto& tier : m_tiers) {
                if (totalVolume < tier.volumeRequired)
                    break;
                idx ++;
            }
            m_agentTiers[agentId][bookId] = std::max(idx, 0);

            auto currentTier = findTierForAgent(bookId, agentId);
            if (currentTier.volumeRequired != initialTier.volumeRequired) {
                m_simulation->logDebug("{} | AGENT #{} BOOK {} : VOL {} | FEE TIER UPDATED FROM [{},{},{}] -> [{},{},{}]",
                    m_simulation->currentTimestamp(), agentId, m_simulation->bookIdCanon(bookId), totalVolume,
                    initialTier.volumeRequired, initialTier.makerFeeRate, initialTier.takerFeeRate,
                    currentTier.volumeRequired, currentTier.makerFeeRate, currentTier.takerFeeRate
                );
            }

            std::move(volumes.begin() + 1, volumes.end(), volumes.begin());
            volumes.back() = {};
        }
    }
}

//-------------------------------------------------------------------------

void TieredFeePolicy::updateHistory(Timestamp timestamp, BookId bookId, AgentId agentId, decimal_t volume, std::optional<bool> isAggressive)
{
    auto itAgent = m_agentVolumes.find(agentId);
    if (itAgent != m_agentVolumes.end()){
        auto it = itAgent->second.find(bookId);
        if (it != itAgent->second.end()) {
            it->second[m_historySlots - 1] += volume;
        } else {
            itAgent->second[bookId] = std::vector<decimal_t>(m_historySlots, 0_dec);
            itAgent->second[bookId][m_historySlots - 1] += volume;
        }
    } else {
        m_agentVolumes[agentId][bookId] = std::vector<decimal_t>(m_historySlots, 0_dec);
        m_agentVolumes[agentId][bookId][m_historySlots - 1] += volume;
    }
}

//-------------------------------------------------------------------------

void TieredFeePolicy::resetHistory() noexcept
{
    for (auto& [agentId, bookTiers]: m_agentTiers) {
        for (auto& [bookId, tierIdx]: bookTiers) {
            ranges::fill(m_agentVolumes[agentId][bookId], 0_dec);
            tierIdx = 0;
        }
    }
}

//-------------------------------------------------------------------------

void TieredFeePolicy::resetHistory(const std::unordered_set<AgentId>& agentIds) noexcept
{
    ranges::for_each(
        m_agentTiers
        | views::filter([&](const auto& item) { return agentIds.contains(item.first); }),
        [this](auto& item) {
            auto& [agentId, bookTiers] = item;
            for (auto& [bookId, tierIdx]: bookTiers) {
                ranges::fill(m_agentVolumes[agentId][bookId], 0_dec);
                tierIdx = 0;
            }
        });
}

//-------------------------------------------------------------------------

Fees TieredFeePolicy::getRates(BookId bookId, AgentId agentId) const
{
    const auto& tier = findTierForAgent(bookId, agentId);
    return { 
        .maker = tier.makerFeeRate,
        .taker = tier.takerFeeRate
    };
}

//-------------------------------------------------------------------------

std::unique_ptr<TieredFeePolicy> TieredFeePolicy::fromXML(pugi::xml_node node, Simulation* simulation) 
{
    static constexpr auto ctx = std::source_location::current().function_name();

    auto getAttr = [](pugi::xml_node node, const char* name) {
        if (pugi::xml_attribute attr = node.attribute(name)) {
            return attr;
        }
        throw std::invalid_argument{fmt::format(
            "{}: Missing required argument '{}'", ctx, name)};
    };

    std::vector<Tier> parsedTiers;
    for (pugi::xml_node tierNode : node.children("Tier")) {
        parsedTiers.push_back({
            .volumeRequired =
                util::double2decimal(getAttr(tierNode, "volumeRequired").as_double()),
            .makerFeeRate = checkFeeRate(getAttr(tierNode, "makerFee").as_double()),
            .takerFeeRate = checkFeeRate(getAttr(tierNode, "takerFee").as_double())
        });
    }

    // A maker in one tier trades with a taker in any other, so the lowest maker rate meets the lowest taker rate.
    if (!parsedTiers.empty()) {
        checkFeeSum(
            std::ranges::min(parsedTiers, {}, &Tier::makerFeeRate).makerFeeRate,
            std::ranges::min(parsedTiers, {}, &Tier::takerFeeRate).takerFeeRate,
            ctx);
    }

    return std::make_unique<TieredFeePolicy>(TFPolicyDesc{
        .simulation = simulation,
        .historySlots = getAttr(node, "historySlots").as_int(),
        .slotPeriod = getAttr(node, "slotPeriod").as_ullong(),
        .tiers = std::move(parsedTiers)
    });
}

//-------------------------------------------------------------------------

const Tier& TieredFeePolicy::findTierForVolume(decimal_t volume) const noexcept
{
    TierIdx idx = -1;
    for (const auto& tier : m_tiers) {
        if (volume < tier.volumeRequired)
            break;
        idx ++;
    }
    return m_tiers.at(std::max(idx, 0));
}

//-------------------------------------------------------------------------

const Tier& TieredFeePolicy::findTierForAgent(BookId bookId, AgentId agentId) const noexcept
{
    auto itAgent = m_agentTiers.find(agentId);
    if (itAgent != m_agentTiers.end()){
        auto it = itAgent->second.find(bookId);
        if (it != itAgent->second.end()){
            const auto& tier = m_tiers.at(it->second);
            return tier;
        }
    }
    return m_tiers.front();
}

//-------------------------------------------------------------------------

}  // namespace taosim::matching

//-------------------------------------------------------------------------
