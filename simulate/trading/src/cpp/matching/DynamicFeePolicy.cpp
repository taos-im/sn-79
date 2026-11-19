/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "Simulation.hpp"

#include <taosim/matching/DynamicFeePolicy.hpp>
#include "common.hpp"

//-------------------------------------------------------------------------

namespace taosim::matching
{

//-------------------------------------------------------------------------

taosim::matching::DynamicFeePolicy::DynamicFeePolicy(const DFPolicyDesc &desc)
    : m_simulation{desc.simulation},
      m_historySlots{desc.historySlots},
      m_makerFee{desc.makerFee},
      m_takerFee{desc.takerFee},
      m_feeMath{desc.maxMakerRate, desc.maxTakerRate, desc.targetMTR, desc.shapeMakerFee, desc.shapeMakerRebate}
{}

//-------------------------------------------------------------------------

Fees DynamicFeePolicy::calculateFees(const TradeDesc &tradeDesc) const
{
    [[maybe_unused]] const auto [bookId, restingAgentId, aggressingAgentId, trade] = tradeDesc;
    const auto volumeWeightedPrice = trade->volume() * trade->price();
    return {
        .maker = getRates(bookId, restingAgentId).maker * volumeWeightedPrice,
        .taker = getRates(bookId, aggressingAgentId).taker * volumeWeightedPrice
    };
}

//-------------------------------------------------------------------------

Fees DynamicFeePolicy::getRates(BookId bookId, AgentId agentId) const
{    
    return exchangeRates(bookId, agentId) + dynamicRates(bookId, agentId);
}

//-------------------------------------------------------------------------

decimal_t DynamicFeePolicy::makerTakerRatio(BookId bookId) const
{
    auto it = m_totalVolumesPrev.find(bookId);
    if (it != m_totalVolumesPrev.end()){
        if (m_volumes.at(bookId).size() < m_historySlots)
            return m_feeMath.targetMTR;
        if (it->second.aggressive + it->second.passive == 0_dec){
            return m_feeMath.targetMTR;
        }
        return it->second.passive / (it->second.passive + it->second.aggressive);
    }
    return m_feeMath.targetMTR;
} 

//-------------------------------------------------------------------------

Fees DynamicFeePolicy::dynamicRates(BookId bookId, AgentId agentId) const noexcept
{
    decimal_t x = makerTakerRatio(bookId);

    if (x == 0_dec) return {};

    return {
        .maker = m_feeMath.calculate(x),
        .taker = -m_feeMath.calculate(x)
    };
}

//-------------------------------------------------------------------------

Fees DynamicFeePolicy::exchangeRates(BookId bookId, AgentId agentId) const noexcept
{
    return {
        .maker = m_makerFee,
        .taker = m_takerFee
    };
}

//-------------------------------------------------------------------------

void DynamicFeePolicy::updateHistory(
    Timestamp timestamp, BookId bookId, AgentId agentId, decimal_t volume, std::optional<bool> isAggressive)
{
    if (agentId < 0) return;

    if (!isAggressive.has_value()){
        throw std::runtime_error(fmt::format(
            "Book {}, agent {}: Dynamic fee policy must define if the agent is aggressive or not", bookId, agentId
        ));
    }

    // Each book keeps its own history: a slot per distinct time at which that book traded, its own running total
    // over those slots, the total as it stood when its newest slot opened, and its own last-update time.
    auto& slots = m_volumes.try_emplace(bookId, m_historySlots).first->second;
    auto& total = m_totalVolumes[bookId];
    const auto last = m_lastUpdate.find(bookId);
    if (last == m_lastUpdate.end() || last->second != timestamp || slots.empty()) {
        if (last != m_lastUpdate.end()) {
            m_totalVolumesPrev[bookId] = total;
        }
        if (slots.full()) {
            const auto& oldest = slots.front();
            total.aggressive -= oldest.aggressive;
            total.passive -= oldest.passive;
        }
        slots.push_back(Volumes{});
    }
    if (isAggressive.value()) {
        slots.back().aggressive += volume;
        total.aggressive += volume;
    } else {
        slots.back().passive += volume;
        total.passive += volume;
    }
    m_lastUpdate[bookId] = timestamp;
}

//-------------------------------------------------------------------------

void DynamicFeePolicy::resetHistory() noexcept
{
    m_volumes.clear();
    m_totalVolumes.clear();
    m_totalVolumesPrev.clear();
    m_lastUpdate.clear();
}

//-------------------------------------------------------------------------

void DynamicFeePolicy::resetHistory(const std::unordered_set<AgentId>& agentIds) noexcept
{}

//-------------------------------------------------------------------------

std::unique_ptr<DynamicFeePolicy> DynamicFeePolicy::fromXML(pugi::xml_node node, Simulation *simulation)
{
    static constexpr auto ctx = std::source_location::current().function_name();

    auto getAttr = [](pugi::xml_node node, const char* name) {
        if (pugi::xml_attribute attr = node.attribute(name)) {
            return attr;
        }
        throw std::invalid_argument{fmt::format(
            "{}: Missing required argument '{}'", ctx, name)};
    };

    // The dynamic part moves a rate between the sides and leaves their sum alone, so the configured pair bounds it.
    const auto makerFee = checkFeeRate(getAttr(node, "makerFee").as_double());
    const auto takerFee = checkFeeRate(getAttr(node, "takerFee").as_double());
    checkFeeSum(makerFee, takerFee, ctx);

    return std::make_unique<DynamicFeePolicy>(DFPolicyDesc{
        .simulation = simulation,
        .historySlots = getAttr(node, "historySlots").as_int(),
        .makerFee = makerFee,
        .takerFee = takerFee,
        .maxMakerRate = util::double2decimal(getAttr(node, "maxMakerRate").as_double()),
        .maxTakerRate = util::double2decimal(getAttr(node, "maxTakerRate").as_double()),
        .targetMTR = util::double2decimal(getAttr(node, "targetMTR").as_double()),
        .shapeMakerFee = util::double2decimal(getAttr(node, "shapeMakerFee").as_double()),
        .shapeMakerRebate = util::double2decimal(getAttr(node, "shapeMakerRebate").as_double())
    });
}

//-------------------------------------------------------------------------

}  // namespace taosim::matching

//-------------------------------------------------------------------------