/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/process/MagneticField.hpp>

#include <Simulation.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

//-------------------------------------------------------------------------

namespace taosim::process
{

//-------------------------------------------------------------------------

MagneticField::MagneticField(const MagneticFieldDesc& desc) noexcept 
 :  m_simulation{desc.simulation},
    m_rows{desc.sqrtNumAgents},
    m_numAgents{desc.sqrtNumAgents * desc.sqrtNumAgents}, 
    m_alpha{desc.alpha},
    m_beta{desc.beta},
    m_J{desc.interactionCoef}
{
    m_updatePeriod = desc.proc.updatePeriod;
    m_state.rng = RNG{desc.seed};

    init();
    
    const auto logFilepath = m_simulation->logDir() / fmt::format("MagneticField-{}.csv", desc.seed);
    m_logger = std::make_unique<MagneticFieldLogger>(logging::LoggerBaseDesc{
        .name = fmt::format("MagneticFieldLogger-{}", logFilepath.stem().c_str()),
        .filepath = logFilepath,
        .header = std::string{MagneticFieldLogger::s_header}
    });
}

//-------------------------------------------------------------------------

MagneticField::Position MagneticField::resolvePosition(uint32_t id) const noexcept
{
    return {
        .x = static_cast<int32_t>(id % m_rows),
        .y = static_cast<int32_t>(id / m_rows)
    };
}

//-------------------------------------------------------------------------

DurationComp MagneticField::getDurationComp(const std::string& agentBaseName)
{
    return m_state.agentBaseNameToDuration[agentBaseName];
}

//-------------------------------------------------------------------------

int32_t MagneticField::asyncUpdate(uint32_t pos)
{
    const auto [i, j] = resolvePosition(pos);
    const auto h_local = localSum(i, j); 
    const auto h_global = m_alpha * atField(i, j) * std::abs(avgMagnetism()); 
    const auto local_field = h_local - h_global; 
    const auto probability = 1.0f / (1.0f + std::exp(-2.0f * m_beta * local_field));
    const auto decision = bool2spin(std::bernoulli_distribution{probability}(m_state.rng));
    m_state.field[pos] = (decision > 0) - (decision < 0);
    return m_state.field[pos];
}

//-------------------------------------------------------------------------

void MagneticField::setValAt(uint32_t pos, int32_t val)
{
    m_state.field[pos] = val;
    asyncUpdate(pos);
}

//-------------------------------------------------------------------------

void MagneticField::insertDurationComp(const std::string& agentBaseName, DurationComp event)
{
    auto& s = m_state.agentBaseNameToStats[agentBaseName];
    ++s.n;
    s.delaySum += event.delay;
    s.delaySumSq += static_cast<double>(event.delay) * event.delay;
    s.delayMin = std::min(s.delayMin, static_cast<double>(event.delay));
    s.delayMax = std::max(s.delayMax, static_cast<double>(event.delay));
    s.psiSum += event.psi;
    s.psiSumSq += static_cast<double>(event.psi) * event.psi;
    m_state.agentBaseNameToDuration[agentBaseName] = std::move(event);
}

void MagneticField::emitDiagnostics(const std::string& agentBaseName, uint32_t bookIdCanon) const
{
    const auto it = m_state.agentBaseNameToStats.find(agentBaseName);
    if (it == m_state.agentBaseNameToStats.end() || it->second.n == 0) return;
    const auto& s = it->second;
    const double n = static_cast<double>(s.n);
    const double delayMean = s.delaySum / n;
    const double delayStd = std::sqrt(std::max(0.0, s.delaySumSq / n - delayMean * delayMean));
    const double psiMean = s.psiSum / n;
    const double psiStd = std::sqrt(std::max(0.0, s.psiSumSq / n - psiMean * psiMean));
    fmt::print(
        "AGENTDIAG {{\"agent\":\"{}\",\"book\":{},\"n\":{},"
        "\"delay_mean\":{},\"delay_std\":{},\"delay_min\":{},\"delay_max\":{},"
        "\"psi_mean\":{},\"psi_std\":{}}}\n",
        agentBaseName, bookIdCanon, s.n,
        delayMean, delayStd, s.delayMin, s.delayMax, psiMean, psiStd);
    std::fflush(stdout);
}


//-------------------------------------------------------------------------

void MagneticField::logState(Timestamp timestamp, uint32_t lastPosition)
{
    if (m_logger) {
        m_logger->log(timestamp, m_state.magnetism, m_state.field, lastPosition);
    }
}

//-------------------------------------------------------------------------

void MagneticField::update(Timestamp timestamp)
{            
    ++m_state.lastCount;
    std::vector<double> weights(m_numAgents, 1.0);
    std::discrete_distribution<uint32_t> dist(weights.begin(), weights.end());
    uint32_t num_updates = m_numAgents - 1; 
    for (int32_t j = 0; j < num_updates; ++j) {
        int32_t pos = dist(m_state.rng);
        asyncUpdate(pos);
    }
    m_state.magnetism = updateMagnetism();
    m_state.value = avgMagnetism();
}

//-------------------------------------------------------------------------

std::unique_ptr<MagneticField> MagneticField::fromXML(
    pugi::xml_node node, taosim::simulation::ISimulation* simulation, uint64_t seed)
{
    return std::make_unique<MagneticField>(MagneticFieldDesc{
        .simulation = simulation,
        .sqrtNumAgents = node.attribute("numRows").as_uint(32),
        .alpha = node.attribute("alpha").as_float(6.0f), 
        .beta = node.attribute("beta").as_float(0.667f),
        .interactionCoef = node.attribute("interactionCoef").as_float(1.0f),
        .seed = seed,
        .proc = {
            .updatePeriod = node.attribute("updatePeriod").as_ullong(10'000'000'000),
        }
    });
}

//-------------------------------------------------------------------------

void MagneticField::init()
{
    m_state.field = views::iota(0u, m_numAgents)
        | views::transform([this](auto) {
            static std::bernoulli_distribution s_bernoulli{0.5};
            return bool2spin(s_bernoulli(m_state.rng));
        })
        | ranges::to<decltype(m_state.field)>;
    m_state.magnetism = updateMagnetism();
    m_state.value = avgMagnetism();
}

//-------------------------------------------------------------------------

float MagneticField::updateMagnetism() 
{
    const auto magnetism = totalMagnetism();
    m_state.magnetismReturn = (magnetism - m_state.magnetism) / m_numAgents;
    return magnetism;
}

//-------------------------------------------------------------------------

float MagneticField::localSum(int32_t i, int32_t j)
{
    static constexpr std::array<Position, 8> s_directions{{
        { 1,  0}, {-1,  0}, { 0,  1}, { 0, -1},
        { 1,  1}, { 1, -1}, {-1,  1}, {-1, -1} 
    }};

    auto validNeighborsView = s_directions
        | views::filter([&](auto&& pos) {
            const int32_t ni = i + pos.x;
            const int32_t nj = j + pos.y;
            return ni >= 0 && ni < m_rows && nj >= 0 && nj < m_rows;
        });

    return ranges::accumulate(
        validNeighborsView
        | std::views::transform([&](auto&& dir) {
            return atField(i + dir.x, j + dir.y) * m_J;
        }),
        0.0
    );
}

//-------------------------------------------------------------------------

}  // namespace taosim::process

//-------------------------------------------------------------------------
