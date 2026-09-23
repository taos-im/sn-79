/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/process/MagneticFieldLogger.hpp>
#include <taosim/process/Process.hpp>
#include <taosim/process/RNG.hpp>
#include <taosim/simulation/ISimulation.hpp>
#include <common.hpp>
#include <GBMValuationModel.hpp>

#include <boost/circular_buffer.hpp>
#include <pugixml.hpp>

#include <limits>

//-------------------------------------------------------------------------

class Simulation;

//-------------------------------------------------------------------------

namespace taosim::process
{

//-------------------------------------------------------------------------

struct MagneticFieldDesc
{
    simulation::ISimulation* simulation;
    uint32_t sqrtNumAgents;
    float alpha;
    float beta;
    float interactionCoef;
    uint64_t seed;
    ProcessDesc proc;
};

struct DurationComp
{
    float delay, psi;
};

struct DurationStats
{
    uint64_t n{};
    double delaySum{}, delaySumSq{};
    double delayMin{std::numeric_limits<double>::infinity()};
    double delayMax{-std::numeric_limits<double>::infinity()};
    double psiSum{}, psiSumSq{};
};

struct MagneticFieldState
{
    RNG rng;
    double logReturn{};
    float magnetism{};
    float magnetismReturn{};
    std::map<std::string, DurationComp> agentBaseNameToDuration;
    std::map<std::string, DurationStats> agentBaseNameToStats;
    std::vector<int32_t> field;
    uint64_t lastCount{};
    double value{};
};

//-------------------------------------------------------------------------

class MagneticField : public Process
{
public:
    struct Position
    {
        int32_t x, y;
    };

    MagneticField() noexcept = default;
    explicit MagneticField(const MagneticFieldDesc& desc) noexcept;

    [[nodiscard]] auto&& state(this auto&& self) noexcept { return self.m_state; }

    [[nodiscard]] uint32_t rows() const noexcept { return m_rows; }
    [[nodiscard]] uint32_t numAgents() const noexcept { return m_numAgents; }
    [[nodiscard]] float magnetism() const noexcept { return m_state.magnetism; }
    [[nodiscard]] float magnetismReturn() const noexcept { return m_state.magnetismReturn; }
    [[nodiscard]] float avgMagnetism() const noexcept { return m_state.magnetism / m_numAgents; }
    [[nodiscard]] int32_t signAt(uint32_t id) const noexcept { return m_state.field.at(id); }   

    [[nodiscard]] Position resolvePosition(uint32_t id) const noexcept;
    [[nodiscard]] DurationComp getDurationComp(const std::string& agentBaseName);
    int32_t asyncUpdate(uint32_t pos);

    void setValAt(uint32_t pos, int32_t val);
    void insertDurationComp(const std::string& name, DurationComp event);
    void emitDiagnostics(const std::string& agentBaseName, uint32_t bookIdCanon) const;
    void logState(Timestamp timestamp, uint32_t lastPosition = 0);

    virtual void update(Timestamp timestamp) override;
    virtual double value() const override { return m_state.value; }
    virtual uint64_t count() const override { return m_state.lastCount; }

    [[nodiscard]] static std::unique_ptr<MagneticField> fromXML(
        pugi::xml_node node, simulation::ISimulation* simulation, uint64_t seed);

private:
    void init();
    float updateMagnetism();
    [[nodiscard]] float localSum(int32_t x, int32_t y);
    
    [[nodiscard]] float totalMagnetism() const noexcept { return ranges::accumulate(m_state.field, 0.0f); }
    [[nodiscard]] int32_t bool2spin(bool val) const noexcept { return val * 2 - 1; }
    [[nodiscard]] int32_t position(int32_t x, int32_t y) const { return x + y * m_rows; }
    [[nodiscard]] int32_t atField(int32_t x, int32_t y) const { return m_state.field.at(position(x, y)); }

    taosim::simulation::ISimulation* m_simulation;
    uint32_t m_rows;
    uint32_t m_numAgents;
    float m_alpha;
    float m_beta;
    float m_J;
    std::unique_ptr<MagneticFieldLogger> m_logger;
    MagneticFieldState m_state;
};

//-------------------------------------------------------------------------

}  // namespace taosim::process

//-------------------------------------------------------------------------