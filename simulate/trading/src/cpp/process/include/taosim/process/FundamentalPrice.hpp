/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/simulation/ISimulation.hpp>
#include <taosim/process/Process.hpp>
#include "common.hpp"

#include <Eigen/Dense>
#include <optional>
#include <string>
#include <vector>
#include <pugixml.hpp>

//-------------------------------------------------------------------------

class Simulation;  // needed for currentTimestamp() in the interpolating value().

//-------------------------------------------------------------------------

namespace taosim::process
{

//-------------------------------------------------------------------------

struct FundamentalPriceDesc
{
    simulation::ISimulation* simulation;
    uint64_t bookId;
    uint64_t seedInterval;
    double X0;
    double mu;
    double sigma;
    double dt;
    double lambda;
    double muJump;
    double sigmaJump;
    double hurst{0.5};
    double epsilon{0.0};
    int interpolate{0};        // 0 = staircase (legacy); 1 = jump-preserving reveal
    // 1 = draw this process's randomness from a PRIVATE
    // generator (seeded per book), instead of reseeding the SHARED simulation RNG
    // every seedInterval — which restarted every agent's random stream ~34,560
    // times per 12h run from a ~400k-state pool (~1,500 collisions/run) and made
    // the "independent" books share randomness. 1 is the default; 0 = legacy.
    int ownRng{1};
    Timestamp gracePeriod{};   // suppress seed-update warnings before this t
    ProcessDesc proc;
    // float, not double: the Cholesky factor is large and read-only here, and SIMU003TASK004's
    // multi-asset work halved its footprint. Accumulations widen to double at the two dot
    // products in cholesky_step().
    const Eigen::MatrixXf* L{};
    // A recorded seed sequence, one `count,price` row per seed step. When set, the step at simulated time t
    // takes row floor(t / seedInterval) of the file (the first row at one seedInterval) instead of the live
    // service's newest row, and a step past the last row stops the run. With rngSeed and ownRng=1 the same
    // config and file reproduce a run offline. Empty: the live file at the run root.
    std::string seedFile{};
};

struct FundamentalPriceState
{
    double dJ{};
    double t{};
    double W{};
    Eigen::VectorXd X;
    Eigen::VectorXd V;
    double BH{};
    int lastCount{};
    uint64_t lastSeed{};
    Timestamp lastSeedTime{};
    double value{};
    // diffusion-only log anchors for the jump-preserving reveal,
    // logDiff_k = ln(value_k) - dJ_k. Checkpoint-serialized alongside the rest of the
    // state; on restore from an old checkpoint without them they are rebuilt from
    // value/dJ (see checkpoint/serialization/process/FundamentalPrice.hpp).
    double logDiffPrev{};
    double logDiffCur{};
};

//-------------------------------------------------------------------------

class FundamentalPrice : public Process
{
public:
    FundamentalPrice() noexcept = default;
    FundamentalPrice(const FundamentalPriceDesc& desc) noexcept;

    [[nodiscard]] auto&& state(this auto&& self) noexcept { return self.m_state; }
    [[nodiscard]] auto&& rng(this auto&& self) noexcept { return self.m_rng; }
    [[nodiscard]] const std::string& seedFile() const noexcept { return m_seedfile; }
    [[nodiscard]] int ownRng() const noexcept { return m_ownRng; }

    virtual void update(Timestamp timestamp) override;
    // interpolated at READ time (see the .cpp). m_state.value always holds the
    // exact seed-step value; agents see valueAt(currentTimestamp()).
    virtual double value() const override;
    // The reveal math at an explicit time — value() delegates here; separate so it is
    // unit-testable without a live Simulation and usable with per-row timestamps.
    [[nodiscard]] double valueAt(Timestamp now) const;
    // The process CSV / replay source records the latent seed path, NOT the reveal:
    // keeps the file's meaning identical whether or not interpolation is on.
    virtual double loggedValue() const override { return m_state.value; }

    // The generator seed of a step whose seed price is `priceCents`, for book `bookId`.
    [[nodiscard]] static uint32_t seedFor(int64_t priceCents, uint64_t bookId) noexcept;
    // Seeds the stream this process draws from (its own with ownRng, else the shared one); a checkpoint
    // restore calls it with the step's recorded seed.
    void reseed(uint64_t seed);

    [[nodiscard]] static std::unique_ptr<FundamentalPrice> fromXML(
        simulation::ISimulation* simulation,
        pugi::xml_node node,
        uint64_t bookId,
        double X0,
        const Eigen::MatrixXf* L);

private:
    struct SeedRow
    {
        int count;
        int64_t cents;
    };

    void cholesky_step(int64_t i);
    [[nodiscard]] std::optional<SeedRow> liveSeed(Timestamp timestamp) const;
    [[nodiscard]] SeedRow recordedSeed(Timestamp timestamp);

    simulation::ISimulation* m_simulation;
    Simulation* m_sim{};  // cached concrete pointer (currentTimestamp() at read time)
    std::mt19937* m_rng;
    uint64_t m_bookId;
    uint64_t m_seedInterval;
    Timestamp m_gracePeriod{};
    std::string m_seedfile;
    bool m_recordedSeeds{};
    std::vector<SeedRow> m_recorded;
    double m_X0, m_mu, m_sigma, m_dt;
    FundamentalPriceState m_state;
    const Eigen::MatrixXf* m_L;
    std::normal_distribution<double> m_gaussian;
    double m_epsilon;
    double m_hurst;
    std::normal_distribution<double> m_fractionalGaussian;
    std::normal_distribution<double> m_jump;
    std::poisson_distribution<int> m_poisson;
    // 0 => legacy staircase; 1 => jump-preserving reveal (diffusion ramps
    // linearly in log space between seed anchors, Poisson jumps apply instantly at the
    // seed instant). Anchors live in m_state (logDiffPrev/Cur) and are checkpointed.
    int m_interpolate{0};
    // D4b: private generator used instead of the shared sim RNG when m_ownRng == 1.
    int m_ownRng{1};
    std::mt19937 m_processRng;
};

//-------------------------------------------------------------------------

}  // namespace taosim::process

//-------------------------------------------------------------------------