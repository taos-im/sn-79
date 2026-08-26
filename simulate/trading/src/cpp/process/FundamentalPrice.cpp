/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/process/FundamentalPrice.hpp>

#include <Simulation.hpp>

#include <algorithm>
#include <cmath>
#include <source_location>

//-------------------------------------------------------------------------

namespace taosim::process
{

//-------------------------------------------------------------------------

FundamentalPrice::FundamentalPrice(const FundamentalPriceDesc& desc) noexcept
    : m_simulation{desc.simulation},
      m_rng{&dynamic_cast<Simulation*>(desc.simulation)->rng()},
      m_bookId{desc.bookId},
      m_seedInterval{desc.seedInterval},
      m_gracePeriod{desc.gracePeriod},
      m_mu{desc.mu},
      m_sigma{desc.sigma},
      m_dt{desc.dt},
      m_gaussian{0.0, std::sqrt(desc.dt)},
      m_X0{desc.X0},
      m_L{desc.L},
      m_poisson{desc.lambda},
      m_jump{desc.muJump, desc.sigmaJump},
      m_hurst{desc.hurst},
      m_epsilon{desc.epsilon},
      m_interpolate{desc.interpolate},
      m_ownRng{desc.ownRng}
{
    m_updatePeriod = desc.proc.updatePeriod;
    m_state.value = m_X0;
    // diffusion-only log anchors start at ln(X0) (dJ == 0 at t = 0), so any
    // pre-first-seed read reveals exactly X0.
    m_state.logDiffPrev = m_state.logDiffCur = std::log(m_X0);
    // D4b: deterministic per-book seed for the private stream (only used when
    // m_ownRng — reseeded from the seed file / fallback path on every seed step).
    m_processRng.seed(0x5EEDBA5Eu + static_cast<unsigned>(desc.bookId));

    const auto sim = dynamic_cast<Simulation*>(m_simulation);
    m_sim = sim;  // cache for currentTimestamp() in value(), which is read-hot
    Timestamp N = sim->duration() / m_updatePeriod;
    const double dtH = std::pow(N, -m_hurst);
    m_state.X = Eigen::VectorXd::Zero(N + 2);
    m_state.V.resize(N + 2);
    // TODO seed
    m_fractionalGaussian = std::normal_distribution<double>{0.0, dtH};

    for (int i = 0; i < 2; i++) {
        m_state.V(i) = m_epsilon * m_fractionalGaussian(m_ownRng ? m_processRng : *m_rng);
    }

    m_state.X(0) = m_state.V(0);
    m_state.X(1) = m_L->row(1).head(2).dot(m_state.V.head(2));

    m_seedfile = (sim->logDir() / "fundamental_seed.csv").generic_string();
}

//-------------------------------------------------------------------------

void FundamentalPrice::update(Timestamp timestamp)
{
    if (m_values.empty()) {
        if (timestamp - m_state.lastSeedTime >= m_seedInterval) {
            int count = m_state.lastCount;
            uint64_t seed = 0;
            if ( fs::exists( m_seedfile ) ) {
                try {
                    std::vector<std::string> lines = taosim::util::getLastLines(m_seedfile, 2);
                    if (lines.size() >= 2) {
                        std::vector<std::string> line = taosim::util::split(lines[lines.size() - 2], ',');
                        if (line.size()== 2) {
                            count = std::stoi(line[0]);
                            seed = static_cast<uint64_t>(round(std::stof(line[1])*100)) + m_bookId*10;
                        } else {
                            fmt::println("FundamentalPrice::update : FAILED TO GET SEED FROM LINE - {}", lines[lines.size() - 2]);
                        }
                    } else {
                        fmt::println("FundamentalPrice::update : FAILED TO GET SEED FROM FILE - NO DATA ({} LINES READ)", lines.size());
                    }
                } catch (const std::exception& exc) {
                    fmt::println("FundamentalPrice::update : ERROR GETTING SEED FROM FILE - {}", exc.what());
                }
                if (count == m_state.lastCount) {
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_int_distribution<> distr(-50, 50);
                    seed = m_state.lastSeed + distr(gen);
                    if (timestamp >= m_gracePeriod) {
                        fmt::println("WARNING : Fundamental price seed not updated - using random seed.  Last Count {} | Count {} | Last Seed {} | Seed {}", m_state.lastCount, count, seed, m_state.lastSeed);
                    }
                }
            } else {
                if (timestamp >= m_gracePeriod) {
                    fmt::println("FundamentalPrice::update : NO SEED FILE PRESENT AT {}.  Using random seed.", m_seedfile);
                }
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> distr(10800000,11200000);
                seed = distr(gen);
            }
            // D4b: with ownRng the seed steers a PRIVATE stream; the shared agent
            // RNG is left alone (legacy reseeded it, restarting every agent's
            // randomness each seedInterval — see the desc field comment).
            std::mt19937& rng = m_ownRng ? m_processRng : *m_rng;
            if (m_ownRng) m_processRng.seed(seed); else m_rng->seed(seed);
            m_state.lastCount = count;
            m_state.lastSeed = seed;
            m_state.lastSeedTime = timestamp;
            m_state.t += m_dt;
            // Jump part
            m_state.dJ += m_poisson(rng) * m_jump(rng);
            //fBM
            int64_t step = timestamp/m_updatePeriod;
            cholesky_step(step);
            m_state.BH += m_state.X(step);
            const double fBM_comp =
                m_epsilon * m_state.BH - (0.5 * m_epsilon * m_epsilon * std::pow(m_state.t, 2 * m_hurst));
            // BM
            m_state.W += m_gaussian(rng);
            // pricing
            m_state.value = m_X0 * std::exp((m_mu - 0.5 * m_sigma * m_sigma) * m_state.t + m_sigma * m_state.W + fBM_comp + m_state.dJ);
            // shift the diffusion-only log anchors (jumps excluded, so the
            // reveal in valueAt() can apply dJ instantly while ramping the diffusion).
            // m_state.value keeps the exact seed value; the seed recursion above only
            // ever reads t/W/BH/dJ, so the reveal cannot perturb it either way — the
            // read-time design is preferred because it keeps the checkpointed value
            // honest and serves sub-interval reads exactly.
            m_state.logDiffPrev = m_state.logDiffCur;
            m_state.logDiffCur = std::log(m_state.value) - m_state.dJ;
        }
    }
    else {
        m_state.value = m_values.at(m_valueIdx);
        m_valueIdx = std::min(m_valueIdx + 1, m_values.size() - 1);
    }
    // The signal carries the latent seed value (like loggedValue()); it has no
    // subscribers in-tree, and the reveal is a read-time presentation, not state.
    m_valueSignal(m_state.value);
}

//-------------------------------------------------------------------------

// jump-preserving continuous reveal between seed anchors.
//
// The seed schedule, the RNG stream, dt, N and the Cholesky factor L are all untouched —
// only the READ is transformed. On [T_k, T_k + seedInterval) the reported value is
//
//     R = exp( (1-u)·logDiffPrev + u·logDiffCur + dJ_k ),   u = (now - T_k)/seedInterval,
//
// where logDiff_k = ln(value_k) - dJ_k is the diffusion-only log anchor. The DIFFUSION
// (GBM + fBm) ramps linearly in log space — killing the 30 s staircase discontinuities
// that bipower statistics misread as jumps — while the compound-Poisson jump dJ applies
// INSTANTLY at the seed instant, so genuine jumps still read as jumps at every sampling
// scale. At u=1 the reveal equals the seed value exactly; within an interval the path is
// continuous, and the only discontinuities are the drawn jumps themselves.
//
// Trade-off (deliberate): the diffusion component lags the latest seed by up to one
// seedInterval, because a causal continuous path cannot ramp toward an anchor it has not
// drawn yet (gradual information revelation, not look-ahead). Jumps are NOT lagged.
//
// m_interpolate == 0 (the default) is the legacy staircase: bit-identical to before.
double FundamentalPrice::value() const
{
    // Replay mode (m_values non-empty) is an externally supplied path — never interpolate
    // it. m_interpolate is checked first so a default-constructed instance (m_sim null,
    // m_interpolate 0) can never reach the dereference below.
    if (!m_interpolate || m_seedInterval == 0 || !m_values.empty()) {
        return m_state.value;
    }
    return valueAt(m_sim->currentTimestamp());
}

//-------------------------------------------------------------------------

double FundamentalPrice::valueAt(Timestamp now) const
{
    if (m_seedInterval == 0) {  // degenerate config: nothing to ramp over
        return m_state.value;
    }
    // Timestamp is unsigned: guard the subtraction rather than letting it wrap. At
    // now == lastSeedTime the ramp fraction is 0, so both branches agree.
    if (now <= m_state.lastSeedTime) {
        return std::exp(m_state.logDiffPrev + m_state.dJ);
    }
    const double frac = std::min(
        1.0,
        static_cast<double>(now - m_state.lastSeedTime) / static_cast<double>(m_seedInterval));
    const double logDiff =
        m_state.logDiffPrev + (m_state.logDiffCur - m_state.logDiffPrev) * frac;
    return std::exp(logDiff + m_state.dJ);
}

//-------------------------------------------------------------------------

void FundamentalPrice::cholesky_step(int64_t i)
{
    m_state.V(i + 1) = m_fractionalGaussian(m_ownRng ? m_processRng : *m_rng);
    m_state.X(i) = m_L->row(i).head(i + 1).dot(m_state.V.head(i + 1));
}

//-------------------------------------------------------------------------

std::unique_ptr<FundamentalPrice> FundamentalPrice::fromXML(
    taosim::simulation::ISimulation* simulation,
    pugi::xml_node node,
    uint64_t bookId,
    double X0,
    const Eigen::MatrixXd* L)
{
    static constexpr auto ctx = std::source_location::current().function_name();

    auto getNonNegativeFloatAttribute = [&](pugi::xml_node node, const char* name) {
        pugi::xml_attribute attr = node.attribute(name);
        if (double value = attr.as_double(); attr.empty() || value < 0.0) {
            throw std::invalid_argument(fmt::format(
                "{}: Attribute '{}' must be non-negative", ctx, name));
        } else {
            return value;
        }
    };

    const auto updatePeriod = node.attribute("updatePeriod").as_ullong(1);
    const auto sim = dynamic_cast<Simulation*>(simulation);
    const float dt = (float) updatePeriod / sim->duration();

    auto getNonNegativeUint64Attribute = [&](pugi::xml_node node, const char* name) {
        pugi::xml_attribute attr = node.attribute(name);
        if (uint64_t value = attr.as_ullong(); attr.empty() || value < 0.0) {
            throw std::invalid_argument(fmt::format(
                "{}: Attribute '{}' must be non-negative", ctx, name));
        } else {
            return value;
        }
    };
    const double hurst = node.attribute("Hurst").as_double(0.5);
    const double epsilon = node.attribute("epsilon").as_double(0.0);
    // 0 (default) = legacy staircase; 1 = jump-preserving reveal.
    const int interpolate = node.attribute("interpolate").as_int(0);
    // D4b: 1 = private RNG for this process (agents' shared stream left intact).
    const int ownRng = node.attribute("ownRng").as_int(0);

    // XML structure: <MultiBookExchangeAgent gracePeriod=...><Books><Processes><FundamentalPrice/>
    // Walk up 3 levels to find gracePeriod.
    const Timestamp gracePeriod =
        node.parent().parent().parent().attribute("gracePeriod").as_ullong();

    return std::make_unique<FundamentalPrice>(FundamentalPriceDesc{
        .simulation = simulation,
        .bookId = bookId,
        .seedInterval = getNonNegativeUint64Attribute(node, "seedInterval"),
        .X0 = X0,
        .mu = getNonNegativeFloatAttribute(node, "mu"),
        .sigma = getNonNegativeFloatAttribute(node, "sigma"),
        .dt = dt,
        .lambda = getNonNegativeFloatAttribute(node, "lambda"),
        .muJump = getNonNegativeFloatAttribute(node, "muJump"),
        .sigmaJump = getNonNegativeFloatAttribute(node, "sigmaJump"),
        .hurst = hurst,
        .epsilon = epsilon,
        .interpolate = interpolate,
        .ownRng = ownRng,
        .gracePeriod = gracePeriod,
        .proc = {
            .updatePeriod = updatePeriod
        },
        .L = L
    });
}

//-------------------------------------------------------------------------

}  // namespace taosim::process

//-------------------------------------------------------------------------