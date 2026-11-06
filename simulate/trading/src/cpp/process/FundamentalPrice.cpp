/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/process/FundamentalPrice.hpp>

#include <Simulation.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <source_location>
#include <stdexcept>
#include <utility>

//-------------------------------------------------------------------------

namespace taosim::process
{

//-------------------------------------------------------------------------

namespace
{

// One row of a seed file: `round,randomness` (the randomness as 64 hex characters, of which the first 8 bytes are
// read), with an optional third column ignored, or the older `count,price` (the price in cents).
std::optional<std::pair<int, int64_t>> parseSeedRow(const std::string& line)
{
    const auto fields = taosim::util::split(line, ',');
    if (fields.size() != 2 && fields.size() != 3) return std::nullopt;
    try {
        const auto& value = fields[1];
        if (value.size() == 64 && value.find_first_not_of("0123456789abcdefABCDEF") == std::string::npos) {
            return std::pair{std::stoi(fields[0]), static_cast<int64_t>(std::stoull(value.substr(0, 16), nullptr, 16))};
        }
        if (fields.size() != 2) return std::nullopt;
        return std::pair{
            std::stoi(fields[0]), static_cast<int64_t>(std::llround(std::stod(fields[1]) * 100.0))};
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

}  // namespace

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
    // L is stored as float; widen the row to double so the accumulation matches V.
    m_state.X(1) = m_L->row(1).head(2).cast<double>().dot(m_state.V.head(2));

    m_recordedSeeds = !desc.seedFile.empty();
    m_seedfile = desc.seedFile;
}

//-------------------------------------------------------------------------

void FundamentalPrice::update(Timestamp timestamp)
{
    if (m_values.empty()) {
        if (timestamp - m_state.lastSeedTime >= m_seedInterval) {
            const auto row = m_recordedSeeds ? std::optional{recordedSeed(timestamp)} : deliveredSeed(timestamp);
            uint64_t seed{};
            if (row) {
                seed = seedFor(row->value, m_bookId);
                m_state.lastCount = row->count;
            } else {
                seed = std::random_device{}();
                ++m_randomSeedSteps;
            }
            reseed(seed);
            std::mt19937& rng = m_ownRng ? m_processRng : *m_rng;
            m_state.lastSeed = seed;
            m_state.lastSeedTime = timestamp;
            m_state.t += m_dt;
            // Jump part, its two draws in a fixed order
            const int jumps = m_poisson(rng);
            const double jumpSize = m_jump(rng);
            m_state.dJ += jumps * jumpSize;
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

// The seed the validator delivered with its batch since this book's last seed step: a public randomness round
// published after the miners' reply deadline, so nobody holds it before replying. With none new the step draws a
// random seed, logged on the first such step after the grace period and every thousandth after.
std::optional<FundamentalPrice::SeedRow> FundamentalPrice::deliveredSeed(Timestamp timestamp)
{
    const auto* seed = m_sim && m_sim->stepSeed() ? &*m_sim->stepSeed() : nullptr;
    if (seed == nullptr || seed->generation == m_seedGeneration) {
        if (timestamp >= m_gracePeriod) {
            if (m_randomSeedStepsAfterGrace % 1000 == 0) {
                fmt::println(
                    "WARNING : no new fundamental seed delivered for book {} at {} - using a random seed ({} so far)",
                    m_bookId, timestamp, m_randomSeedStepsAfterGrace + 1);
            }
            ++m_randomSeedStepsAfterGrace;
        }
        return std::nullopt;
    }
    m_seedGeneration = seed->generation;
    return SeedRow{static_cast<int>(seed->round), static_cast<int64_t>(seed->word)};
}

//-------------------------------------------------------------------------

FundamentalPrice::SeedRow FundamentalPrice::recordedSeed(Timestamp timestamp)
{
    if (m_recorded.empty()) {
        std::ifstream ifs{m_seedfile};
        if (!ifs) {
            throw std::runtime_error{fmt::format(
                "FundamentalPrice: the recorded seed file {} cannot be read", m_seedfile)};
        }
        for (std::string line; std::getline(ifs, line);) {
            if (line.empty()) continue;
            const auto parsed = parseSeedRow(line);
            if (!parsed) {
                throw std::runtime_error{fmt::format(
                    "FundamentalPrice: row {} of the recorded seed file {} is neither round,randomness nor "
                    "count,price: {}",
                    m_recorded.size() + 1, m_seedfile, line)};
            }
            m_recorded.push_back({parsed->first, parsed->second});
        }
    }
    const uint64_t row = m_seedInterval == 0 ? 0 : timestamp / m_seedInterval;
    if (row == 0 || row > m_recorded.size()) {
        throw std::runtime_error{fmt::format(
            "FundamentalPrice: the recorded seed file {} has {} rows, and the seed step at {} needs row {}",
            m_seedfile, m_recorded.size(), timestamp, row)};
    }
    return m_recorded[row - 1];
}

//-------------------------------------------------------------------------

uint32_t FundamentalPrice::seedFor(int64_t value, uint64_t bookId) noexcept
{
    // The value and the book each pass through a full avalanche, so distinct books and distinct values
    // seed distinct streams.
    auto mix = [](uint64_t z) {
        z += 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    const uint64_t z = mix(mix(static_cast<uint64_t>(value)) ^ bookId);
    return static_cast<uint32_t>(z ^ (z >> 32));
}

//-------------------------------------------------------------------------

void FundamentalPrice::reseed(uint64_t seed)
{
    (m_ownRng ? m_processRng : *m_rng).seed(static_cast<std::mt19937::result_type>(seed));
    // A normal distribution keeps the second value of each pair it generates; started afresh here, every
    // draw of a seed step comes from that step's seed.
    m_gaussian.reset();
    m_fractionalGaussian.reset();
    m_jump.reset();
    m_poisson.reset();
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
    // The step's own innovation is drawn now, under its own seed; X(i) combines it with the earlier ones.
    m_state.V(i) = m_fractionalGaussian(m_ownRng ? m_processRng : *m_rng);
    m_state.X(i) = m_L->row(i).head(i + 1).cast<double>().dot(m_state.V.head(i + 1));
}

//-------------------------------------------------------------------------

std::unique_ptr<FundamentalPrice> FundamentalPrice::fromXML(
    taosim::simulation::ISimulation* simulation,
    pugi::xml_node node,
    uint64_t bookId,
    double X0,
    const Eigen::MatrixXf* L)
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

    // Required and strictly positive: the per-step process-update loop strides by this,
    // so a defaulted ~1ns period means ~1e9 updates per step (and an OOM-scale buffer
    // of process values) instead of a config error.
    const auto updatePeriod = node.attribute("updatePeriod").as_ullong();
    if (updatePeriod == 0) {
        throw std::invalid_argument{
            "FundamentalPrice::fromXML: attribute 'updatePeriod' must be present and > 0"};
    }
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
    // 1 = draw this process's randomness from a PRIVATE per-book generator instead of
    // reseeding the SHARED simulation RNG every seedInterval. Legacy (0) reseeded the shared
    // stream ~34,560 times per 12 h run from a ~400k-state pool, which restarted every agent's
    // randomness and made nominally independent books share draws. 1 is the default; 0 = legacy.
    // An element without the attribute is named at load, so that no file relies on the default.
    const pugi::xml_attribute ownRngAttr = node.attribute("ownRng");
    if (ownRngAttr.empty()) {
        fmt::println(
            "FundamentalPrice::fromXML : WARNING : book {} has no ownRng attribute; it draws from its own "
            "generator (ownRng=1, the default). Set ownRng explicitly.",
            bookId);
    }
    const int ownRng = ownRngAttr.as_int(1);

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
        .L = L,
        .seedFile = node.attribute("seedFile").as_string("")
    });
}

//-------------------------------------------------------------------------

}  // namespace taosim::process

//-------------------------------------------------------------------------