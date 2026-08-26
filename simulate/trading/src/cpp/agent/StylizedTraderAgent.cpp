/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/agent/StylizedTraderAgent.hpp>

#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/message/MessagePayload.hpp>
#include "DistributionFactory.hpp"
#include "RayleighDistribution.hpp"
#include "Simulation.hpp"
#include <taosim/util/RootFinding.hpp>

#include <boost/algorithm/string/regex.hpp>
#include <boost/random.hpp>
#include <boost/random/laplace_distribution.hpp>

//-------------------------------------------------------------------------

namespace br = boost::random;

//-------------------------------------------------------------------------

namespace taosim::agent
{

//-------------------------------------------------------------------------

[[nodiscard]] static double investmentPosition(
    double price,
    double forecast,
    double variance,
    double base,
    double quote,
    double risk,
    double constant)
{
    // constant is a subsistence wealth level, so it belongs inside the risk division.
    return std::log(forecast / price) / (variance * price)
        * ((base * price + quote + constant) / risk);
};

// Returns 0 when no non-crossing price exists; the caller must skip the order.
[[nodiscard]] static double slideToRest(
    double price, OrderDirection direction, const TopLevel& topLevel, double tick)
{
    if (direction == OrderDirection::BUY) {
        if (topLevel.ask > 0.0 && price >= topLevel.ask) {
            const double slid = topLevel.ask - tick;
            return slid > 0.0 ? std::round(slid / tick) * tick : 0.0;
        }
    }
    else {
        if (topLevel.bid > 0.0 && price <= topLevel.bid) {
            const double slid = topLevel.bid + tick;
            return std::round(slid / tick) * tick;
        }
    }
    return price;
}

//-------------------------------------------------------------------------

StylizedTraderAgent::StylizedTraderAgent(Simulation* simulation) noexcept
    : Agent{simulation}
{}

//-------------------------------------------------------------------------

void StylizedTraderAgent::configure(const pugi::xml_node& node)
{
    Agent::configure(node);

    m_rng = &simulation()->rng();

    pugi::xml_attribute attr;
    static constexpr auto ctx = std::source_location::current().function_name();

    if (attr = node.attribute("exchange"); attr.empty()) {
        throw std::invalid_argument(fmt::format(
            "{}: missing required attribute 'exchange'", ctx));
    }
    m_exchange = attr.as_string();

    if (simulation()->exchange() == nullptr) {
        throw std::runtime_error(fmt::format(
            "{}: exchange must be configured a priori", ctx));
    }
    m_bookCount = simulation()->exchange()->books().size();

    if (attr = node.attribute("sigmaF"); attr.empty() || attr.as_double() < 0.0f) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'sigmaF' should have a value of at least 0.0f", ctx));
    }
    const double sigmaF = attr.as_double();
    if (attr = node.attribute("sigmaC"); attr.empty() || attr.as_double() < 0.0f) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'sigmaC' should have a value of at least 0.0f", ctx));
    }
    const double sigmaC = attr.as_double();
    if (attr = node.attribute("sigmaN"); attr.empty() || attr.as_double() < 0.0f) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'sigmaN' should have a value of at least 0.0f", ctx));
    }
    const double sigmaN = attr.as_double();
    m_weight = {
        .F = std::abs(br::laplace_distribution{sigmaF, sigmaF}(*m_rng)),
        .C = std::abs(br::laplace_distribution{sigmaC, sigmaC}(*m_rng)),
        .N = std::abs(br::laplace_distribution{sigmaN, sigmaN}(*m_rng))
    };
    m_weightNormalizer = 1.0f / (m_weight.F + m_weight.C + m_weight.N);
    if (isnan(m_weightNormalizer)) {
         throw std::invalid_argument(fmt::format(
            "{}: attribute 'weightDraw error'", ctx));
    }

    m_priceF0 = simulation()->exchange()->process("fundamental", BookId{})->value();

    m_price0 = taosim::util::decimal2double(simulation()->exchange()->config2().initialPrice);
    
    if (attr = node.attribute("tauF"); attr.empty() || attr.as_double() <= 0.0) {
            throw std::invalid_argument(fmt::format(
            "{}: attribute 'tauF' should have a value greater than 0.0", ctx));
    }
    m_tauF = std::vector<double>(m_bookCount, attr.as_double()); 
    m_tauFOrig = attr.as_double();

    if (attr = node.attribute("sigmaEps"); attr.empty() || attr.as_double() <= 0.0f) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'sigmaEps' should have a value greater than 0.0f", ctx));
    }
    m_sigmaEps = attr.as_double();
    if (attr = node.attribute("HARA"); attr.empty() || attr.as_double() < 0.0) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'HARA' should have a value of at least 0.0", ctx));
    }
    m_hara = attr.as_double();

    if (attr = node.attribute("r_aversion"); attr.empty() || attr.as_double() <= 0.0f) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'r_aversion' should have a value greater than 0.0f", ctx));
    }
    
    m_riskAversion0 = attr.as_double();
    const double riskAversionCoef = m_riskAversion0* (1.0f + sigmaC)/(1.0f + sigmaF);
    m_riskAversion = riskAversionCoef * (1.0f + m_weight.F) / (1.0f + m_weight.C);

    attr = node.attribute("volGuard");
    if (attr.empty() || attr.as_double() <= 0.0f) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'volatility Guard (volGuard)' should have a value greater than 0.0f", ctx));
    }
    m_volatilityGuard =  attr.as_double();
    const float p_low  = m_volatilityGuard;
    const float p_high = 1- m_volatilityGuard;    
    const float L1 = std::log((1 - m_volatilityGuard) / m_volatilityGuard);
    const float L2 = std::log(m_volatilityGuard / (1- m_volatilityGuard));
    m_slopeVolGuard =  (L1 - L2) / (10*m_volatilityGuard - m_volatilityGuard/100);
    m_volGuardX0 = m_volatilityGuard/10 + L1/m_slopeVolGuard;

    if (attr = node.attribute("minOPLatency"); attr.as_ullong() == 0) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'minLatency' should have a value greater than 0", ctx));
    }
    m_opl.min = attr.as_ullong();
    if (attr = node.attribute("maxOPLatency"); attr.as_ullong() == 0) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'maxLatency' should have a value greater than 0", ctx));
    }
    m_opl.max = attr.as_ullong();
    if (m_opl.min >= m_opl.max) {
        throw std::invalid_argument(fmt::format(
            "{}: 'minOPLatency' ({}) should be strictly less 'maxOPLatency' ({})",
            ctx, m_opl.min, m_opl.max));
    }

    m_price = m_priceF0;

    m_orderFlag = std::vector<bool>(m_bookCount, false);

    m_priceIncrement = 1 / std::pow(10, simulation()->exchange()->config().parameters().priceIncrementDecimals);
    m_volumeIncrement = 1 / std::pow(10, simulation()->exchange()->config().parameters().volumeIncrementDecimals);

    m_debug = node.attribute("debug").as_bool();

    m_regimeChangeFlag = node.attribute("regimeChangeFlag").as_bool();
    if (m_regimeChangeFlag) {
        if (attr = node.attribute("sigmaFRegime"); attr.empty() || attr.as_double() < 0.0f) {
            throw std::invalid_argument(fmt::format(
                "{}: attribute 'sigmaFRegime' should have a value of at least 0.0f", ctx));
        }
        const double sigmaFRegime = attr.as_double();
        if (attr = node.attribute("sigmaCRegime"); attr.empty() || attr.as_double() < 0.0f) {
            throw std::invalid_argument(fmt::format(
                "{}: attribute 'sigmaCRegime' should have a value of at least 0.0f", ctx));
        }
        const double sigmaCRegime = attr.as_double();
        if (attr = node.attribute("sigmaNRegime"); attr.empty() || attr.as_double() < 0.0f) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'sigmaNRegime' should have a value of at least 0.0f", ctx));
        }
        const double sigmaNRegime = attr.as_double();
        m_weightRegime = {
            .F = std::abs(br::laplace_distribution{sigmaFRegime, sigmaFRegime}(*m_rng)),
            .C = std::abs(br::laplace_distribution{sigmaCRegime, sigmaCRegime}(*m_rng)),
            .N = std::abs(br::laplace_distribution{sigmaNRegime, sigmaNRegime}(*m_rng))
        };
        m_regimeChangeProb = std::vector<float>(m_bookCount, std::clamp(node.attribute("regimeProb").as_float(), 0.0f, 1.0f));
        if (attr = node.attribute("tauFRegime"); attr.empty() || attr.as_double() <= 0.0) {
            throw std::invalid_argument(fmt::format(
            "{}: attribute 'tauFRegime' should have a value greater than 0.0", ctx));
        }
        m_tauFRegime = attr.as_double();
    } else {
        m_weightRegime = m_weight;
        m_regimeChangeProb = std::vector<float>(m_bookCount, 0.0f);
        m_tauFRegime = 1.0;
    }
    
    m_regimeState = std::vector<RegimeState>(m_bookCount, RegimeState::NORMAL);

    m_weightOrig = m_weight;

    if (attr = node.attribute("pO_alpha"); attr.empty() || attr.as_double() < 0.0f || attr.as_double() >= 1.0f){
        m_alpha = 0.0;
    } else {
        m_alpha = attr.as_double();
    }

    m_marketFeedLatencyDistribution = std::normal_distribution<double>{
        [&] {
            static constexpr const char* name = "MFLmean";
            if (auto attr = node.attribute(name); attr.empty()) {
                throw std::invalid_argument{fmt::format(
                    "{}: Missing attribute '{}'", ctx, name)};
            } else {
                return attr.as_double();
            }
        }(),
        [&] {
            static constexpr const char* name = "MFLstd";
            if (auto attr = node.attribute(name); attr.empty()) {
                throw std::invalid_argument{fmt::format(
                    "{}: Missing attribute '{}'", ctx, name)};
            } else {
                return attr.as_double();
            }
        }()
    };
    m_decisionMakingDelayDistribution = std::normal_distribution<double>{
        [&] {
            static constexpr const char* name = "delayMean";
            if (auto attr = node.attribute(name); attr.empty()) {
                throw std::invalid_argument{fmt::format(
                    "{}: Missing attribute '{}'", ctx, name)};
            } else {
                return attr.as_double();
            }
        }(),
        [&] {
            static constexpr const char* name = "delaySTD";
            if (auto attr = node.attribute(name); attr.empty()) {
                throw std::invalid_argument{fmt::format(
                    "{}: Missing attribute '{}'", ctx, name)};
            } else {
                return attr.as_double();
            }
        }()
    };

    if (attr = node.attribute("tau"); attr.empty() || attr.as_ullong() == 0) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'tau' should have a value greater than 0", ctx));
    }
    m_tau0 = attr.as_ullong();
    const double tauCoef = (m_tau0* (1.0f + sigmaC)/(1.0f + sigmaF));
    m_tau = std::clamp(
        static_cast<Timestamp>(std::ceil(
            tauCoef * (1.0f + m_weight.F) / (1.0f + m_weight.C))),static_cast<Timestamp>(1'000'000'000),
        static_cast<Timestamp>(3'600'000'000'000));

    if (attr = node.attribute("tauF"); attr.empty() || attr.as_double() == 0.0) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'tauF' should have a value greater than 0.0", ctx));
    }
    
    if (attr = node.attribute("tauHist"); attr.empty() || attr.as_ullong() == 0) {
        throw std::invalid_argument(fmt::format(
            "{}: attribute 'tauHist' should have a value greater than 0", ctx));
    }

    m_tauHist = attr.as_ullong();
    const Timestamp averageStepsCoef = static_cast<Timestamp>( m_tauHist * (1.0f + sigmaC)/(1.0f + sigmaF)); 
    m_historySize = std::clamp(static_cast<Timestamp>(std::ceil(averageStepsCoef* (1.0 + m_weight.F) / (1.0 + m_weight.C))),
                        static_cast<Timestamp>(10),
                        static_cast<Timestamp>(1000)
                );

    m_horizonSeconds = (m_tau / 1e9) * std::max(1.0, std::log(static_cast<double>(m_historySize)));
    attr = node.attribute("GBM_X0");
    const double gbmX0 = (attr.empty() || attr.as_double() <= 0.0f) ?  0.001 : attr.as_double();
    attr = node.attribute("GBM_mu");
    const double gbmMu = (attr.empty() || attr.as_double() < 0.0f) ? 0 : attr.as_double();
    attr = node.attribute("GBM_sigma");
    const double gbmSigma = (attr.empty() || attr.as_double() < 0.0f) ? 0.1 : attr.as_double();
    attr = node.attribute("GBM_seed");
    const uint64_t gbmSeed = attr.as_ullong(10000); 

    for (BookId bookId = 0; bookId < m_bookCount; ++bookId) {
        m_topLevel.push_back(TopLevel{});
        const auto& Xt = simulation()->getOrComputeGbmPath(
            gbmSeed + bookId + 1, m_price0, gbmMu, gbmSigma, 86400);
        double price = Xt[86400-1];
        
        uint64_t stepLen = m_tau / 1'000'000'000;
        double prevPrice = 0.0;
        for (uint32_t i = 1; i < 86400; i += stepLen) {
            price = Xt[86400-i];
            prevPrice = price;
        }
        m_lastMid.push_back(prevPrice);
        m_fundamental.push_back(simulation()->exchange()->process("fundamental", bookId));
        m_varEst.push_back({});
        m_varEst.back().configure(m_varHalflifeSeconds, m_varJumpRobust);
    }

    attr = node.attribute("opLatencyScaleRay"); 
    const double scale = (attr.empty() || attr.as_double() == 0.0) ? 0.235 : attr.as_double();
    const double percentile = 1-std::exp(-1/(2*scale*scale));
    m_orderPlacementLatencyDistribution =  std::make_unique<taosim::stats::RayleighDistribution>(scale, percentile); 


    // ACD Things
    m_acdDelayDist = std::weibull_distribution<float>{1.0, 1.0}; 
    if (attr = node.attribute("acdOmega"); attr.empty()) {
        throw std::invalid_argument{fmt::format(
                    "{}: Missing attribute 'acdOmega'", ctx)};
    }
    m_omegaDu = attr.as_float();
    if (attr = node.attribute("acdAlpha"); attr.empty() || attr.as_float() < 0.0f) {
            throw std::invalid_argument{fmt::format(
                    "{}: Missing attribute 'acdAlpha'", ctx)};
    }
    m_alphaDu =  attr.as_float(); 
    if (attr = node.attribute("acdBeta"); attr.empty() || attr.as_float() < 0.0f || m_alphaDu + attr.as_float() > 1) {
            throw std::invalid_argument{fmt::format(
                    "{}: Missing attribute 'acdBeta' or values need to be checked", ctx)};
    }
    m_betaDu =  attr.as_float(); 
    attr = node.attribute("maxDMD");
    m_maxDelay = (attr.empty() || attr.as_ullong() < 1'000'000'000) ? static_cast<Timestamp>(450'000'000'000) : attr.as_ullong();
    attr = node.attribute("minDMD");
    m_minDelay = (attr.empty() || attr.as_ullong() < 100'000'000) ? static_cast<Timestamp>(100'000'000) : attr.as_ullong();
    m_feeReserveFrac = std::clamp(node.attribute("feeReserveFrac").as_double(0.01), 0.0, 0.5);
    m_varHalflifeSeconds = node.attribute("varHalflifeSeconds").as_double(60.0);
    m_varJumpRobust = node.attribute("varJumpRobust").as_bool(true);

    m_baseName = [&] {
        std::string res = name();
        boost::algorithm::erase_regex(res, boost::regex("(_\\d+)$"));
        return res;
    }();

    size_t pos = name().find_last_not_of("0123456789");
    if (pos != std::string::npos && pos + 1 < name().size()) {
        std::string numStr = name().substr(pos + 1);
        m_catUId = static_cast<uint64_t>(std::stoul(numStr));
    }
}

//-------------------------------------------------------------------------

void StylizedTraderAgent::receiveMessage(Message::Ptr msg)
{
    if (msg->type == "EVENT_SIMULATION_START") {
        handleSimulationStart();
    }
    else if (msg->type == "EVENT_SIMULATION_END") {
        handleSimulationStop();
    }
    else if (msg->type == "RESPONSE_SUBSCRIBE_EVENT_TRADE") {
        handleTradeSubscriptionResponse();
    }
    else if (msg->type == "RESPONSE_RETRIEVE_L1_EXT") {
        handleRetrieveL1ExtResponse(msg);
    }
    else if (msg->type == "RESPONSE_PLACE_ORDER_LIMIT") {
        handleLimitOrderPlacementResponse(msg);
    }
    else if (msg->type == "ERROR_RESPONSE_PLACE_ORDER_LIMIT") {
        handleLimitOrderPlacementErrorResponse(msg);
    }
    else if (msg->type == "RESPONSE_CANCEL_ORDERS") {
        handleCancelOrdersResponse(msg);
    }
    else if (msg->type == "ERROR_RESPONSE_CANCEL_ORDERS") {
        handleCancelOrdersErrorResponse(msg);
    }
    else if (msg->type == "EVENT_TRADE") {
        handleTrade(msg);
    } else if (msg->type == "WAKEUP") {
        handleWakeup(msg);
    }
}

//-------------------------------------------------------------------------

void StylizedTraderAgent::handleSimulationStart()
{
    for (BookId bookId = 0; bookId < m_bookCount; ++bookId) {
        simulation()->dispatchMessage(
            simulation()->currentTimestamp(),
            1,
            name(),
            m_exchange,
            "RETRIEVE_L1_EXT",
            MessagePayload::create<RetrieveL1ExtPayload>(bookId));
        if (m_catUId == 0) {
            auto chosenAgent = selectTurn();
            Timestamp initDelay = marketFeedLatency();
            simulation()->dispatchMessage(
            simulation()->currentTimestamp(),
                initDelay,
                name(),
                fmt::format("{}_{}", m_baseName, chosenAgent),
                "WAKEUP",
                MessagePayload::create<RetrieveL1Payload>(bookId));
            const float initPsi = m_omegaDu / (1.0f - m_alphaDu - m_betaDu);
            simulation()->exchange()->acdClocks().at(bookId).insert(
                m_baseName, taosim::book::AcdClock{.delay=initPsi, .psi=initPsi});
        }
    }
}

//-------------------------------------------------------------------------

void StylizedTraderAgent::handleSimulationStop()
{
    if (m_catUId != 0) return;
    for (BookId bookId = 0; bookId < m_bookCount; ++bookId) {
        const auto s = simulation()->exchange()->acdClocks().at(bookId).stats(m_baseName);
        if (s.n == 0) continue;
        const double n = static_cast<double>(s.n);
        const double delayMean = s.delaySum / n;
        const double delayStd = std::sqrt(std::max(0.0, s.delaySumSq / n - delayMean * delayMean));
        const double psiMean = s.psiSum / n;
        const double psiStd = std::sqrt(std::max(0.0, s.psiSumSq / n - psiMean * psiMean));
        fmt::print(
            "AGENTDIAG {{\"agent\":\"{}\",\"book\":{},\"n\":{},"
            "\"delay_mean\":{},\"delay_std\":{},\"delay_min\":{},\"delay_max\":{},"
            "\"psi_mean\":{},\"psi_std\":{}}}\n",
            m_baseName, bookId, s.n,
            delayMean, delayStd, s.delayMin, s.delayMax, psiMean, psiStd);
        std::fflush(stdout);
    }
}

//-------------------------------------------------------------------------

void StylizedTraderAgent::handleTradeSubscriptionResponse()
{

}

//-------------------------------------------------------------------------

void StylizedTraderAgent::handleRetrieveL1ExtResponse(Message::Ptr msg)
{
    // Guards a null payload, not a type mismatch: static_pointer_cast cannot fail.
    const auto payload = std::static_pointer_cast<RetrieveL1ExtResponsePayload>(msg->payload);
    if (payload == nullptr) return;
    m_varEst.at(payload->bookId).update(payload->tradeStats, simulation()->currentTimestamp());

    const BookId bookId = payload->bookId;

    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        marketFeedLatency() + m_tau,
        name(),
        m_exchange,
        "RETRIEVE_L1_EXT",
        MessagePayload::create<RetrieveL1ExtPayload>(bookId));

    auto& topLevel = m_topLevel[bookId];
    topLevel.bid = taosim::util::decimal2double(payload->bestBidPrice);
    topLevel.ask = taosim::util::decimal2double(payload->bestAskPrice);
    
    if  (topLevel.bid == 0.0) topLevel.bid = m_lastMid.at(bookId);
    if  (topLevel.ask == 0.0) topLevel.ask = topLevel.bid + m_priceIncrement;
    const double midQuote = 0.5 * (topLevel.bid + topLevel.ask);

    m_lastMid.at(bookId) = midQuote;
}

//-------------------------------------------------------------------------

void StylizedTraderAgent::handleLimitOrderPlacementResponse(Message::Ptr msg)
{
    const auto payload = std::static_pointer_cast<PlaceOrderLimitResponsePayload>(msg->payload);

    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
        static_cast<Timestamp>(m_tau*std::max(1.0,std::log(m_historySize))),
        name(),
        m_exchange,
        "CANCEL_ORDERS",
        MessagePayload::create<CancelOrdersPayload>(
            std::vector{taosim::event::Cancellation(payload->id)}, payload->requestPayload->bookId));

    m_orderFlag[payload->requestPayload->bookId] = false;
}

//-------------------------------------------------------------------------

void StylizedTraderAgent::handleLimitOrderPlacementErrorResponse(Message::Ptr msg)
{
    const auto payload =
        std::static_pointer_cast<PlaceOrderLimitErrorResponsePayload>(msg->payload);

    const BookId bookId = payload->requestPayload->bookId;

    m_orderFlag[bookId] = false;
}

//-------------------------------------------------------------------------

void StylizedTraderAgent::handleCancelOrdersResponse(Message::Ptr msg)
{}

//-------------------------------------------------------------------------

void StylizedTraderAgent::handleCancelOrdersErrorResponse(Message::Ptr msg)
{}

//-------------------------------------------------------------------------

void StylizedTraderAgent::handleTrade(Message::Ptr msg)
{
    const auto payload = std::static_pointer_cast<EventTradePayload>(msg->payload);
}

//-------------------------------------------------------------------------

StylizedTraderAgent::ForecastResult StylizedTraderAgent::forecast(BookId bookId)
{
    const double pf = getProcessValue(bookId, "fundamental");

    m_price = m_lastMid.at(bookId);
    if (isnan(m_price) || m_price <= 0.0) {
        // Error recovery
        m_price = m_price0;
    }
    double compF =  1.0 / m_tauF[bookId] * std::log(pf/ m_price);
    // Error recovery, just in case
    if (isnan(compF)) {
        compF = 0.0;
    }
    double compC = m_varEst.at(bookId).meanLogReturn();
    if (isnan(compC)) compC = 0.0;
    const double compN = std::normal_distribution{0.0, m_sigmaEps}(*m_rng);
    double logReturnForecast = std::clamp(m_weightNormalizer
        * (m_weight.F * compF + m_weight.C * compC + m_weight.N * compN), -1.0, 1.0);
    double varLastLogs = m_varEst.at(bookId).value() * m_horizonSeconds;
    // Error recovery
    if (isnan(varLastLogs)) {
        varLastLogs = std::abs(std::log(pf/m_price)*0.33);
    }
    if (const double relTick = m_priceIncrement / m_price;
        varLastLogs < relTick * relTick) {
        varLastLogs = relTick * relTick;
    }
    if (isnan(logReturnForecast)) {
        logReturnForecast = 0.0000001;
    }
    const double forecastPrice =
        m_price * std::exp(logReturnForecast*std::max(1.0,std::log(m_historySize)));
    return {
        .price = forecastPrice,
        .varianceOfLastLogReturns =  varLastLogs};
}

//-------------------------------------------------------------------------

void StylizedTraderAgent::placeOrderChiarella(BookId bookId)
{
    const ForecastResult forecastResult = forecast(bookId);

    const auto freeBase =
        taosim::util::decimal2double(simulation()->account(name()).at(bookId).base.getFree());
    const auto freeQuote =
        taosim::util::decimal2double(simulation()->account(name()).at(bookId).quote->getFree());

    if (m_riskAversion * forecastResult.varianceOfLastLogReturns == 0.0) {
        // ERROR recovery rebalance in order to keep things flowing
        double quoteValue = freeQuote/m_price;
        double ordQty =  std::uniform_real_distribution<double>{0.1,std::abs(quoteValue-freeBase)} (*m_rng);
        if (freeBase < quoteValue) {
            simulation()->dispatchMessage(
                simulation()->currentTimestamp(),
                orderPlacementLatency(),
                name(),
                m_exchange,
                "PLACE_ORDER_MARKET",
                MessagePayload::create<PlaceOrderMarketPayload>(
                    OrderDirection::BUY,
                    taosim::util::double2decimal(ordQty),
                    bookId));
        } else if (freeBase > quoteValue) {
            simulation()->dispatchMessage(
                simulation()->currentTimestamp(),
                orderPlacementLatency(),
                name(),
                m_exchange,
                "PLACE_ORDER_MARKET",
                MessagePayload::create<PlaceOrderMarketPayload>(
                    OrderDirection::SELL,
                    taosim::util::double2decimal(ordQty),
                    bookId));
        }
        return;
    }

    const auto [indifferencePrice, indifferencePriceConverged] =
        calculateIndifferencePrice(forecastResult, freeBase, freeQuote);
    if (!indifferencePriceConverged) {
        return;
    }

    const auto [minimumPrice, minimumPriceConverged] =
        calculateMinimumPrice(forecastResult, freeBase, freeQuote, indifferencePrice);
    if (!minimumPriceConverged) {
        return;
    }

    const auto maximumPrice = forecastResult.price;

    const double sampleLow = std::max(minimumPrice, m_priceIncrement);
    const double sampleHigh = maximumPrice;
    if (sampleLow >= sampleHigh) {
        return;
    }

    const double sampledPrice = std::uniform_real_distribution{sampleLow, sampleHigh}(*m_rng);
    if (sampledPrice < indifferencePrice) {
        // Due to the error recoveries, technical adjustments in placements
        placeLimitBuy(bookId, forecastResult, sampledPrice, freeBase, freeQuote);
    }
    else if (sampledPrice > indifferencePrice) {
        placeLimitSell(bookId, forecastResult, sampledPrice, freeBase, freeQuote);
    }
}

//-------------------------------------------------------------------------

StylizedTraderAgent::OptimizationResult StylizedTraderAgent::calculateIndifferencePrice(
    const StylizedTraderAgent::ForecastResult& forecastResult, double freeBase, double freeQuote)
{
    auto residual = [&](double x) {
        return investmentPosition(x, forecastResult.price,
            forecastResult.varianceOfLastLogReturns,
            freeBase, freeQuote, m_riskAversion, m_hara) - freeBase;
    };
    const auto root =
        util::solveScalarBracketed(residual, m_priceIncrement, forecastResult.price);
    return {.value = root.value, .converged = root.converged};
}

//-------------------------------------------------------------------------

StylizedTraderAgent::OptimizationResult StylizedTraderAgent::calculateMinimumPrice(
    const StylizedTraderAgent::ForecastResult& forecastResult,
    double freeBase,
    double freeQuote,
    double indifferencePrice)
{
    auto residual = [&](double x) {
        return x * (investmentPosition(x, forecastResult.price,
            forecastResult.varianceOfLastLogReturns,
            freeBase, freeQuote, m_riskAversion, m_hara) - freeBase) - freeQuote;
    };
    const auto root =
        util::solveScalarBracketed(residual, m_priceIncrement, indifferencePrice);
    if (!root.converged && root.status == util::RootStatus::SameSignNegative) {
        return {.value = m_priceIncrement, .converged = true};
    }
    return {.value = root.value, .converged = root.converged};
}

//-------------------------------------------------------------------------

double StylizedTraderAgent::calcPositionPrice(
    const StylizedTraderAgent::ForecastResult& forecastResult,
    double price,
    double freeBase,
    double freeQuote)
{
    return investmentPosition(
        price,
        forecastResult.price,
        forecastResult.varianceOfLastLogReturns,
        freeBase,
        freeQuote,
        m_riskAversion,
        m_hara);
}

//-------------------------------------------------------------------------

void StylizedTraderAgent::placeLimitBuy(
    BookId bookId,
    const StylizedTraderAgent::ForecastResult& forecastResult,
    double sampledPrice,
    double freeBase,
    double freeQuote)
{
    const double price = std::round(sampledPrice / m_priceIncrement) * m_priceIncrement;

    double volume = calcPositionPrice(forecastResult,sampledPrice,freeBase,freeQuote) - freeBase;
    if (const auto attainableVolume = freeQuote / price * (1.0 - m_feeReserveFrac);
        volume > attainableVolume) {
        volume = attainableVolume;
    }
    volume = std::floor(volume / m_volumeIncrement) * m_volumeIncrement;
    if (volume <= 0.0) {
        return;
    }

    m_orderFlag[bookId] = true;

    const float postOnlyProb = std::max(1.0/(1.0 + std::exp(-m_slopeVolGuard* (forecastResult.varianceOfLastLogReturns - m_volGuardX0))), m_alpha);
    const bool postOnly = std::bernoulli_distribution{postOnlyProb}(*m_rng);
    if ((sampledPrice > m_price*(1.0 + m_wealthFrac) 
			    && std::bernoulli_distribution{std::pow((sampledPrice-m_price)/m_price,0.20)} (*m_rng)) 
		    && !postOnly) {
        simulation()->dispatchMessage(
            simulation()->currentTimestamp(),
            orderPlacementLatency(),
            name(),
            m_exchange,
            "PLACE_ORDER_MARKET",
            MessagePayload::create<PlaceOrderMarketPayload>(
                OrderDirection::BUY,
                taosim::util::double2decimal(volume),
                bookId));
    } else {
        double restPrice = price;
        if (postOnly) {
            restPrice = slideToRest(price, OrderDirection::BUY, m_topLevel.at(bookId), m_priceIncrement);
            if (restPrice <= 0.0) return;
        }
        simulation()->dispatchMessage(
            simulation()->currentTimestamp(),
            orderPlacementLatency(),
            name(),
            m_exchange,
            "PLACE_ORDER_LIMIT",
            MessagePayload::create<PlaceOrderLimitPayload>(
                OrderDirection::BUY,
                taosim::util::double2decimal(volume),
                taosim::util::double2decimal(restPrice),
                bookId,
                Currency::BASE,
                std::nullopt,
                postOnly));
    }
}

//-------------------------------------------------------------------------

void StylizedTraderAgent::placeLimitSell(
    BookId bookId,
    const StylizedTraderAgent::ForecastResult& forecastResult,
    double sampledPrice,
    double freeBase,
    double freeQuote)
{
    const double price = std::round(sampledPrice / m_priceIncrement) * m_priceIncrement;

    double volume = freeBase - calcPositionPrice(forecastResult, sampledPrice,freeBase,freeQuote);
    if (volume > freeBase) {
        volume = freeBase;
    }
    volume = std::floor(volume / m_volumeIncrement) * m_volumeIncrement;
    if (volume <= 0.0) {
        return;
    }

    m_orderFlag[bookId] = true;
    const float postOnlyProb = std::max(1.0/(1.0 + std::exp(-m_slopeVolGuard* (forecastResult.varianceOfLastLogReturns - m_volGuardX0))), m_alpha);
    const bool postOnly = std::bernoulli_distribution{postOnlyProb}(*m_rng);
    if (!postOnly &&
	(sampledPrice < m_price*(1.0 - m_wealthFrac) && 
    	std::bernoulli_distribution{std::pow((m_price-sampledPrice)/m_price,0.20)}(*m_rng))) {
        simulation()->dispatchMessage(
            simulation()->currentTimestamp(),
            orderPlacementLatency(),
            name(),
            m_exchange,
            "PLACE_ORDER_MARKET",
            MessagePayload::create<PlaceOrderMarketPayload>(
                OrderDirection::SELL,
                taosim::util::double2decimal(volume),
                bookId));
    } else {
        double restPrice = price;
        if (postOnly) {
            restPrice = slideToRest(price, OrderDirection::SELL, m_topLevel.at(bookId), m_priceIncrement);
            if (restPrice <= 0.0) return;
        }
        simulation()->dispatchMessage(
            simulation()->currentTimestamp(),
            orderPlacementLatency(),
            name(),
            m_exchange,
            "PLACE_ORDER_LIMIT",
            MessagePayload::create<PlaceOrderLimitPayload>(
                OrderDirection::SELL,
                taosim::util::double2decimal(volume),
                taosim::util::double2decimal(restPrice),
                bookId,
                Currency::BASE,
                std::nullopt,
                postOnly));
    }
}

uint64_t StylizedTraderAgent::selectTurn() {
    const auto& agentBaseNamesToCounts = simulation()->localAgentManager()->roster()->baseNamesToCounts();
    return  std::uniform_int_distribution<uint64_t>{0, agentBaseNamesToCounts.at(m_baseName) - 1}(*m_rng);
}

//------------------------------------------------------------------------

void StylizedTraderAgent::handleWakeup(Message::Ptr &msg)
{
    const auto payload = std::static_pointer_cast<RetrieveL1Payload>(msg->payload);

    const BookId bookId = payload->bookId;
    auto chosenAgent = selectTurn();

    simulation()->dispatchMessage(
        simulation()->currentTimestamp(),
            decisionMakingDelay(bookId),
            name(),
            fmt::format("{}_{}", m_baseName, chosenAgent),
            "WAKEUP",
            MessagePayload::create<RetrieveL1Payload>(bookId));
    placeOrderChiarella(bookId);
}

//-------------------------------------------------------------------------

double StylizedTraderAgent::getProcessValue(BookId bookId, const std::string& name)
{
    if (name == "fundamental") return m_fundamental[bookId]->value();
    return simulation()->exchange()->process(name, bookId)->value();
}

//-------------------------------------------------------------------------

void StylizedTraderAgent::updateRegime(BookId bookId)
{

}

//-------------------------------------------------------------------------

Timestamp StylizedTraderAgent::orderPlacementLatency()
{
    return static_cast<Timestamp>(
        std::lerp(m_opl.min, m_opl.max, m_orderPlacementLatencyDistribution->sample(*m_rng)));
}

//-------------------------------------------------------------------------

Timestamp StylizedTraderAgent::marketFeedLatency()
{
    return static_cast<Timestamp>(std::min(
        std::abs(m_marketFeedLatencyDistribution(*m_rng)),
        + m_marketFeedLatencyDistribution.mean() + 3 * m_marketFeedLatencyDistribution.stddev()));
}
//-------------------------------------------------------------------------

Timestamp StylizedTraderAgent::decisionMakingDelay(BookId bookId)
{
    const std::string& key = m_baseName;
    auto& clocks = simulation()->exchange()->acdClocks().at(bookId);
    const auto lastDurationComp = clocks.get(key);
    float lastDelay = lastDurationComp.delay;
    float psi_prev = lastDurationComp.psi;
    float psi_next = m_omegaDu + m_alphaDu * lastDelay + m_betaDu *psi_prev;
    if (!std::isfinite(psi_next)) {
        psi_next = m_omegaDu/(1-m_alphaDu - m_betaDu);
    }
    double delay = static_cast<double>(std::exp(psi_next)) * m_acdDelayDist(*m_rng);
    if (!std::isfinite(delay) || delay > static_cast<double>(m_maxDelay)) {
        delay = static_cast<double>(m_maxDelay);
    }
    Timestamp delay_timestamped = std::clamp(static_cast<Timestamp>(delay), m_minDelay, m_maxDelay);
    clocks.insert(
        key,
        taosim::book::AcdClock{.delay=std::log((float) delay_timestamped), .psi=psi_next});
    return delay_timestamped;
}

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------
