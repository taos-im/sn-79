/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/agent/common.hpp>
#include "Agent.hpp"
#include "GBMValuationModel.hpp"
#include "Distribution.hpp"
#include "Order.hpp"

#include <taosim/util/TradeStatsEstimator.hpp>


//-------------------------------------------------------------------------

// Forward declaration to avoid pulling the Process header into the public agent
// header just for a cached-pointer member type.
namespace taosim::process { class Process; }

namespace taosim::agent
{

//-------------------------------------------------------------------------
    
class StylizedTraderAgent : public Agent
{
public:
    enum RegimeState {
        NORMAL,
        REGIME_A,
        NEWS    
    };

    StylizedTraderAgent() noexcept = default;
    StylizedTraderAgent(Simulation* simulation) noexcept;

    // TODO: Wrap state into a struct and provide a single access point here.
    [[nodiscard]] auto&& tauF(this auto&& self) noexcept { return self.m_tauF; }
    [[nodiscard]] auto&& orderFlag(this auto&& self) noexcept { return self.m_orderFlag; }
    [[nodiscard]] auto&& regimeChangeProb(this auto&& self) noexcept { return self.m_regimeChangeProb; }
    [[nodiscard]] auto&& regimeState(this auto&& self) noexcept { return self.m_regimeState; }
    [[nodiscard]] auto&& topLevel(this auto&& self) noexcept { return self.m_topLevel; }
    [[nodiscard]] auto&& lastMid(this auto&& self) noexcept { return self.m_lastMid; }
    [[nodiscard]] auto&& tradePrice(this auto&& self) noexcept { return self.m_tradePrice; }

    virtual void configure(const pugi::xml_node& node) override;
    virtual void receiveMessage(Message::Ptr msg) override;

private:
    struct ForecastResult
    {
        double price;
        double varianceOfLastLogReturns;
    };

    struct OptimizationResult
    {
        double value;
        bool converged;
    };

    struct Weight
    {
        double F, C, N;
    };

    void handleSimulationStart();
    void handleSimulationStop();
    void handleTradeSubscriptionResponse();
    void handleWakeup(Message::Ptr &msg);
    void handleRetrieveL1Response(Message::Ptr msg);
    void handleRetrieveL1ExtResponse(Message::Ptr msg);
    void handleLimitOrderPlacementResponse(Message::Ptr msg);
    void handleLimitOrderPlacementErrorResponse(Message::Ptr msg);
    void handleCancelOrdersResponse(Message::Ptr msg);
    void handleCancelOrdersErrorResponse(Message::Ptr msg);
    void handleTrade(Message::Ptr msg);
    uint64_t selectTurn();

    ForecastResult forecast(BookId bookId);
    void placeOrderChiarella(BookId bookId);
    OptimizationResult calculateIndifferencePrice(
        const ForecastResult& forecastResult, double freeBase, double freeQuote);
    OptimizationResult calculateMinimumPrice(
        const ForecastResult& forecastResult,
        double freeBase,
        double freeQuote,
        double indifferencePrice);
    double calcPositionPrice(const ForecastResult& forecastResult, double price, double freeBase, double freeQuote);
    void placeLimitBuy(
        BookId bookId,
        const ForecastResult& forecastResult,
        double sampledPrice,
        double freeBase,
        double freeQuote);
    void placeLimitSell(
        BookId bookId,
        const ForecastResult& forecastResult,
        double sampledPrice,
        double freeBase,
        double freeQuote);
    double getProcessValue(BookId bookId, const std::string& name);
    void updateRegime(BookId bookId);
    Timestamp orderPlacementLatency();
    Timestamp marketFeedLatency();
    Timestamp decisionMakingDelay(BookId bookId);

    // Parameters, injections.
    std::mt19937* m_rng;
    std::string m_exchange;
    uint32_t m_bookCount;
    Weight m_weight;
    double m_weightNormalizer;
    double m_priceF0;
    double m_price0;
    double m_tauFOrig;
    double m_sigmaEps;
    double m_hara;
    double m_riskAversion0;
    double m_riskAversion;
    float m_volatilityGuard;
    float m_slopeVolGuard;
    float m_volGuardX0;
    DelayBounds m_opl;
    double m_price;
    double m_priceIncrement;
    double m_volumeIncrement;
    bool m_debug;
    bool m_regimeChangeFlag;
    Weight m_weightRegime;
    double m_tauFRegime;
    Weight m_weightOrig;
    double m_alpha;
    std::normal_distribution<double> m_marketFeedLatencyDistribution;
    std::normal_distribution<double> m_decisionMakingDelayDistribution;
    Timestamp m_tau0;
    Timestamp m_tau;
    Timestamp m_tauHist;
    Timestamp m_historySize;
    double m_horizonSeconds{};
    std::unique_ptr<taosim::stats::Distribution> m_orderPlacementLatencyDistribution;
    std::string m_baseName;
    uint32_t m_catUId;
    double m_wealthFrac{0.01};
    double m_feeReserveFrac{0.01};
    float m_omegaDu;
    float m_alphaDu;
    float m_betaDu;
    float m_gammaDu;
    Timestamp m_maxDelay;
    Timestamp m_minDelay;
    std::weibull_distribution<float> m_acdDelayDist;
    
    // State.
    std::vector<double> m_tauF;
    std::vector<bool> m_orderFlag;
    std::vector<float> m_regimeChangeProb;
    std::vector<RegimeState> m_regimeState;
    std::vector<TopLevel> m_topLevel;
    std::vector<double> m_lastMid;
    std::vector<taosim::util::TradeStatsEstimator> m_varEst;
    double m_varHalflifeSeconds{};
    bool m_varJumpRobust{};
    std::vector<taosim::process::Process*> m_fundamental;
};

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------
