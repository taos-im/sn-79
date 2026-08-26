/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "Agent.hpp"
#include "Distribution.hpp"
#include "Order.hpp"
#include "common.hpp"

//-------------------------------------------------------------------------

namespace taosim::process { class MagneticField; }

namespace taosim::agent
{

//-------------------------------------------------------------------------

struct NoiseTraderAgentState
{
    std::vector<bool> orderFlag;
};

//-------------------------------------------------------------------------

class NoiseTraderAgent : public Agent
{
public:
    explicit NoiseTraderAgent(Simulation* simulation) noexcept;

    [[nodiscard]] auto&& state(this auto&& self) noexcept { return self.m_state; }
    
    virtual void configure(const pugi::xml_node& node) override;
    virtual void receiveMessage(Message::Ptr msg) override;

private:
    struct OptimizationResult
    {
        double value;
        bool converged;
    };

    struct ForecastResult
    {
        double price;
        double varianceOfLastLogReturns;
    };

    void handleSimulationStart();
    void handleSimulationStop();
    void handleTradeSubscriptionResponse();
    void handleWakeup(Message::Ptr &msg);
    void handleRetrieveL1Response(Message::Ptr msg);
    void handleMarketOrderPlacementResponse(Message::Ptr msg);
    void handleMarketOrderPlacementErrorResponse(Message::Ptr msg);
    void handleLimitOrderPlacementResponse(Message::Ptr msg);
    void handleLimitOrderPlacementErrorResponse(Message::Ptr msg);
    void handleCancelOrdersResponse(Message::Ptr msg);
    void handleCancelOrdersErrorResponse(Message::Ptr msg);
    void handleTrade(Message::Ptr msg);
    uint64_t selectTurn();

    void placeOrder(BookId bookId);
    double samplePrice(double minP, double indiffP, double maxP, int sign, double magnetism);
    OptimizationResult calculateIndifferencePrice(
        const ForecastResult& forecastResult, double freeBase, double freeQuote);
    OptimizationResult calculateMinimumPrice(
        const ForecastResult& forecastResult,
        double freeBase,
        double freeQuote,
        double indifferencePrice);
    double calcPositionPrice(
        const ForecastResult& forecastResult, double price, double freeBase, double freeQuote);
    void placeBid(BookId bookId,double volume, double price);
    void placeBuy(BookId bookId,double volume);
    void placeAsk(BookId bookId, double volume, double price);
    void placeSell(BookId bookId, double volume);
    Timestamp orderPlacementLatency();
    Timestamp marketFeedLatency();

    // Parameters, injections.
    //General params
    std::string m_baseName;
    uint32_t m_bookCount;
    uint32_t m_catUId;
    bool m_debug;
    bool m_logFlag;
    std::string m_exchange;
    std::mt19937* m_rng;
    double m_priceIncrement;
    double m_volumeIncrement;

    // Order placement
    double m_volumeConst;
    double m_balanceCoef;
    double m_feeReserveFrac;
    double m_price;
    Timestamp m_tau;
    double m_sigma;
    double m_mWeight;

    // Delays, latencys activations and more
    float m_omegaDu;
    float m_alphaDu;
    float m_betaDu;
    float m_gammaDu;
    Timestamp m_maxDelay;
    Timestamp m_minDelay;
    std::weibull_distribution<float> m_acdDelayDist;
    DelayBounds m_opl;
    std::normal_distribution<double> m_marketFeedLatencyDistribution;
    std::unique_ptr<stats::Distribution> m_orderPlacementLatencyDistribution;
    std::unique_ptr<stats::Distribution> m_priceShiftDistribution;

    // State.
    NoiseTraderAgentState m_state;
    // Cached per-bookId MagneticField pointer — eliminates per-tick string
    // lookup + RTTI cast in handleWakeup/handleRetrieveL1Response paths.
    std::vector<taosim::process::MagneticField*> m_magneticField;
};

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------
