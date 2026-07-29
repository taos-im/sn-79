/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/agent/common.hpp>
#include <taosim/dsa/PriorityQueue.hpp>
#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include "Agent.hpp"
#include "Distribution.hpp"
#include "Order.hpp"
#include "Trade.hpp"

#include <deque>
#include <memory>
#include <queue>
#include <random>

//-------------------------------------------------------------------------

namespace taosim::agent
{

//-------------------------------------------------------------------------

enum class ALGOTraderStatus : uint32_t
{
    ASLEEP,
    READY,
    EXECUTING
};

struct TimestampedVolume
{
    Timestamp timestamp;
    decimal_t volume;
    decimal_t price; // VWAP for logrets

    [[nodiscard]] auto operator<=>(const TimestampedVolume& other) const noexcept
    {
        return timestamp <=> other.timestamp;
    }
};

struct BookStat 
{
    double bid, ask;
};

struct ALGOTraderVolumeStatsDesc
{
    Timestamp period;
    double alpha;
    double beta;
    double omega;
    double gamma; 
    double initPrice;
    size_t depth;
};

class ALGOTraderVolumeStats
{
public:
    ALGOTraderVolumeStats() noexcept = default;
    explicit ALGOTraderVolumeStats(const ALGOTraderVolumeStatsDesc& desc);

    void push(const Trade& trade);
    void push(TimestampedVolume timestampedVolume);
    void pushLevels(
        Timestamp timestamp, std::span<const BookLevel> bids, std::span<const BookLevel> asks);
    
    [[nodiscard]] double estimatedVolatility() const noexcept;
    [[nodiscard]] double bidSlope() noexcept { return lastSlopes().bid; }
    [[nodiscard]] double askSlope() noexcept { return lastSlopes().ask; }
    [[nodiscard]] double bidVolume() const { return lastVolume().bid; }
    [[nodiscard]] double askVolume() const { return lastVolume().ask; }

    // TODO: Wrap state into a struct and provide a single access point here.
    [[nodiscard]] auto&& queue(this auto&& self) noexcept { return self.m_queue; }
    [[nodiscard]] auto&& rollingSum(this auto&& self) noexcept { return self.m_rollingSum; }
    [[nodiscard]] auto&& priceLast(this auto&& self) noexcept { return self.m_priceLast; }
    [[nodiscard]] auto&& variance(this auto&& self) noexcept { return self.m_variance; }
    [[nodiscard]] auto&& estimatedVol(this auto&& self) noexcept { return self.m_estimatedVol; }
    [[nodiscard]] auto&& lastSeq(this auto&& self) noexcept { return self.m_lastSeq; }
    [[nodiscard]] auto&& bookSlopes(this auto&& self) noexcept { return self.m_bookSlopes; }
    [[nodiscard]] auto&& bookVolumes(this auto&& self) noexcept { return self.m_bookVolumes; }

    [[nodiscard]] static ALGOTraderVolumeStats fromXML(pugi::xml_node node, double initPrice, size_t depth);

private:
    [[nodiscard]] double slopeOLS(std::span<const BookLevel> side);
    [[nodiscard]] double volumeSum(std::span<const BookLevel> side, size_t depth = 5);
    [[nodiscard]] const BookStat& lastSlopes() const { return m_bookSlopes.at(m_lastSeq); }
    [[nodiscard]] const BookStat& lastVolume() const { return m_bookVolumes.at(m_lastSeq); }

    // Incremental volume-window state for the log-return variance. A VWAP bucket
    // per distinct timestamp in the m_period window; the variance is maintained
    // from running Σlogret / Σlogret² instead of rescanning the whole window on
    // every push. See ALGOTraderAgent.cpp.
    struct VolBucket
    {
        Timestamp ts;
        decimal_t sumVol;       // Σ volume at this exact timestamp
        decimal_t sumPriceVol;  // Σ price*volume at this timestamp (VWAP numerator)
    };
    [[nodiscard]] double bucketPriceDouble(const VolBucket& b) const noexcept;
    [[nodiscard]] double logretBetween(const VolBucket& prev, const VolBucket& cur) const noexcept;
    void addTradeIncremental(const TimestampedVolume& tv);
    void rebuildIncremental();
    void recomputeVariance() noexcept;

    // Parameters, injections.
    Timestamp m_period;
    double m_alpha;
    double m_beta;
    double m_omega;
    double m_gamma;
    double m_initPrice;
    size_t m_depth;

    // State.
    dsa::PriorityQueue<
        TimestampedVolume,
        std::vector<TimestampedVolume>,
        std::greater<TimestampedVolume>> m_queue;
    decimal_t m_rollingSum{};
    double m_priceLast;
    double m_variance{0.0};
    double m_estimatedVol;
    // Incremental window state (NOT checkpoint-serialized; rebuilt lazily from
    // m_queue on the first push, including after a checkpoint restore).
    std::deque<VolBucket> m_buckets;
    double m_logRetSum{0.0};
    double m_logRetSumSq{0.0};
    bool m_incBuilt{false};
    Timestamp m_lastSeq;
    std::map<Timestamp, BookStat> m_bookSlopes; 
    std::map<Timestamp, BookStat> m_bookVolumes;
};

struct ALGOTraderState
{
    ALGOTraderStatus status;
    Timestamp marketFeedLatency;
    ALGOTraderVolumeStats volumeStats;
    decimal_t volumeToBeExecuted;
    OrderDirection direction;
    Timestamp statusChangeTime;
    Timestamp statusChangeEndTime;
};

//-------------------------------------------------------------------------

class ALGOTraderAgent : public Agent
{
public:
    ALGOTraderAgent() noexcept = default;
    explicit ALGOTraderAgent(Simulation* simulation) noexcept;

    virtual void configure(const pugi::xml_node& node) override;
    virtual void receiveMessage(Message::Ptr msg) override;

    // TODO: Wrap state into a struct and provide a single access point here.
    [[nodiscard]] auto&& state(this auto&& self) noexcept { return self.m_state; }
    [[nodiscard]] auto&& lastPrice(this auto&& self) noexcept { return self.m_lastPrice; }
    [[nodiscard]] auto&& topLevel(this auto&& self) noexcept { return self.m_topLevel; }

private:
    struct VolatilityBounds
    {
        double activationRate, activationMidpoint, activationCapacity;
    };

    void handleSimulationStart(Message::Ptr msg);
    void handleTrade(Message::Ptr msg);
    void handleWakeup(Message::Ptr msg);
    void handleMarketOrderResponse(Message::Ptr msg);
    void handleMarketOrderPlacementErrorResponse(Message::Ptr msg);
    void handleBookResponse(Message::Ptr msg);
    void handleL1Response(Message::Ptr msg);

    void execute(BookId bookId, ALGOTraderState& state);
    decimal_t drawNewVolume(uint32_t baseDecimals);
    double getProcessValue(BookId bookId, const std::string& name);
    uint64_t getProcessCount(BookId bookId, const std::string& name);
    double wakeupProb(ALGOTraderState& state, double fundDist);
    Timestamp orderPlacementLatency();
    Timestamp marketFeedLatency();
    Timestamp decisionMakingDelay();

    // Parameters, injections.
    std::mt19937* m_rng;
    std::string m_exchange;
    uint32_t m_bookCount;
    double m_volumeMin;
    std::unique_ptr<stats::Distribution> m_volumeDistribution;
    double m_opLatencyScaleRay;
    DelayBounds m_opl;
    std::normal_distribution<double> m_marketFeedLatencyDistribution;
    std::unique_ptr<stats::Distribution> m_orderPlacementLatencyDistribution;
    std::unique_ptr<stats::Distribution> m_volumeDrawDistribution;
    std::normal_distribution<double> m_departureThreshold;
    VolatilityBounds m_volatilityBounds;
    double m_deviationProbCoef;
    double m_reversionDeadband{};
    double m_reversionPower{1.0};
    double m_timeActivationCoef;
    Timestamp m_period;
    size_t m_depth;
    std::normal_distribution<double> m_delay;
    double m_immediateBase;
    
    // State.
    std::vector<ALGOTraderState> m_state;
    std::vector<decimal_t> m_lastPrice;
    std::vector<TopLevel> m_topLevel;
};

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------
