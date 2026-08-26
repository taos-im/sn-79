/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/agent/ALGOTraderAgent.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <optional>

//-------------------------------------------------------------------------

using taosim::decimal_t;
using namespace taosim::agent;
using namespace testing;

//-------------------------------------------------------------------------

// The rolling-window window is the only thing under test here, but the constructor now
// takes the whole GARCH-ish descriptor and validates it, so the volatility parameters
// have to be present and legal (alpha >= 0, beta >= 0, omega > 0) even though they do
// not influence rollingSum().
namespace
{

[[nodiscard]] ALGOTraderVolumeStatsDesc descWithPeriod(Timestamp period) noexcept
{
    return ALGOTraderVolumeStatsDesc{
        .period = period,
        .alpha = 0.1,
        .beta = 0.8,
        .omega = 1e-6,
        .gamma = 0.0,
        .initPrice = 100.0,
        .depth = 5};
}

}  // namespace

using VolumeStatsFixture =
    TestWithParam<std::tuple<Timestamp, std::vector<TimestampedVolume>, decimal_t>>;

struct VolumeStatsTest : public VolumeStatsFixture
{
    virtual void SetUp() override
    {
        const auto& [period, timestampedVolumes, referenceSum] = GetParam();
        // ALGOTraderVolumeStats has no default constructor, so it cannot be a bare
        // member assigned in SetUp() the way the old size_t ctor allowed.
        volumeStats.emplace(descWithPeriod(period));
        for (const auto& item : timestampedVolumes) {
            volumeStats->push(item);
        }
        this->referenceSum = referenceSum;
    }

    std::optional<ALGOTraderVolumeStats> volumeStats;
    decimal_t referenceSum;
};

INSTANTIATE_TEST_SUITE_P(
    ALGOTraderTest,
    VolumeStatsTest,
    Values(
        std::tuple{
            5,
            std::vector<TimestampedVolume>{{.timestamp = 0, .volume = 1_dec}},
            1_dec},
        std::tuple{
            5,
            std::vector<TimestampedVolume>{
                {.timestamp = 0, .volume = 1_dec},
                {.timestamp = 0, .volume = DEC(2.5)},
                {.timestamp = 4, .volume = DEC(3.75)},
                {.timestamp = 5, .volume = 10_dec}},
            DEC(13.75)},
        std::tuple{
            10,
            std::vector<TimestampedVolume>{
                {.timestamp = 0, .volume = 1_dec},
                {.timestamp = 0, .volume = DEC(2.5)},
                {.timestamp = 4, .volume = DEC(3.75)},
                {.timestamp = 5, .volume = 10_dec},
                {.timestamp = 10, .volume = DEC(4.2)},
                {.timestamp = 15, .volume = 20_dec},
                {.timestamp = 18, .volume = 2_dec},},
            DEC(26.2)}));

TEST_P(VolumeStatsTest, WorksCorrectly)
{
    EXPECT_EQ(volumeStats->rollingSum(), referenceSum);
}

//-------------------------------------------------------------------------

TEST(ALGOTraderTest, ThrowsCorrectly)
{
    EXPECT_THROW(ALGOTraderVolumeStats{descWithPeriod(0)}, std::invalid_argument);

    // The descriptor grew three more validated fields when the volatility estimate
    // moved in; pin each one so a future reorder of the checks cannot silently drop a
    // guard the way the period check was nearly lost here.
    auto badAlpha = descWithPeriod(5); badAlpha.alpha = -1.0;
    EXPECT_THROW(ALGOTraderVolumeStats{badAlpha}, std::invalid_argument);
    auto badBeta = descWithPeriod(5); badBeta.beta = -1.0;
    EXPECT_THROW(ALGOTraderVolumeStats{badBeta}, std::invalid_argument);
    auto badOmega = descWithPeriod(5); badOmega.omega = 0.0;
    EXPECT_THROW(ALGOTraderVolumeStats{badOmega}, std::invalid_argument);
}

//-------------------------------------------------------------------------
