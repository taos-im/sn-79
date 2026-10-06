// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT
//
// The multi-asset wrapper's `seed` must actually reach the engine RNG.
//
// Simulation reads exactly one attribute for its RNG: `rngSeed`. The orchestrator used to stamp
// each realization's derived seed as `seed`, which nothing reads, so no multi-asset run was
// reproducible. The rule now: a wrapper that DECLARES a seed (any value, 0 included) pins every
// realization's `rngSeed` to realizationSeed(master, background, instance); a wrapper without one
// leaves the background's <Simulation> untouched, so a background carrying its own `rngSeed`
// (run/config/band_det/*) keeps it and one without stays device-seeded, as in a single-config run.

#include <gtest/gtest.h>

#include <taosim/simulation/multiasset/MultiAssetConfig.hpp>
#include <taosim/simulation/multiasset/MultiAssetError.hpp>
#include <taosim/simulation/multiasset/helpers.hpp>

#include <pugixml.hpp>

#include <cstdint>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>

using taosim::simulation::multiasset::MultiAssetConfig;
using taosim::simulation::multiasset::MultiAssetError;
using taosim::simulation::multiasset::adjustedBackgroundDoc;
using taosim::simulation::multiasset::realizationSeed;

namespace
{

pugi::xml_document parse(std::string_view xml)
{
    pugi::xml_document doc;
    const auto result = doc.load_buffer(xml.data(), xml.size());
    if (!result) throw std::runtime_error{result.description()};
    return doc;
}

// The wrapper with the given attribute text spliced in; one background suffices for fromXML.
MultiAssetConfig wrapperConfig(std::string_view attrs)
{
    const auto doc = parse(
        "<MultiAssetSimulation " + std::string{attrs} + " step=\"1\" duration=\"1\">"
        "<Background name=\"hv\" path=\"hv.xml\" instanceCount=\"3\"/>"
        "</MultiAssetSimulation>");
    return MultiAssetConfig::fromXML(
        doc.child("MultiAssetSimulation"), std::filesystem::temp_directory_path());
}

// A background in the shape adjustedBackgroundDoc expects, optionally with its own rngSeed.
pugi::xml_document background(std::string_view simuAttrs)
{
    return parse(
        "<Simulation start=\"0\" step=\"1\" duration=\"1\" " + std::string{simuAttrs} + ">"
        "<Agents><MultiBookExchangeAgent/></Agents>"
        "</Simulation>");
}

}  // namespace

//-------------------------------------------------------------------------

TEST(MultiAssetSeed, WrapperWithoutSeedLeavesMasterSeedUnset)
{
    const auto cfg = wrapperConfig("threads=\"2\"");
    EXPECT_FALSE(cfg.masterSeed.has_value());
}

TEST(MultiAssetSeed, WithoutSeedTheBackgroundsOwnRngSeedIsKeptVerbatim)
{
    const auto bg = background("rngSeed=\"31337\"");
    const auto adjusted = adjustedBackgroundDoc(bg.child("Simulation"), std::nullopt, "hv-0");
    const auto simu = adjusted.child("Simulation");

    EXPECT_STREQ(simu.attribute("rngSeed").as_string(), "31337");
    EXPECT_TRUE(simu.attribute("seed").empty()) << "the dead `seed` stamp must not come back";
    EXPECT_STREQ(simu.attribute("id").as_string(), "hv-0");
    // Quote balances are per book, as in a single market: the wrapper no longer stamps a shared
    // wallet (removed 30 September 2026; it gave a miner one book's quote for a whole realization).
    EXPECT_TRUE(simu.child("Agents").child("MultiBookExchangeAgent")
        .attribute("sharedQuoteBalances").empty());
}

TEST(MultiAssetSeed, WithoutSeedAnUnseededBackgroundGetsNoRngSeed)
{
    const auto bg = background("");
    const auto adjusted = adjustedBackgroundDoc(bg.child("Simulation"), std::nullopt, "hv-0");
    const auto simu = adjusted.child("Simulation");

    EXPECT_TRUE(simu.attribute("rngSeed").empty());
    EXPECT_TRUE(simu.attribute("seed").empty());
}

TEST(MultiAssetSeed, WrapperSeedIsParsed)
{
    const auto cfg = wrapperConfig("seed=\"20260101\"");
    ASSERT_TRUE(cfg.masterSeed.has_value());
    EXPECT_EQ(*cfg.masterSeed, 20260101u);
}

TEST(MultiAssetSeed, WithSeedRngSeedIsStampedFromRealizationSeedAndDistinctPerRealization)
{
    const uint64_t master = 20260101u;
    const auto bg = background("rngSeed=\"31337\"");  // must be overridden, not kept

    std::set<uint64_t> stamped;
    for (uint32_t bgIdx = 0; bgIdx < 2; ++bgIdx) {
        for (uint32_t instance = 0; instance < 3; ++instance) {
            const auto expected = realizationSeed(master, bgIdx, instance);
            const auto adjusted =
                adjustedBackgroundDoc(bg.child("Simulation"), expected, "hv-x");
            const auto simu = adjusted.child("Simulation");

            const auto rngSeed = simu.attribute("rngSeed");
            ASSERT_FALSE(rngSeed.empty());
            EXPECT_EQ(rngSeed.as_ullong(), expected) << "bg " << bgIdx << " inst " << instance;
            EXPECT_TRUE(simu.attribute("seed").empty());
            stamped.insert(rngSeed.as_ullong());
        }
    }
    EXPECT_EQ(stamped.size(), 6u) << "realizations must not share an rngSeed";
    EXPECT_EQ(stamped.count(31337u), 0u) << "the background's own rngSeed must be overridden";
}

TEST(MultiAssetSeed, WithSeedTheSourceBackgroundNodeIsLeftUntouched)
{
    const auto bg = background("rngSeed=\"31337\"");
    std::ignore = adjustedBackgroundDoc(bg.child("Simulation"), uint64_t{7}, "hv-0");
    const auto source = bg.child("Simulation");
    EXPECT_STREQ(source.attribute("rngSeed").as_string(), "31337");
    EXPECT_TRUE(source.attribute("id").empty());
    EXPECT_TRUE(source.child("Agents").child("MultiBookExchangeAgent")
        .attribute("sharedQuoteBalances").empty());
}

TEST(MultiAssetSeed, ANonNumericSeedIsRefusedNotReadAsZero)
{
    // as_ullong() would read "abc" as 0 and pin every realization to realizationSeed(0, ...)
    // without a word; a declared seed that is not an unsigned integer is a config error.
    EXPECT_THROW(wrapperConfig("seed=\"abc\""), MultiAssetError);
    EXPECT_THROW(wrapperConfig("seed=\"-1\""), MultiAssetError);
    EXPECT_THROW(wrapperConfig("seed=\"12x\""), MultiAssetError);
    EXPECT_THROW(wrapperConfig("seed=\"\""), MultiAssetError) << "declared but blank is a typo too";
}

TEST(MultiAssetSeed, SeedZeroIsADeclaredSeed)
{
    const auto cfg = wrapperConfig("seed=\"0\"");
    ASSERT_TRUE(cfg.masterSeed.has_value());
    EXPECT_EQ(*cfg.masterSeed, 0u);

    const auto bg = background("");
    const auto adjusted = adjustedBackgroundDoc(
        bg.child("Simulation"), realizationSeed(*cfg.masterSeed, 0, 0), "hv-0");
    const auto rngSeed = adjusted.child("Simulation").attribute("rngSeed");
    ASSERT_FALSE(rngSeed.empty());
    EXPECT_EQ(rngSeed.as_ullong(), realizationSeed(0, 0, 0));
}
