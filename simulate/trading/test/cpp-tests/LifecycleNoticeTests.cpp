// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT
//
// The lifecycle notices the multi-asset orchestrator posts to the validator must be the ones the
// validator already parses: the /account route walks {"messages": [...]} and hands each
// EVENT_SIMULATION_START (payload.logDir) to on_start and EVENT_SIMULATION_END to on_end. The
// multi-asset run sends one START per realization in one batch; these pin that batch's shape and
// the wrapper attributes that address it, without a network or an engine.

#include <gtest/gtest.h>

#include <taosim/simulation/LifecycleNotice.hpp>
#include <taosim/simulation/multiasset/MultiAssetConfig.hpp>

#include <pugixml.hpp>
#include <rapidjson/document.h>

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using taosim::simulation::makeEndNotice;
using taosim::simulation::makeNoticeBatch;
using taosim::simulation::makeStartNotice;
using taosim::simulation::multiasset::MultiAssetConfig;

namespace
{

pugi::xml_document parse(std::string_view xml)
{
    pugi::xml_document doc;
    const auto result = doc.load_buffer(xml.data(), xml.size());
    if (!result) throw std::runtime_error{result.description()};
    return doc;
}

MultiAssetConfig wrapperConfig(std::string_view attrs)
{
    const auto doc = parse(
        "<MultiAssetSimulation " + std::string{attrs} + " step=\"1\" duration=\"1\">"
        "<Background name=\"hv\" path=\"hv.xml\" instanceCount=\"3\"/>"
        "</MultiAssetSimulation>");
    return MultiAssetConfig::fromXML(
        doc.child("MultiAssetSimulation"), std::filesystem::temp_directory_path());
}

}  // namespace

//-------------------------------------------------------------------------

TEST(LifecycleNotice, StartBatchCarriesOneMessagePerRealizationInTheValidatorsShape)
{
    const std::vector<std::string> logDirs{
        "/run/20260917/simulation_0-0", "/run/20260917/highVol-0", "/run/20260917/highVol-1"};
    std::vector<Message::Ptr> notices;
    for (const auto& dir : logDirs) {
        notices.push_back(makeStartNotice(0, dir));
    }

    const auto batch = makeNoticeBatch(notices);
    ASSERT_TRUE(batch.IsObject());
    ASSERT_TRUE(batch.HasMember("messages"));
    const auto& messages = batch["messages"];
    ASSERT_TRUE(messages.IsArray());
    ASSERT_EQ(messages.Size(), 3u);

    for (rapidjson::SizeType i = 0; i < messages.Size(); ++i) {
        const auto& message = messages[i];
        ASSERT_TRUE(message.IsObject()) << "message " << i;
        EXPECT_STREQ(message["type"].GetString(), "EVENT_SIMULATION_START");
        EXPECT_EQ(message["timestamp"].GetUint64(), 0u);
        EXPECT_EQ(message["delay"].GetUint64(), 0u);
        EXPECT_STREQ(message["source"].GetString(), "SIMULATION");
        EXPECT_STREQ(message["target"].GetString(), "*");
        ASSERT_TRUE(message.HasMember("payload") && message["payload"].IsObject());
        EXPECT_STREQ(message["payload"]["logDir"].GetString(), logDirs[i].c_str());
    }
}

TEST(LifecycleNotice, StartTimestampIsTheGridStartAndDelayStaysZero)
{
    // `delay` serializes as arrival - occurrence on an unsigned type: with arrival 0 and a
    // non-zero start it would wrap to 2^64 - start. Both notices pin it at 0 for any start.
    const std::vector notices{
        makeStartNotice(1'700'000'000'000ull, "/run/x/bg0-0"), makeEndNotice(1'700'000'000'000ull)};
    const auto batch = makeNoticeBatch(notices);
    for (rapidjson::SizeType i = 0; i < 2; ++i) {
        EXPECT_EQ(batch["messages"][i]["timestamp"].GetUint64(), 1'700'000'000'000ull);
        EXPECT_EQ(batch["messages"][i]["delay"].GetUint64(), 0u) << "message " << i;
    }
}

TEST(LifecycleNotice, EndNoticeNamesNoLogDir)
{
    const std::vector notices{makeEndNotice(0)};
    const auto batch = makeNoticeBatch(notices);
    ASSERT_EQ(batch["messages"].Size(), 1u);
    const auto& message = batch["messages"][0];
    EXPECT_STREQ(message["type"].GetString(), "EVENT_SIMULATION_END");
    EXPECT_STREQ(message["source"].GetString(), "SIMULATION");
    EXPECT_STREQ(message["target"].GetString(), "*");
    EXPECT_FALSE(message.HasMember("payload") && message["payload"].IsObject()
        && message["payload"].HasMember("logDir"));
}

TEST(LifecycleNotice, AnEmptyBatchIsStillAWellFormedEnvelope)
{
    const std::vector<Message::Ptr> none;
    const auto batch = makeNoticeBatch(none);
    ASSERT_TRUE(batch.HasMember("messages"));
    EXPECT_EQ(batch["messages"].Size(), 0u);
}

//-------------------------------------------------------------------------

TEST(LifecycleNotice, WrapperWithoutHostAndPortIsOffline)
{
    const auto cfg = wrapperConfig("useMessagePack=\"1\"");
    EXPECT_TRUE(cfg.netInfo.host.empty());
    EXPECT_TRUE(cfg.netInfo.port.empty());
    EXPECT_EQ(cfg.generalMsgEndpoint, "/");
    // SimulationManager's defaults, so one validator config serves either run mode.
    EXPECT_EQ(cfg.netInfo.resolveTimeout, 1);
    EXPECT_EQ(cfg.netInfo.connectTimeout, 3);
    EXPECT_EQ(cfg.netInfo.writeTimeout, 15);
    EXPECT_EQ(cfg.netInfo.readTimeout, 60);
}

TEST(LifecycleNotice, WrapperNetworkAttributesAreReadFromTheRoot)
{
    const auto cfg = wrapperConfig(
        "host=\"localhost\" port=\"8000\" generalMsgEndpoint=\"/account\" readTimeout=\"7\"");
    EXPECT_EQ(cfg.netInfo.host, "localhost");
    EXPECT_EQ(cfg.netInfo.port, "8000");
    EXPECT_EQ(cfg.generalMsgEndpoint, "/account");
    EXPECT_EQ(cfg.netInfo.readTimeout, 7);
    EXPECT_EQ(cfg.netInfo.connectTimeout, 3) << "an attribute left out keeps its default";
}

// THE SL/TP TRACER IS A WRAPPER SETTING. SimulationManager reads `sltpDebug` from its own root, but
// a multi-asset run never builds through that path, so the attribute was dead under the wrapper and
// in every background file that set it: the 7 October acceptance run could not show the trigger
// table its SL/TP check needed. The wrapper root carries it now, like the other wrapper settings.
TEST(LifecycleNotice, WrapperReadsTheSltpTracerFlag)
{
    EXPECT_TRUE(wrapperConfig("sltpDebug=\"1\"").sltpDebug);
    EXPECT_FALSE(wrapperConfig("").sltpDebug) << "off unless the wrapper asks for it";
}

// THE STEP DIAGNOSTICS ARE WRAPPER SETTINGS TOO. traceTime prints the sim clock each step and
// measureStepWallClockTime the PROCESSED wall-clock breakdown; SimulationManager reads both from its root,
// a multi-asset run never builds through it, and the background files that set them were ignored, so a
// 0.6.3 validator's engine log carried neither line.
TEST(LifecycleNotice, WrapperReadsTheStepDiagnostics)
{
    const auto on = wrapperConfig("traceTime=\"1\" measureStepWallClockTime=\"1\"");
    EXPECT_TRUE(on.traceTime);
    EXPECT_TRUE(on.measureStepWallClockTime);
    const auto off = wrapperConfig("");
    EXPECT_FALSE(off.traceTime);
    EXPECT_FALSE(off.measureStepWallClockTime);
}

