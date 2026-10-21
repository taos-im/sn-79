// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT

#include <taosim/simulation/LifecycleNotice.hpp>

#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/message/MessagePayload.hpp>

#include <boost/asio/io_context.hpp>

#include <algorithm>
#include <vector>

//-------------------------------------------------------------------------

namespace taosim::simulation
{

//-------------------------------------------------------------------------

// Occurrence and arrival are both the grid start, so the serialized `delay`
// (arrival - occurrence, unsigned) is 0 for any start; the single-config manager
// passes 0 as the arrival, which only happens to serialize as 0 when start is 0.

Message::Ptr makeStartNotice(uint64_t start, const std::filesystem::path& logDir)
{
    return Message::create(
        start,
        start,
        "SIMULATION",
        "*",
        "EVENT_SIMULATION_START",
        MessagePayload::create<StartSimulationPayload>(logDir.generic_string()));
}

//-------------------------------------------------------------------------

Message::Ptr makeEndNotice(uint64_t start)
{
    return Message::create(
        start,
        start,
        "SIMULATION",
        "*",
        "EVENT_SIMULATION_END",
        MessagePayload::create<EmptyPayload>());
}

//-------------------------------------------------------------------------

void SimulationFailedPayload::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("realization", rapidjson::Value{realization}, allocator);
        json.AddMember("background", rapidjson::Value{background.c_str(), allocator}, allocator);
        json.AddMember("time", rapidjson::Value{time}, allocator);
        json.AddMember("error", rapidjson::Value{error.c_str(), allocator}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

Message::Ptr makeFailureNotice(
    uint32_t realization, std::string_view background, uint64_t time, std::string_view error)
{
    // Back off to the start of a UTF-8 sequence, so the clipped text is still valid JSON text.
    size_t length = std::min(error.size(), kFailureNoticeErrorLength);
    if (length < error.size()) {
        while (length > 0 && (static_cast<unsigned char>(error[length]) & 0xC0) == 0x80) --length;
    }
    return Message::create(
        time,
        time,
        "SIMULATION",
        "*",
        "EVENT_SIMULATION_FAILED",
        MessagePayload::create<SimulationFailedPayload>(
            realization, std::string{background}, time, std::string{error.substr(0, length)}));
}

//-------------------------------------------------------------------------

rapidjson::Document makeNoticeBatch(std::span<const Message::Ptr> messages)
{
    rapidjson::Document json{rapidjson::kObjectType};
    auto& allocator = json.GetAllocator();
    rapidjson::Document messagesJson{rapidjson::kArrayType, &allocator};
    for (const auto& message : messages) {
        rapidjson::Document messageJson{&allocator};
        message->jsonSerialize(messageJson);
        messagesJson.PushBack(messageJson, allocator);
    }
    json.AddMember("messages", messagesJson.Move(), allocator);
    return json;
}

//-------------------------------------------------------------------------

void postGeneralMessage(
    const Simulation& logger,
    const net::NetworkingInfo& netInfo,
    std::string_view endpoint,
    const rapidjson::Document& body,
    bool failFast)
{
    rapidjson::Document res;
    net::asio::io_context ctx;
    net::asio::co_spawn(
        ctx,
        net::asyncSendOverNetwork(net::AsyncSendContext{
            .logger = logger,
            .netInfo = netInfo,
            .reqBody = body,
            .endpoint = endpoint,
            .resJson = res,
            .failFast = failFast
        }),
        net::asio::detached);
    ctx.run();
}

//-------------------------------------------------------------------------

}  // namespace taosim::simulation

//-------------------------------------------------------------------------
