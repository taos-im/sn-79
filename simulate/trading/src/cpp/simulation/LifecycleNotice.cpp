// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT

#include <taosim/simulation/LifecycleNotice.hpp>

#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/message/MessagePayload.hpp>

#include <boost/asio/io_context.hpp>

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
    const rapidjson::Document& body)
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
            .resJson = res
        }),
        net::asio::detached);
    ctx.run();
}

//-------------------------------------------------------------------------

}  // namespace taosim::simulation

//-------------------------------------------------------------------------
