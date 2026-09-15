/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "AgentReset.hpp"

//-------------------------------------------------------------------------

void AgentResetLogContext::L3Serialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        rapidjson::Value reset{rapidjson::kObjectType};
        reset.AddMember("a", rapidjson::Value{agentId}, allocator);
        reset.AddMember("n", rapidjson::Value{cancelled}, allocator);
        json.AddMember("r", reset, allocator);
        rapidjson::Value group{rapidjson::kObjectType};
        group.AddMember("a", rapidjson::Value{agentId}, allocator);
        group.AddMember("b", rapidjson::Value{bookId}, allocator);
        group.AddMember("j", rapidjson::Value{timestamp}, allocator);
        json.AddMember("g", group, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void AgentResetLogContext::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember("agentId", rapidjson::Value{agentId}, allocator);
        json.AddMember("bookId", rapidjson::Value{bookId}, allocator);
        json.AddMember("timestamp", rapidjson::Value{timestamp}, allocator);
        json.AddMember("cancelled", rapidjson::Value{cancelled}, allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------
