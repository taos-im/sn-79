/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "CheckpointSerializable.hpp"

#include <memory>
#include <span>
#include <unordered_set>
#include <vector>

//-------------------------------------------------------------------------

namespace taosim::util
{

//-------------------------------------------------------------------------

template<typename T>
class SubscriptionRegistry : public CheckpointSerializable
{
public:
    SubscriptionRegistry() noexcept = default;

    [[nodiscard]] std::span<const T> subs() const noexcept { return m_subs; }

    [[nodiscard]] bool contains(const T& sub) const noexcept { return m_registry.contains(sub); }

    bool add(const T& sub) noexcept
    {
        if (m_registry.contains(sub)) {
            return false;
        }
        m_subs.push_back(sub);
        m_registry.insert(sub);
        return true;
    }

    auto begin() const noexcept { return m_subs.begin(); }
    auto end() const noexcept { return m_subs.end(); }

    virtual void checkpointSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override
    {
        auto serialize = [this](rapidjson::Document& json) {
            json.SetArray();
            auto& allocator = json.GetAllocator();
            for (const T& sub : *this) {
                if constexpr (requires(T t) { t.checkpointSerialize(json); }) {
                    rapidjson::Document subJson{&allocator};
                    sub.checkpointSerialize(subJson);
                    json.PushBack(subJson, allocator);
                } else if constexpr (requires(T t) { t.c_str(); }) {
                    json.PushBack(rapidjson::Value{sub.c_str(), allocator}, allocator);
                } else if constexpr (requires(T t) { json.PushBack(t, allocator); }) {
                    json.PushBack(sub, allocator);
                } else {
                    static_assert(false);
                }
            }
        };
        taosim::json::serializeHelper(json, key, serialize);
    }

    [[nodiscard]] static SubscriptionRegistry<T> fromJson(const rapidjson::Value& json)
    {
        SubscriptionRegistry reg;
        for (const rapidjson::Value& subJson : json.GetArray()) {
            if constexpr (std::signed_integral<T>) {
                reg.add(subJson.GetInt());
            } else if constexpr (std::unsigned_integral<T>) {
                reg.add(subJson.GetUint());
            } else if constexpr (std::convertible_to<const char*, T>) {
                reg.add(subJson.GetString());
            } else {
                static_assert(false);
            }
        }
        return reg;
    }

private:
    std::vector<T> m_subs;
    std::unordered_set<T> m_registry;
};

//-------------------------------------------------------------------------

}  // namespace taosim::util

//-------------------------------------------------------------------------
