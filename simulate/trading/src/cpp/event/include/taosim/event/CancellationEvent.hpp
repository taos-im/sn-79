/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/event/Cancellation.hpp>

//-------------------------------------------------------------------------

namespace taosim::event
{

//-------------------------------------------------------------------------

struct CancellationEvent : public JsonSerializable
{
    Cancellation cancellation;
    Timestamp timestamp;
    taosim::decimal_t price;
    // The cancelled order's side, published as "s" so a consumer replaying the book need not guess it.
    OrderDirection direction{OrderDirection::BUY};
    // The cancelled order's leverage, published as "l" as an order event does: the book rests a leveraged
    // order at volume x (1 + leverage), and a consumer replaying a cancel in a later update than the order
    // has no other way to know how much the level loses.
    taosim::decimal_t leverage{};

    CancellationEvent() noexcept = default;

    CancellationEvent(
        Cancellation cancellation, Timestamp timestamp, taosim::decimal_t price,
        OrderDirection direction, taosim::decimal_t leverage = {}) noexcept
        : cancellation{cancellation}, timestamp{timestamp}, price{price}, direction{direction},
          leverage{leverage}
    {}

    virtual void jsonSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;
};

//-------------------------------------------------------------------------

}  // namespace taosim::event

//-------------------------------------------------------------------------
