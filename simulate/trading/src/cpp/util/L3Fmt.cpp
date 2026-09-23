/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "L3Fmt.hpp"

#include <taosim/message/ExchangeAgentMessagePayloads.hpp>

#include <magic_enum/magic_enum.hpp>

#include <optional>
#include <string_view>
#include <utility>
#include <variant>

//-------------------------------------------------------------------------
// A thin JSON-value layer over the types' own formatters: strings/enum names get
// quotes, empty optionals become null, settle flags pick quoted-enum or plain id,
// and decimals go through fmt::formatter<decimal_t> (quantum digits, trailing
// zeros trimmed) — values are rounded upstream by their own increment schemes,
// so no formatting-side rounding happens here.

namespace
{

template<typename T>
struct Json
{
    const T& value;
};

template<typename T>
Json(const T&) -> Json<T>;

}  // namespace

template<typename T>
struct fmt::formatter<Json<T>>
{
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    auto format(const Json<T>& json, FormatContext& ctx) const
    {
        const auto& value = json.value;
        if constexpr (requires { std::string_view{value}; }) {
            return fmt::format_to(ctx.out(), "\"{}\"", std::string_view{value});
        }
        else if constexpr (requires { value.has_value(); }) {
            return value.has_value()
                ? fmt::format_to(ctx.out(), "{}", Json{*value})
                : fmt::format_to(ctx.out(), "null");
        }
        else if constexpr (std::same_as<T, SettleFlag>) {
            return std::visit(
                [&](auto&& flag) {
                    using F = std::remove_cvref_t<decltype(flag)>;
                    if constexpr (std::same_as<F, SettleType>) {
                        return fmt::format_to(
                            ctx.out(), "{}", Json{magic_enum::enum_name(flag)});
                    } else {
                        return fmt::format_to(ctx.out(), "{}", flag);
                    }
                },
                value);
        }
        else if constexpr (std::is_enum_v<T>) {
            return fmt::format_to(ctx.out(), "{}", Json{magic_enum::enum_name(value)});
        }
        else {
            return fmt::format_to(ctx.out(), "{}", value);
        }
    }
};

//-------------------------------------------------------------------------

namespace taosim::l3fmt
{

//-------------------------------------------------------------------------

namespace
{

void orderBody(fmt::memory_buffer& out, const Order& order)
{
    fmt::format_to(
        fmt::appender{out},
        R"("i":{},"j":{},"v":{},"d":{},"l":{},"s":{},"f":{},"n":{})",
        order.id(), order.timestamp(), order.volume(),
        std::to_underlying(order.direction()), order.leverage(),
        Json{order.stpFlag()}, Json{order.settleFlag()}, Json{order.currency()}
    );
}

void orderTail(fmt::memory_buffer& out, const Order& order)
{
    fmt::format_to(
        fmt::appender{out},
        R"(,"sl":{},"tp":{},"ph":{})",
        Json{order.stopLoss()}, Json{order.takeProfit()}, Json{order.placeholder()}
    );
}

}  // namespace

//-------------------------------------------------------------------------

void formatItem(
    fmt::memory_buffer& out,
    const OrderWithLogContext& item,
    BookId bookIdCanon,
    uint64_t eventId)
{
    fmt::format_to(fmt::appender{out}, R"({{"o":{{)");
    orderBody(out, *item.order);
    if (const auto limitOrder = std::dynamic_pointer_cast<LimitOrder>(item.order)) {
        fmt::format_to(
            fmt::appender{out},
            R"(,"p":{},"y":{},"r":{},"x":{})",
            limitOrder->price(), limitOrder->postOnly(),
            Json{limitOrder->timeInForce()}, Json{limitOrder->expiryPeriod()}
        );
    }
    orderTail(out, *item.order);
    fmt::format_to(
        fmt::appender{out},
        R"(}},"g":{{"a":{},"b":{}}},"k":{}}})",
        item.logContext->agentId, bookIdCanon, eventId
    );
}

//-------------------------------------------------------------------------

void formatItem(
    fmt::memory_buffer& out,
    const CancellationWithLogContext& item,
    BookId bookIdCanon,
    uint64_t eventId)
{
    fmt::format_to(
        fmt::appender{out},
        R"({{"c":{{"e":"cancel","i":{},"v":{}}},"g":{{"a":{},"b":{},"j":{}}},"k":{}}})",
        item.cancellation.id, Json{item.cancellation.volume},
        item.logContext->agentId, bookIdCanon, item.logContext->timestamp, eventId
    );
}

//-------------------------------------------------------------------------

void formatItem(
    fmt::memory_buffer& out,
    const TradeWithLogContext& item,
    BookId bookIdCanon,
    uint64_t eventId)
{
    const auto& trade = *item.trade;
    const auto& ctx = *item.logContext;
    fmt::format_to(
        fmt::appender{out},
        R"({{"t":{{"m":{},"j":{},"d":{},"ai":{},"ri":{},"v":{},"p":{}}},)"
        R"("g":{{"aa":{},"ra":{},"b":{},"fs":{{"mk":{},"tk":{}}}}},"k":{}}})",
        trade.id(), trade.timestamp(), std::to_underlying(trade.direction()),
        trade.aggressingOrderID(), trade.restingOrderID(), trade.volume(), trade.price(),
        ctx.aggressingAgentId, ctx.restingAgentId, bookIdCanon,
        ctx.fees.maker, ctx.fees.taker, eventId
    );
}

//-------------------------------------------------------------------------

void formatItem(
    fmt::memory_buffer& out,
    const AgentResetLogContext& item,
    BookId bookIdCanon,
    uint64_t eventId)
{
    // The record carries its own book and timestamp rather than a log context: a reset is
    // recorded once per book, including for an agent with nothing resting there.
    fmt::format_to(
        fmt::appender{out},
        R"({{"r":{{"a":{},"n":{}}},"g":{{"a":{},"b":{},"j":{}}},"k":{}}})",
        item.agentId, item.cancelled,
        item.agentId, bookIdCanon, item.timestamp, eventId
    );
}

//-------------------------------------------------------------------------

void formatItem(
    fmt::memory_buffer& out, const taosim::InstructionLogContext& item, uint64_t eventId)
{
    fmt::format_to(fmt::appender{out}, R"({{"in":{{)");
    std::visit(
        [&](auto&& pld) {
            using T = typename std::remove_cvref_t<decltype(pld)>::element_type;
            if constexpr (std::same_as<T, PlaceOrderMarketPayload>) {
                fmt::format_to(
                    fmt::appender{out},
                    R"("d":{},"v":{},"b":{},"n":{},"ci":{},"s":{},"l":{},"f":{})",
                    std::to_underlying(pld->direction), pld->volume, pld->bookId,
                    std::to_underlying(pld->currency), Json{pld->clientOrderId},
                    Json{pld->stpFlag}, pld->leverage, Json{pld->settleFlag});
            } else {
                fmt::format_to(
                    fmt::appender{out},
                    R"("d":{},"v":{},"p":{},"l":{},"b":{},"n":{},"ci":{},)"
                    R"("y":{},"r":{},"x":{},"s":{},"f":{})",
                    std::to_underlying(pld->direction), pld->volume, pld->price,
                    pld->leverage, pld->bookId, std::to_underlying(pld->currency),
                    Json{pld->clientOrderId}, pld->postOnly, Json{pld->timeInForce},
                    Json{pld->expiryPeriod}, Json{pld->stpFlag}, Json{pld->settleFlag});
            }
            fmt::format_to(
                fmt::appender{out},
                R"(,"sl":{},"tp":{},"ph":{})",
                Json{pld->stopLoss}, Json{pld->takeProfit}, Json{pld->placeholder});
        },
        item.payload
    );
    fmt::format_to(
        fmt::appender{out},
        R"(}},"a":{},"i":{},"k":{}}})",
        item.agentId, item.orderId, eventId
    );
}

//-------------------------------------------------------------------------

}  // namespace taosim::l3fmt

//-------------------------------------------------------------------------
