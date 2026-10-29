/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/simulation/util.hpp>

#include "Simulation.hpp"
#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/message/MessagePayload.hpp>
#include <taosim/message/MultiBookMessagePayloads.hpp>

#include <fmt/core.h>

//-------------------------------------------------------------------------

namespace taosim::simulation
{

//-------------------------------------------------------------------------

Message::Ptr canonize(Message::Ptr msg, uint32_t blockIdx, uint32_t blockDim)
{
    const auto payload = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);

    if (payload == nullptr) return msg;

    auto canonizeBookId = [=](BookId& bookId) -> void {
        bookId = blockIdx * blockDim + bookId;
    };

    if (const auto pld = std::dynamic_pointer_cast<PlaceOrderMarketPayload>(payload->payload)) {
        canonizeBookId(pld->bookId);
    }
    else if (const auto pld = std::dynamic_pointer_cast<PlaceOrderMarketResponsePayload>(payload->payload)) {
        canonizeBookId(pld->requestPayload->bookId);
    }
    else if (const auto pld = std::dynamic_pointer_cast<PlaceOrderMarketErrorResponsePayload>(payload->payload)) {
        canonizeBookId(pld->requestPayload->bookId);
    }
    else if (const auto pld = std::dynamic_pointer_cast<PlaceOrderLimitPayload>(payload->payload)) {
        canonizeBookId(pld->bookId);
    }
    else if (const auto pld = std::dynamic_pointer_cast<PlaceOrderLimitResponsePayload>(payload->payload)) {
        canonizeBookId(pld->requestPayload->bookId);
    }
    else if (const auto pld = std::dynamic_pointer_cast<PlaceOrderLimitErrorResponsePayload>(payload->payload)) {
        canonizeBookId(pld->requestPayload->bookId);
    }
    else if (const auto pld = std::dynamic_pointer_cast<RetrieveOrdersPayload>(payload->payload)) {
        canonizeBookId(pld->bookId);
    }
    else if (const auto pld = std::dynamic_pointer_cast<RetrieveOrdersResponsePayload>(payload->payload)) {
        canonizeBookId(pld->bookId);
    }
    else if (const auto pld = std::dynamic_pointer_cast<CancelOrdersPayload>(payload->payload)) {
        canonizeBookId(pld->bookId);
    }
    else if (const auto pld = std::dynamic_pointer_cast<ClosePositionsPayload>(payload->payload)) {
        canonizeBookId(pld->bookId);
    }
    else if (const auto pld = std::dynamic_pointer_cast<CancelOrdersResponsePayload>(payload->payload)) {
        canonizeBookId(pld->requestPayload->bookId);
    }
    else if (const auto pld = std::dynamic_pointer_cast<CancelOrdersErrorResponsePayload>(payload->payload)) {
        canonizeBookId(pld->requestPayload->bookId);
    }
    else if (const auto pld = std::dynamic_pointer_cast<RetrieveL1Payload>(payload->payload)) {
        canonizeBookId(pld->bookId);
    }
    else if (const auto pld = std::dynamic_pointer_cast<RetrieveL1ResponsePayload>(payload->payload)) {
        canonizeBookId(pld->bookId);
    }
    else if (const auto pld = std::dynamic_pointer_cast<RetrieveL1ExtPayload>(payload->payload)) {
        canonizeBookId(pld->bookId);
    }
    else if (const auto pld = std::dynamic_pointer_cast<RetrieveL1ExtResponsePayload>(payload->payload)) {
        canonizeBookId(pld->bookId);
    }
    else if (const auto pld = std::dynamic_pointer_cast<RetrieveL2Payload>(payload->payload)) {
        canonizeBookId(pld->bookId);
    }
    else if (const auto pld = std::dynamic_pointer_cast<RetrieveL2ResponsePayload>(payload->payload)) {
        canonizeBookId(pld->bookId);
    }
    else if (const auto pld = std::dynamic_pointer_cast<EventTradePayload>(payload->payload)) {
        canonizeBookId(pld->bookId);
        canonizeBookId(pld->context.bookId);
    }

    return msg;
}

//-------------------------------------------------------------------------

namespace
{

// Applies `decanonizeBookId` to the book id(s) a validator instruction names and hands
// back the canonical id it carried, or nothing when the payload names no book.
template<typename Rewrite>
std::optional<BookId> rewriteBookId(
    const std::shared_ptr<DistributedAgentResponsePayload>& payload, Rewrite&& decanonizeBookId)
{
    return [&] -> std::optional<BookId> {
        if (const auto pld = std::dynamic_pointer_cast<PlaceOrderMarketPayload>(payload->payload)) {
            return decanonizeBookId(pld->bookId);
        }
        if (const auto pld = std::dynamic_pointer_cast<PlaceOrderMarketResponsePayload>(payload->payload)) {
            return decanonizeBookId(pld->requestPayload->bookId);
        }
        if (const auto pld = std::dynamic_pointer_cast<PlaceOrderMarketErrorResponsePayload>(payload->payload)) {
            return decanonizeBookId(pld->requestPayload->bookId);
        }
        if (const auto pld = std::dynamic_pointer_cast<PlaceOrderLimitPayload>(payload->payload)) {
            return decanonizeBookId(pld->bookId);
        }
        if (const auto pld = std::dynamic_pointer_cast<PlaceOrderLimitResponsePayload>(payload->payload)) {
            return decanonizeBookId(pld->requestPayload->bookId);
        }
        if (const auto pld = std::dynamic_pointer_cast<PlaceOrderLimitErrorResponsePayload>(payload->payload)) {
            return decanonizeBookId(pld->requestPayload->bookId);
        }
        if (const auto pld = std::dynamic_pointer_cast<RetrieveOrdersPayload>(payload->payload)) {
            return decanonizeBookId(pld->bookId);
        }
        if (const auto pld = std::dynamic_pointer_cast<RetrieveOrdersResponsePayload>(payload->payload)) {
            return decanonizeBookId(pld->bookId);
        }
        if (const auto pld = std::dynamic_pointer_cast<CancelOrdersPayload>(payload->payload)) {
            return decanonizeBookId(pld->bookId);
        }
        if (const auto pld = std::dynamic_pointer_cast<ClosePositionsPayload>(payload->payload)) {
            return decanonizeBookId(pld->bookId);
        }
        if (const auto pld = std::dynamic_pointer_cast<CancelOrdersResponsePayload>(payload->payload)) {
            return decanonizeBookId(pld->requestPayload->bookId);
        }
        if (const auto pld = std::dynamic_pointer_cast<CancelOrdersErrorResponsePayload>(payload->payload)) {
            return decanonizeBookId(pld->requestPayload->bookId);
        }
        if (const auto pld = std::dynamic_pointer_cast<RetrieveL1Payload>(payload->payload)) {
            return decanonizeBookId(pld->bookId);
        }
        if (const auto pld = std::dynamic_pointer_cast<RetrieveL1ResponsePayload>(payload->payload)) {
            return decanonizeBookId(pld->bookId);
        }
        if (const auto pld = std::dynamic_pointer_cast<RetrieveL1ExtPayload>(payload->payload)) {
            return decanonizeBookId(pld->bookId);
        }
        if (const auto pld = std::dynamic_pointer_cast<RetrieveL1ExtResponsePayload>(payload->payload)) {
            return decanonizeBookId(pld->bookId);
        }
        if (const auto pld = std::dynamic_pointer_cast<RetrieveL2Payload>(payload->payload)) {
            return decanonizeBookId(pld->bookId);
        }
        if (const auto pld = std::dynamic_pointer_cast<RetrieveL2ResponsePayload>(payload->payload)) {
            return decanonizeBookId(pld->bookId);
        }
        if (const auto pld = std::dynamic_pointer_cast<EventTradePayload>(payload->payload)) {
            decanonizeBookId(pld->bookId);
            return decanonizeBookId(pld->context.bookId);
        }
        return {};
    }();
}

}  // namespace

//-------------------------------------------------------------------------

DecanonizeResult decanonize(Message::Ptr msg, uint32_t blockDim, uint32_t blockCount)
{
    const auto payload = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);

    if (payload == nullptr) return {.msg = msg, .blockIdx = {}};

    std::optional<uint32_t> block;
    const auto bookIdCanon = rewriteBookId(payload, [&](BookId& bookId) -> BookId {
        if (bookId / blockDim >= blockCount) return bookId;
        block = bookId / blockDim;
        return std::exchange(bookId, bookId % blockDim);
    });

    return {
        .msg = msg,
        .blockIdx = block,
        .unrouted = bookIdCanon.has_value() && !block.has_value()
    };
}

//-------------------------------------------------------------------------

std::optional<RealizationRoute> routeCanonicalBookId(
    uint32_t canon, std::span<const uint32_t> bases, std::span<const uint32_t> counts)
{
    for (size_t i = bases.size(); i-- > 0;) {
        if (canon < bases[i]) continue;
        if (canon - bases[i] < counts[i]) {
            return RealizationRoute{.idx = static_cast<uint32_t>(i), .local = canon - bases[i]};
        }
        return {};
    }
    return {};
}

//-------------------------------------------------------------------------

DecanonizeResult decanonize(
    Message::Ptr msg, std::span<const uint32_t> bases, std::span<const uint32_t> counts)
{
    const auto payload = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);

    if (payload == nullptr) return {.msg = msg, .blockIdx = {}};

    std::optional<uint32_t> realization;
    const auto bookIdCanon = rewriteBookId(payload, [&](BookId& bookId) -> BookId {
        if (const auto route = routeCanonicalBookId(bookId, bases, counts)) {
            realization = route->idx;
            return std::exchange(bookId, route->local);
        }
        return bookId;
    });

    return {
        .msg = msg,
        .blockIdx = realization,
        .unrouted = bookIdCanon.has_value() && !realization.has_value()
    };
}

//-------------------------------------------------------------------------

void queueValidatorResponse(
    Message::Ptr response, const std::vector<std::unique_ptr<Simulation>>& blocks, uint32_t blockDim)
{
    const auto [msg, blockIdx, unrouted] =
        decanonize(std::move(response), blockDim, static_cast<uint32_t>(blocks.size()));
    if (unrouted) {
        fmt::println("Dropping a validator instruction that names a book no block owns");
        return;
    }
    if (!blockIdx) {
        for (const auto& block : blocks) {
            block->queueMessage(msg);
        }
        return;
    }
    blocks.at(*blockIdx)->queueMessage(msg);
}

//-------------------------------------------------------------------------

std::optional<std::vector<CappedPair>> readCappedPairs(const msgpack::object& batch)
{
    if (batch.type != msgpack::type::MAP) return std::nullopt;
    const msgpack::object* capped{};
    for (const auto& [k, v] : batch.via.map) {
        if (k.type == msgpack::type::STR && k.as<std::string_view>() == "capped") capped = &v;
    }
    if (capped == nullptr) return std::nullopt;
    std::vector<CappedPair> pairs;
    if (capped->type != msgpack::type::ARRAY) return pairs;
    for (const auto& entry : capped->via.array) {
        if (entry.type != msgpack::type::ARRAY || entry.via.array.size != 2) continue;
        const auto& agent = entry.via.array.ptr[0];
        const auto& book = entry.via.array.ptr[1];
        if (agent.type != msgpack::type::POSITIVE_INTEGER || book.type != msgpack::type::POSITIVE_INTEGER) continue;
        pairs.emplace_back(agent.as<AgentId>(), book.as<BookId>());
    }
    return pairs;
}

//-------------------------------------------------------------------------

std::vector<std::vector<CappedPair>> splitCappedPairs(
    std::span<const CappedPair> pairs, std::span<const uint32_t> bases, std::span<const uint32_t> counts)
{
    std::vector<std::vector<CappedPair>> split(bases.size());
    for (const auto& [agent, book] : pairs) {
        if (const auto route = routeCanonicalBookId(static_cast<uint32_t>(book), bases, counts)) {
            split[route->idx].emplace_back(agent, route->local);
        }
    }
    return split;
}

//-------------------------------------------------------------------------

std::vector<std::vector<CappedPair>> splitCappedPairs(
    std::span<const CappedPair> pairs, uint32_t blockDim, uint32_t blockCount)
{
    std::vector<std::vector<CappedPair>> split(blockCount);
    for (const auto& [agent, book] : pairs) {
        if (book / blockDim < blockCount) split[book / blockDim].emplace_back(agent, book % blockDim);
    }
    return split;
}

//-------------------------------------------------------------------------

}  // namespace taosim::simulation

//-------------------------------------------------------------------------