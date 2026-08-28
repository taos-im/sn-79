/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/accounting/Account.hpp>

//-------------------------------------------------------------------------

namespace taosim::accounting
{

//-------------------------------------------------------------------------

Account::Account(uint32_t bookCount, std::optional<Balances> balances) noexcept
{
    m_holdings = balances.has_value()
        ? views::iota(decltype(bookCount){}, bookCount)
            | views::transform([&](auto) -> Balances { return balances.value(); })
            | ranges::to<std::vector>
        : decltype(m_holdings)(bookCount);
    m_activeOrders.resize(bookCount);
}

//-------------------------------------------------------------------------

Account::Account(Account::Holdings balances) noexcept
{
    m_holdings = std::move(balances);
    m_activeOrders.resize(m_holdings.size());
}

//-------------------------------------------------------------------------

void Account::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        json::serializeHelper(
            json,
            "holdings",
            [this](rapidjson::Document& json) {
                json.SetArray();
                auto& allocator = json.GetAllocator();
                for (const auto& balances : *this) {
                    rapidjson::Document balancesJson{&allocator};
                    balances.jsonSerialize(balancesJson);
                    json.PushBack(balancesJson, allocator);
                }
            });
        json::serializeHelper(
            json,
            "activeOrders",
            [this](rapidjson::Document& json) {
                json.SetArray();
                auto& allocator = json.GetAllocator();
                for (const auto& activeOrders : m_activeOrders) {
                    rapidjson::Document orderArrayJson{rapidjson::kArrayType, &allocator};
                    for (const auto order : activeOrders) {
                        rapidjson::Document orderJson{&allocator};
                        order->jsonSerialize(orderJson);
                        orderArrayJson.PushBack(orderJson, allocator);
                    }
                    json.PushBack(orderArrayJson, allocator);
                }
            });
    };
    json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

std::ostream& operator<<(std::ostream& os, const Account& account)
{
    BookId bookId = 0;
    for (const auto& balances : account) {
        os << fmt::format("Book {}\n{}", bookId++, balances);
    }
    return os;
}

//-------------------------------------------------------------------------

Account Account::fromJson(const rapidjson::Value& json)
{
    return Account{
        [&] -> Account::Holdings {
            // One quote per AGENT, shared across books, matching helpers::makeHoldings. Balances::fromJson
            // allocates a new quote per entry, so restoring without this gives every book its own TAO
            // balance and only book 0 ever gets reconciled to the chain. See AccountRegistry::registerJson.
            Account::Holdings holdings;
            const rapidjson::Value& holdingsArrayJson = json["holdings"].GetArray();
            const uint32_t bookCount = holdingsArrayJson.Size();
            std::shared_ptr<Balance> sharedQuote;
            for (BookId bookId = 0; bookId < bookCount; ++bookId) {
                auto bal = Balances::fromJson(holdingsArrayJson[bookId]);
                if (!sharedQuote) {
                    sharedQuote = bal.quote;
                } else {
                    bal.quote = sharedQuote;
                }
                holdings.push_back(std::move(bal));
            }
            return holdings;
        }(),
        [&] -> Account::ActiveOrders {
            Account::ActiveOrders activeOrders;
            const rapidjson::Value& activeOrdersArrayJson = json["activeOrders"].GetArray();
            const uint32_t bookCount = activeOrdersArrayJson.Size();
            for (BookId bookId = 0; bookId < activeOrdersArrayJson.Size(); ++bookId) {
                activeOrders.emplace_back();
                for (const rapidjson::Value& orderJson : activeOrdersArrayJson[bookId].GetArray()) {
                    // Assumption: All active orders are of type limit at checkpoint time.
                    // TODO: Deserialization should also yield the decimal parameters below.
                    activeOrders.back().insert(LimitOrder::fromJson(orderJson, 8, 8));
                }
            }
            return activeOrders;
        }()
    };
}

//-------------------------------------------------------------------------

Account::Account(Holdings holdings, ActiveOrders activeOrders) noexcept
    : m_holdings{std::move(holdings)}, m_activeOrders{std::move(activeOrders)}
{}

//-------------------------------------------------------------------------

}  // namespace taosim::accounting

//-------------------------------------------------------------------------
