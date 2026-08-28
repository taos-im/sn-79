/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/accounting/Balance.hpp>

//-------------------------------------------------------------------------

namespace taosim::accounting
{

//-------------------------------------------------------------------------

Balance::Balance(
    decimal_t total,
    const std::string& symbol,
    uint32_t roundingDecimals)
    : m_symbol{symbol}, m_roundingDecimals{roundingDecimals}
{
    total = roundAmount(total);

    if (total < 0_dec) {
        throw std::invalid_argument(fmt::format(
            "{}: Initial balance must be non-negative, was {}",
            std::source_location::current().function_name(),
            total));
    }

    m_free = total;
    m_total = total;
    m_initial = total;
}

//-------------------------------------------------------------------------

Balance::Balance(Balance&& other) noexcept
{
    move(std::move(other));
}

//-------------------------------------------------------------------------

Balance& Balance::operator=(Balance&& other) noexcept
{
    move(std::move(other));
    return *this;
}

//-------------------------------------------------------------------------

FreeInfo Balance::canFree(OrderID id, std::optional<decimal_t> amount) const noexcept
{
    amount = roundAmount(amount);
    auto it = m_reservations.find(id);
    if (it == m_reservations.end()) {
        return {
            .orderId = id,
            .amount = amount,
            .reservation = std::nullopt,
            .status = amount.has_value()
                ? amount.value() > 0_dec
                    ? FreeStatus::NONEXISTENT_RESERVATION
                    : FreeStatus::NONEXISTENT_RESERVATION_AND_NEGATIVE_AMOUNT
                : FreeStatus::NONEXISTENT_RESERVATION_AND_AMOUNT
        };
    }
    if (amount.has_value()) {
        return {
            .orderId = id,
            .amount = amount,
            .reservation = std::make_optional(it->second),
            .status = [&] -> FreeStatus {
                if (amount.value() > it->second) {
                    return FreeStatus::AMOUNT_EXCEEDS_RESERVATION;
                } else if (amount.value() < 0_dec) {
                    return FreeStatus::NEGATIVE_AMOUNT;
                } else {
                    return FreeStatus::FREEABLE;
                }
            }()
        };
    }
    return {
        .orderId = id,
        .amount = amount,
        .reservation = std::make_optional(it->second),
        .status = FreeStatus::FREEABLE
    };
}

//-------------------------------------------------------------------------

bool Balance::canReserve(decimal_t amount) const noexcept
{
    amount = roundAmount(amount);
    return amount >= 0_dec && amount <= m_free;
}

//-------------------------------------------------------------------------

std::optional<decimal_t> Balance::getReservation(OrderID id) const noexcept
{
    if (auto it = m_reservations.find(id); it != m_reservations.end()) {
        return it->second;
    }
    return std::nullopt;
}

//-------------------------------------------------------------------------

void Balance::deposit(decimal_t amount, BookId bookId)
{
    amount = roundAmount(amount);
    m_free = m_free + amount;
    m_total = m_total + amount;
    checkConsistency(std::source_location::current(), bookId);
}

//-------------------------------------------------------------------------

decimal_t Balance::makeReservation(OrderID id, decimal_t amount, BookId bookId)
{
    static constexpr auto ctx = std::source_location::current().function_name();

    if (amount < 0_dec) {
        throw std::invalid_argument{
            fmt::format("{}: BOOK {} | Reservation amount for order {} cannot be negative {} | {} ", ctx, bookId, id, amount, *this)};
    }

    amount = roundAmount(amount);

    if (amount == 0_dec) {
        return 0_dec;
    }
    if (!canReserve(amount)) {
        throw std::invalid_argument{fmt::format(
            "{}: Cannot reserve {} with current balance of {}", ctx, amount, *this)};
    }

    m_free -= amount;
    m_reserved += amount;
    auto it = m_reservations.find(id);
    if (it != m_reservations.end()) {
        it->second += amount;
    } else {
        m_reservations.insert({id, amount});
    }

    checkConsistency(std::source_location::current(), bookId);

    if (auto reserved = ranges::accumulate(m_reservations | views::values, 0_dec);
        reserved != m_reserved) {
        throw std::runtime_error{fmt::format(
            "{}: total reservation {} does not match the sum of reservations {} "
            "after makeReservation {} amount for order #{} | {}.",
            ctx, m_reserved, reserved,
            amount, id, *this)};
    }
    return amount;
}

//-------------------------------------------------------------------------

decimal_t Balance::freeReservation(OrderID id, BookId bookId, std::optional<decimal_t> amount)
{
    static constexpr auto ctx = std::source_location::current().function_name();

    amount = roundAmount(amount);

    if (FreeInfo info = canFree(id, amount); info.status != FreeStatus::FREEABLE) {
        // Include the whole balance, not just the amount pair. Diagnosing the price-band abort from
        // "amount X exceeds reservation Y" alone was impossible: the two figures matched neither the
        // order's base volume nor its quote notional, so which ledger and which call site were
        // involved could not be determined without re-running. The balance shows total/reserved/free
        // and every outstanding reservation, which identifies both.
        throw FreeException{fmt::format("{}: In book #{}, {} || balance {}",
            ctx, bookId, info.toString(), *this)};
    }

    if (!amount.has_value()) {
        auto it = m_reservations.find(id);
        amount = m_reservations.size() == 1 ? m_reserved : it->second;
        m_reservations.erase(it);
        m_reserved -= amount.value();
        m_reserved = roundAmount(m_reserved);
        m_free = roundAmount(m_total - m_reserved);
    }
    else {
        auto it = m_reservations.find(id);
        auto& reservation = it->second;
        reservation -= amount.value();
        if (reservation < 0_dec) {
            throw std::runtime_error{fmt::format(
                "{}: In book #{}, negative reservation {} for order #{} by amount {} || {}",
                ctx, bookId, reservation, id, amount.value(), *this)};
        }
        if (reservation == 0_dec) {
            m_reservations.erase(it);
        }   
        m_reserved -= amount.value();
        m_free += amount.value();
        m_reserved = roundAmount(m_reserved);
        m_free = roundAmount(m_free);
    }

    if (m_reserved > 0_dec && m_reservations.empty()) {
        if (util::round(m_reserved, m_roundingDecimals - 1) == 0_dec){
            m_reserved = 0_dec;
            fmt::println(
                "{}: In book #{}, some residual after freeing reservation of {} for order #{} : "
                "No reservations left but amount reserved is {}. m_reserved corrected!",
                ctx, bookId, amount.value_or(-1_dec), id, m_reserved);
        } else {
            throw std::runtime_error{fmt::format(
                "{}: In book #{}, Unable to free reservation of {} for order #{} : "
                "No reservations left but amount reserved is {}",
                ctx, bookId, amount.value_or(-1_dec), id, m_reserved)};
        }
    }

    checkConsistency(std::source_location::current(), bookId);

    return amount.value();
}

//-------------------------------------------------------------------------

decimal_t Balance::tryFreeReservation(OrderID orderId, BookId bookId, std::optional<decimal_t> amount)
{
    try {
        return freeReservation(orderId, bookId, amount);
    }
    catch (const FreeException& exc) {
        return 0_dec;
    }
    catch (...) {
        fmt::println(
            "Trying to free (amount:{}) reservation for order {} in book {}", 
            orderId, 
            amount.value_or(0_dec),
            bookId
        );
        throw;
    }
}

//-------------------------------------------------------------------------

void Balance::voidReservation(OrderID id, BookId bookId, std::optional<decimal_t> amount)
{
    if (getReservation(id).has_value()) {
        amount = freeReservation(id, bookId, amount);
        amount = roundAmount(amount);
        m_free -= amount.value();
        m_total -= amount.value();
        checkConsistency(std::source_location::current(), bookId);
    }
}

//-------------------------------------------------------------------------

void Balance::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();
        json.AddMember(
            "initial", rapidjson::Value{util::decimal2double(m_initial)}, allocator);
        json.AddMember(
            "free", rapidjson::Value{util::decimal2double(m_free)}, allocator);
        json.AddMember(
            "reserved", rapidjson::Value{util::decimal2double(m_reserved)}, allocator);
        json.AddMember(
            "total", rapidjson::Value{util::decimal2double(m_total)}, allocator);
        json.AddMember(
            "symbol",
            !m_symbol.empty()
                ? rapidjson::Value{m_symbol.c_str(), allocator}.Move()
                : rapidjson::Value{}.SetNull(),
            allocator);
        json.AddMember(
            "roundingDecimals", rapidjson::Value{m_roundingDecimals}, allocator);
    };
    json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

std::ostream& operator<<(std::ostream& os, const Balance& bal) noexcept
{
    return os << fmt::format("{}", bal);
}

//-------------------------------------------------------------------------

Balance Balance::fromXML(pugi::xml_node node, uint32_t roundingDecimals)
{
    return Balance{
        util::double2decimal(node.attribute("total").as_double(), roundingDecimals),
        node.attribute("symbol").as_string(),
        roundingDecimals};
}

//-------------------------------------------------------------------------

Balance Balance::fromJson(const rapidjson::Value& json)
{
    Balance balance;
    balance.m_roundingDecimals = json["roundingDecimals"].GetUint();
    balance.m_initial = balance.roundAmount(json::getDecimal(json["initial"]));
    balance.m_free = balance.roundAmount(json::getDecimal(json["free"]));
    balance.m_reserved = balance.roundAmount(json::getDecimal(json["reserved"]));
    balance.m_total = balance.roundAmount(json::getDecimal(json["total"]));
    balance.m_symbol = !json["symbol"].IsNull() ? json["symbol"].GetString() : "";
    for (const auto& member : json["reservations"].GetObject()) {
        const char* name = member.name.GetString();
        balance.m_reservations[std::stoull(name)] =
            balance.roundAmount(json::getDecimal(member.value));
    }
    return balance;
}

//-------------------------------------------------------------------------

void Balance::move(Balance&& other) noexcept
{
    m_initial = std::exchange(other.m_initial, {});
    m_free = std::exchange(other.m_free, {});
    m_total = std::exchange(other.m_total, {});
    m_reserved = std::exchange(other.m_reserved, {});
    m_reservations = std::move(other.m_reservations);
    m_roundingDecimals = other.m_roundingDecimals;
}

//-------------------------------------------------------------------------

decimal_t Balance::roundAmount(decimal_t amount) const
{
    return util::round(amount, m_roundingDecimals);
}

//-------------------------------------------------------------------------

std::optional<decimal_t> Balance::roundAmount(std::optional<decimal_t> amount) const
{
    return amount.transform([this](decimal_t val) { return roundAmount(val); });
}

//-------------------------------------------------------------------------

void Balance::checkConsistency(std::source_location sl, BookId bookId)
{
    if (m_total != m_free + m_reserved) {
        fmt::println(
            "{} : In book {} there is inconsistency in the accounting where total {}"
            " is not equal to free {} +  reserved {} = {}",
            sl.function_name(), bookId, m_total, m_free, m_reserved, m_free + m_reserved);
            
        if (m_total > m_free + m_reserved && util::round(m_total - m_free - m_reserved, m_roundingDecimals - 2) == 0_dec){
            m_free = m_total - m_reserved;
        } else if (m_total < m_free + m_reserved && util::round(m_free + m_reserved - m_total, m_roundingDecimals - 2)  == 0_dec){
            m_total = m_free + m_reserved;
        } else {
            throw std::runtime_error{fmt::format(
                "{} : In book {} there is inconsistency in the accounting where total {}"
                " is not equal to free {} +  reserved {} = {}",
                sl.function_name(), bookId, m_total, m_free, m_reserved, m_free + m_reserved)};
        }
    }
    if (m_total < 0_dec || m_free < 0_dec || m_reserved < 0_dec) {
        throw std::runtime_error{fmt::format(
            "{} : In book {} there is negative values in accounting  {} ( {} | {})",
            sl.function_name(), bookId, m_total, m_free, m_reserved)};
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::accounting

//-------------------------------------------------------------------------