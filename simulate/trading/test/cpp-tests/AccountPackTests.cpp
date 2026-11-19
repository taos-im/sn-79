/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "MultiBookExchangeAgent.hpp"
#include "Order.hpp"
#include "Simulation.hpp"
#include "util.hpp"
#include <taosim/book/Book.hpp>
#include <taosim/matching/ClearingManager.hpp>
#include <taosim/message/PayloadFactory.hpp>
#include <taosim/serialization/msgpack/common.hpp>
#include <taosim/simulation/serialization/ValidatorRequest.hpp>

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <msgpack.hpp>

#include <array>
#include <filesystem>
#include <map>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>

//-------------------------------------------------------------------------
// The state update's accounts under the exchange's dirtyAccountsOnly: an entry unchanged on its
// book since the last publish the validator answered is left out under an untiered policy (the
// book's rates travel once, in "bookFees"), and carries only its ids and fees when the agent's
// fees are its own (a tiered policy). Whatever changes it there (an order, a fill on either side,
// a cancel) makes the next publish carry it in full again, until one is answered.

using namespace taosim::literals;
using taosim::decimal_t;
using taosim::TimeInForce;
using testing::Test;

namespace fs = std::filesystem;

namespace
{

const auto kTestDataPath = fs::path{__FILE__}.parent_path() / "data";

constexpr AgentId kMaker{0}, kTaker{1};
constexpr BookId kBookId{}, kOtherBookId{1};
constexpr std::array kOmittable{"bb", "qb", "bl", "ql", "bc", "qc", "o", "l"};

Timestamp nextTimestamp() noexcept
{
    static Timestamp ts{};
    return ++ts;
}

//-------------------------------------------------------------------------

[[nodiscard]] const msgpack::object& at(const msgpack::object& map, std::string_view key)
{
    for (const auto& kv : std::span{map.via.map.ptr, map.via.map.size}) {
        if (kv.key.type == msgpack::type::STR && kv.key.as<std::string_view>() == key) {
            return kv.val;
        }
    }
    throw std::out_of_range{fmt::format("no key '{}'", key)};
}

[[nodiscard]] const msgpack::object& at(const msgpack::object& map, int64_t key)
{
    for (const auto& kv : std::span{map.via.map.ptr, map.via.map.size}) {
        if (kv.key.type == msgpack::type::POSITIVE_INTEGER && kv.key.as<int64_t>() == key) {
            return kv.val;
        }
    }
    throw std::out_of_range{fmt::format("no key {}", key)};
}

//-------------------------------------------------------------------------

class AccountPackTest : public Test
{
protected:
    taosim::util::Nodes nodes;
    std::vector<std::unique_ptr<Simulation>> simulations;
    MultiBookExchangeAgent* exchange{};

    // The fixture's file prices remote agents on a tiered policy, whose fees follow each agent's own volume.
    virtual void setFeePolicy(pugi::xml_node) {}

    void SetUp() override
    {
        nodes = taosim::util::parseSimulationFile(kTestDataPath / "MultiAgentFees.xml");
        nodes.exchange.child("Books").attribute("instanceCount").set_value(2);  // a book left alone
        setFeePolicy(nodes.exchange);
        simulations.push_back(std::make_unique<Simulation>());
        simulations.front()->setDebug(false);
        simulations.front()->configure(nodes.simulation);
        exchange = simulations.front()->exchange();
        exchange->dirtyAccountsOnly() = true;
        exchange->fullAccountsEvery() = 0;  // no rotation unless a test sets one
    }

    // A state sent; the validator has not answered it.
    [[nodiscard]] msgpack::object_handle pack() const
    {
        taosim::serialization::HumanReadableStream buffer;
        msgpack::pack(buffer, taosim::simulation::serialization::ValidatorRequest{
            .simulations = simulations,
            .logDir = fs::temp_directory_path()
        });
        return msgpack::unpack(buffer.data(), buffer.size());
    }

    // A state sent and answered.
    [[nodiscard]] msgpack::object_handle publish() const
    {
        auto state = pack();
        exchange->accountsDelivered();
        return state;
    }

    [[nodiscard]] static const msgpack::object& entry(
        const msgpack::object_handle& state, AgentId agentId, BookId bookId = kBookId)
    {
        return at(at(at(state.get(), "accounts"), agentId), bookId);
    }

    [[nodiscard]] bool dirty(AgentId agentId) { return exchange->accounts().at(agentId).dirty(kBookId); }

    LimitOrder::Ptr placeLimit(
        AgentId agentId, OrderDirection direction, decimal_t volume, decimal_t price)
    {
        const auto payload = MessagePayload::create<PlaceOrderLimitPayload>(
            direction, volume, price, 0_dec, kBookId, Currency::BASE, std::nullopt, false,
            TimeInForce::GTC, std::nullopt, STPFlag::CO);
        const auto result = exchange->clearingManager().handleOrder(
            taosim::matching::LimitOrderDesc{.agentId = agentId, .payload = payload});
        if (result.ec != OrderErrorCode::VALID) return nullptr;
        return exchange->books()[kBookId]->placeLimitOrder(
            OrderClientContext{agentId},
            nextTimestamp(),
            result.orderSize,
            direction,
            price,
            0_dec,
            STPFlag::CO);
    }
};

}  // namespace

//-------------------------------------------------------------------------

TEST_F(AccountPackTest, AFreshAccountIsPublishedInFull)
{
    const auto state = publish();
    const auto& maker = entry(state, kMaker);

    EXPECT_EQ(at(maker, "i").as<AgentId>(), kMaker);
    for (std::string_view key : kOmittable) {
        EXPECT_FALSE(at(maker, key).is_nil()) << key;
    }
    EXPECT_FALSE(dirty(kMaker)) << "an answered publish leaves the account clean";
}

TEST_F(AccountPackTest, AnUnchangedAccountCarriesOnlyItsIdsAndFees)
{
    std::ignore = publish();
    const auto state = publish();
    const auto& maker = entry(state, kMaker);

    EXPECT_EQ(maker.via.map.size, 11u);
    EXPECT_EQ(at(maker, "i").as<AgentId>(), kMaker);
    EXPECT_EQ(at(maker, "b").as<BookId>(), kBookId);
    const auto& fees = at(maker, "f");
    ASSERT_EQ(fees.type, msgpack::type::MAP);
    EXPECT_FALSE(at(fees, "m").is_nil());
    EXPECT_FALSE(at(fees, "t").is_nil());
    for (std::string_view key : kOmittable) {
        EXPECT_TRUE(at(maker, key).is_nil()) << key;
    }
}

TEST_F(AccountPackTest, WithTheToggleOffEveryPublishIsFull)
{
    exchange->dirtyAccountsOnly() = false;
    std::ignore = publish();
    const auto state = publish();

    for (std::string_view key : kOmittable) {
        EXPECT_FALSE(at(entry(state, kMaker), key).is_nil()) << key;
    }
}

TEST_F(AccountPackTest, AnOrderRepublishesItsAccountOnly)
{
    std::ignore = publish();
    ASSERT_NE(placeLimit(kMaker, OrderDirection::BUY, 1_dec, 290_dec), nullptr);
    const auto state = publish();

    const auto& orders = at(entry(state, kMaker), "o");
    ASSERT_EQ(orders.type, msgpack::type::ARRAY);
    EXPECT_EQ(orders.via.array.size, 1u);
    EXPECT_TRUE(at(entry(state, kTaker), "bb").is_nil());
}

TEST_F(AccountPackTest, AFillDirtiesBothSides)
{
    ASSERT_NE(placeLimit(kMaker, OrderDirection::SELL, 1_dec, 310_dec), nullptr);
    std::ignore = publish();
    ASSERT_FALSE(dirty(kMaker));
    ASSERT_FALSE(dirty(kTaker));

    ASSERT_NE(placeLimit(kTaker, OrderDirection::BUY, 1_dec, 310_dec), nullptr);

    EXPECT_TRUE(dirty(kMaker)) << "the resting side was filled";
    EXPECT_TRUE(dirty(kTaker));
    const auto state = publish();
    EXPECT_FALSE(at(entry(state, kMaker), "bb").is_nil());
}

TEST_F(AccountPackTest, ACancelDirtiesTheAccount)
{
    const auto order = placeLimit(kMaker, OrderDirection::BUY, 1_dec, 290_dec);
    ASSERT_NE(order, nullptr);
    std::ignore = publish();
    ASSERT_FALSE(dirty(kMaker));

    ASSERT_TRUE(exchange->books()[kBookId]->cancelOrder(order->id(), DEC(0.4)));
    EXPECT_TRUE(dirty(kMaker)) << "a partial cancel frees part of the reservation";

    std::ignore = publish();
    ASSERT_TRUE(exchange->books()[kBookId]->cancelOrder(order->id()));
    EXPECT_TRUE(dirty(kMaker));
}

//-------------------------------------------------------------------------

TEST_F(AccountPackTest, AnOrderMarksOnlyItsOwnBook)
{
    std::ignore = publish();
    ASSERT_NE(placeLimit(kMaker, OrderDirection::BUY, 1_dec, 290_dec), nullptr);
    const auto state = publish();

    EXPECT_FALSE(at(entry(state, kMaker), "o").is_nil());
    EXPECT_TRUE(at(entry(state, kMaker, kOtherBookId), "bb").is_nil()) << "nothing changed on the other book";
}

TEST_F(AccountPackTest, AStateLeftUnansweredKeepsItsChanges)
{
    std::ignore = publish();
    ASSERT_NE(placeLimit(kMaker, OrderDirection::BUY, 1_dec, 290_dec), nullptr);
    std::ignore = pack();

    const auto again = publish();
    EXPECT_FALSE(at(entry(again, kMaker), "o").is_nil()) << "listed again: no answered state carried it";
    const auto after = publish();
    EXPECT_TRUE(at(entry(after, kMaker), "o").is_nil());
}

TEST_F(AccountPackTest, TheRotationListsEveryAccountInFullWithinItsPeriod)
{
    exchange->fullAccountsEvery() = 2;
    std::ignore = publish();

    std::map<AgentId, int> listedInFull;
    for (int i = 0; i < 2; ++i) {
        const auto state = publish();
        for (AgentId agentId : {kMaker, kTaker}) {
            listedInFull[agentId] += !at(entry(state, agentId, kOtherBookId), "bb").is_nil();
        }
    }
    EXPECT_EQ(listedInFull[kMaker], 1);
    EXPECT_EQ(listedInFull[kTaker], 1);
}

TEST(AccountPackConfig, TheRotationIsOnEveryThirtyUnlessTheConfigSaysOtherwise)
{
    pugi::xml_document doc;
    ASSERT_TRUE(doc.load_string(R"(<MultiBookExchangeAgent dirtyAccountsOnly="1"/>)"));
    EXPECT_EQ(MultiBookExchangeAgent::fullAccountsEveryOf(doc.first_child()), 30u);
    ASSERT_TRUE(doc.load_string(R"(<MultiBookExchangeAgent dirtyAccountsOnly="1" fullAccountsEvery="0"/>)"));
    EXPECT_EQ(MultiBookExchangeAgent::fullAccountsEveryOf(doc.first_child()), 0u) << "0 turns it off";
    ASSERT_TRUE(doc.load_string(R"(<MultiBookExchangeAgent dirtyAccountsOnly="1" fullAccountsEvery="12"/>)"));
    EXPECT_EQ(MultiBookExchangeAgent::fullAccountsEveryOf(doc.first_child()), 12u);
}

TEST_F(AccountPackTest, AValidatorThatAsksGetsEveryAccountInFullOnce)
{
    std::ignore = publish();
    exchange->requestFullAccounts();

    const auto asked = publish();
    EXPECT_FALSE(at(entry(asked, kTaker, kOtherBookId), "bb").is_nil());
    const auto next = publish();
    EXPECT_TRUE(at(entry(next, kTaker, kOtherBookId), "bb").is_nil());
}

//-------------------------------------------------------------------------

TEST_F(AccountPackTest, TheStateSaysWhenEntriesMayBeLeftOut)
{
    EXPECT_TRUE(at(publish().get(), "dirtyAccountsOnly").as<bool>());

    exchange->dirtyAccountsOnly() = false;
    const auto state = publish();
    EXPECT_THROW(std::ignore = at(state.get(), "dirtyAccountsOnly"), std::out_of_range) << "the state as before";
    EXPECT_THROW(std::ignore = at(state.get(), "bookFees"), std::out_of_range) << "the state as before";
}

//-------------------------------------------------------------------------
// Under mainnet's untiered dynamic policy every agent without a policy of its own pays the book's rates, so an entry
// unchanged on its book is left out of the state entirely and the book's rates travel once, in "bookFees": the
// skeletons of ids, fees and nils were 4.6 MB of a 6.3 MB state at 256 books with no account changed.

class AccountPackUntieredTest : public AccountPackTest
{
protected:
    void setFeePolicy(pugi::xml_node exchangeNode) override
    {
        exchangeNode.remove_child("FeePolicy");
        auto policy = exchangeNode.prepend_child("FeePolicy");
        for (const auto [name, value] : std::array<std::pair<const char*, const char*>, 9>{{
                 {"type", "dynamic"}, {"historySlots", "300"}, {"makerFee", "0.000"}, {"takerFee", "0.00023"},
                 {"maxMakerRate", "0.015"}, {"maxTakerRate", "0.015"}, {"targetMTR", "0.4"},
                 {"shapeMakerFee", "2"}, {"shapeMakerRebate", "2"}}}) {
            policy.append_attribute(name).set_value(value);
        }
    }

    [[nodiscard]] static const msgpack::object& agentEntries(const msgpack::object_handle& state, AgentId agentId)
    {
        return at(at(state.get(), "accounts"), agentId);
    }
};

TEST_F(AccountPackUntieredTest, AnUnchangedEntryIsLeftOutAndItsBooksRatesTravelOnce)
{
    const auto first = publish();
    const auto state = publish();

    EXPECT_EQ(agentEntries(state, kMaker).via.map.size, 0u);
    EXPECT_EQ(agentEntries(state, kTaker).via.map.size, 0u);
    const auto& bookFees = at(state.get(), "bookFees");
    ASSERT_EQ(bookFees.via.map.size, 2u);
    for (const BookId bookId : {kBookId, kOtherBookId}) {
        const auto& rates = at(bookFees, bookId);
        const auto& charged = at(entry(first, kMaker, bookId), "f");
        EXPECT_EQ(at(rates, "m").as<double>(), at(charged, "m").as<double>()) << bookId;
        EXPECT_EQ(at(rates, "t").as<double>(), at(charged, "t").as<double>()) << bookId;
    }
}

TEST_F(AccountPackUntieredTest, AnOrderRepublishesOnlyItsBooksEntry)
{
    std::ignore = publish();
    ASSERT_NE(placeLimit(kMaker, OrderDirection::BUY, 1_dec, 290_dec), nullptr);
    const auto state = publish();

    ASSERT_EQ(agentEntries(state, kMaker).via.map.size, 1u);
    EXPECT_EQ(at(entry(state, kMaker), "o").via.array.size, 1u);
    EXPECT_EQ(agentEntries(state, kTaker).via.map.size, 0u);
}

TEST_F(AccountPackUntieredTest, TheRotationAndARequestStillListEveryEntry)
{
    std::ignore = publish();
    exchange->requestFullAccounts();
    const auto asked = publish();
    EXPECT_EQ(agentEntries(asked, kTaker).via.map.size, 2u);
    EXPECT_FALSE(at(entry(asked, kTaker, kOtherBookId), "bb").is_nil());
}

TEST_F(AccountPackTest, ATieredPolicyKeepsEveryEntryWithItsOwnFees)
{
    std::ignore = publish();
    const auto state = publish();
    EXPECT_EQ(at(at(state.get(), "accounts"), kMaker).via.map.size, 2u) << "an agent's tiered fees are its own";
    EXPECT_EQ(at(state.get(), "bookFees").via.map.size, 0u);
}

//-------------------------------------------------------------------------
