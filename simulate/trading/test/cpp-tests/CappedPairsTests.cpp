// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT
//
// The validator's capped set as the engine reads it from a reply batch: the "capped" member beside the responses,
// [[agent, canonical book], ...], the full set each round. An absent member keeps the engine's last set; an empty
// one clears it. The set is then split over the realizations, or the blocks, that own the books.

#include <taosim/simulation/util.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <msgpack.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

//-------------------------------------------------------------------------

using taosim::simulation::CappedPair;
using taosim::simulation::readCappedPairs;
using taosim::simulation::splitCappedPairs;

using namespace testing;

//-------------------------------------------------------------------------

namespace
{

// A reply batch with an empty responses list and, when given, the packer's own "capped" member.
template<typename Pack>
[[nodiscard]] msgpack::object_handle batch(Pack&& packCapped)
{
    msgpack::sbuffer buf;
    msgpack::packer<msgpack::sbuffer> pk{buf};
    pk.pack_map(2);
    pk.pack(std::string{"responses"});
    pk.pack_array(0);
    pk.pack(std::string{"capped"});
    packCapped(pk);
    return msgpack::unpack(buf.data(), buf.size());
}

[[nodiscard]] msgpack::object_handle batchWithoutCapped()
{
    msgpack::sbuffer buf;
    msgpack::packer<msgpack::sbuffer> pk{buf};
    pk.pack_map(1);
    pk.pack(std::string{"responses"});
    pk.pack_array(0);
    return msgpack::unpack(buf.data(), buf.size());
}

}  // namespace

//-------------------------------------------------------------------------

TEST(CappedPairsTest, ABatchWithoutTheMemberKeepsTheLastSet)
{
    const auto oh = batchWithoutCapped();
    EXPECT_EQ(readCappedPairs(oh.get()), std::nullopt);
}

TEST(CappedPairsTest, AnEmptyMemberClearsTheSet)
{
    const auto oh = batch([](auto& pk) { pk.pack_array(0); });
    const auto pairs = readCappedPairs(oh.get());
    ASSERT_TRUE(pairs.has_value());
    EXPECT_THAT(*pairs, IsEmpty());
}

TEST(CappedPairsTest, ThePairsAreReadInOrder)
{
    const auto oh = batch([](auto& pk) { pk.pack(std::vector<std::vector<int64_t>>{{3, 70}, {5, 0}}); });
    const auto pairs = readCappedPairs(oh.get());
    ASSERT_TRUE(pairs.has_value());
    EXPECT_THAT(*pairs, ElementsAre(CappedPair{3, 70}, CappedPair{5, 0}));
}

// One bad entry costs that entry, not the round's set.
TEST(CappedPairsTest, AnEntryThatIsNotAPairOfNonNegativeIntegersIsSkipped)
{
    const auto oh = batch([](auto& pk) {
        pk.pack_array(6);
        pk.pack(std::vector<int64_t>{3});
        pk.pack(std::string{"x"});
        pk.pack(std::vector<int64_t>{4, -1});
        pk.pack(std::vector<int64_t>{6, 2, 9});
        pk.pack(std::vector<int64_t>{-2, 4});
        pk.pack(std::vector<int64_t>{7, 12});
    });
    const auto pairs = readCappedPairs(oh.get());
    ASSERT_TRUE(pairs.has_value());
    EXPECT_THAT(*pairs, ElementsAre(CappedPair{7, 12}));
}

TEST(CappedPairsTest, AMemberThatIsNotAListIsAnEmptySet)
{
    const auto oh = batch([](auto& pk) { pk.pack(std::string{"all"}); });
    const auto pairs = readCappedPairs(oh.get());
    ASSERT_TRUE(pairs.has_value());
    EXPECT_THAT(*pairs, IsEmpty());
}

//-------------------------------------------------------------------------

// Realizations of 16, 16 and 16 books: canonical ids 0-15, 16-31, 32-47.
TEST(CappedPairsTest, ThePairsGoToTheRealizationThatOwnsTheBookWithItsLocalId)
{
    const std::vector<CappedPair> pairs{{3, 0}, {3, 17}, {4, 47}, {5, 48}};
    const std::vector<uint32_t> bases{0, 16, 32};
    const std::vector<uint32_t> counts{16, 16, 16};

    const auto split = splitCappedPairs(pairs, bases, counts);

    ASSERT_EQ(split.size(), 3uz);
    EXPECT_THAT(split[0], ElementsAre(CappedPair{3, 0}));
    EXPECT_THAT(split[1], ElementsAre(CappedPair{3, 1}));
    EXPECT_THAT(split[2], ElementsAre(CappedPair{4, 15}));
}

// A realization with no capped pair still receives a set: the empty one that clears its last.
TEST(CappedPairsTest, EveryRealizationReceivesASet)
{
    const std::vector<CappedPair> pairs{{3, 20}};
    const std::vector<uint32_t> bases{0, 16};
    const std::vector<uint32_t> counts{16, 16};

    const auto split = splitCappedPairs(pairs, bases, counts);

    ASSERT_EQ(split.size(), 2uz);
    EXPECT_THAT(split[0], IsEmpty());
    EXPECT_THAT(split[1], ElementsAre(CappedPair{3, 4}));
}

TEST(CappedPairsTest, ThePairsGoToTheBlockThatOwnsTheBook)
{
    const std::vector<CappedPair> pairs{{3, 5}, {3, 20}, {3, 40}};

    const auto split = splitCappedPairs(pairs, 16u, 2u);

    ASSERT_EQ(split.size(), 2uz);
    EXPECT_THAT(split[0], ElementsAre(CappedPair{3, 5}));
    EXPECT_THAT(split[1], ElementsAre(CappedPair{3, 4}));
}

//-------------------------------------------------------------------------
