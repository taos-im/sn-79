// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT
//
// A single-market run hands each validator response to the block that owns the book it names, by its
// canonical id, and a response that names no book to every block. A response naming a book no block owns
// is dropped with one line, as the multi-asset run drops one naming a book no realization owns, and the
// responses after it are still handed on.

#include <taosim/simulation/util.hpp>
#include <taosim/xml/helpers.hpp>

#include "MultiBookExchangeAgent.hpp"
#include "Simulation.hpp"
#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include <taosim/message/MultiBookMessagePayloads.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

//-------------------------------------------------------------------------

using namespace taosim::literals;

using testing::Test;

namespace fs = std::filesystem;

//-------------------------------------------------------------------------

namespace
{

const auto kFixture = fs::path{__FILE__}.parent_path().parent_path() / "data" / "WakeupChain.xml";

// The fixture declares two books per block.
constexpr uint32_t kBooksPerBlock = 2;
constexpr uint32_t kBlockCount = 2;
constexpr AgentId kAgent{0};

class ValidatorResponseRoutingTest : public Test
{
protected:
    void SetUp() override
    {
        for (uint32_t blockIdx{}; blockIdx < kBlockCount; ++blockIdx) {
            auto& doc = docs.emplace_back(std::make_unique<pugi::xml_document>());
            ASSERT_TRUE(doc->load_file(kFixture.c_str()));
            auto& block = blocks.emplace_back(std::make_unique<Simulation>(
                blockIdx, kBooksPerBlock, fs::temp_directory_path(), nullptr));
            block->setDebug(false);
            block->configure(taosim::xml::rootNode(*doc));
            ASSERT_TRUE(block->messageQueue().empty());
        }
    }

    // A response as the manager builds it from the validator's, naming a canonical book id.
    [[nodiscard]] static Message::Ptr limitResponse(BookId canonicalBookId)
    {
        return Message::create(
            Timestamp{},
            Timestamp{},
            "DISTRIBUTED_PROXY_AGENT",
            "EXCHANGE",
            "DISTRIBUTED_PLACE_ORDER_LIMIT",
            MessagePayload::create<DistributedAgentResponsePayload>(
                kAgent,
                MessagePayload::create<PlaceOrderLimitPayload>(
                    OrderDirection::BUY, 1_dec, 100_dec, canonicalBookId)));
    }

    [[nodiscard]] static Message::Ptr resetResponse()
    {
        return Message::create(
            Timestamp{},
            Timestamp{},
            "DISTRIBUTED_PROXY_AGENT",
            "EXCHANGE",
            "DISTRIBUTED_RESET_AGENT",
            MessagePayload::create<DistributedAgentResponsePayload>(
                kAgent, MessagePayload::create<ResetAgentsPayload>(std::vector{kAgent})));
    }

    [[nodiscard]] std::vector<Message::Ptr> queued(uint32_t blockIdx) const
    {
        std::vector<Message::Ptr> res;
        for (const auto& entry : blocks.at(blockIdx)->messageQueue().queue().underlying()) {
            res.push_back(entry.pmsg.msg);
        }
        return res;
    }

    [[nodiscard]] static BookId limitBookId(const Message::Ptr& msg)
    {
        const auto wrapped = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
        const auto limit = wrapped ? std::dynamic_pointer_cast<PlaceOrderLimitPayload>(wrapped->payload) : nullptr;
        return limit ? limit->bookId : BookId(-1);
    }

    std::vector<std::unique_ptr<pugi::xml_document>> docs;
    std::vector<std::unique_ptr<Simulation>> blocks;
};

}  // namespace

//-------------------------------------------------------------------------

TEST_F(ValidatorResponseRoutingTest, AResponseGoesToTheBlockThatOwnsItsBook)
{
    taosim::simulation::queueValidatorResponse(limitResponse(3), blocks, kBooksPerBlock);

    EXPECT_TRUE(queued(0).empty());
    const auto onSecond = queued(1);
    ASSERT_EQ(onSecond.size(), 1uz);
    EXPECT_EQ(limitBookId(onSecond.front()), BookId{1}) << "the book id was not made local to its block";
}

TEST_F(ValidatorResponseRoutingTest, AResponseNamingNoBookGoesToEveryBlock)
{
    taosim::simulation::queueValidatorResponse(resetResponse(), blocks, kBooksPerBlock);

    EXPECT_EQ(queued(0).size(), 1uz);
    EXPECT_EQ(queued(1).size(), 1uz);
}

TEST_F(ValidatorResponseRoutingTest, AResponseNamingABookNoBlockOwnsIsDroppedAndTheRestStillRoute)
{
    const std::vector responses{
        limitResponse(kBlockCount * kBooksPerBlock),
        limitResponse(1'000'000),
        limitResponse(2),
        resetResponse()};

    for (const auto& response : responses) {
        ASSERT_NO_THROW(taosim::simulation::queueValidatorResponse(response, blocks, kBooksPerBlock));
    }

    const auto onFirst = queued(0);
    const auto onSecond = queued(1);
    ASSERT_EQ(onFirst.size(), 1uz) << "only the reset belongs to the first block";
    EXPECT_EQ(onFirst.front()->type, "DISTRIBUTED_RESET_AGENT");
    ASSERT_EQ(onSecond.size(), 2uz) << "the limit on canonical book 2 and the reset belong to the second block";
    for (const auto& block : blocks) {
        ASSERT_NO_THROW(block->step()) << "the step did not complete";
    }
}

//-------------------------------------------------------------------------
