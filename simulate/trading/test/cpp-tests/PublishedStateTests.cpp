// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT
//
// What a multi-asset run publishes to its validator, read as the validator reads it.
//
// The engine packs every realization's part of the published state; the workers do that packing, and the
// clearing of filled orders before it, each on its own realizations rather than in the barrier's single-threaded
// completion. These tests run a publishing run in a child process on private IPC names, answer each state as a
// validator does (an empty response list) and keep the bytes, so what was published can be checked. A realization
// that fails in continue mode no longer steps on its worker; its part must still be its state, frozen at the failure,
// and the other realizations must go on publishing. Some runs also answer with miners' orders, so the miners' part of
// the state (their accounts, open orders and notices), which an empty answer leaves empty, is checked too.

#include "MultiAssetRunFixture.hpp"

#include <taosim/ipc/PosixMessageQueue.hpp>
#include <taosim/serialization/msgpack/utils.hpp>
#include <taosim/simulation/SimulationManager.hpp>
#include <taosim/simulation/multiasset/serialization/MultiAssetValidatorRequest.hpp>
#include <taosim/simulation/serialization/ValidatorRequest.hpp>

#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

//-------------------------------------------------------------------------

using namespace taosim::test::multiasset_run;

namespace bipc = boost::interprocess;

namespace
{

constexpr std::string_view kChildTest = "PublishedStateChild.DISABLED_RunsAndPublishes";
constexpr std::string_view kIpcSuffixVar = "TAOSIM_IPC_SUFFIX";
constexpr std::string_view kPortVar = "TAOSIM_TEST_PUB_PORT";
constexpr std::string_view kFailVar = "TAOSIM_TEST_PUB_FAIL";
constexpr std::string_view kOutVar = "TAOSIM_TEST_PUB_OUT";
constexpr std::string_view kMinersVar = "TAOSIM_TEST_PUB_MINERS";
constexpr std::string_view kThreadsVar = "TAOSIM_TEST_PUB_THREADS";
constexpr std::string_view kStepsVar = "TAOSIM_TEST_PUB_STEPS";
constexpr std::string_view kContinue = R"(stopOnRealizationFailure="0")";
constexpr uint32_t kRealizations = 3;
constexpr uint64_t kSteps = 6;
constexpr uint64_t kFailTime = 2 * kStep;
constexpr int32_t kMiners = 2;

// Answers the run's lifecycle posts with 200, as the validator's notice route does.
class LifecycleStub
{
public:
    LifecycleStub()
    {
        accept();
        m_thread = std::jthread{[this] { m_ctx.run(); }};
    }

    ~LifecycleStub() { m_ctx.stop(); }

    [[nodiscard]] uint16_t port() const { return m_acceptor.local_endpoint().port(); }

private:
    void accept()
    {
        m_acceptor.async_accept([this](boost::beast::error_code ec, boost::asio::ip::tcp::socket socket) {
            if (ec) return;
            namespace http = boost::beast::http;
            boost::beast::flat_buffer buffer;
            http::request<http::string_body> req;
            http::read(socket, buffer, req, ec);
            if (!ec) {
                http::response<http::string_body> res{http::status::ok, req.version()};
                res.set(http::field::content_type, "application/json");
                res.body() = "null";
                res.prepare_payload();
                http::write(socket, res, ec);
            }
            socket.shutdown(boost::asio::ip::tcp::socket::shutdown_send, ec);
            accept();
        });
    }

    boost::asio::io_context m_ctx;
    boost::asio::ip::tcp::acceptor m_acceptor{
        m_ctx, boost::asio::ip::tcp::endpoint{boost::asio::ip::make_address("127.0.0.1"), 0}};
    std::jthread m_thread;
};

[[nodiscard]] std::optional<double> number(const msgpack::object& o)
{
    switch (o.type) {
        case msgpack::type::FLOAT32:
        case msgpack::type::FLOAT64: return o.via.f64;
        case msgpack::type::POSITIVE_INTEGER: return static_cast<double>(o.via.u64);
        case msgpack::type::NEGATIVE_INTEGER: return static_cast<double>(o.via.i64);
        default: return std::nullopt;
    }
}

[[nodiscard]] const msgpack::object* member(const msgpack::object& map, std::string_view key)
{
    if (map.type != msgpack::type::MAP) return nullptr;
    for (const auto& [k, v] : map.via.map) {
        if (k.type == msgpack::type::STR && k.as<std::string_view>() == key) return &v;
    }
    return nullptr;
}

[[nodiscard]] const msgpack::object* entry(const msgpack::object& map, int64_t key)
{
    if (map.type != msgpack::type::MAP) return nullptr;
    for (const auto& [k, v] : map.via.map) {
        if ((k.type == msgpack::type::POSITIVE_INTEGER || k.type == msgpack::type::NEGATIVE_INTEGER)
            && k.as<int64_t>() == key) {
            return &v;
        }
    }
    return nullptr;
}

// Each realization's part of a published state, in flat order.
[[nodiscard]] std::vector<const msgpack::object*> parts(const msgpack::object& state)
{
    std::vector<const msgpack::object*> out;
    const auto* backgrounds = member(state, "backgrounds");
    if (backgrounds == nullptr || backgrounds->type != msgpack::type::ARRAY) return out;
    for (const auto& bg : backgrounds->via.array) {
        const auto* instances = member(bg, "instances");
        if (instances == nullptr || instances->type != msgpack::type::ARRAY) continue;
        for (const auto& instance : instances->via.array) out.push_back(&instance);
    }
    return out;
}

[[nodiscard]] std::optional<double> bestPrice(const msgpack::object& book, std::string_view side)
{
    const auto* levels = member(book, side);
    if (levels == nullptr || levels->type != msgpack::type::ARRAY || levels->via.array.size == 0) return std::nullopt;
    const auto* price = member(levels->via.array.ptr[0], "p");
    return price ? number(*price) : std::nullopt;
}

// The miners' orders for one published state, on a fixed schedule keyed on the step and priced off each realization's
// own book, so the answer is a function of the state alone and two runs that publish the same states send the same
// orders. On one book of each realization, rotating with the step: miner 0 rests a buy below the mid and miner 1 a
// sell above it; every third step miner 0 crosses miner 1's sell with a limit buy and miner 1 sells at market into
// the bids; every other step each miner cancels its oldest open order in the realization, on whichever book it rests
// (by time, then id, never by its place in the published list: see withOpenOrdersById).
[[nodiscard]] std::string minerOrders(const msgpack::object& state, uint64_t step)
{
    struct Order
    {
        int32_t agentId;
        std::string type;
        std::function<void(msgpack::packer<taosim::serialization::BinaryStream>&)> payload;
    };
    std::vector<Order> orders;
    uint32_t clientOrderId = static_cast<uint32_t>(step * 64);
    auto limit = [&](int32_t agentId, uint32_t direction, double price, uint32_t bookId) {
        orders.push_back({agentId, "PLACE_ORDER_LIMIT", [=, id = clientOrderId++](auto& pk) {
            pk.pack_map(12);
            pk.pack("direction"); pk.pack(direction);
            pk.pack("volume"); pk.pack(0.25);
            pk.pack("price"); pk.pack(price);
            pk.pack("bookId"); pk.pack(bookId);
            pk.pack("clientOrderId"); pk.pack(id);
            pk.pack("postOnly"); pk.pack(false);
            pk.pack("timeInForce"); pk.pack(0u);
            pk.pack("expiryPeriod"); pk.pack_nil();
            pk.pack("stp"); pk.pack(1u);
            pk.pack("currency"); pk.pack(0u);
            pk.pack("leverage"); pk.pack(0.0);
            pk.pack("settleFlag"); pk.pack(int32_t{-1});
        }});
    };
    auto market = [&](int32_t agentId, uint32_t direction, uint32_t bookId) {
        orders.push_back({agentId, "PLACE_ORDER_MARKET", [=, id = clientOrderId++](auto& pk) {
            pk.pack_map(9);
            pk.pack("direction"); pk.pack(direction);
            pk.pack("volume"); pk.pack(0.25);
            pk.pack("bookId"); pk.pack(bookId);
            pk.pack("clientOrderId"); pk.pack(id);
            pk.pack("stp"); pk.pack(1u);
            pk.pack("currency"); pk.pack(0u);
            pk.pack("leverage"); pk.pack(0.0);
            pk.pack("settleFlag"); pk.pack(int32_t{-1});
            pk.pack("max_slippage"); pk.pack(0.0);
        }});
    };
    auto cancel = [&](int32_t agentId, uint32_t orderId, uint32_t bookId) {
        orders.push_back({agentId, "CANCEL_ORDERS", [=](auto& pk) {
            pk.pack_map(2);
            pk.pack("cancellations");
            pk.pack_array(1);
            pk.pack_map(2);
            pk.pack("orderId"); pk.pack(orderId);
            pk.pack("volume"); pk.pack_nil();
            pk.pack("bookId"); pk.pack(bookId);
        }});
    };
    auto cents = [](double price) { return std::round(price * 100.0) / 100.0; };
    for (const auto* part : parts(state)) {
        const auto* books = member(*part, "books");
        if (books == nullptr || books->type != msgpack::type::MAP || books->via.map.size == 0) continue;
        std::vector<uint32_t> ids;
        for (const auto& [k, v] : books->via.map) ids.push_back(k.as<uint32_t>());
        std::ranges::sort(ids);
        const uint32_t bookId = ids[step % ids.size()];
        const auto* book = entry(*books, bookId);
        if (book == nullptr) continue;
        const auto bid = bestPrice(*book, "b");
        const auto ask = bestPrice(*book, "a");
        const double mid = bid && ask ? (*bid + *ask) / 2 : bid ? *bid : ask ? *ask : 300.0;
        limit(0, 0, cents(mid - 0.05), bookId);
        limit(1, 1, cents(mid + 0.05), bookId);
        if (step % 3 == 1) {
            limit(0, 0, cents(mid + 0.10), bookId);
            market(1, 1, bookId);
        }
        if (step % 2 == 1) {
            const auto* accounts = member(*part, "accounts");
            for (int32_t miner = 0; miner < kMiners; ++miner) {
                const auto* account = accounts ? entry(*accounts, miner) : nullptr;
                if (account == nullptr || account->type != msgpack::type::MAP) continue;
                std::optional<std::tuple<uint64_t, uint32_t, uint32_t>> oldest;  // time, order id, book
                for (const auto& [book, onBook] : account->via.map) {
                    const auto* open = member(onBook, "o");
                    if (open == nullptr || open->type != msgpack::type::ARRAY) continue;
                    for (const auto& order : open->via.array) {
                        const auto* t = member(order, "t");
                        const auto* id = member(order, "i");
                        if (t == nullptr || id == nullptr) continue;
                        const std::tuple key{t->as<uint64_t>(), id->as<uint32_t>(), book.as<uint32_t>()};
                        if (!oldest || key < *oldest) oldest = key;
                    }
                }
                if (oldest) cancel(miner, std::get<1>(*oldest), std::get<2>(*oldest));
            }
        }
    }
    taosim::serialization::BinaryStream reply;
    msgpack::packer pk{reply};
    pk.pack_map(1);
    pk.pack("responses");
    pk.pack_array(static_cast<uint32_t>(orders.size()));
    for (size_t i = 0; i < orders.size(); ++i) {
        const auto& order = orders[i];
        pk.pack_map(4);
        pk.pack("agentId"); pk.pack(order.agentId);
        // Distinct delays keep the orders' arrival order the order they are listed in.
        pk.pack("delay"); pk.pack(static_cast<uint64_t>(1'000'000 + 100'000 * i));
        pk.pack("type"); pk.pack(order.type);
        pk.pack("payload"); order.payload(pk);
    }
    return {reply.data(), reply.size()};
}

// Reads each published state from the state segment as the validator does, keeps its bytes, and answers with an
// empty response list, or, with miners, with their orders for that state.
class StandInValidator
{
public:
    explicit StandInValidator(std::string suffix, bool miners = false)
        : m_suffix{std::move(suffix)},
          m_miners{miners},
          m_req{desc("taosim-req")},
          m_res{desc("taosim-res")}
    {
        m_thread = std::jthread{[this](std::stop_token stop) { serve(stop); }};
    }

    ~StandInValidator()
    {
        m_thread.request_stop();
        m_thread.join();
        for (const auto* base : {"taosim-req", "taosim-res"}) {
            taosim::ipc::PosixMessageQueue::remove(fmt::format("/{}{}", base, m_suffix));
        }
        bipc::shared_memory_object::remove(fmt::format("state{}", m_suffix).c_str());
        bipc::shared_memory_object::remove(fmt::format("responses{}", m_suffix).c_str());
    }

    [[nodiscard]] std::vector<std::string> states() const
    {
        const std::scoped_lock lock{m_mutex};
        return m_states;
    }

private:
    taosim::ipc::PosixMessageQueueDesc desc(std::string_view base) const
    {
        taosim::ipc::PosixMessageQueueDesc d{.name = fmt::format("{}{}", base, m_suffix)};
        d.timeout = std::optional<size_t>{static_cast<size_t>(std::chrono::nanoseconds{std::chrono::milliseconds{200}}.count())};
        return d;
    }

    void serve(std::stop_token stop)
    {
        taosim::serialization::BinaryStream empty;
        msgpack::packer pk{empty};
        pk.pack_map(1);
        pk.pack("responses");
        pk.pack_array(0);
        while (!stop.stop_requested()) {
            size_t size{};
            if (m_req.receive(std::span{std::bit_cast<char*>(&size), sizeof(size)}) == -1) continue;
            std::string orders;
            {
                bipc::shared_memory_object shm{bipc::open_only, fmt::format("state{}", m_suffix).c_str(), bipc::read_only};
                bipc::mapped_region region{shm, bipc::read_only};
                const std::scoped_lock lock{m_mutex};
                m_states.emplace_back(static_cast<const char*>(region.get_address()), size);
                if (m_miners) {
                    const auto& bytes = m_states.back();
                    orders = minerOrders(msgpack::unpack(bytes.data(), bytes.size()).get(), m_states.size() - 1);
                }
            }
            const std::string_view reply =
                m_miners ? std::string_view{orders} : std::string_view{empty.data(), empty.size()};
            bipc::shared_memory_object out{
                bipc::open_or_create, fmt::format("responses{}", m_suffix).c_str(), bipc::read_write};
            out.truncate(static_cast<bipc::offset_t>(reply.size()));
            bipc::mapped_region region{out, bipc::read_write};
            std::memcpy(region.get_address(), reply.data(), reply.size());
            const size_t replySize = reply.size();
            m_res.send(std::span{std::bit_cast<const char*>(&replySize), sizeof(replySize)});
        }
    }

    std::string m_suffix;
    bool m_miners;
    taosim::ipc::PosixMessageQueue m_req;
    taosim::ipc::PosixMessageQueue m_res;
    mutable std::mutex m_mutex;
    std::vector<std::string> m_states;
    std::jthread m_thread;
};

struct Published
{
    uint64_t timestamp{};
    std::vector<std::string> instances;  // each realization's part, re-packed, in flat order
    std::vector<std::string> comparable;  // the same parts with the child's process id replaced
};

[[nodiscard]] std::string repack(const msgpack::object& o)
{
    taosim::serialization::BinaryStream stream;
    msgpack::packer pk{stream};
    pk.pack(o);
    return {stream.data(), stream.size()};
}

// A part with the child's temporary directory, which carries its process id (the logDir, and the start notice's run
// directory), replaced by one placeholder, so two runs compare byte for byte.
[[nodiscard]] std::string comparable(std::string bytes)
{
    static const std::regex pid{R"(taosim-test-ma-[0-9]+-child)"};
    return std::regex_replace(bytes, pid, "taosim-test-ma-PID-child");
}

// A part with every account's open orders sorted by order id: the form the engine publishes (Account::ActiveOrders is
// ordered by id), against which a published part is compared.
[[nodiscard]] std::string withOpenOrdersById(const std::string& bytes)
{
    const auto oh = msgpack::unpack(bytes.data(), bytes.size());
    const msgpack::object part = oh.get();
    const auto* accounts = member(part, "accounts");
    if (accounts != nullptr && accounts->type == msgpack::type::MAP) {
        for (const auto& [agent, books] : accounts->via.map) {
            if (books.type != msgpack::type::MAP) continue;
            for (const auto& [book, onBook] : books.via.map) {
                const auto* open = member(onBook, "o");
                if (open == nullptr || open->type != msgpack::type::ARRAY) continue;
                auto* first = const_cast<msgpack::object*>(open->via.array.ptr);
                std::sort(first, first + open->via.array.size, [](const msgpack::object& x, const msgpack::object& y) {
                    const auto* xi = member(x, "i");
                    const auto* yi = member(y, "i");
                    return (xi ? xi->as<uint64_t>() : 0) < (yi ? yi->as<uint64_t>() : 0);
                });
            }
        }
    }
    return repack(part);
}

[[nodiscard]] uint64_t fnv1a(std::string_view bytes)
{
    uint64_t h = 1469598103934665603ull;
    for (const unsigned char c : bytes) h = (h ^ c) * 1099511628211ull;
    return h;
}

[[nodiscard]] Published parse(const std::string& bytes)
{
    Published out;
    const auto oh = msgpack::unpack(bytes.data(), bytes.size());
    const auto& top = oh.get();
    EXPECT_EQ(top.type, msgpack::type::MAP);
    if (top.type != msgpack::type::MAP) return out;
    for (const auto& [k, v] : top.via.map) {
        const auto key = k.as<std::string_view>();
        if (key == "timestamp") out.timestamp = v.as<uint64_t>();
        if (key != "backgrounds") continue;
        for (const auto& bg : v.via.array) {
            for (const auto& [bk, bv] : bg.via.map) {
                if (bk.as<std::string_view>() != "instances") continue;
                for (const auto& instance : bv.via.array) {
                    EXPECT_EQ(instance.type, msgpack::type::MAP) << "a published part is not a validator request";
                    out.instances.push_back(repack(instance));
                    out.comparable.push_back(comparable(out.instances.back()));
                }
            }
        }
    }
    return out;
}

struct PublishingRun
{
    std::vector<Published> published;
    std::vector<std::string> finalParts;  // each realization packed after the run, re-packed
    ChildExit exit;
};

[[nodiscard]] PublishingRun runPublishingInChild(
    std::string_view name, std::optional<uint32_t> fail, bool miners = false, uint32_t threads = 1,
    uint64_t steps = kSteps)
{
    const auto dir = fs::temp_directory_path() / fmt::format("taosim-test-pub-{}-{}", ::getpid(), name);
    fs::remove_all(dir);
    fs::create_directories(dir);
    const auto suffix = fmt::format("-test-pub-{}-{}", ::getpid(), name);
    LifecycleStub lifecycle;
    PublishingRun out;
    {
        StandInValidator validator{suffix, miners};
        out.exit = runTestInChild(
            kChildTest,
            {{std::string{kIpcSuffixVar}, suffix},
             {std::string{kPortVar}, std::to_string(lifecycle.port())},
             {std::string{kFailVar}, fail ? std::to_string(*fail) : std::string{}},
             {std::string{kOutVar}, (dir / "parts").string()},
             {std::string{kMinersVar}, miners ? std::to_string(kMiners) : std::string{}},
             {std::string{kThreadsVar}, std::to_string(threads)},
             {std::string{kStepsVar}, std::to_string(steps)}},
            dir / "child.log");
        for (const auto& state : validator.states()) out.published.push_back(parse(state));
    }
    if (out.exit.clean()) {
        const auto bytes = readBytes(dir / "parts");
        std::vector<std::string> raw;
        msgpack::unpack(bytes.data(), bytes.size()).get().convert(raw);
        for (const auto& part : raw) out.finalParts.push_back(repack(msgpack::unpack(part.data(), part.size()).get()));
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
    return out;
}

// What the miners' part of a run's published states carried: their notices, by kind, and the states in which a miner
// had an order open.
struct MinerActivity
{
    uint64_t placed{};
    uint64_t refused{};
    uint64_t traded{};
    uint64_t cancelled{};
    uint64_t restingStates{};
};

[[nodiscard]] MinerActivity minerActivity(const PublishingRun& run)
{
    MinerActivity out;
    for (const auto& state : run.published) {
        bool resting = false;
        for (const auto& bytes : state.instances) {
            const auto oh = msgpack::unpack(bytes.data(), bytes.size());
            const auto& part = oh.get();
            const auto* notices = member(part, "notices");
            const auto* accounts = member(part, "accounts");
            for (int32_t miner = 0; miner < kMiners; ++miner) {
                const auto* list = notices ? entry(*notices, miner) : nullptr;
                if (list != nullptr && list->type == msgpack::type::ARRAY) {
                    for (const auto& notice : list->via.array) {
                        const auto* y = member(notice, "y");
                        if (y == nullptr || y->type != msgpack::type::STR) continue;
                        const auto kind = y->as<std::string_view>();
                        if (kind == "RDPOL" || kind == "RDPOM") ++out.placed;
                        else if (kind == "ERDPOL" || kind == "ERDPOM") ++out.refused;
                        else if (kind == "ET") ++out.traded;
                        else if (kind == "RDCO") ++out.cancelled;
                    }
                }
                const auto* account = accounts ? entry(*accounts, miner) : nullptr;
                if (account == nullptr || account->type != msgpack::type::MAP) continue;
                for (const auto& [book, onBook] : account->via.map) {
                    const auto* open = member(onBook, "o");
                    if (open != nullptr && open->type == msgpack::type::ARRAY && open->via.array.size > 0) resting = true;
                }
            }
        }
        if (resting) ++out.restingStates;
    }
    return out;
}

// The most open orders any account had published on one book in any of a run's states.
[[nodiscard]] size_t longestOpenOrderList(const PublishingRun& run)
{
    size_t longest{};
    for (const auto& state : run.published) {
        for (const auto& bytes : state.instances) {
            const auto oh = msgpack::unpack(bytes.data(), bytes.size());
            const auto* accounts = member(oh.get(), "accounts");
            if (accounts == nullptr || accounts->type != msgpack::type::MAP) continue;
            for (const auto& [agent, books] : accounts->via.map) {
                if (books.type != msgpack::type::MAP) continue;
                for (const auto& [book, onBook] : books.via.map) {
                    const auto* open = member(onBook, "o");
                    if (open != nullptr && open->type == msgpack::type::ARRAY) {
                        longest = std::max<size_t>(longest, open->via.array.size);
                    }
                }
            }
        }
    }
    return longest;
}

}  // namespace

//-------------------------------------------------------------------------

TEST(PublishedState, EveryStepPublishesEveryRealizationsPart)
{
    const auto run = runPublishingInChild("all", std::nullopt);
    ASSERT_TRUE(run.exit.clean()) << run.exit.log;
    ASSERT_EQ(run.published.size(), kSteps) << "one state per step, past a grace period of zero";
    for (const auto& state : run.published) {
        ASSERT_EQ(state.instances.size(), kRealizations);
        for (uint32_t i = 0; i < kRealizations; ++i) {
            EXPECT_TRUE(state.instances[i].contains(fmt::format("/bg-{}", i)))
                << "at " << state.timestamp << " part " << i << " is not realization " << i << "'s";
        }
    }
    for (uint32_t i = 0; i < kRealizations; ++i) {
        EXPECT_NE(run.published.front().instances[i], run.published.back().instances[i])
            << "realization " << i << "'s part did not move over the run";
    }
}

TEST(PublishedState, AFailedRealizationPublishesItsFrozenStateAndTheOthersGoOn)
{
    const auto run = runPublishingInChild("failed", 1);
    ASSERT_TRUE(run.exit.clean()) << run.exit.log;
    ASSERT_EQ(run.finalParts.size(), kRealizations);
    ASSERT_EQ(run.published.size(), kSteps) << "the other realizations stopped publishing after the failure";
    std::optional<size_t> firstAfter;
    for (size_t i = 0; i < run.published.size(); ++i) {
        const auto& state = run.published[i];
        ASSERT_EQ(state.instances.size(), kRealizations);
        if (state.timestamp <= kFailTime) continue;
        if (!firstAfter) firstAfter = i;
        EXPECT_EQ(state.instances[1], run.finalParts[1])
            << "at " << state.timestamp << " the failed realization published something other than its frozen state";
    }
    ASSERT_TRUE(firstAfter.has_value() && *firstAfter + 1 < run.published.size());
    for (const uint32_t i : {0u, 2u}) {
        EXPECT_NE(run.published.back().instances[i], run.published[*firstAfter].instances[i])
            << "realization " << i << " did not step on beside the failure";
    }
}

// The seeded run publishes the same parts every time, and their digests, printed, compare one build with another:
// the parity of a change to how the state is packed (onto the workers, for one) is that the digests of a
// build before it and a build after it agree, step for step.
TEST(PublishedState, TheSeededRunPublishesTheSamePartsEveryTime)
{
    const auto a = runPublishingInChild("parity-a", std::nullopt);
    const auto b = runPublishingInChild("parity-b", std::nullopt);
    ASSERT_TRUE(a.exit.clean() && b.exit.clean()) << a.exit.log << b.exit.log;
    ASSERT_EQ(a.published.size(), b.published.size());
    for (size_t step = 0; step < a.published.size(); ++step) {
        EXPECT_EQ(a.published[step].comparable, b.published[step].comparable) << "step " << step << " differs";
        std::string line = fmt::format("published-digest step={} t={}", step, a.published[step].timestamp);
        for (const auto& part : a.published[step].comparable) line += fmt::format(" {:016x}", fnv1a(part));
        fmt::println("{}", line);
    }
}

// The same with miners trading: the stand-in answers each state with the miners' orders for it, so the miners' part of
// the state (their accounts, open orders and notices) is filled, and it too is published the same every time and,
// printed, compares one build with another.
TEST(PublishedState, TheSeededRunWithMinersPublishesTheSamePartsEveryTime)
{
    const auto a = runPublishingInChild("miners-a", std::nullopt, true);
    const auto b = runPublishingInChild("miners-b", std::nullopt, true);
    ASSERT_TRUE(a.exit.clean() && b.exit.clean()) << a.exit.log << b.exit.log;
    ASSERT_EQ(a.published.size(), b.published.size());
    for (size_t step = 0; step < a.published.size(); ++step) {
        EXPECT_EQ(a.published[step].comparable, b.published[step].comparable) << "step " << step << " differs";
        std::string line = fmt::format("published-digest-miners step={} t={}", step, a.published[step].timestamp);
        for (const auto& part : a.published[step].comparable) line += fmt::format(" {:016x}", fnv1a(part));
        fmt::println("{}", line);
    }
    const auto activity = minerActivity(a);
    fmt::println(
        "miner-activity placed={} refused={} traded={} cancelled={} resting-states={} of {}", activity.placed,
        activity.refused, activity.traded, activity.cancelled, activity.restingStates, a.published.size());
    EXPECT_GT(activity.placed, 0u) << "no miner order was placed";
    EXPECT_GT(activity.traded, 0u) << "no miner order traded";
    EXPECT_GT(activity.cancelled, 0u) << "no miner order was cancelled";
    EXPECT_GT(activity.restingStates, 0u) << "no state showed a miner's open order";
}

// A caller that packs the request whole, without the parts its workers pack, still gets every realization, in order:
// the request then packs each one itself. (Packing a realization consumes its proxy's notices, so the two forms
// cannot be compared on one state; the digests above compare the published bytes of a build before the change.)
TEST(PublishedState, ARequestPackedWholeStillCarriesEveryRealization)
{
    const MultiAssetRun run{
        "whole", {.realizations = kRealizations, .threads = 1, .steps = kSteps, .ckptIntervalInSteps = 0}};
    const auto orchestrator = run.build();
    orchestrator->run();
    taosim::serialization::HumanReadableStream whole;
    msgpack::pack(whole, taosim::simulation::multiasset::serialization::MultiAssetValidatorRequest{
        .simulations = orchestrator->checkpointBlocks(),
        .config = &orchestrator->config(),
        .logDir = orchestrator->config().baseDir});
    const auto state = parse(std::string{whole.data(), whole.size()});
    ASSERT_EQ(state.instances.size(), kRealizations);
    for (uint32_t i = 0; i < kRealizations; ++i) {
        EXPECT_TRUE(state.instances[i].contains(fmt::format("/bg-{}", i))) << "part " << i << " is not realization " << i << "'s";
    }
}

// Each account's open orders on a book are published in order-id order, however the engine allocated them: every
// published part equals itself with its open orders sorted by id, on one thread and on three, and two seeded runs on
// three threads, where the orders' addresses differ run to run, publish the same bytes at every step. Long enough for
// the miners to hold several open orders on a book at once, which the test checks so that the order is tested.
constexpr uint64_t kOrderIdSteps = 120;

TEST(PublishedState, AnAccountsOpenOrdersArePublishedInOrderIdOrder)
{
    auto byId = [](const Published& state) {
        std::vector<std::string> parts;
        for (const auto& part : state.instances) parts.push_back(comparable(withOpenOrdersById(part)));
        return parts;
    };
    const auto one = runPublishingInChild("ids-1", std::nullopt, true, 1, kOrderIdSteps);
    const auto a = runPublishingInChild("ids-3a", std::nullopt, true, 3, kOrderIdSteps);
    const auto b = runPublishingInChild("ids-3b", std::nullopt, true, 3, kOrderIdSteps);
    ASSERT_TRUE(one.exit.clean() && a.exit.clean() && b.exit.clean()) << one.exit.log << a.exit.log << b.exit.log;
    for (const auto* run : {&one, &a, &b}) {
        EXPECT_GE(longestOpenOrderList(*run), 2u) << "no account held two open orders on a book at once";
        for (size_t step = 0; step < run->published.size(); ++step) {
            EXPECT_EQ(run->published[step].comparable, byId(run->published[step]))
                << "step " << step << ": an open-order list is not in order-id order";
        }
    }
    ASSERT_EQ(a.published.size(), b.published.size());
    for (size_t step = 0; step < a.published.size(); ++step) {
        EXPECT_EQ(a.published[step].comparable, b.published[step].comparable) << "step " << step << " differs";
    }
}

// The child side of the tests above, run only when re-executed by them.
TEST(PublishedStateChild, DISABLED_RunsAndPublishes)
{
    const char* suffix = std::getenv(kIpcSuffixVar.data());
    const char* port = std::getenv(kPortVar.data());
    const char* fail = std::getenv(kFailVar.data());
    const char* out = std::getenv(kOutVar.data());
    const char* miners = std::getenv(kMinersVar.data());
    const char* threads = std::getenv(kThreadsVar.data());
    const char* steps = std::getenv(kStepsVar.data());
    if (suffix == nullptr || *suffix == '\0' || port == nullptr || fail == nullptr || out == nullptr) {
        GTEST_SKIP() << "runs only as the child of a PublishedState test";
    }
    ASSERT_TRUE(taosim::simulation::SimulationManager::s_statePublishShmName.ends_with(suffix));
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    const MultiAssetRun run{
        "child", {.realizations = kRealizations,
                  .threads = threads && *threads != '\0' ? static_cast<uint32_t>(std::stoul(threads)) : 1u,
                  .steps = steps && *steps != '\0' ? std::stoull(steps) : kSteps, .ckptIntervalInSteps = 0,
                  .extraAttrs = fmt::format(
                      R"(useMessagePack="1" host="127.0.0.1" port="{}" generalMsgEndpoint="/account" {})", port,
                      kContinue),
                  .remoteAgentCount = miners && *miners != '\0' ? static_cast<uint32_t>(std::stoul(miners)) : 1u}};
    const auto orchestrator = run.build();
    if (*fail != '\0') {
        const auto* sim = orchestrator->checkpointBlocks()[std::stoul(fail)].get();
        sim->signals().step.connect([sim] {
            if (sim->currentTimestamp() >= kFailTime) throw std::runtime_error{"injected failure"};
        });
    }
    orchestrator->run();

    const auto& blocks = orchestrator->checkpointBlocks();
    std::vector<std::string> parts;
    for (size_t i = 0; i < blocks.size(); ++i) {
        // The stream the engine publishes through: decimals as doubles, as the validator reads them.
        taosim::serialization::HumanReadableStream stream{1uz << 22};
        msgpack::pack(stream, taosim::simulation::serialization::ValidatorRequest{
            .simulations = std::span{blocks}.subspan(i, 1), .logDir = blocks[i]->logDir()});
        parts.emplace_back(stream.data(), stream.size());
    }
    taosim::serialization::BinaryStream stream;
    msgpack::packer pk{stream};
    pk.pack(parts);
    std::ofstream{out, std::ios::binary}.write(stream.data(), stream.size());
}

//-------------------------------------------------------------------------
