// SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
// SPDX-License-Identifier: MIT
//
// The engine publishes a state and waits for the validator's answer, sending the state again after
// each full queue timeout. A stop is acted on only once a step completes, so a validator that has
// already gone away kept the engine in that wait until it was killed, and the shutdown checkpoint was
// never written. The wait now asks between short slices whether to give up. Test-only queue names,
// never the production /taosim-* ones.

#include <gtest/gtest.h>

#include <taosim/ipc/PosixMessageQueue.hpp>
#include <taosim/simulation/SimulationManager.hpp>
#include <taosim/simulation/multiasset/helpers.hpp>

#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <fmt/format.h>
#include <rapidjson/document.h>

extern "C" {
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
}

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

extern char** environ;

using taosim::ipc::PosixMessageQueue;
using taosim::ipc::PosixMessageQueueDesc;
using namespace std::chrono_literals;

namespace
{

std::string testQueueName(const char* role)
{
    return "/taosim-test-publish-stop-" + std::to_string(::getpid()) + "-" + role;
}

PosixMessageQueueDesc testDesc(const std::string& name, std::chrono::nanoseconds timeout)
{
    PosixMessageQueueDesc desc{.name = name};
    desc.timeout = std::optional<size_t>{static_cast<size_t>(timeout.count())};
    return desc;
}

}  // namespace

TEST(PublishStop, AnAnsweredRequestReturnsTheReply)
{
    PosixMessageQueue req{testDesc(testQueueName("req-a"), 2s)};
    PosixMessageQueue res{testDesc(testQueueName("res-a"), 2s)};
    std::thread validator{[&] {
        size_t got{};
        if (req.receive(std::span{std::bit_cast<char*>(&got), sizeof(got)}) != -1) {
            const size_t reply = got + 35;
            res.send(std::span{std::bit_cast<const char*>(&reply), sizeof(reply)});
        }
    }};

    const auto reply = taosim::ipc::sendAndAwaitReply(req, res, 7, [] { return false; });
    validator.join();

    ASSERT_TRUE(reply.has_value());
    EXPECT_EQ(*reply, 42u);
}

TEST(PublishStop, AStopIsHonouredWhileNoValidatorAnswers)
{
    PosixMessageQueue req{testDesc(testQueueName("req-b"), 30s)};
    PosixMessageQueue res{testDesc(testQueueName("res-b"), 30s)};
    std::atomic_bool stop{false};
    std::thread signaller{[&] {
        std::this_thread::sleep_for(300ms);
        stop.store(true);
    }};

    const auto t0 = std::chrono::steady_clock::now();
    const auto reply = taosim::ipc::sendAndAwaitReply(req, res, 7, [&] { return stop.load(); });
    const auto waited = std::chrono::steady_clock::now() - t0;
    signaller.join();

    EXPECT_FALSE(reply.has_value());
    EXPECT_LT(waited, 5s) << "the wait ran on towards the 30 s queue timeout instead of stopping";
}

TEST(PublishStop, ARequestUnansweredForAFullTimeoutIsSentAgain)
{
    PosixMessageQueue req{testDesc(testQueueName("req-c"), 200ms)};
    PosixMessageQueue res{testDesc(testQueueName("res-c"), 200ms)};
    int retries = 0;

    const auto reply = taosim::ipc::sendAndAwaitReply(
        req, res, 7, [&] { return retries >= 2; }, [&](std::string_view) { ++retries; });

    EXPECT_FALSE(reply.has_value());
    EXPECT_GE(retries, 2);
}

//-------------------------------------------------------------------------
// The validator takes EVENT_SIMULATION_END as the end of the run and opens the next one, so the end is
// published only by a run that reached its duration. A run that leaves on a stop request resumes from
// its shutdown checkpoint and publishes no end. Each run below is a child process (this binary,
// re-executed) with private IPC names, posting its lifecycle notices to a recorder in the parent.

namespace
{

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace fs = std::filesystem;
namespace http = beast::http;
using tcp = asio::ip::tcp;

constexpr std::string_view kRunConfigVar = "TAOSIM_TEST_RUN_CONFIG";
constexpr std::string_view kRunDirVar = "TAOSIM_TEST_RUN_DIR";
constexpr std::string_view kIpcSuffixVar = "TAOSIM_IPC_SUFFIX";
constexpr std::string_view kChildTest = "PublishStopChild.DISABLED_RunsTheConfigInItsEnvironment";

// Answers each notice batch as the validator's /account route does and keeps the type of every
// message in it.
class NoticeRecorder
{
public:
    NoticeRecorder()
    {
        accept();
        m_thread = std::jthread{[this] { m_ctx.run(); }};
    }

    ~NoticeRecorder() { m_ctx.stop(); }

    [[nodiscard]] uint16_t port() const { return m_acceptor.local_endpoint().port(); }

    [[nodiscard]] std::vector<std::string> types() const
    {
        const std::scoped_lock lock{m_mutex};
        return m_types;
    }

    [[nodiscard]] size_t count(std::string_view type) const
    {
        const auto all = types();
        return static_cast<size_t>(std::ranges::count(all, type));
    }

private:
    void accept()
    {
        m_acceptor.async_accept([this](beast::error_code ec, tcp::socket socket) {
            if (ec) return;
            serve(std::move(socket));
            accept();
        });
    }

    void serve(tcp::socket socket)
    {
        beast::flat_buffer buffer;
        http::request<http::string_body> req;
        beast::error_code ec;
        http::read(socket, buffer, req, ec);
        if (ec) return;
        rapidjson::Document body;
        body.Parse(req.body().c_str());
        if (body.IsObject() && body.HasMember("messages") && body["messages"].IsArray()) {
            const std::scoped_lock lock{m_mutex};
            for (const auto& message : body["messages"].GetArray()) {
                if (message.IsObject() && message.HasMember("type") && message["type"].IsString()) {
                    m_types.emplace_back(message["type"].GetString());
                }
            }
        }
        http::response<http::string_body> res{http::status::ok, req.version()};
        res.set(http::field::content_type, "application/json");
        res.body() = "{}";
        res.prepare_payload();
        http::write(socket, res, ec);
        socket.shutdown(tcp::socket::shutdown_send, ec);
    }

    asio::io_context m_ctx;
    tcp::acceptor m_acceptor{m_ctx, tcp::endpoint{asio::ip::make_address("127.0.0.1"), 0}};
    mutable std::mutex m_mutex;
    std::vector<std::string> m_types;
    std::jthread m_thread;
};

// A grid long enough that only a stop ends it, and a grace period past any grid, so no state is ever
// published and the only traffic is the lifecycle notices.
constexpr std::string_view kEndlessDuration = "1000000000000000000";
constexpr std::string_view kShortDuration = "5000000000";
constexpr std::string_view kNoGracePeriodEnd = "18000000000000000000";

std::string backgroundXml(std::string_view duration, std::string_view rootAttrs)
{
    return fmt::format(
        R"(<Simulation id="run" start="0" step="1000000000" duration="{}" {}>
    <Agents>
        <MultiBookExchangeAgent name="EXCHANGE" priceDecimals="2" volumeDecimals="4" baseDecimals="4"
            quoteDecimals="8" remoteAgentCount="1" gracePeriod="{}" initialPrice="100.0">
            <Books instanceCount="1" algorithm="PriceTime"/>
            <Balances>
                <Base total="100.0"/>
                <Quote total="10000.0"/>
            </Balances>
        </MultiBookExchangeAgent>
        <DistributedProxyAgent/>
    </Agents>
</Simulation>
)",
        duration, rootAttrs, kNoGracePeriodEnd);
}

std::string wrapperXml(std::string_view duration, std::string_view rootAttrs)
{
    return fmt::format(
        R"(<MultiAssetSimulation useMessagePack="1" threads="1" start="0" step="1000000000" duration="{}"
    gracePeriod="{}" {}>
    <Background name="bg" path="background.xml"/>
</MultiAssetSimulation>
)",
        duration, kNoGracePeriodEnd, rootAttrs);
}

std::string readFile(const fs::path& path)
{
    std::ifstream ifs{path};
    return {std::istreambuf_iterator<char>{ifs}, std::istreambuf_iterator<char>{}};
}

enum class RunKind { SINGLE, MULTI_ASSET };

// One child run: its directory, the recorder it posts to, and its process.
class ChildRun
{
public:
    ChildRun(RunKind kind, std::string_view duration, std::string_view name)
        : m_dir{fs::temp_directory_path() / fmt::format("taosim-test-run-exit-{}-{}", ::getpid(), name)},
          m_ipcSuffix{fmt::format("-test-run-exit-{}-{}", ::getpid(), name)}
    {
        fs::remove_all(m_dir);
        fs::create_directories(m_dir);
        // traceTime prints a line per step, which is how the parent sees that a step has completed.
        const auto rootAttrs = fmt::format(
            R"(traceTime="1" host="127.0.0.1" port="{}" generalMsgEndpoint="/account")",
            m_recorder.port());
        const auto config = m_dir / "config.xml";
        if (kind == RunKind::SINGLE) {
            std::ofstream{config} << backgroundXml(duration, rootAttrs);
        } else {
            std::ofstream{m_dir / "background.xml"} << backgroundXml(duration, "");
            std::ofstream{config} << wrapperXml(duration, rootAttrs);
        }
        spawn(config);
    }

    ~ChildRun()
    {
        if (m_pid > 0) {
            ::kill(m_pid, SIGKILL);
            ::waitpid(m_pid, nullptr, 0);
        }
        for (const auto* base : {"taosim-req", "taosim-res"}) {
            PosixMessageQueue::remove(fmt::format("/{}{}", base, m_ipcSuffix));
        }
        std::error_code ec;
        fs::remove_all(m_dir, ec);
    }

    [[nodiscard]] const NoticeRecorder& recorder() const noexcept { return m_recorder; }
    [[nodiscard]] std::string log() const { return readFile(m_dir / "child.log"); }

    [[nodiscard]] bool waitForOutput(std::string_view needle, std::chrono::seconds limit) const
    {
        const auto deadline = std::chrono::steady_clock::now() + limit;
        while (std::chrono::steady_clock::now() < deadline) {
            if (log().contains(needle)) return true;
            std::this_thread::sleep_for(10ms);
        }
        return false;
    }

    void signal(int sig) const { ::kill(m_pid, sig); }

    // The child's wait status, or nothing if it outlived the limit (it is then killed).
    [[nodiscard]] std::optional<int> waitForExit(std::chrono::seconds limit)
    {
        const auto deadline = std::chrono::steady_clock::now() + limit;
        int status{};
        while (std::chrono::steady_clock::now() < deadline) {
            if (::waitpid(m_pid, &status, WNOHANG) == m_pid) {
                m_pid = -1;
                return status;
            }
            std::this_thread::sleep_for(10ms);
        }
        return std::nullopt;
    }

private:
    void spawn(const fs::path& config)
    {
        const auto self = fs::read_symlink("/proc/self/exe").string();
        const auto filter = fmt::format("--gtest_filter={}", kChildTest);
        std::vector<std::string> args{self, filter, "--gtest_also_run_disabled_tests"};

        // The child's own gtest settings come from its command line only: an inherited GTEST_OUTPUT
        // would have it overwrite the parent's report.
        std::vector<std::string> env;
        for (char** e = environ; *e != nullptr; ++e) {
            const std::string_view entry{*e};
            if (entry.starts_with("GTEST_") || entry.starts_with(kIpcSuffixVar)
                || entry.starts_with(kRunConfigVar) || entry.starts_with(kRunDirVar)) {
                continue;
            }
            env.emplace_back(entry);
        }
        env.push_back(fmt::format("{}={}", kIpcSuffixVar, m_ipcSuffix));
        env.push_back(fmt::format("{}={}", kRunConfigVar, config.string()));
        env.push_back(fmt::format("{}={}", kRunDirVar, (m_dir / "logs").string()));

        auto pointers = [](std::vector<std::string>& strings) {
            std::vector<char*> out;
            for (auto& s : strings) out.push_back(s.data());
            out.push_back(nullptr);
            return out;
        };
        auto argv = pointers(args);
        auto envp = pointers(env);

        const auto logPath = (m_dir / "child.log").string();
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_addopen(
            &actions, STDOUT_FILENO, logPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
        const auto rc = ::posix_spawn(&m_pid, self.c_str(), &actions, nullptr, argv.data(), envp.data());
        posix_spawn_file_actions_destroy(&actions);
        if (rc != 0) m_pid = -1;
    }

    NoticeRecorder m_recorder;
    fs::path m_dir;
    std::string m_ipcSuffix;
    pid_t m_pid{-1};
};

[[nodiscard]] bool exitedCleanly(const std::optional<int>& status)
{
    return status && WIFEXITED(*status) && WEXITSTATUS(*status) == 0;
}

void expectTheEndIsPublished(RunKind kind, std::string_view name)
{
    ChildRun run{kind, kShortDuration, name};

    const auto status = run.waitForExit(120s);

    ASSERT_TRUE(exitedCleanly(status)) << run.log();
    EXPECT_GE(run.recorder().count("EVENT_SIMULATION_START"), 1u) << run.log();
    EXPECT_EQ(run.recorder().count("EVENT_SIMULATION_END"), 1u)
        << "a run that reached its duration must tell the validator it ended\n" << run.log();
}

void expectAStopPublishesNoEnd(RunKind kind, std::string_view name)
{
    ChildRun run{kind, kEndlessDuration, name};
    // A step has completed, so the run is past its start notice and its stop handler is installed.
    ASSERT_TRUE(run.waitForOutput("TIME :", 120s)) << run.log();

    run.signal(SIGTERM);
    const auto status = run.waitForExit(120s);

    ASSERT_TRUE(exitedCleanly(status)) << "the run did not leave on the stop\n" << run.log();
    EXPECT_GE(run.recorder().count("EVENT_SIMULATION_START"), 1u) << run.log();
    EXPECT_EQ(run.recorder().count("EVENT_SIMULATION_END"), 0u)
        << "a stopped run published an end, so a validator would open a new run\n" << run.log();
    EXPECT_TRUE(run.log().contains("Stopped on request; no end notice published")) << run.log();
}

}  // namespace

//-------------------------------------------------------------------------

TEST(PublishStop, ARunThatReachesItsDurationPublishesItsEnd)
{
    expectTheEndIsPublished(RunKind::SINGLE, "single-end");
}

TEST(PublishStop, AStoppedRunPublishesNoEnd)
{
    expectAStopPublishesNoEnd(RunKind::SINGLE, "single-stop");
}

TEST(PublishStop, AMultiAssetRunThatReachesItsDurationPublishesItsEnd)
{
    expectTheEndIsPublished(RunKind::MULTI_ASSET, "multi-end");
}

TEST(PublishStop, AStoppedMultiAssetRunPublishesNoEnd)
{
    expectAStopPublishesNoEnd(RunKind::MULTI_ASSET, "multi-stop");
}

//-------------------------------------------------------------------------
// The child side of the tests above, run only when re-executed by them. It refuses to run without a
// private IPC suffix, so it can never open another engine's queues.

TEST(PublishStopChild, DISABLED_RunsTheConfigInItsEnvironment)
{
    const char* config = std::getenv(kRunConfigVar.data());
    const char* dir = std::getenv(kRunDirVar.data());
    const char* suffix = std::getenv(kIpcSuffixVar.data());
    if (config == nullptr || dir == nullptr || suffix == nullptr || *suffix == '\0') {
        GTEST_SKIP() << "runs only as the child of a PublishStop run test";
    }
    ASSERT_TRUE(taosim::simulation::SimulationManager::s_validatorReqMessageQueueName.ends_with(suffix));
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    auto orchestrator = taosim::simulation::multiasset::makeSimulationOrchestrator(config, dir);
    orchestrator->run();
}

//-------------------------------------------------------------------------
