/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/checkpoint/helpers.hpp>

#include <range/v3/action/sort.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/subrange.hpp>
#include <range/v3/view/transform.hpp>

#include <taosim/checkpoint/CheckpointError.hpp>
#include <taosim/checkpoint/CheckpointManager.hpp>
#include <taosim/checkpoint/serialization/accounting/AccountRegistry.hpp>
#include <taosim/checkpoint/serialization/agent/ALGOTraderAgent.hpp>
#include <taosim/checkpoint/serialization/agent/FuturesTraderAgent.hpp>
#include <taosim/checkpoint/serialization/agent/HighFrequencyTraderAgent.hpp>
#include <taosim/checkpoint/serialization/agent/NoiseTraderAgent.hpp>
#include <taosim/checkpoint/serialization/agent/RandomTraderAgent.hpp>
#include <taosim/checkpoint/serialization/agent/StylizedTraderAgent.hpp>
#include <taosim/checkpoint/serialization/book/Book.hpp>
#include <taosim/checkpoint/serialization/book/BookProcessManager.hpp>
#include <taosim/checkpoint/serialization/matching/ClearingManager.hpp>
#include <taosim/checkpoint/serialization/matching/ExchangeSignals.hpp>
#include <taosim/checkpoint/serialization/matching/SLTPContainer.hpp>
#include <taosim/event/serialization/L3RecordContainer.hpp>
#include <taosim/filesystem/utils.hpp>
#include <taosim/message/serialization/MessageQueue.hpp>
#include <taosim/serialization/msgpack/utils.hpp>
#include <taosim/simulation/SimulationManager.hpp>
#include <taosim/simulation/SimulationState.hpp>
#include <taosim/util/serialization/SubscriptionRegistry.hpp>

#include <fmt/format.h>

#include <cstdio>
#include <latch>

//-------------------------------------------------------------------------

namespace fs = std::filesystem;

//-------------------------------------------------------------------------

namespace
{

void printProgress(size_t cur, size_t tot, std::string_view label)
{
    constexpr int W = 30;
    const int filled = tot > 0 ? static_cast<int>(W * cur / tot) : 0;
    const std::string bar =
        std::string(filled > 0 ? filled - 1 : 0, '=')
        + (filled > 0 ? ">" : "")
        + std::string(W - filled, ' ');
    fmt::print(
        "\r  {:<20} [{}] {:>3}%  {}/{}",
        label, bar, tot > 0 ? 100 * cur / tot : 0, cur, tot);
    std::fflush(stdout);
    if (cur == tot) fmt::println("");
}

}  // namespace

//-------------------------------------------------------------------------

namespace taosim::checkpoint
{

//-------------------------------------------------------------------------

void restoreSignalCounters(MultiBookExchangeAgent* exchange, const msgpack::object& section)
{
    if (exchange == nullptr) return;
    if (section.type != msgpack::type::MAP) {
        throw CheckpointError{"checkpoint: signals section is not a map"};
    }
    auto& signalsMap = exchange->signals();
    for (uint32_t i = 0; i < section.via.map.size; ++i) {
        BookId bookId{};
        section.via.map.ptr[i].key.convert(bookId);
        const auto it = signalsMap.find(bookId);
        if (it == signalsMap.end() || !it->second) continue;
        // Plain integer into the EXISTING object. Never a convert of the object or of the map.
        uint64_t counter{};
        section.via.map.ptr[i].val.convert(counter);
        it->second->eventCounter = counter;
    }
}

//-------------------------------------------------------------------------

CheckpointToken postProcessToken(const CheckpointToken& token)
{
    if (s_specialTokens.contains(token)) {
        return token;
    }

    const auto path = fs::absolute(token);

    if (!fs::exists(path)) {
        throw CheckpointError{fmt::format("No such path: '{}'", path.c_str())};
    }
    if (!fs::is_directory(path)) {
        throw CheckpointError{fmt::format("Path '{}' not a valid directory", path.c_str())};
    }

    return path.string();
}

//-------------------------------------------------------------------------

fs::path runDirFromToken(const CheckpointToken& token)
{
    if (s_specialTokens.contains(token)) {
        const auto baseDir = fs::current_path() / "logs";
        if (!fs::exists(baseDir)) {
            throw CheckpointError{fmt::format("No such directory: '{}'", baseDir.c_str())};
        }
        return std::invoke(s_tokenToRunDirFactory.at(token), baseDir);
    }

    auto checkCkptStoreDir = [&](const fs::path& p) {
        if (!fs::is_directory(p / CheckpointManager::s_storeDirName)) {
            throw CheckpointError{fmt::format(
                "Run directory inferred from '{}' ('{}') is ill-formed; missing directory '{}'",
                token,
                p.c_str(),
                CheckpointManager::s_storeDirName
            )};
        }
    };

    if (token.ends_with(CheckpointManager::s_dirExtension)) {
        const auto runDir = fs::path{token}.parent_path().parent_path();
        checkCkptStoreDir(runDir);
        return runDir;
    }

    const fs::path runDir{token};
    checkCkptStoreDir(runDir);
    return runDir;
}

//-------------------------------------------------------------------------

fs::path runDirLatest(const fs::path& baseDir)
{
    // A valid simulation run directory must have a ckpt/ subdirectory containing
    // at least one .ckptd directory with common.ckpt (the simulation checkpoint
    // format). This excludes exchange-service directories which use state.ckpt.
    const auto isSimRunDir = [](const fs::path& p) {
        if (!fs::is_directory(p)) return false;
        const auto ckptStoreDir = p / CheckpointManager::s_storeDirName;
        if (!fs::is_directory(ckptStoreDir)) return false;
        const auto commonFile =
            fmt::format("common{}", CheckpointManager::s_fileExtension);
        for (const auto& entry : fs::directory_iterator{ckptStoreDir}) {
            if (entry.is_directory() && fs::exists(entry.path() / commonFile))
                return true;
        }
        return false;
    };
    const auto runDirsSortedByWriteTime =
        filesystem::collectMatchingPaths(baseDir, isSimRunDir)
        | ranges::actions::sort([](auto&& lhs, auto&& rhs) {
            return fs::last_write_time(lhs) < fs::last_write_time(rhs);
        });
    if (runDirsSortedByWriteTime.empty()) {
        throw CheckpointError{fmt::format(
            "No run directories containing '{}' found in '{}'",
            CheckpointManager::s_storeDirName, baseDir.c_str())};
    }
    return runDirsSortedByWriteTime.back();
}

//-------------------------------------------------------------------------

fs::path ckptDirLatest(const fs::path& runDir)
{
    const auto ckptStoreDir = runDir / CheckpointManager::s_storeDirName;

    if (!fs::exists(ckptStoreDir)) {
        throw CheckpointError{fmt::format(
            "Run directory '{}' missing directory '{}'",
            runDir.c_str(), CheckpointManager::s_storeDirName
        )};
    }

    return ckptDirsSortedByWriteTime(ckptStoreDir).back();
}

//-------------------------------------------------------------------------

fs::path ckptDirFromToken(const CheckpointToken& token)
{
    if (s_specialTokens.contains(token)) {
        const auto baseDir = fs::current_path() / "logs";
        if (!fs::exists(baseDir)) {
            throw CheckpointError{fmt::format("No such directory: '{}'", baseDir.c_str())};
        }
        return std::invoke(
            s_tokenToCkptDirFactory.at(token),
            std::invoke(s_tokenToRunDirFactory.at(token), baseDir));
    }

    if (token.ends_with(CheckpointManager::s_dirExtension)) {
        return token;
    }

    return ckptDirLatest(token);
}

//-------------------------------------------------------------------------

std::vector<fs::path> ckptDirsSortedByWriteTime(const fs::path& path)
{
    return ranges::subrange(fs::directory_iterator{path}, fs::directory_iterator{})
        | views::filter([](auto&& e) {
            return fs::is_directory(e)
                && e.path().extension() == CheckpointManager::s_dirExtension;
        })
        | views::transform([](auto&& p) { return p.path(); })
        | ranges::to<std::vector>
        | ranges::actions::sort([](auto&& lhs, auto&& rhs) {
            return fs::last_write_time(lhs) < fs::last_write_time(rhs);
        });
}

//-------------------------------------------------------------------------

static void setupAgents(const msgpack::object& o, Simulation& simu, size_t blockIdx)
{
    if (o.type != msgpack::type::MAP) {
        throw taosim::checkpoint::CheckpointError{std::to_string(blockIdx)};
    }

    for (const auto& [k, val] : o.via.map) {
        auto key = k.as<std::string_view>();

        auto agentIt = ranges::lower_bound(
            simu.agents(), key, {}, [](auto&& agent) { return agent->name(); });
        if (agentIt == simu.agents().end()) {
            throw taosim::checkpoint::CheckpointError{fmt::format(
                "{}: agent named {} not found", blockIdx, key
            )};
        }

        if (key.starts_with("ALGO_TRADER_AGENT")) {
            auto agent = dynamic_cast<taosim::agent::ALGOTraderAgent*>(agentIt->get());
            val.convert(*agent);
        }
        else if (key.starts_with("FUTURES_TRADER_AGENT")) {
            auto agent = dynamic_cast<taosim::agent::FuturesTraderAgent*>(agentIt->get());
            val.convert(*agent);
        }
        else if (key.starts_with("HIGH_FREQUENCY_TRADER_AGENT")) {
            auto agent = dynamic_cast<taosim::agent::HighFrequencyTraderAgent*>(agentIt->get());
            val.convert(*agent);
        }
        else if (key.starts_with("NOISE_TRADER_AGENT")) {
            auto agent = dynamic_cast<taosim::agent::NoiseTraderAgent*>(agentIt->get());
            val.convert(*agent);
        }
        else if (key.starts_with("RANDOM_TRADER_AGENT")) {
            auto agent = dynamic_cast<taosim::agent::RandomTraderAgent*>(agentIt->get());
            val.convert(*agent);
        }
        else if (key.starts_with("STYLIZED_TRADER_AGENT")) {
            auto agent = dynamic_cast<taosim::agent::StylizedTraderAgent*>(agentIt->get());
            val.convert(*agent);
        }
    }
}

//-------------------------------------------------------------------------

static void setupExchange(const msgpack::object& o, Simulation& simu, size_t blockIdx)
{
    if (o.type != msgpack::type::MAP) {
        throw taosim::checkpoint::CheckpointError{std::to_string(blockIdx)};
    }

    auto setupBooks = [&](const msgpack::object& o) {
        if (o.type != msgpack::type::ARRAY) {
            throw taosim::checkpoint::CheckpointError{std::to_string(blockIdx)};
        }
        const auto& arr = o.via.array;
        for (size_t i{}; i < arr.size; ++i) {
            const auto& val = arr.ptr[i];
            val.convert(*simu.exchange()->books().at(i));
        }
    };

    auto setupSignals = [&](const msgpack::object& o) {
        restoreSignalCounters(simu.exchange(), o);
    };

    for (const auto& [k, val] : o.via.map) {
        auto key = k.as<std::string_view>();

        if (key == "accounts") {
            val.convert(simu.exchange()->accounts());
        }
        else if (key == "books") {
            setupBooks(val);
        }
        else if (key == "signals") {
            // Restore eventCounter in-place on existing ExchangeSignals objects.
            // The default unique_ptr<ExchangeSignals> convert would create new
            // objects and destroy the originals, severing the L3EventLogger
            // signal connections established during fromConfig.
            if (val.type == msgpack::type::MAP) {
                auto& signalsMap = simu.exchange()->signals();
                for (uint32_t i = 0; i < val.via.map.size; ++i) {
                    BookId bookId;
                    val.via.map.ptr[i].key.convert(bookId);
                    const auto it = signalsMap.find(bookId);
                    if (it != signalsMap.end() && it->second) {
                        val.via.map.ptr[i].val.convert(*it->second);
                    }
                }
            }
        }
        else if (key == "bookProcessManager") {
            val.convert(simu.exchange()->bookProcessManager());
        }
        else if (key == "clearingManager") {
            val.convert(simu.exchange()->clearingManager());
        }
        else if (key == "L3Record") {
            val.convert(simu.exchange()->L3Record());
        }
        else if (key == "marginCallCounter") {
            val.convert(simu.exchange()->marginCallCounter());
        }
        else if (key == "localMarketOrderSubs") {
            val.convert(simu.exchange()->localMarketOrderSubs());
        }
        else if (key == "localLimitOrderSubs") {
            val.convert(simu.exchange()->localLimitOrderSubs());
        }
        else if (key == "localTradeSubs") {
            val.convert(simu.exchange()->localTradeSubs());
        }
        else if (key == "localTradeByOrderSubs") {
            val.convert(simu.exchange()->localTradeByOrderSubs());
        }
        else if (key == "localOwnTradeSubs") {
            val.convert(simu.exchange()->localOwnTradeSubs());
        }
        else if (key == "bookTradeStats") {
            val.convert(simu.exchange()->bookTradeStats());
        }
        else if (key == "sltpContainer") {
            // Restore the per-book trigger state in place; the container's
            // runtime wiring was already established during fromConfig and is
            // intentionally left untouched here.
            val.convert(simu.exchange()->sltpContainer());
        }
    }

    for (const auto& book : simu.exchange()->books()) {
        auto setupActiveOrders = [&](const auto& side) {
            for (const auto& level : side) {
                for (const auto& order : level) {
                    book->orderIdMap().insert({
                        order->id(), std::dynamic_pointer_cast<LimitOrder>(order)
                    });
                    const auto owningAgentId = book->orderToClientInfo().at(order->id()).agentId;
                    simu.exchange()->accounts()
                        .at(owningAgentId)
                        .activeOrders()
                        .at(book->id())
                        .insert(order);
                }
            }
        };
        setupActiveOrders(book->buyQueue());
        setupActiveOrders(book->sellQueue());
    }
}

//-------------------------------------------------------------------------

static void setupMessageQueue(const msgpack::object& o, Simulation& simu)
{
    simu.messageQueue() = o.as<taosim::message::MessageQueue>();
}

//-------------------------------------------------------------------------

static void setupBlocks(
    std::span<msgpack::object_handle> blockObjHandles,
    taosim::simulation::SimulationManager* simuMngr)
{
    using taosim::serialization::msgpackMapKeysToString;

    const size_t nBlocks = blockObjHandles.size();
    for (auto&& [blockIdx, oh] : views::enumerate(blockObjHandles)) {

        const msgpack::object obj = oh.get();

        if (obj.type != msgpack::type::MAP) {
            throw taosim::checkpoint::CheckpointError{};
        }

        auto& simu = *simuMngr->simulations().at(blockIdx);

        auto agentsObj = taosim::serialization::msgpackFindMapObj(obj, "agents");
        if (!agentsObj) {
            throw taosim::checkpoint::CheckpointError{fmt::format(
                "{}: {}", blockIdx, msgpackMapKeysToString(obj)
            )};
        }
        setupAgents(*agentsObj, simu, blockIdx);

        auto exchangeObj = taosim::serialization::msgpackFindMapObj(obj, "exchange");
        if (!exchangeObj) {
            throw taosim::checkpoint::CheckpointError{fmt::format(
                "{}: {}", blockIdx, msgpackMapKeysToString(obj)
            )};
        }
        setupExchange(*exchangeObj, simu, blockIdx);

        auto messageQueueObj = taosim::serialization::msgpackFindMapObj(obj, "messageQueue");
        if (!messageQueueObj) {
            throw taosim::checkpoint::CheckpointError{fmt::format(
                "{}: {}", blockIdx, msgpackMapKeysToString(obj)
            )};
        }
        setupMessageQueue(*messageQueueObj, simu);
        printProgress(blockIdx + 1, nBlocks, "Restoring blocks");
    }
}

//-------------------------------------------------------------------------

static void setupLogFiles(
    const msgpack::object& o, const taosim::simulation::SimulationManager& simuMngr)
{
    if (o.type != msgpack::type::MAP) {
        throw taosim::checkpoint::CheckpointError{};
    }

    for (const auto& [key, val] : o.via.map) {
        fs::resize_file(simuMngr.logDir() / key.as<std::string>(), val.as<size_t>());
    }
}

//-------------------------------------------------------------------------

void setupUsingCkptData(
    taosim::simulation::SimulationManager* simuMngr,
    const msgpack::object& commonObj,
    std::span<msgpack::object_handle> blockObjHandles)
{
    using taosim::serialization::msgpackMapKeysToString;

    setupBlocks(blockObjHandles, simuMngr);

    auto logFileSizesObj = taosim::serialization::msgpackFindMapObj(commonObj, "logFileSizes");
    if (!logFileSizesObj) {
        throw taosim::checkpoint::CheckpointError{msgpackMapKeysToString(commonObj)};
    }
    setupLogFiles(*logFileSizesObj, *simuMngr);

    for (auto&& simu : simuMngr->simulations()) {
        simu->state() = taosim::simulation::SimulationState::STARTED;
    }

    if (auto& ckptMngr = simuMngr->checkpointManager()) {
        ckptMngr->stepCounter() = {};
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::checkpoint

//-------------------------------------------------------------------------