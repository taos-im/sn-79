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
#include <taosim/checkpoint/CheckpointFormat.hpp>
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
#include <fstream>
#include <regex>
#include <unordered_set>
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
    // NEWEST BY RUN ID, NOT BY THE DIRECTORY'S WRITE TIME. A run directory is named by its start time
    // (YYYYMMDD_HHMMSS), and its own write time moves whenever an entry is created or removed in it, so a
    // day-rollover file or a cleanup could make an older run "latest". Same rule as run/start_simulator.sh's
    // _newest_real_rundir.
    const auto runDirsSortedByName =
        filesystem::collectMatchingPaths(baseDir, isSimRunDir)
        | ranges::actions::sort([](auto&& lhs, auto&& rhs) {
            return lhs.filename().string() < rhs.filename().string();
        });
    if (runDirsSortedByName.empty()) {
        throw CheckpointError{fmt::format(
            "No run directories containing '{}' found in '{}'",
            CheckpointManager::s_storeDirName, baseDir.c_str())};
    }
    fmt::println("Resuming the newest run by id: {}", runDirsSortedByName.back().c_str());
    return runDirsSortedByName.back();
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
        | ranges::views::filter([](auto&& e) {
            return fs::is_directory(e)
                && e.path().extension() == CheckpointManager::s_dirExtension;
        })
        | ranges::views::transform(&fs::directory_entry::path)
        | ranges::to<std::vector>
        | ranges::actions::sort([](auto&& lhs, auto&& rhs) {
            return fs::last_write_time(lhs) < fs::last_write_time(rhs);
        });
}

//-------------------------------------------------------------------------
// Stream exceptions are enabled so a failed write (e.g. disk full) throws instead
// of silently setting failbit — atomicWrite must not rename a truncated temp file
// over a good checkpoint. The explicit close() surfaces final-flush errors the
// destructor would swallow. No fsync: the goal is atomicity via the rename, not
// power-loss durability; per-file fsync costs multi-second stalls per checkpoint.

void atomicWrite(const fs::path& path, std::span<const char> data)
{
    const auto tmp = fs::path{path.native() + ".tmp"};

    try {
        std::ofstream ofs;
        ofs.exceptions(std::ios::failbit | std::ios::badbit);
        ofs.open(tmp, std::ios::binary | std::ios::trunc);
        ofs.write(data.data(), std::ssize(data));
        ofs.close();
    }
    catch (const std::exception& e) {
        std::error_code removeEc;
        fs::remove(tmp, removeEc);
        throw CheckpointError{fmt::format("writing {} failed: {}", tmp.string(), e.what())};
    }

    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        std::error_code removeEc;
        fs::remove(tmp, removeEc);
        throw CheckpointError{fmt::format(
            "rename({} -> {}) failed: {}", tmp.string(), path.string(), ec.message())};
    }
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
// In the shared-quote portfolio model (multi-asset, and any single-config run with
// sharedQuoteBalances) the books of one exchange share, by shared_ptr, three things:
// one quote Balance per account, and one order-id and one trade-id counter for the
// whole portfolio (so ids are unique across its books). A restore deserializes each of
// these per book independently, severing the sharing — every book gets its own quote
// and its own counters reset to the (identical) checkpointed value. On resume the
// un-shared counters then hand out colliding ids and corrupt the shared quote's
// reservation map. Re-link every book to the first book's shared objects; pre-checkpoint
// they were one object each, so the first is canonical. No-op when quote is not shared.

void reestablishSharedTopology(MultiBookExchangeAgent& exchange)
{
    if (!exchange.sharedQuoteBalances()) return;

    // One order-id and trade-id counter shared across the portfolio's books.
    auto&& books = exchange.books();
    if (!books.empty()) {
        // The exchange's own order counter is canonical, not the first book's: it is the one
        // mintPoolOrderId draws from, and the restore leaves it at its construction value.
        auto& orderIdCounter = exchange.orderIdCounter();
        for (auto&& book : books) {
            if (book->orderIdCounter()) {
                *orderIdCounter = std::max(*orderIdCounter, *book->orderIdCounter());
            }
        }
        const auto tradeIdCounter = books.front()->tradeIdCounter();
        for (auto&& book : books) {
            book->orderIdCounter() = orderIdCounter;
        }
        for (auto&& book : books | views::drop(1)) {
            book->tradeIdCounter() = tradeIdCounter;
        }
    }

    // One quote Balance shared across each account's books.
    size_t discarded = 0;
    for (auto&& [agentId, account] : exchange.accounts()) {
        auto&& holdings = account.holdings();
        if (holdings.empty()) continue;
        const auto quote = holdings.front().quote;
        for (auto&& balances : holdings | views::drop(1)) {
            // A reservation here that the canonical (front) quote lacks means the
            // pre-checkpoint account did NOT actually share its quote — re-linking
            // discards that state, so make the loss loud rather than silent.
            for (const auto& [orderId, amount] : balances.quote->getReservations()) {
                if (!quote->getReservation(orderId).has_value()) ++discarded;
            }
            balances.quote = quote;
        }
    }
    if (discarded > 0) {
        fmt::println(
            "reestablishSharedTopology: WARNING — discarded {} quote reservation(s) found"
            " only on non-canonical books; the checkpointed account topology was not"
            " actually shared (pre-fix checkpoint?)",
            discarded);
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

    // Books and accounts are now restored, but their per-book shared_ptrs (quote and
    // the id counters) were deserialized independently; re-link them where the exchange
    // shares quote across its books.
    reestablishSharedTopology(*simu.exchange());

    for (const auto& book : simu.exchange()->books()) {
        auto setupActiveOrders = [&](const auto& side) {
            for (const auto& level : side) {
                for (const auto& order : level) {
                    book->orderIdMap().insert({
                        order->id(), std::dynamic_pointer_cast<LimitOrder>(order)
                    });
                    // Ghosts (zero-volume fills kept in-book for reconciliation) keep
                    // their id-map identity but are no longer active — fill time already
                    // removed them from activeOrders and released their reservations.
                    if (order->totalVolume() == 0_dec) { continue; }
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
    std::span<const std::unique_ptr<Simulation>> simulations,
    std::span<msgpack::object_handle> blockObjHandles)
{
    using taosim::serialization::msgpackMapKeysToString;

    const size_t nBlocks = blockObjHandles.size();
    for (auto&& [blockIdx, oh] : views::enumerate(blockObjHandles)) {

        const msgpack::object obj = oh.get();

        if (obj.type != msgpack::type::MAP) {
            throw taosim::checkpoint::CheckpointError{};
        }

        auto& simu = *simulations[blockIdx];

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

namespace
{

// The end of the window a log file covers, from its name: "<stem>.<DDHHMMSS>-<DDHHMMSS>.log", the
// simulation's own time. Nothing for a name without one.
std::optional<std::chrono::seconds> logWindowEnd(const std::string& name)
{
    static const std::regex pattern{R"(^.+\.(\d{8})-(\d{8})\.log$)"};
    std::smatch m;
    if (!std::regex_match(name, m, pattern)) return std::nullopt;
    const std::string end = m[2].str();
    const auto field = [&](size_t at) { return std::stoll(end.substr(at, 2)); };
    return std::chrono::seconds{field(0) * 86400 + field(2) * 3600 + field(4) * 60 + field(6)};
}

std::chrono::seconds simulationOffset(const Simulation& simu, uint64_t timestamp)
{
    return std::chrono::duration_cast<std::chrono::seconds>(
        taosim::simulation::timescaleToConverter(simu.config().time().scale)(timestamp));
}

}  // namespace

void restoreLogFileSizes(const msgpack::object& sizes, const fs::path& logDir, std::chrono::seconds ckptOffset)
{
    if (sizes.type != msgpack::type::MAP) {
        throw taosim::checkpoint::CheckpointError{};
    }

    for (const auto& [key, val] : sizes.via.map) {
        const auto name = key.as<std::string>();
        const auto path = logDir / name;
        if (fs::exists(path)) {
            fs::resize_file(path, val.as<size_t>());
            continue;
        }
        const auto windowEnd = logWindowEnd(name);
        if (windowEnd && *windowEnd <= ckptOffset) {
            fmt::println("  {} is not on disk; its window had closed by the checkpoint, so it is "
                "complete in the run's archive and is left there", path.c_str());
            continue;
        }
        throw taosim::checkpoint::CheckpointError{fmt::format(
            "{} is missing, and the checkpoint needs it: {}. Put the file back (a closed window is "
            "packed into a zip at the run's root) and resume again.",
            path.c_str(),
            windowEnd ? "its window was still open at the checkpoint" : "its name carries no window")};
    }
}

//-------------------------------------------------------------------------

size_t setAsideLogsWrittenAfter(const fs::path& runDir, const msgpack::object& commonObj, uint64_t ckptTimestamp)
{
    auto sizesObj = taosim::serialization::msgpackFindMapObj(commonObj, "logFileSizes");
    if (!sizesObj) return 0;
    const msgpack::object& sizes = sizesObj->get();

    std::unordered_set<std::string> recorded;
    auto collect = [&](const msgpack::object& map) {
        if (map.type != msgpack::type::MAP) return;
        for (const auto& [key, _] : map.via.map) recorded.insert(key.as<std::string>());
    };
    std::vector<fs::path> dirs;
    if (sizes.type == msgpack::type::ARRAY) {
        for (const auto& map : std::span{sizes.via.array.ptr, sizes.via.array.size}) collect(map);
        // A realization's own directory is named <class>-<instance>; the run's root holds shared
        // inputs (the seed files) that no realization records, and the checkpoint store.
        static const std::regex realizationDir{R"(^[A-Za-z_][A-Za-z0-9_]*-\d+$)"};
        for (const auto& entry : fs::directory_iterator{runDir}) {
            if (entry.is_directory() && std::regex_match(entry.path().filename().string(), realizationDir)) {
                dirs.push_back(entry.path());
            }
        }
    } else {
        collect(sizes);
        dirs.push_back(runDir);
    }

    size_t moved = 0;
    for (const auto& dir : dirs) {
        for (const auto& entry : fs::directory_iterator{dir}) {
            const auto name = entry.path().filename().string();
            if (!entry.is_regular_file() || recorded.contains(name)
                || !std::regex_match(name, CheckpointManager::s_relevantLogFilePattern)) {
                continue;
            }
            fs::rename(entry.path(), dir / fmt::format("{}.superseded-{}", name, ckptTimestamp));
            ++moved;
        }
    }
    if (moved) {
        fmt::println("  set aside {} log file(s) written after the checkpoint, as *.superseded-{}",
            moved, ckptTimestamp);
    }
    return moved;
}

//-------------------------------------------------------------------------

void setupUsingCkptData(
    taosim::simulation::SimulationManager* simuMngr,
    const msgpack::object& commonObj,
    std::span<msgpack::object_handle> blockObjHandles)
{
    using taosim::serialization::msgpackMapKeysToString;

    setupBlocks(simuMngr->simulations(), blockObjHandles);

    auto logFileSizesObj = taosim::serialization::msgpackFindMapObj(commonObj, "logFileSizes");
    if (!logFileSizesObj) {
        throw taosim::checkpoint::CheckpointError{msgpackMapKeysToString(commonObj)};
    }
    const auto ckptTimestamp = taosim::serialization::msgpackFindMap<uint64_t>(commonObj, "timestamp");
    if (!ckptTimestamp) {
        throw taosim::checkpoint::CheckpointError{msgpackMapKeysToString(commonObj)};
    }
    restoreLogFileSizes(
        *logFileSizesObj, simuMngr->logDir(),
        simulationOffset(*simuMngr->simulations().front(), *ckptTimestamp));

    for (auto&& simu : simuMngr->simulations()) {
        simu->state() = taosim::simulation::SimulationState::STARTED;
    }

    if (auto& ckptMngr = simuMngr->checkpointManager()) {
        ckptMngr->stepCounter() = {};
    }
}

//-------------------------------------------------------------------------
// Multi-asset counterpart: the "common" carries a per-realization array of log file
// sizes (each realization owns a subdirectory of the run root), and the realizations
// are matched to block files by flat index. Block restore re-establishes the
// shared-quote topology (see setupExchange / reestablishSharedTopology).

void setupUsingCkptDataMultiAsset(
    std::span<const std::unique_ptr<Simulation>> simulations,
    CheckpointManager* checkpointManager,
    const msgpack::object& commonObj,
    std::span<msgpack::object_handle> blockObjHandles)
{
    using taosim::serialization::msgpackMapKeysToString;

    setupBlocks(simulations, blockObjHandles);

    auto logFileSizesObj = taosim::serialization::msgpackFindMapObj(commonObj, "logFileSizes");
    if (!logFileSizesObj || logFileSizesObj->get().type != msgpack::type::ARRAY) {
        throw taosim::checkpoint::CheckpointError{msgpackMapKeysToString(commonObj)};
    }
    const auto ckptTimestamp = taosim::serialization::msgpackFindMap<uint64_t>(commonObj, "timestamp");
    if (!ckptTimestamp) {
        throw taosim::checkpoint::CheckpointError{msgpackMapKeysToString(commonObj)};
    }
    const auto& perRealization = logFileSizesObj->get().via.array;
    for (size_t idx = 0; idx < simulations.size() && idx < perRealization.size; ++idx) {
        restoreLogFileSizes(
            perRealization.ptr[idx], simulations[idx]->logDir(),
            simulationOffset(*simulations[idx], *ckptTimestamp));
    }

    for (auto&& simu : simulations) {
        simu->state() = taosim::simulation::SimulationState::STARTED;
    }

    if (checkpointManager) {
        checkpointManager->stepCounter() = {};
    }
}

//-------------------------------------------------------------------------
// Read a checkpoint directory's common file and its numbered block files into msgpack
// object handles (which own their backing buffers). Shared by both restore paths.

LoadedCheckpoint loadCheckpointObjects(const fs::path& ckptDir)
{
    auto readObject = [](const fs::path& file) {
        const size_t byteSize = fs::file_size(file);
        std::vector<char> buffer(byteSize);
        std::ifstream ifs{file, std::ios::binary};
        ifs.read(buffer.data(), byteSize);
        return msgpack::unpack(buffer.data(), byteSize);
    };

    LoadedCheckpoint loaded{
        .common = readObject(
            ckptDir / fmt::format("common{}", CheckpointManager::s_fileExtension))
    };
    requireReadableFormat(loaded.common.get());

    static const std::regex blockCkptPattern{
        fmt::format("^\\d+\\{}$", CheckpointManager::s_fileExtension)};
    const auto blockCkptFilesSorted = filesystem::collectMatchingPaths(
        ckptDir,
        [&](auto&& p) { return std::regex_match(p.filename().string(), blockCkptPattern); })
        | ranges::actions::sort([](auto&& lhs, auto&& rhs) {
            return std::stoul(lhs.stem().string()) < std::stoul(rhs.stem().string());
        });
    for (const auto& ckptFile : blockCkptFilesSorted) {
        loaded.blocks.push_back(readObject(ckptFile));
    }

    return loaded;
}

//-------------------------------------------------------------------------

}  // namespace taosim::checkpoint

//-------------------------------------------------------------------------