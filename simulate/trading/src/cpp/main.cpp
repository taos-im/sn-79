/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/checkpoint/helpers.hpp>
#include <taosim/replay/ReplayDesc.hpp>
#include <taosim/replay/helpers.hpp>
#include <taosim/simulation/SimulationManager.hpp>
#include <taosim/simulation/multiasset/helpers.hpp>

#include <CLI/CLI.hpp>
#include <cstdio>
#include <fmt/format.h>
#ifdef OVERRIDE_NEW_DELETE
#include <mimalloc-new-delete.h>
#endif

//-------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    // Force line-buffered stdout/stderr so pm2/pipe environments don't batch log lines.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::setvbuf(stderr, nullptr, _IOLBF, 0);

    CLI::App app{"ExchangeSimulator v2.0"};

    auto initGroup = app.add_option_group("Init");

    fs::path configPath;
    auto optConfigPath = initGroup->add_option(
        "-f,--config-file", configPath, "Simulation config file")
        ->check(CLI::ExistingFile)
        ->transform([](auto&& p) { return fs::absolute(p); });

    fs::path baseDir;
    app.add_option("-d,--dir", baseDir, "Base directory in which to preserve the run artifacts")
        ->needs(optConfigPath)
        ->default_val(fs::current_path() / "logs")
        ->transform([](auto&& p) { return fs::absolute(p); });

    taosim::checkpoint::CheckpointToken ckptToken;
    initGroup->add_option(
        "-c,--load-checkpoint",
        ckptToken,
        fmt::format(
            "Checkpoint directory"
            " OR run directory containing checkpoint directories under '{}'"
            " OR special token (one of [{}], assumes run directories in 'logs/')",
            taosim::checkpoint::CheckpointManager::s_storeDirName,
            fmt::join(taosim::checkpoint::s_specialTokens, ", ")))
        ->transform(&taosim::checkpoint::postProcessToken);

    fs::path exchangeSvcConfigPath;

    taosim::replay::ReplayDesc replayDesc;

    auto optReplayPath = initGroup->add_option(
        "-r,--replay-dir", replayDesc.dir, "Log directory to use in a replay context")
        ->check(CLI::ExistingDirectory)
        ->transform(&taosim::replay::helpers::cleanReplayPath);

    app.add_option("--book-id", replayDesc.bookId, "Book to replay")
        ->needs(optReplayPath);

    app.add_option(
        "--replaced-agents",
        replayDesc.replacedAgents,
        "Comma-separated list of agent base names which to replace during replay")
        ->delimiter(',')
        ->needs(optReplayPath);

    app.add_flag(
        "--adjust-limit-prices",
        replayDesc.adjustLimitPrices,
        "Adjust limit prices of passive agents using historical mid price data")
        ->needs(optReplayPath);

    initGroup->require_option(1);

    CLI11_PARSE(app, argc, argv);

    fmt::println("{}", app.get_description());

    if (!configPath.empty()) {
        // Same -f flag, routed by the config's root element (multi-asset wrapper
        // vs single <Simulation>); both kinds run through the same interface.
        auto orchestrator = taosim::simulation::multiasset::makeSimulationOrchestrator(configPath, baseDir);
        orchestrator->run();
    }
    else if (!ckptToken.empty()) {
        // Same -c flag for both run modes; routed by the persisted run-dir config root.
        auto orchestrator = taosim::simulation::multiasset::makeOrchestratorFromCheckpoint(ckptToken);
        orchestrator->run();
    }
    else if (!replayDesc.dir.empty()) {
        auto orchestrator = taosim::simulation::SimulationManager::fromReplay(replayDesc);
        orchestrator->run();
    }

    fmt::println(" - all simulations finished, exiting");

    return 0;
}

//-------------------------------------------------------------------------
