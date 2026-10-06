/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/simulation/multiasset/helpers.hpp>

#include <taosim/checkpoint/helpers.hpp>
#include <taosim/simulation/multiasset/MultiAssetConfig.hpp>
#include <taosim/simulation/multiasset/MultiAssetSimulationOrchestrator.hpp>
#include <taosim/simulation/SimulationManager.hpp>
#include <taosim/xml/helpers.hpp>

#include <algorithm>
#include <optional>
#include <string>

//-------------------------------------------------------------------------

namespace taosim::simulation::multiasset
{

//-------------------------------------------------------------------------

bool isMultiAssetConfig(const std::filesystem::path& configPath)
{
    const pugi::xml_document doc = xml::loadDocument(configPath);
    return doc.child(kRootElement.data());
}

//-------------------------------------------------------------------------

std::unique_ptr<simulation::SimulationOrchestrator> makeSimulationOrchestrator(
    const std::filesystem::path& configPath, const std::filesystem::path& baseDir)
{
    if (isMultiAssetConfig(configPath)) {
        return MultiAssetSimulationOrchestrator::fromConfig(configPath, baseDir);
    }
    return simulation::SimulationManager::fromConfig(configPath, baseDir);
}

//-------------------------------------------------------------------------

std::unique_ptr<simulation::SimulationOrchestrator> makeOrchestratorFromCheckpoint(
    const checkpoint::CheckpointToken& ckptToken)
{
    const auto configPath = checkpoint::runDirFromToken(ckptToken) / "config.xml";
    if (isMultiAssetConfig(configPath)) {
        return MultiAssetSimulationOrchestrator::fromCheckpoint(ckptToken);
    }
    return simulation::SimulationManager::fromCheckpoint(ckptToken);
}

//-------------------------------------------------------------------------

ResourceAllocation computeResourceAllocation(const ResourceAllocationDesc& desc)
{
    // Thread budget left after reserving checkpoint I/O workers.
    const auto hardware = std::max(desc.hardwareConcurrency, 1u);
    const auto budget =
        hardware > desc.checkpointWorkers ? hardware - desc.checkpointWorkers : 1u;

    // Honor the request when given; never exceed the budget; never spawn more
    // workers than there are Simulations to drain (the rest would idle).
    const auto workers = [&] {
        const auto requested = desc.requestedWorkers != 0 ? desc.requestedWorkers : budget;
        return std::max(std::min({requested, budget, desc.simulationCount}), 1u);
    }();

    // Spread Simulations evenly; the first `remainder` workers take one extra,
    // so cohort sizes differ by at most one.
    const auto base = desc.simulationCount / workers;
    const auto remainder = desc.simulationCount % workers;

    return ResourceAllocation{
        .workerCount = workers,
        .simulationCount = desc.simulationCount,
        .threadPoolSize = workers + desc.checkpointWorkers,
        .cohortSizes = ranges::views::iota(0u, workers)
            | ranges::views::transform([base, remainder](auto w) {
                return base + (w < remainder ? 1u : 0u);
            })
            | ranges::to<std::vector>
    };
}

//-------------------------------------------------------------------------

uint64_t realizationSeed(uint64_t masterSeed, uint32_t backgroundIdx, uint32_t instanceIdx)
{
    // splitmix64 over the master seed mixed with the (background, instance) key.
    uint64_t x = masterSeed
        + 0x9E3779B97F4A7C15ull * ((static_cast<uint64_t>(backgroundIdx) << 32) | instanceIdx);
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

//-------------------------------------------------------------------------

pugi::xml_document adjustedBackgroundDoc(
    const pugi::xml_node& backgroundNode, std::optional<uint64_t> seed, std::string_view id)
{
    pugi::xml_document doc;
    doc.append_copy(backgroundNode);

    const auto simuNode = doc.child("Simulation");
    xml::setAttribute(simuNode, "id", std::string{id}.c_str());
    if (seed) {
        xml::setAttribute(simuNode, "rngSeed", *seed);
    }
    // Quote balances stay per book, as in a single-market run. Until 30 September 2026 this stamped
    // sharedQuoteBalances onto every realization, which gave a miner ONE quote wallet per realization
    // holding one book's template quote (25k for sixteen books, where mainnet gives 25k per book): the
    // probe miner in the first multi-asset test emptied two realizations' wallets in the first sim hour
    // and never traded there again, and the validator's balance totals summed the wallet once per book.
    // A portfolio wallet may come back later as a sized, declared option, not as a stamp.

    return doc;
}

//-------------------------------------------------------------------------

}  // namespace taosim::simulation::multiasset

//-------------------------------------------------------------------------
