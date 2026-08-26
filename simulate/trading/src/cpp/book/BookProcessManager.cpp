/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/book/BookProcessManager.hpp>

#include <taosim/exchange/ExchangeConfig.hpp>
#include "Simulation.hpp"

#include <rapidcsv.h>

//-------------------------------------------------------------------------

namespace taosim::book
{

//-------------------------------------------------------------------------

BookProcessManager::BookProcessManager(
    BookProcessManager::ProcessContainer container,
    BookProcessManager::LoggerContainer loggers,
    std::unique_ptr<taosim::process::ProcessFactory> processFactory,
    decltype(taosim::simulation::SimulationSignals::time)& timeSignal)
    : m_container{std::move(container)},
      m_loggers{std::move(loggers)},
      m_processFactory{std::move(processFactory)}
{
    m_feed = timeSignal.connect([this](Timespan timespan) {
        updateProcesses(timespan);
    });
    for (const auto& [name, bookProcesses] : m_container) {
        const auto& representativeProcess = bookProcesses.front();
        m_updateCounters[name] = UpdateCounter{representativeProcess->updatePeriod()};
    }
}

//-------------------------------------------------------------------------

void BookProcessManager::updateProcesses(Timespan timespan)
{
    for (const auto& [name, bookId2Process] : m_container) {
        std::map<BookId, std::vector<double>> bookId2ProcessValues;
        std::vector<Timestamp> timestamps;
        auto& updateCounter = m_updateCounters.at(name);
        const Timestamp stepsUntilUpdate = updateCounter.stepsUntilUpdate();
        if (const auto len = timespan.end - timespan.begin; len < stepsUntilUpdate) {
            updateCounter.setState(updateCounter.state() + len + 1);
            continue;
        }
        const Timestamp begin = timespan.begin + stepsUntilUpdate;
        const Timestamp stride = updateCounter.period();
        for (const auto& [bookId, process] : views::enumerate(bookId2Process)) {
            std::vector<double> processValues;
            for (Timestamp t = begin; t <= timespan.end; t += stride) {
                process->update(t);
                // loggedValue(): the latent state at t. value() would evaluate at
                // currentTimestamp() (== timespan.end for every t in this loop) and,
                // for read-time-interpolating processes, record presentation rather
                // than state — the CSV is the replay source and must stay latent.
                processValues.push_back(process->loggedValue());
            }
            bookId2ProcessValues.insert({bookId, std::move(processValues)});
        }
        for (Timestamp t = begin; t <= timespan.end; t += stride) {
            timestamps.push_back(t);
        }
        m_loggers.at(name)->log(bookId2ProcessValues, timestamps);
        updateCounter.setState((timespan.end - begin) % updateCounter.period());
    }
}

//-------------------------------------------------------------------------

std::unique_ptr<BookProcessManager> BookProcessManager::fromXML(
    pugi::xml_node node,
    Simulation* simulation,
    taosim::exchange::ExchangeConfig* exchangeConfig,
    const taosim::simulation::SharedResources* sharedResources)
{
    static constexpr auto ctx = std::source_location::current().function_name();

    if (std::string_view name = node.name(); name != "Books") {
        throw std::invalid_argument{fmt::format(
            "{}: Instantiation node should be 'Books', was '{}'", ctx, name)};
    }

    const uint32_t bookCount = node.attribute("instanceCount").as_uint(1);
    const std::pair bookIdCanonRange{
        simulation->blockIdx() * bookCount,
        simulation->blockIdx() * bookCount + bookCount - 1
    };

    auto processFactory =
        std::make_unique<taosim::process::ProcessFactory>(simulation, exchangeConfig, sharedResources);

    ProcessContainer container;
    LoggerContainer loggers;

    for (pugi::xml_node processNode : node.child("Processes")) {
        pugi::xml_attribute attr;
        if (attr = processNode.attribute("name"); attr.empty()) {
            throw std::invalid_argument{fmt::format(
                "{}: Node '{}' missing required attribute 'name'", ctx, processNode.name())};
        }
        const std::string name = attr.as_string();
        const auto processLogFileName =
            fmt::format("{}.{}-{}.csv", name, bookIdCanonRange.first, bookIdCanonRange.second);
        ProcessContainer::mapped_type bookId2Process(bookCount);
        auto replayCsv = [&] -> std::optional<rapidcsv::Document> {
            if (!simulation->replayMode()) return {};
            const auto path = simulation->replayDesc().dir / processLogFileName;
            return std::make_optional<rapidcsv::Document>(path);
        }();
        for (BookId bookId = 0; bookId < bookCount; ++bookId) {
            auto process = processFactory->createFromXML(
                processNode, simulation->blockIdx() * bookCount + bookId);
            if (replayCsv) {
                const auto values = (*replayCsv).GetColumn<double>(
                    std::to_string(simulation->bookIdCanon(bookId)));
                process->values() = values | views::drop(1) | ranges::to<std::vector>;
            }
            bookId2Process[bookId] = std::move(process);
        }
        container[name] = std::move(bookId2Process);
        loggers[name] = std::make_unique<BookProcessLogger>(
            simulation->logDir() / processLogFileName,
            container.at(name)
            | views::transform([](const auto& p) { return p->loggedValue(); })
            | ranges::to<std::vector>,
            simulation);
    }

    return std::make_unique<BookProcessManager>(
        std::move(container),
        std::move(loggers),
        std::move(processFactory),
        simulation->signals().time);
}

//-------------------------------------------------------------------------

}  // namespace taosim::book

//-------------------------------------------------------------------------