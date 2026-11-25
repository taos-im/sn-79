// `-c latest` resumes the newest RUN, named by its start time, not the run directory touched last.
//
// A run directory's own write time moves whenever any entry is created or removed directly in it (a
// day-rollover file, a zip, a cleanup), so ranking run directories by it can resume an older run in place
// of the newest one, and the validator then saves the mixed state.

#include <taosim/checkpoint/CheckpointManager.hpp>
#include <taosim/checkpoint/helpers.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace fs = std::filesystem;

namespace
{

fs::path makeRun(const fs::path& base, const std::string& name)
{
    const auto ckpt = base / name / taosim::checkpoint::CheckpointManager::s_storeDirName / "100.ckptd";
    fs::create_directories(ckpt);
    std::ofstream{ckpt / fmt::format("common{}", taosim::checkpoint::CheckpointManager::s_fileExtension)}
        << "x";
    return base / name;
}

struct TempDir
{
    fs::path path{fs::temp_directory_path() / fmt::format("taosim-rundir-{}", ::getpid())};
    TempDir() { fs::remove_all(path); fs::create_directories(path); }
    ~TempDir() { fs::remove_all(path); }
};

}  // namespace

TEST(RunDirSelection, TheNewestRunByNameWinsOverAnOlderRunTouchedLater)
{
    TempDir tmp;
    const auto older = makeRun(tmp.path, "20261002_120000");
    const auto newer = makeRun(tmp.path, "20261006_120000");
    const auto now = fs::file_time_type::clock::now();
    fs::last_write_time(newer, now - std::chrono::hours{48});
    fs::last_write_time(older, now);

    EXPECT_EQ(taosim::checkpoint::runDirLatest(tmp.path), newer);
}

TEST(RunDirSelection, AnExchangeStateDirectoryIsNeverChosen)
{
    TempDir tmp;
    const auto run = makeRun(tmp.path, "20261006_120000");
    fs::create_directories(tmp.path / "20261007_000000" / "ckpt");
    std::ofstream{tmp.path / "20261007_000000" / "ckpt" / "state.ckpt"} << "x";

    EXPECT_EQ(taosim::checkpoint::runDirLatest(tmp.path), run);
}
