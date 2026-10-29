/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

// A fatal signal leaves its name, its address and the faulting thread's backtrace on stderr, and the process still dies
// by that signal, so a crash with no core kept still says where it happened. A forked child installs the handler and
// faults inside a known function; the trace it writes must name that function once its frame offsets are read with
// addr2line.

#include <taosim/util/FatalSignal.hpp>

#include <gtest/gtest.h>

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <climits>
#include <cstdlib>
#include <cstdio>
#include <regex>
#include <string>

//-------------------------------------------------------------------------

extern "C" __attribute__((noinline)) void taosimFatalSignalTestFaultsHere(int sig)
{
    ::raise(sig);
    asm volatile("");
}

namespace
{

struct Death
{
    int signal{};
    std::string stderrText;
};

Death dieOf(int sig)
{
    int fds[2];
    if (::pipe(fds) != 0) return {};
    const pid_t pid = ::fork();
    if (pid == 0) {
        ::dup2(fds[1], STDERR_FILENO);
        ::close(fds[0]);
        taosim::util::installFatalSignalHandler();
        taosimFatalSignalTestFaultsHere(sig);
        ::_exit(0);
    }
    ::close(fds[1]);
    Death death;
    std::array<char, 4096> buf{};
    for (ssize_t n; (n = ::read(fds[0], buf.data(), buf.size())) > 0;) death.stderrText.append(buf.data(), n);
    ::close(fds[0]);
    int status = 0;
    ::waitpid(pid, &status, 0);
    death.signal = WIFSIGNALED(status) ? WTERMSIG(status) : 0;
    return death;
}

// The functions addr2line names for the frames that lie in this test binary.
std::string namedFrames(const std::string& trace)
{
    std::array<char, PATH_MAX> exe{};
    const auto len = ::readlink("/proc/self/exe", exe.data(), exe.size() - 1);
    if (len <= 0) return {};
    const std::string self{exe.data(), static_cast<std::size_t>(len)};
    std::string offsets;
    const std::regex frame{R"(^(\S+)\((?:[^)+]*)\+(0x[0-9a-f]+)\))"};
    std::size_t start = 0;
    while (start < trace.size()) {
        const auto end = trace.find('\n', start);
        const auto line = trace.substr(start, end == std::string::npos ? std::string::npos : end - start);
        // A frame names the binary as it was launched (pm2 runs the engine as ../build/src/cpp/taosim), so
        // compare the resolved path, not the text.
        std::array<char, PATH_MAX> resolved{};
        if (std::smatch m; std::regex_search(line, m, frame)
            && ::realpath(m[1].str().c_str(), resolved.data()) != nullptr && self == resolved.data()) {
            offsets += " " + m[2].str();
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    if (offsets.empty()) return {};
    std::string names;
    if (FILE* p = ::popen(("addr2line -f -C -e '" + self + "'" + offsets + " 2>/dev/null").c_str(), "r")) {
        std::array<char, 512> line{};
        while (std::fgets(line.data(), line.size(), p) != nullptr) names += line.data();
        ::pclose(p);
    }
    return names;
}

}  // namespace

//-------------------------------------------------------------------------

TEST(FatalSignal, ACrashLeavesItsSignalAndBacktraceAndStillDiesByTheSignal)
{
    for (const int sig : {SIGSEGV, SIGABRT}) {
        const auto death = dieOf(sig);
        EXPECT_EQ(death.signal, sig) << "the process did not die by the signal it raised";
        EXPECT_NE(death.stderrText.find(sig == SIGSEGV ? "FATAL SIGNAL SIGSEGV" : "FATAL SIGNAL SIGABRT"),
            std::string::npos) << death.stderrText;
        EXPECT_NE(death.stderrText.find("end of backtrace"), std::string::npos) << death.stderrText;
        if (::system("command -v addr2line >/dev/null 2>&1") == 0) {
            EXPECT_NE(namedFrames(death.stderrText).find("taosimFatalSignalTestFaultsHere"), std::string::npos)
                << "addr2line does not name the faulting function from the trace:\n" << death.stderrText;
        }
    }
}

//-------------------------------------------------------------------------
