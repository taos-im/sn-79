/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/util/FatalSignal.hpp>

#include <execinfo.h>
#include <signal.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstring>

//-------------------------------------------------------------------------

namespace taosim::util
{

namespace
{

constexpr std::array kSignals{SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT};

// The handler runs here, so a stack overflow can still be reported.
alignas(16) char g_altStack[64 * 1024];

// Only async-signal-safe calls below: write(2), backtrace_symbols_fd(3) and raise(3), no allocation, no stdio.
void put(const char* s) noexcept
{
    if (::write(STDERR_FILENO, s, std::strlen(s)) < 0) {}
}

void putHex(std::uintptr_t v) noexcept
{
    char buf[19] = "0x";
    for (int i = 0; i < 16; ++i) {
        buf[2 + i] = "0123456789abcdef"[(v >> (60 - 4 * i)) & 0xf];
    }
    buf[18] = '\0';
    put(buf);
}

const char* name(int sig) noexcept
{
    switch (sig) {
        case SIGSEGV: return "SIGSEGV";
        case SIGBUS: return "SIGBUS";
        case SIGFPE: return "SIGFPE";
        case SIGILL: return "SIGILL";
        case SIGABRT: return "SIGABRT";
        default: return "signal";
    }
}

void onFatalSignal(int sig, siginfo_t* info, void*) noexcept
{
    put("\nFATAL SIGNAL ");
    put(name(sig));
    put(" at address ");
    putHex(reinterpret_cast<std::uintptr_t>(info != nullptr ? info->si_addr : nullptr));
    put("\nbacktrace of the faulting thread (addr2line -f -C -e <binary> <offset> names each frame):\n");
    std::array<void*, 64> frames{};
    const int n = ::backtrace(frames.data(), static_cast<int>(frames.size()));
    ::backtrace_symbols_fd(frames.data(), n, STDERR_FILENO);
    put("end of backtrace\n");
    // SA_RESETHAND has restored the default action, so the exit status and any core are the signal's own.
    ::raise(sig);
}

}  // namespace

//-------------------------------------------------------------------------

void installFatalSignalHandler() noexcept
{
    // backtrace() loads libgcc on its first call, which allocates: that call is made here, not in the handler.
    std::array<void*, 1> warm{};
    (void)::backtrace(warm.data(), 1);

    stack_t stack{};
    stack.ss_sp = g_altStack;
    stack.ss_size = sizeof g_altStack;
    ::sigaltstack(&stack, nullptr);

    struct sigaction action{};
    action.sa_sigaction = onFatalSignal;
    action.sa_flags = SA_SIGINFO | SA_RESETHAND | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    for (const int sig : kSignals) {
        ::sigaction(sig, &action, nullptr);
    }
}

}  // namespace taosim::util

//-------------------------------------------------------------------------
