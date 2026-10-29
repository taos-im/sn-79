/*
 * SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

//-------------------------------------------------------------------------

namespace taosim::util
{

// On SIGSEGV, SIGBUS, SIGFPE, SIGILL or SIGABRT, writes the signal, the faulting address and the faulting thread's
// backtrace to stderr (which pm2 keeps), then re-raises with the default action, so the exit status and any core are
// the signal's own. Frames are written as the binary and an offset in it: `addr2line -f -C -e <binary> <offset>` names
// the function and line. Call once, early in main, before other threads start.
void installFatalSignalHandler() noexcept;

}  // namespace taosim::util

//-------------------------------------------------------------------------
