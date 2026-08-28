# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Route stdlib `logging` into `bt.logging`, so every log line actually reaches pm2.

WHY THIS EXISTS. The validator uses two logging systems. `bt.logging` reaches pm2; stdlib
`logging.getLogger(__name__)` does NOT, and there are 250 such call sites across 11 modules. A line
written to the stdlib logger is not "quiet", it is INVISIBLE, and the code around it reads as though it
reports something.

The failure mode is a silent one: grepping pm2 for a line that a module does emit finds nothing, and the
absence reads as "the code did not run" rather than "the line cannot appear". Converting individual call
sites fixes instances, not the class.

A mass rewrite of 250 sites is the wrong fix: stdlib logging is lazy (`logger.info("x=%s", v)`) and
mechanically turning those into f-strings at 250 places would introduce formatting bugs in code paths
that only run in production. Bridging is one change, it is reversible, and it cannot alter a message.

Install it as early as possible, before the modules that construct their loggers are used.
"""

import logging

import bittensor as bt

_INSTALLED = False


class _BtLoggingHandler(logging.Handler):
    """Forward a stdlib record to the matching bt.logging level.

    Formats via the handler (so `%s`-style lazy args are rendered exactly as the caller intended) and
    hands bt.logging a finished string, which is what it expects.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # DO NOT FEED bt.logging BACK INTO ITSELF. bt.logging writes THROUGH stdlib logging, so a
        # handler on the root logger receives bittensor's own records and forwards them to bt.logging
        # again, which re-emits them, each pass adding another prefix. Observed immediately after this
        # bridge was installed on 2026-08-09:
        #     INFO | [bittensor] [bittensor] [bittensor] [bittensor] [bittensor]
        # which also buried this module's own install announcement, so the check that was supposed to
        # confirm the bridge was live reported it absent.
        """Forward one stdlib logging record into ``bt.logging`` at the matching level."""
        name = record.name or ""
        if name == "bittensor" or name.startswith("bittensor."):
            return
        try:
            msg = self.format(record)
        except Exception:
            # A broken format string must not take down the caller: that would turn a logging bug into
            # an outage in whatever path happened to log.
            try:
                msg = f"{record.name}: {record.msg!r} (unformattable args)"
            except Exception:
                return
        try:
            if record.levelno >= logging.ERROR:
                bt.logging.error(msg)
            elif record.levelno >= logging.WARNING:
                bt.logging.warning(msg)
            elif record.levelno >= logging.INFO:
                bt.logging.info(msg)
            else:
                bt.logging.debug(msg)
        except Exception:
            pass


def install(level: int = logging.INFO) -> None:
    """Attach the bridge to the root logger. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    handler = _BtLoggingHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    handler.setLevel(level)

    root = logging.getLogger()
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)

    # Third-party libraries that are chatty at INFO stay at WARNING: the point is to surface OUR lines,
    # not to bury them under someone else's handshake logs.
    for noisy in ("websockets", "aiohttp", "asyncio", "urllib3", "substrateinterface", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _INSTALLED = True
    bt.logging.info(
        "log bridge installed: stdlib logging now reaches pm2 "
        "(250 call sites across 11 modules were previously invisible)"
    )
