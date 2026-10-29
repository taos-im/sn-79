# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A client holding a connection to the validator API does not hold up the stop save.

In simulation mode the API server runs on the main thread, so it takes the stop signal first: it shuts down, and only
then hands the signal on to the validator's own handler, which saves the state. Its graceful shutdown waits for every
request in progress to finish. Without a bound, one request that never finishes would keep the validator from saving
until pm2's kill timeout ran out, and the restart would load the last periodic save.

The server here is the real one, built from the validator's own options and serving on the main thread, so the
signal takes the same path it takes in the validator.
"""
import asyncio
import os
import signal
import socket
import threading
import time
from types import SimpleNamespace

import uvicorn

from taos.im.validator.http_surface import api_server_options, make_app

# Past this the test stops the server itself, so a server with no bound fails the test instead of hanging it.
_GIVE_UP_S = 15.0


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_held_connection_does_not_delay_the_stop_save_past_the_bound():
    port = _free_port()
    options = api_server_options(SimpleNamespace(port=port))
    app = make_app()
    in_progress = threading.Event()

    @app.get("/held")
    async def _held():
        in_progress.set()
        await asyncio.sleep(3600)

    server = uvicorn.Server(uvicorn.Config(app, log_config=None, **options))
    signalled, saving = [], []

    def _validator_handler(signum, frame):
        saving.append(time.monotonic())

    def _client():
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        conn = socket.create_connection(("127.0.0.1", port))
        conn.sendall(b"GET /held HTTP/1.1\r\nHost: validator\r\n\r\n")
        in_progress.wait(5)
        signalled.append(time.monotonic())
        os.kill(os.getpid(), signal.SIGINT)
        while not saving and time.monotonic() - signalled[0] < _GIVE_UP_S:
            time.sleep(0.05)
        if not saving:
            server.force_exit = True
        time.sleep(0.5)
        conn.close()

    previous = signal.signal(signal.SIGINT, _validator_handler)
    try:
        client = threading.Thread(target=_client, daemon=True)
        client.start()
        server.run()
        client.join(timeout=5)
    finally:
        signal.signal(signal.SIGINT, previous)

    assert in_progress.is_set(), "the held request never reached the server"
    assert saving, "the server never handed the stop signal on to the validator"
    waited = saving[0] - signalled[0]
    bound = options.get("timeout_graceful_shutdown")
    assert bound is not None, f"the API server has no shutdown bound; the held connection held the stop {waited:.1f}s"
    assert waited <= bound + 2.0, f"the stop save began {waited:.1f}s after the signal, past the {bound}s bound"
