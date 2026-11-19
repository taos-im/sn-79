# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A report cut off by the validator's own shutdown is a normal stop, not an alert.

cleanup() sets _cleanup_done and then shuts the executors down; a report scheduled just before it reaches
run_in_executor on a closed executor and raises "cannot schedule new futures after shutdown". The listener loop
already treats that as the end of the run; the report sender paged PagerDuty with it on every planned stop,
after the stop save had completed. A real failure while running still pages.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

V = pytest.importorskip("taos.im.neurons.validator")


def _validator(cleanup_done, executor):
    pages = []
    v = SimpleNamespace(
        _reporting=False, step=7, _cleanup_done=cleanup_done,
        reporting_process=SimpleNamespace(poll=lambda: None),
        reporting_ipc_executor=executor,
        reporting_response_queue=SimpleNamespace(receive=lambda timeout=None: None),
        pagerduty_alert=lambda message, details=None: pages.append(message),
    )
    return v, pages


def _closed_executor():
    ex = ThreadPoolExecutor(max_workers=1)
    ex.shutdown(wait=True)
    return ex


def test_a_report_cut_off_by_the_shutdown_does_not_page():
    v, pages = _validator(cleanup_done=True, executor=_closed_executor())
    asyncio.run(V.Validator._report(v))
    assert pages == [], "a planned stop pages nobody"
    assert v._reporting is False, "the report's flag is cleared as before"


def test_the_executor_closing_first_is_recognised_without_the_flag():
    v, pages = _validator(cleanup_done=False, executor=_closed_executor())
    asyncio.run(V.Validator._report(v))
    assert pages == [], "the teardown's own error is a stop, not a failure"


def test_a_real_failure_while_running_still_pages():
    class _Broken:
        def receive(self, timeout=None):
            raise OSError("reporting queue gone")
    v, pages = _validator(cleanup_done=False, executor=ThreadPoolExecutor(max_workers=1))
    v.reporting_response_queue = _Broken()
    asyncio.run(V.Validator._report(v))
    assert len(pages) == 1 and "Error sending to reporting service" in pages[0]
