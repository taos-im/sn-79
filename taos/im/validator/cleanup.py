# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
from __future__ import annotations

import time
import asyncio
import time
from concurrent.futures import TimeoutError as FuturesTimeout
import traceback
import subprocess
from typing import TYPE_CHECKING

import bittensor as bt

if TYPE_CHECKING:
    from taos.im.neurons.validator import Validator


def cleanup_ipc(self: Validator):
    """
    Shuts down the query service and releases all POSIX IPC resources.

    Behavior:
        - Attempts to send a shutdown message to the query service.
        - Waits for graceful termination, falling back to terminate/kill.
        - Closes memory maps and shared memory file descriptors.
        - Closes message queues.
        - Logs detailed warnings for any partial cleanup failures.

    Returns:
        None
    """
    try:
        if hasattr(self, 'engine'):
            self.engine.stop()

        bt.logging.info("Cleaning up query service...")
        if hasattr(self, 'request_queue'):
            try:
                self.request_queue.send(b'shutdown', timeout=1.0)
                bt.logging.info("Sent shutdown command to query service")
            except Exception as e:
                bt.logging.warning(f"Failed to send shutdown command: {e}")
        if hasattr(self, 'query_process') and self.query_process:
            try:
                self.query_process.wait(timeout=5.0)
                bt.logging.info(f"Query service exited with code {self.query_process.returncode}")
            except subprocess.TimeoutExpired:
                bt.logging.warning("Query service did not exit gracefully, terminating...")
                self.query_process.terminate()
                try:
                    self.query_process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    bt.logging.error("Query service did not terminate, killing...")
                    self.query_process.kill()

        if hasattr(self, 'request_mem'):
            try:
                self.request_mem.close()
                bt.logging.debug("Closed request memory map")
            except Exception as e:
                bt.logging.warning(f"Error closing request memory map: {e}")

        if hasattr(self, 'response_mem'):
            try:
                self.response_mem.close()
                bt.logging.debug("Closed response memory map")
            except Exception as e:
                bt.logging.warning(f"Error closing response memory map: {e}")

        if hasattr(self, 'request_shm'):
            try:
                self.request_shm.close_fd()
                bt.logging.debug("Closed request shared memory fd")
            except Exception as e:
                bt.logging.warning(f"Error closing request shared memory fd: {e}")

        if hasattr(self, 'response_shm'):
            try:
                self.response_shm.close_fd()
                bt.logging.debug("Closed response shared memory fd")
            except Exception as e:
                bt.logging.warning(f"Error closing response shared memory fd: {e}")

        if hasattr(self, 'request_queue'):
            try:
                self.request_queue.close()
                bt.logging.debug("Closed request queue")
            except Exception as e:
                bt.logging.warning(f"Error closing request queue: {e}")

        if hasattr(self, 'response_queue'):
            try:
                self.response_queue.close()
                bt.logging.debug("Closed response queue")
            except Exception as e:
                bt.logging.warning(f"Error closing response queue: {e}")

        bt.logging.info("Query service cleanup complete")

        bt.logging.info("Cleaning up reporting service...")

        if hasattr(self, 'reporting_request_queue'):
            try:
                self.reporting_request_queue.send(b'shutdown', timeout=1.0)
                bt.logging.info("Sent shutdown command to reporting service")
            except Exception as e:
                bt.logging.warning(f"Failed to send shutdown command to reporting: {e}")

        if hasattr(self, 'reporting_process') and self.reporting_process:
            try:
                self.reporting_process.wait(timeout=5.0)
                bt.logging.info(f"Reporting service exited with code {self.reporting_process.returncode}")
            except subprocess.TimeoutExpired:
                bt.logging.warning("Reporting service did not exit gracefully, terminating...")
                self.reporting_process.terminate()
                try:
                    self.reporting_process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    bt.logging.error("Reporting service did not terminate, killing...")
                    self.reporting_process.kill()

        if hasattr(self, 'reporting_request_mem'):
            try:
                self.reporting_request_mem.close()
                bt.logging.debug("Closed reporting request memory map")
            except Exception as e:
                bt.logging.warning(f"Error closing reporting request memory map: {e}")

        if hasattr(self, 'reporting_response_mem'):
            try:
                self.reporting_response_mem.close()
                bt.logging.debug("Closed reporting response memory map")
            except Exception as e:
                bt.logging.warning(f"Error closing reporting response memory map: {e}")

        if hasattr(self, 'reporting_request_shm'):
            try:
                self.reporting_request_shm.close_fd()
                bt.logging.debug("Closed reporting request shared memory fd")
            except Exception as e:
                bt.logging.warning(f"Error closing reporting request shared memory fd: {e}")

        if hasattr(self, 'reporting_response_shm'):
            try:
                self.reporting_response_shm.close_fd()
                bt.logging.debug("Closed reporting response shared memory fd")
            except Exception as e:
                bt.logging.warning(f"Error closing reporting response shared memory fd: {e}")

        if hasattr(self, 'reporting_request_queue'):
            try:
                self.reporting_request_queue.close()
                bt.logging.debug("Closed reporting request queue")
            except Exception as e:
                bt.logging.warning(f"Error closing reporting request queue: {e}")

        if hasattr(self, 'reporting_response_queue'):
            try:
                self.reporting_response_queue.close()
                bt.logging.debug("Closed reporting response queue")
            except Exception as e:
                bt.logging.warning(f"Error closing reporting response queue: {e}")

        bt.logging.info("Reporting service cleanup complete")

        bt.logging.info("Cleaning up seed service...")
        if hasattr(self, 'seed_process') and self.seed_process:
            try:
                self.seed_process.terminate()
                self.seed_process.wait(timeout=5.0)
                bt.logging.info(f"Seed service exited with code {self.seed_process.returncode}")
            except subprocess.TimeoutExpired:
                bt.logging.warning("Seed service did not exit gracefully, killing...")
                self.seed_process.kill()
        bt.logging.info("Seed service cleanup complete")

    except Exception as e:
        bt.logging.error(f"Error during validator cleanup: {e}")
        bt.logging.error(traceback.format_exc())


def cleanup_executors(self: Validator):
    """
    Shuts down thread and process executors used by the validator.

    Executors cleaned:
        - reward_executor (ProcessPoolExecutor)
        - save_state_executor (ThreadPoolExecutor)
        - maintenance_executor (ThreadPoolExecutor)
        - multiprocessing manager (if present)

    Behavior:
        - Each executor is shut down gracefully with wait=True
        - For ProcessPoolExecutor, attempts graceful shutdown first
        - Falls back to immediate termination if graceful fails
        - Logs success or failure for each executor

    Returns:
        None
    """
    if getattr(self, '_mg_worker', None) is not None:
        try:
            bt.logging.info("Stopping metagraph sync worker...")
            self._mg_worker.stop()
        except Exception as ex:
            bt.logging.warning(f"Error stopping metagraph sync worker: {ex}")
        self._mg_worker = None

    if getattr(self, '_scoring_shadow', None) is not None:
        try:
            bt.logging.info("Stopping scoring shadow...")
            self._scoring_shadow.stop()
        except Exception as ex:
            bt.logging.warning(f"Error stopping scoring shadow: {ex}")
        self._scoring_shadow = None

    if hasattr(self, 'reward_executor') and self.reward_executor is not None:
        try:
            bt.logging.info("Shutting down reward_executor...")
            self.reward_executor.shutdown(wait=True, cancel_futures=False)
            bt.logging.info("reward_executor shut down successfully")
        except Exception as ex:
            bt.logging.error(f"Error shutting down reward_executor: {ex}")
            try:
                bt.logging.warning("Attempting to terminate reward_executor processes...")
                for process in self.reward_executor._processes.values():
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=2.0)
                        if process.is_alive():
                            process.kill()
                bt.logging.info("reward_executor processes terminated")
            except Exception as term_ex:
                bt.logging.error(f"Error terminating reward_executor: {term_ex}")

    if hasattr(self, 'query_ipc_executor') and self.query_ipc_executor is not None:
        try:
            bt.logging.info("Shutting down query_ipc_executor...")
            self.query_ipc_executor.shutdown(wait=True, cancel_futures=False)
            bt.logging.info("query_ipc_executor shut down successfully")
        except Exception as ex:
            bt.logging.error(f"Error shutting down query_ipc_executor: {ex}")

    if hasattr(self, 'reporting_ipc_executor') and self.reporting_ipc_executor is not None:
        try:
            bt.logging.info("Shutting down reporting_ipc_executor...")
            self.reporting_ipc_executor.shutdown(wait=True, cancel_futures=False)
            bt.logging.info("reporting_ipc_executor shut down successfully")
        except Exception as ex:
            bt.logging.error(f"Error shutting down reporting_ipc_executor: {ex}")

    thread_executors = {
        'save_state_executor': getattr(self, 'save_state_executor', None),
        'maintenance_executor': getattr(self, 'maintenance_executor', None),
        '_mvtrx_push_executor': getattr(self, '_mvtrx_push_executor', None),
    }

    for name, executor in thread_executors.items():
        if executor is not None:
            try:
                bt.logging.info(f"Shutting down {name}...")
                executor.shutdown(wait=True, cancel_futures=False)
                bt.logging.info(f"{name} shut down successfully")
            except Exception as ex:
                bt.logging.error(f"Error shutting down {name}: {ex}")

    if hasattr(self, 'manager'):
        try:
            bt.logging.info("Shutting down multiprocessing manager...")
            self.manager.shutdown()
            bt.logging.info("Manager shut down successfully")
        except Exception as ex:
            bt.logging.error(f"Error shutting down manager: {ex}")

    bt.logging.info("Executor cleanup complete")


# The push tasks are named rather than imported, so this module keeps no reference back to the
# validator module. Kept in step with validator.PUSH_TASK_NAME by
# tests/test_a_stopping_validator_finishes_the_block_it_built.py.
PUSH_TASK_NAME = "mvtrx-push"
# One block of state is worth a few seconds of shutdown and no more: past that the data is stale and
# a push that has not completed is not going to.
PUSH_DRAIN_TIMEOUT_S = 10.0
# Cancellation is cooperative: a task that swallows CancelledError would hold shutdown open forever,
# so both waits are bounded and say so rather than hanging.
PENDING_CANCEL_TIMEOUT_S = 10.0
LOOP_STOP_TIMEOUT_S = 5.0


def cleanup_event_loop(self: Validator):
    """
    Gracefully shuts down the main event loop and any pending tasks.

    Behavior:
        - Cancels all pending tasks in the main loop
        - Waits for task cancellation to complete
        - Stops the event loop if still running
        - Closes the event loop

    Returns:
        None
    """
    try:
        if hasattr(self, 'main_loop') and self.main_loop and not self.main_loop.is_closed():
            bt.logging.info("Shutting down main event loop...")

            # FINISH THE BLOCK BEFORE CANCELLING THE LOOPS.
            #
            # Cancelling is right for the long-lived tasks: the query service, the reporting service
            # and the metagraph sync all exist to be stopped. It is wrong for a detached state push,
            # which is one block of data already built and already on its way. Cancel it and its
            # trades never reach the tape, while the service has already written the same fills down
            # its per-fill path -- so the fills read as floating free of a tape that was simply never
            # sent. The losses cluster in the minutes that contain a validator restart, which is
            # what separates this from a steady-state ingest fault.
            _pushes = [t for t in asyncio.all_tasks(self.main_loop)
                       if not t.done() and (t.get_name() or "").startswith(PUSH_TASK_NAME)]
            if _pushes:
                bt.logging.info(
                    f"Waiting up to {PUSH_DRAIN_TIMEOUT_S:.0f}s for {len(_pushes)} in-flight "
                    f"state push(es) so their blocks are not discarded..."
                )
                try:
                    # ACROSS A THREAD, because the loop belongs to another one. main_loop runs in its
                    # own daemon thread and this cleanup is called from a different thread, so
                    # run_until_complete raises "this event loop is already running" without waiting
                    # for anything -- the drain then logs a warning and the block is cancelled anyway,
                    # which is the outcome it exists to prevent. run_coroutine_threadsafe hands the
                    # wait to the loop that owns these tasks and blocks this thread for its result.
                    if self.main_loop.is_running():
                        _fut = asyncio.run_coroutine_threadsafe(
                            asyncio.wait(_pushes, timeout=PUSH_DRAIN_TIMEOUT_S), self.main_loop
                        )
                        # A margin over the inner timeout, so the bound that reports which pushes were
                        # left is the inner one rather than this.
                        _done, _left = _fut.result(timeout=PUSH_DRAIN_TIMEOUT_S + 5)
                    else:
                        _done, _left = self.main_loop.run_until_complete(
                            asyncio.wait(_pushes, timeout=PUSH_DRAIN_TIMEOUT_S)
                        )
                    if _left:
                        bt.logging.warning(
                            f"{len(_left)} state push(es) did not finish within "
                            f"{PUSH_DRAIN_TIMEOUT_S:.0f}s; their blocks are NOT ingested"
                        )
                    else:
                        bt.logging.info("All in-flight state pushes completed before shutdown")
                except Exception as ex:
                    bt.logging.warning(f"Could not drain in-flight state pushes: {ex!r}")

            pending = asyncio.all_tasks(self.main_loop)
            _threaded = self.main_loop.is_running()
            if pending:
                bt.logging.info(f"Cancelling {len(pending)} pending tasks...")
                # EVERY STEP CROSSES A THREAD, for the reason the drain above already gives.
                # Task.cancel() from a foreign thread is not safe, and the gather cannot be driven
                # with run_until_complete on a loop that is already running -- it raised, so the
                # cancellations were never awaited and the two statements below never ran. Shutdown
                # ended in an error with the loop still open, surviving only because its thread is a
                # daemon and the process was exiting regardless.
                if _threaded:
                    for task in pending:
                        self.main_loop.call_soon_threadsafe(task.cancel)
                    # BUILT ON THE LOOP'S OWN THREAD. gather() returns a Future, not a coroutine,
                    # and constructing it here would bind it to whatever loop this thread has --
                    # which is not the one holding these tasks. So the gather happens inside a
                    # coroutine that run_coroutine_threadsafe schedules on the right loop.
                    async def _await_cancelled(_tasks=tuple(pending)):
                        await asyncio.gather(*_tasks, return_exceptions=True)

                    try:
                        asyncio.run_coroutine_threadsafe(
                            _await_cancelled(), self.main_loop
                        ).result(timeout=PENDING_CANCEL_TIMEOUT_S)
                    except FuturesTimeout:
                        bt.logging.warning(
                            f"{len(pending)} task(s) did not finish cancelling within "
                            f"{PENDING_CANCEL_TIMEOUT_S:.0f}s; closing anyway"
                        )
                else:
                    for task in pending:
                        task.cancel()
                    self.main_loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )

            if self.main_loop.is_running():
                # stop() is not thread-safe either, and close() on a loop that has not finished
                # stopping raises. Ask the loop to stop, then wait for it to actually be stopped.
                self.main_loop.call_soon_threadsafe(self.main_loop.stop)
                _deadline = time.time() + LOOP_STOP_TIMEOUT_S
                while self.main_loop.is_running() and time.time() < _deadline:
                    time.sleep(0.02)
                if self.main_loop.is_running():
                    bt.logging.warning("Main event loop did not stop; leaving it open rather than closing it")
                    return

            self.main_loop.close()
            bt.logging.info("Main event loop shut down successfully")
    except Exception as ex:
        bt.logging.error(f"Error shutting down main event loop: {ex}")
        bt.logging.error(traceback.format_exc())


def cleanup(self: Validator):
    """
    Performs full resource cleanup for the validator during shutdown.
    """
    if self._cleanup_done:
        bt.logging.debug("Cleanup already completed, skipping")
        return

    bt.logging.info("Starting validator cleanup...")
    self._cleanup_done = True

    try:
        bt.logging.info("Waiting for active operations to complete...")
        wait_timeout = 30.0
        wait_start = time.time()

        while (self.shared_state_rewarding or
            self.shared_state_saving or
            self.shared_state_reporting or
            self.maintaining or
            self.compressing or
            self.querying):

            elapsed = time.time() - wait_start
            if elapsed > wait_timeout:
                bt.logging.warning(
                    f"Timeout waiting for operations after {elapsed:.2f}s"
                )
                break
            time.sleep(0.1)

        cleanup_executors(self)
        cleanup_ipc(self)
        cleanup_event_loop(self)

        bt.logging.success("Validator cleanup completed successfully")

    except Exception as ex:
        bt.logging.error(f"Error during cleanup: {ex}")
        bt.logging.error(traceback.format_exc())
