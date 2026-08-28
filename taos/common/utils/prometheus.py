# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Prometheus metrics server wrapper: starts a singleton HTTP server on a configured
port and exposes a `prometheus` class for server lifecycle management.
"""
import argparse
import bittensor as bt
from typing import Union
from prometheus_client import start_http_server
from enum import Enum

class prometheus:
    """Namespace for prometheus tooling."""

    # Prometheus global logging levels.
    class level(Enum):
        """Prometheus verbosity levels: OFF, INFO, DEBUG."""
        OFF = "OFF"
        INFO = "INFO"
        DEBUG = "DEBUG"

        def __str__(self):
            return self.value

    # Prometheus Global state.
    port: int = None
    started: bool = False

    def __new__(
        cls,
        config: "bt.Config" = None,
        port: int = None,
        level: Union[str, "prometheus.level"] = None,
        start_server: bool = True
    ):
        """
        Configure or start the global Prometheus metrics server.

        Each server is uniquely identified by its port. If `start_server` is False,
        the port is recorded for use by an external server (e.g. FastAPI) without
        starting a dedicated HTTP listener.

        Args:
            config (bt.Config, optional): Bittensor config object. Defaults to
                the result of `prometheus.config()`.
            port (int, optional): Port for the Prometheus HTTP server. Overrides
                `config.prometheus.port` when provided.
            level (str or prometheus.level, optional): Logging level — 'OFF',
                'INFO', or 'DEBUG'. Overrides `config.prometheus.level` when provided.
            start_server (bool): If True, start the built-in HTTP server. If False,
                configure for use with an external server. Defaults to True.
        """
        if config is None:
            config = prometheus.config()

        if isinstance(level, prometheus.level):
            level = level.name  # Convert ENUM to str.

        config.prometheus.port = port if port is not None else config.prometheus.port
        config.prometheus.level = level if level is not None else config.prometheus.level

        if isinstance(config.prometheus.level, str):
            config.prometheus.level = (
                config.prometheus.level.upper()
            )  # Convert str to upper case.

        cls.check_config(config)
        
        if not start_server:
            prometheus.port = config.prometheus.port
            bt.logging.success(
                f"Prometheus: Configured for custom server on port {config.prometheus.port}"
            )
            return True

        return cls.serve(
            cls,
            port=config.prometheus.port,
            level=config.prometheus.level,
        )

    def serve(cls, port, level) -> bool:
        """
        Start the Prometheus HTTP server on the given port.

        Args:
            port (int): Port number to bind the server to.
            level (str): Prometheus logging level name; if 'OFF', the server is not started.

        Returns:
            bool: Always True.
        """
        if level == prometheus.level.OFF.name:  # If prometheus is off, return true.
            bt.logging.success("Prometheus:".ljust(20) + "<red>OFF</red>")
            return True
        else:
            try:
                start_http_server(port)
            except OSError:
                # The singleton process is likely already running.
                bt.logging.error(
                    f"Prometheus: Port {port} Already in use!"
                )
            prometheus.started = True
            prometheus.port = port
            bt.logging.success(
                f"Prometheus: ON using: [::]:{port}"
            )
            return True

    @classmethod
    def config(cls) -> "bt.Config":
        """
        Get config from the argument parser

        Returns: bt.Config object
        """
        parser = argparse.ArgumentParser()
        cls.add_args(parser=parser)
        return bt.Config(parser)

    @classmethod
    def help(cls):
        """Print help to stdout"""
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        print(cls.__new__.__doc__)
        parser.print_help()

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None):
        """Accept specific arguments from parser

        Args:
            parser: The argument parser to extend.
            prefix: Optional prefix for the argument names.
        """
        try:
            parser.add_argument(
                "--prometheus.port",
                type=int,
                required=False,
                default=9001,
                help="""Prometheus serving port.""",
            )
            parser.add_argument(
                "--prometheus.level",
                required=False,
                type=str,
                choices=[l.name for l in list(prometheus.level)],
                default="INFO",
                help="""Prometheus logging level. <OFF | INFO | DEBUG>""",
            )
        except argparse.ArgumentError:
            pass

    @classmethod
    def check_config(cls, config: "bt.Config"):
        """Check config for wallet name/hotkey/path/hotkeys/sort_by"""
        assert "prometheus" in config
        assert config.prometheus.level in [
            l.name for l in list(prometheus.level)
        ], "config.prometheus.level must be in: {}".format(
            [l.name for l in list(prometheus.level)]
        )
        assert (
            config.prometheus.port > 1024 and config.prometheus.port < 65535
        ), "config.prometheus.port must be in range [1024, 65535]"
        if "axon" in config and "port" in config.axon:
            assert (
                config.prometheus.port != config.axon.port
            ), "config.prometheus.port != config.axon.port"