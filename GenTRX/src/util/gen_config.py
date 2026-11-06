#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Generate proxy test config.json from command-line args.

All paths are resolved relative to GENTRX_ROOT (env var, set by the run script).
TAOS_PROXY and TAOS_BUILD also come from the environment (sourced from .env).

Usage:
    GENTRX_ROOT=/path/to/taos-im/sn-79 TAOS_PROXY=agents/proxy TAOS_BUILD=simulate/trading/build \\
        python gen_config.py --sim-xml /path/to/sim.xml --output config.json
"""

import argparse
import json
import os
from pathlib import Path


def main():
    GenTRX_ROOT = os.environ.get("GENTRX_ROOT")
    if not GenTRX_ROOT:
        raise EnvironmentError("GENTRX_ROOT env var is not set")
    G = Path(GenTRX_ROOT).resolve()

    parser = argparse.ArgumentParser(description="Generate proxy test config")
    parser.add_argument("--sim-xml", required=True, help="Simulation XML path")
    parser.add_argument("--grad-port", type=int, default=8100)
    parser.add_argument("--output", default="config.json")
    parser.add_argument(
        "--n-agents", type=int, default=2, help="Number of training agents"
    )
    parser.add_argument(
        "--agent-mode",
        default="gentrx",
        choices=["gentrx", "mixed"],
        help="gentrx: all HybridTrainingAgent. mixed: 1 RandomMakerAgent + 1 RandomTakerAgent (training enabled)",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/GenTRX/best.pt",
        help="Relative to GENTRX_ROOT",
    )
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--train-seq-len", type=int, default=256)
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Gradient server poll/aggregation interval (wall clock s)",
    )
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument(
        "--sim-delay",
        type=int,
        default=0,
        help="Seconds to wait before starting simulator (sleep for proxy + agents + gradient startup)",
    )
    args = parser.parse_args()

    sim_xml = str(Path(args.sim_xml).resolve())
    ckpt_dir = G / "checkpoints" / "GenTRX"
    val_data = G / "data" / "proxy_test" / "server"
    grad_url = f"http://localhost:{args.grad_port}/gentrx"

    # Common training params (gtx_-prefixed)
    _train_params = {
        "gtx_training_enabled": "true",
        "gtx_collect_data": "false",
        "gtx_train_steps": args.train_steps,
        "gtx_train_batch_size": args.train_batch_size,
        "gtx_train_seq_len": args.train_seq_len,
        "gtx_top_k_frac": 0.05,
        "gtx_train_lr": 1e-4,
        "gtx_label_smooth_sigma": 1.0,
    }

    # Trading params for mixed mode
    _trading_params = {
        "min_quantity": 0.1,
        "max_quantity": 1.0,
        "min_leverage": 0.0,
        "max_leverage": 1.0,
        "expiry_period": 200_000_000_000,
        "max_fee_rate": 0.005,
    }

    agents_section: dict = {
        "path": str(G / "agents"),
        "start_port": 8888,
    }

    if args.agent_mode == "mixed":
        # 1 RandomMakerAgent + 1 RandomTakerAgent with training enabled
        agents_section["RandomMakerAgent"] = [
            {
                "_comment": "Maker: limit orders + GenTRX training",
                "params": {
                    "data_dir": str(G / "data" / "proxy_test" / "agent_0" / "legacy"),
                    "gtx_output_dir": str(G / "data" / "proxy_test" / "agent_0"),
                    "gtx_checkpoint": str(G / args.checkpoint),
                    **_train_params,
                    **_trading_params,
                },
                "count": 1,
            }
        ]
        agents_section["RandomTakerAgent"] = [
            {
                "_comment": "Taker: market orders + GenTRX training",
                "params": {
                    "data_dir": str(G / "data" / "proxy_test" / "agent_1" / "legacy"),
                    "gtx_output_dir": str(G / "data" / "proxy_test" / "agent_1"),
                    "gtx_checkpoint": str(G / args.checkpoint),
                    **_train_params,
                    **{k: v for k, v in _trading_params.items() if k != "expiry_period"},
                },
                "count": 1,
            }
        ]
    else:
        # HybridTrainingAgent: imbalance-signal maker/taker with training on by default
        agents = []
        for i in range(args.n_agents):
            agents.append(
                {
                    "_comment": f"Agent {i}: train from S3 (no local collection)",
                    "params": {
                        "data_dir": str(
                            G / "data" / "proxy_test" / f"agent_{i}" / "legacy"
                        ),
                        "gtx_output_dir": str(G / "data" / "proxy_test" / f"agent_{i}"),
                        "gtx_checkpoint": str(G / args.checkpoint),
                        **_train_params,
                    },
                    "count": 1,
                }
            )
        agents_section["HybridTrainingAgent"] = agents

    config = {
        "proxy": {
            "port": 8000,
            "simulation_xml": sim_xml,
            "timeout": 5,
            "gradient_server_url": grad_url,
        },
        "agents": agents_section,
        "taosim": {
            "bin": "taosim",
            "delay": args.sim_delay,
        },
        "training": {
            "n_agents": args.n_agents,
            "gradient_server": {
                "port": args.grad_port,
                "checkpoint": str(G / args.checkpoint),
                "val_data": str(val_data),
                "output": str(ckpt_dir / "latest.pt"),
                "interval": args.interval,
                "min_score": args.min_score,
                "mode": "simulation",
                "log": str(G / "data" / "proxy_test" / "gradient_server.log"),
                "miner_buckets": os.environ.get("GENTRX_MINER_BUCKETS", ""),
            },
            "minio": {
                "port": 9000,
                "console": 9091,
            },
        },
        "gradient_server": {
            "interval": args.interval,
            "log": str(G / "data" / "proxy_test" / "gradient_server.log"),
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(config, f, indent=4)
    agent_desc = (
        "1 maker + 1 taker (mixed)"
        if args.agent_mode == "mixed"
        else f"{args.n_agents} GenTRX"
    )
    print(f"Config written: {args.output} ({agent_desc} agents)")


if __name__ == "__main__":
    main()
