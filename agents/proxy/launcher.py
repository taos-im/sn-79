# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from functools import wraps
from pathlib import Path

from loguru import logger


def _coro(fn):
    @wraps(fn)
    def wrap(*a, **k):
        return asyncio.run(fn(*a, **k))
    return wrap


async def _watch(name, stream, is_err=False):
    while True:
        try:
            line = await stream.readline()
        except ValueError:
            # One child line longer than the stream limit. readline has already dropped the
            # oversized chunk; carry on reading rather than letting the exception kill this
            # watcher, which would leave the child's pipe undrained and block it on write.
            logger.warning(f"{name} | <log line over the stream limit, dropped>")
            continue
        if not line:
            return
        text = line.decode(errors="replace").rstrip()
        parts = text.split("|")
        parsed = "|".join(parts[2:]) if datetime.now().strftime("%Y-%m-%d") in text and len(parts) >= 3 else text
        if is_err:
            logger.error(f"{name} | {parsed}")
        elif "| SUCCESS  |" in text:
            logger.success(f"{name} | {parsed}")
        else:
            logger.info(f"{name} | {parsed}")


async def _spawn(name, argv, env=None, cwd=None):
    try:
        logger.info(f"Launching {name}: {' '.join(argv)}")
        proc = await asyncio.create_subprocess_exec(
            *argv,
            limit=1024 * 512,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=cwd,
        )
        await asyncio.gather(_watch(name, proc.stdout), _watch(name, proc.stderr, True))
    except asyncio.CancelledError:
        proc.terminate()
        await proc.wait()
        raise
    except Exception as ex:
        logger.error(f"{name} failed to launch: {ex}")


def _resolve(repo_root: Path, p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    if p.startswith("..") or p.startswith("./"):
        script_dir = Path(__file__).resolve().parent
        return (script_dir / p).resolve()
    return (repo_root / p).resolve()


def _agent_env(agent_id: int, endpoint: str, access: str, secret: str) -> dict:
    return {
        "GENTRX_AGENT_S3_BUCKET": f"agent-{agent_id}",
        "GENTRX_AGENT_S3_ENDPOINT_URL": endpoint,
        "GENTRX_AGENT_S3_ACCESS_KEY": access,
        "GENTRX_AGENT_S3_SECRET_KEY": secret,
        "GENTRX_AGENT_S3_READ_ACCESS_KEY": access,
        "GENTRX_AGENT_S3_READ_SECRET_KEY": secret,
        "GENTRX_AGENT_S3_REGION": "us-east-1",
    }


@_coro
async def main(config_path: str, train: bool, sim_xml_override: str | None):
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    cfg = json.load(open(config_path))
    py = sys.executable

    sim_xml = sim_xml_override or cfg["proxy"]["simulation_xml"]
    sim_xml = str(_resolve(repo_root, sim_xml))
    if not Path(sim_xml).is_file():
        logger.error(f"Simulation XML not found: {sim_xml}")
        return

    # A bare name (the default) stays a bare name, so it resolves on PATH the way the
    # `run` wrapper arranges. Anything path-shaped is resolved like simulation_xml is,
    # letting a config point straight at a build tree without an absolute path in it.
    taosim_bin = cfg.get("taosim", {}).get("bin", "taosim")
    if os.sep in taosim_bin:
        taosim_bin = str(_resolve(repo_root, taosim_bin))
    sim_delay = cfg.get("taosim", {}).get("delay", 5)

    endpoint = os.environ.get("GENTRX_CHAIN_ENDPOINT_OVERRIDE", "http://localhost:9000")
    access = os.environ.get("GENTRX_ACCESS_KEY", "minioadmin")
    secret = os.environ.get("GENTRX_SECRET_KEY", "minioadmin")

    gs = None
    if train:
        gs = cfg.get("training", {}).get("gradient_server")
        if not gs:
            logger.error(f"--train requires a 'training.gradient_server' block in {config_path}. Use agents/proxy/config.train.example.json as a template.")
            return

    agents_cfg = cfg.get("agents", {})
    agents_dir = agents_cfg.get("path", "..")
    if not os.path.isabs(agents_dir):
        agents_dir = str((script_dir / agents_dir).resolve())
    port = agents_cfg.get("start_port", 8888)
    agent_id = 0

    tasks = []

    proxy_env = os.environ.copy()
    tasks.append(_spawn(
        "PROXY",
        [py, "proxy.py", "--config", config_path],
        env=proxy_env,
        cwd=str(script_dir),
    ))

    async def _delayed_sim():
        await asyncio.sleep(sim_delay)
        await _spawn("TAOSIM", [taosim_bin, "-f", sim_xml])
    tasks.append(_delayed_sim())

    if train:
        gs_argv = [
            py, "-m", "GenTRX.src.gradient_server",
            "--checkpoint", str(_resolve(repo_root, gs["checkpoint"])),
            "--val-data",   str(_resolve(repo_root, gs["val_data"])),
            "--output",     str(_resolve(repo_root, gs["output"])),
            "--port", str(gs["port"]),
            "--interval", str(gs.get("interval", 60)),
            "--min-score", str(gs.get("min_score", -0.1)),
            "--mode", gs.get("mode", "simulation"),
        ]
        miner_buckets = os.environ.get("GENTRX_MINER_BUCKETS", "")
        if miner_buckets:
            gs_argv += ["--miner-buckets", miner_buckets]
        tasks.append(_spawn("GRAD", gs_argv, env=os.environ.copy(), cwd=str(repo_root)))

    for cls, cfgs in agents_cfg.items():
        if cls in ("start_port", "path"):
            continue
        for entry in cfgs:
            for _ in range(entry.get("count", 1)):
                params = [f"{k}={v}" for k, v in entry["params"].items()]
                name = f"{cls}_{port}"
                argv = [py, f"{agents_dir}/{cls}.py",
                        "--port", str(port),
                        "--agent_id", str(agent_id),
                        "--params"] + params
                env = os.environ.copy()
                if train:
                    env.update(_agent_env(agent_id, endpoint, access, secret))
                tasks.append(_spawn(name, argv, env=env))
                port += 1
                agent_id += 1

    logger.info(
        f"Proxy port={cfg['proxy']['port']}, "
        f"agents start_port={agents_cfg.get('start_port',8888)}, "
        f"mode={'training' if train else 'trading'}"
    )
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.success("Shut down.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.json"))
    p.add_argument("--train", action="store_true")
    p.add_argument("--sim-xml", default=None, help="Override proxy.simulation_xml")
    args = p.parse_args()
    main(args.config, args.train, args.sim_xml)
