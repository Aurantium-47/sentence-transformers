"""Run 18 fresh real two-rank Trainer jobs without overwriting recorded evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

from verify_runtime import verify

ROOT = Path(__file__).resolve().parent


def run_outer(command, env, log, limit):
    """Track descendant processes, including the harness's separately-sessioned launchers."""
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    parent = psutil.Process(process.pid)
    tracked = {}
    deadline = time.monotonic() + limit
    timed_out = False
    while True:
        try:
            for child in parent.children(recursive=True):
                tracked[(child.pid, child.create_time())] = child
        except psutil.NoSuchProcess:
            pass
        if process.poll() is not None:
            break
        if time.monotonic() >= deadline:
            timed_out = True
            break
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    # A torchrun child starts a new session in the original harness, so killing
    # only the outer process group would leave it alive. psutil Process objects
    # retain process identity and guard against PID reuse when sending signals.
    if timed_out or process.returncode:
        for child in reversed(list(tracked.values())):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
        psutil.wait_procs(list(tracked.values()), timeout=10)
    return {"returncode": process.returncode, "timed_out": timed_out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="New, nonexistent output directory")
    args = parser.parse_args()
    verify()
    if args.output.exists():
        raise ValueError("Refusing to mix prior evidence and new runs; choose a new output directory")
    args.output.mkdir(parents=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", ACCELERATE_TORCH_DEVICE="cpu")
    records = []
    for family, harness, limit in (("standard", "native_trainer_validation.py", 660),
                                   ("tail", "native_tail_trainer_validation.py", 360)):
        for strategy in ("round_robin", "proportional"):
            target = args.output.resolve() / family / strategy
            command = [sys.executable, str(ROOT / harness), "--options-json",
                       str(ROOT / f"{strategy}_options.json"), "--output-dir", str(target)]
            if family == "tail":
                command += ["--g", "2"]
            record = {"fixture": family, "strategy": strategy, "outer_timeout_seconds": limit,
                      "started_utc": datetime.now(timezone.utc).isoformat()}
            with (args.output / f"{family}_{strategy}.log").open("w") as log:
                record.update(run_outer(command, env, log, limit))
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            records.append(record)
            (args.output / "outer_runs.json").write_text(json.dumps(records, indent=2) + "\n")
            if record["returncode"]:
                raise RuntimeError(f"Run failed; retain logs and stop: {record}")
    print("All 18 fresh two-rank Trainer jobs completed; outputs remain in the requested directory.")


if __name__ == "__main__":
    main()
