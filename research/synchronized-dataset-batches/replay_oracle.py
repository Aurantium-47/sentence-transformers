"""Recompute the native loss/gradient/SGD/resume oracles from JSON only.

No training, subprocesses, pickle, checkpoint loading, or network calls are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import native_tail_trainer_validation as tail
import native_trainer_validation as standard
from verify_runtime import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, default=Path(__file__).resolve().parent / "traces")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"runtime": verify(), "scope": "JSON-only native-objective replay; not a new DDP run", "groups": []}
    for family, harness, groups in (("standard", standard, (1, 2)), ("tail", tail, (2,))):
        for strategy in ("round_robin", "proportional"):
            folder = args.traces / family / strategy
            for g in groups:
                result = harness.verify_group(folder, g)
                assert result["native_oracle_pass"] and result["second_epoch_pass"] and result["checkpoint_resume_pass"]
                report["groups"].append({"fixture": family, "strategy": strategy, **result})
    report["passed"] = True
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized)
    print(json.dumps({"passed": True, "groups": len(report["groups"]), "scope": report["scope"]}))


if __name__ == "__main__":
    main()
