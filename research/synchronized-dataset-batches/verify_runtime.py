"""Check normal installed package bytes against the independently validated runtime."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify():
    expected = json.loads((ROOT / "evidence/runtime_provenance.json").read_text())["runtime_sha256"]
    roots = {name: Path(importlib.import_module(name).__file__).parent
             for name in ("sentence_transformers", "accelerate")}
    mismatches = []
    for name, sha in expected.items():
        package, relative = name.split("/", 1)
        path = roots[package] / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != sha:
            mismatches.append({"file": name, "expected": sha, "actual": actual})
    if mismatches:
        raise RuntimeError(f"Runtime differs from recorded evidence: {mismatches}")
    return {"matched_recorded_package_files": len(expected)}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
