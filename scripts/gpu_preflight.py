#!/usr/bin/env python3
"""Run the mandatory read-only GPU/data/smoke gate for formal experiments.

This command deliberately performs no formal pretraining, frozen-head
evaluation, benchmark, resume, checkpoint creation, or run-directory setup.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _run(label: str, command: list[str]) -> None:
    print(f"\n==> {label}", flush=True)
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    if completed.returncode:
        raise SystemExit(f"Preflight failed during {label} (exit {completed.returncode}).")


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    python = sys.executable
    _run("NVIDIA driver visibility", ["nvidia-smi"])
    _run("project CUDA kernel", [python, "scripts/check_cuda.py"])
    _run("primary HDF5 schema and finite-value audit", [python, "scripts/audit_hdf5.py", "--scan-finite"])
    for method in ("p0", "p1", "p2"):
        _run(f"{method.upper()} CUDA smoke", [python, f"scripts/{method}_smoke.py", "--device", "cuda", "--steps", "2"])
    print("\nGPU preflight passed. No formal experiment was launched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
