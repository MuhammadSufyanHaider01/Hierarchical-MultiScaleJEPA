#!/usr/bin/env python3
"""Audit the local Phase 6 OOD HDF5 datasets without copying or modifying them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phase6 import audit_ood_datasets, default_ood_root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ood-root", type=Path, default=default_ood_root(REPOSITORY_ROOT))
    parser.add_argument("--scan-finite", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Optional local JSON output path.")
    args = parser.parse_args()
    report = audit_ood_datasets(args.ood_root, scan_finite=args.scan_finite)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
