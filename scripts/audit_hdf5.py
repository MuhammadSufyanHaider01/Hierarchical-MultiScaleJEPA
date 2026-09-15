#!/usr/bin/env python3
"""Audit the eight primary local HDF5 files without modifying them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


EXPECTED = {
    "data/pretraining/spectrogram-pretrain.h5": {
        "spectrogram": ((3332, 1, 224, 224), "float32"),
    },
    "data/pretraining/rfp.h5": {
        "normalized_spectrograms": ((5932, 1, 224, 224), "float32"),
    },
    "data/pretraining/wifi-csi.h5": {
        "normalized_csi": ((840, 3, 224, 224), "float32"),
    },
    "data/pretraining/nrpos-indoor.h5": {
        "normalized_csi": ((14448, 5, 224, 224), "float64"),
    },
    "data/downstream/rfs.h5": {
        "image": ((3624, 1, 224, 224), "float32"),
        "label": ((3624,), "int64"),
    },
    "data/downstream/nrpos-outdoor.h5": {
        "features": ((11628, 4, 224, 224), "float32"),
        "label": ((11628, 3), "float32"),
    },
    "data/downstream/deepmimo.h5": {
        "sample": ((14840, 2, 32, 32), "float32"),
        "label_beam_16": ((14840,), "int64"),
    },
    "../has.h5": {
        "csi": ((1200, 3, 224, 224), "float32"),
        "label": ((1200,), "int64"),
    },
}


def finite_status(dataset: h5py.Dataset, scan_finite: bool) -> dict[str, object]:
    if not np.issubdtype(dataset.dtype, np.inexact):
        return {"checked": False, "finite": None, "mode": "not_applicable"}

    if scan_finite:
        chunk = max(1, min(dataset.shape[0], 128))
        for start in range(0, dataset.shape[0], chunk):
            values = dataset[start : start + chunk]
            if not np.isfinite(values).all():
                return {"checked": True, "finite": False, "mode": "full"}
        return {"checked": True, "finite": True, "mode": "full"}

    indices = sorted({0, max(0, dataset.shape[0] - 1)})
    values = dataset[indices]
    return {
        "checked": True,
        "finite": bool(np.isfinite(values).all()),
        "mode": "first_last_sample",
    }


def audit_file(root: Path, relative_path: str, expected: dict[str, tuple[tuple[int, ...], str]], scan_finite: bool) -> dict[str, object]:
    path = root / relative_path
    result: dict[str, object] = {"path": relative_path, "exists": path.is_file(), "datasets": {}, "errors": []}
    if not path.is_file():
        result["errors"].append("missing file")
        return result

    with h5py.File(path, "r") as h5:
        actual_keys = sorted(h5.keys())
        result["keys"] = actual_keys
        result["datasets"] = {}
        for key, (shape, dtype) in expected.items():
            if key not in h5:
                result["errors"].append(f"missing dataset: {key}")
                continue
            dataset = h5[key]
            observed = {
                "shape": list(dataset.shape),
                "dtype": str(dataset.dtype),
                "count": int(dataset.shape[0]) if dataset.shape else 1,
                "finite": finite_status(dataset, scan_finite),
            }
            result["datasets"][key] = observed
            if tuple(dataset.shape) != shape:
                result["errors"].append(f"{key}: shape {tuple(dataset.shape)} != {shape}")
            if str(dataset.dtype) != dtype:
                result["errors"].append(f"{key}: dtype {dataset.dtype} != {dtype}")
            if observed["finite"]["finite"] is False:
                result["errors"].append(f"{key}: non-finite values detected")
    result["valid"] = not result["errors"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scan-finite", action="store_true", help="scan all numeric values instead of first/last samples")
    args = parser.parse_args()

    files = [audit_file(args.root, path, expected, args.scan_finite) for path, expected in EXPECTED.items()]
    report = {
        "root": str(args.root.resolve()),
        "finite_scan": args.scan_finite,
        "file_count": len(files),
        "valid": all(item.get("valid", False) for item in files),
        "files": files,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
