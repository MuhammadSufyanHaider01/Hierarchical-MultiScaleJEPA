#!/usr/bin/env python3
"""Verify that the project environment can execute a CUDA kernel."""

from __future__ import annotations

import torch


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return 1

    device = torch.device("cuda:0")
    left = torch.randn((2048, 2048), device=device)
    right = torch.randn((2048, 2048), device=device)
    result = left @ right
    torch.cuda.synchronize(device)

    print(f"torch={torch.__version__}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"capability={torch.cuda.get_device_capability(0)}")
    print(f"result_shape={tuple(result.shape)}")
    print(f"finite={bool(torch.isfinite(result).all())}")
    return 0 if bool(torch.isfinite(result).all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
