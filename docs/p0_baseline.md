# Phase 1: exact P0 baseline

P0 is the pinned LatentWave JEPA path. It uses the context encoder on encoder
masks, the full-input EMA target encoder under `no_grad`, the original
predictor and prediction masks, final target LayerNorm, MSE loss, AdamW
parameter groups, warmup/cosine learning-rate schedule, cosine weight-decay
schedule, and the original EMA momentum schedule. No hierarchical loss or
early-exit behavior is enabled.

The primary configuration is `configs/p0.yml`. Its paths are relative to the
repository root and point to the four local pretraining caches plus the three
downstream files. The downstream foundation weights are taken from the
`target_encoder` checkpoint namespace and are consumed by the existing
`latentwfm` finetuning path.

## Checks

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python scripts/p0_smoke.py --device cpu --steps 1
PYTHONPATH=src .venv/bin/python scripts/p0_smoke.py --device cuda --steps 2
PYTHONPATH=src .venv/bin/python src/pretrain.py --help
```

P0 checkpoints include the pinned source commit, repository revision, complete
configuration, seed, HDF5 dataset manifest, Python/PyTorch/CUDA environment,
and detected GPU/driver metadata. HDF5 contents are described in the manifest
without being copied into the checkpoint or Git.
