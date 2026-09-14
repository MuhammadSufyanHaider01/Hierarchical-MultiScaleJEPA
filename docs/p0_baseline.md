# Phase 1: exact P0 baseline

P0 is the pinned LatentWave JEPA path. It uses the context encoder on encoder
masks, the full-input EMA target encoder under `no_grad`, the original
predictor and prediction masks, final target LayerNorm, MSE loss, AdamW
parameter groups, warmup/cosine learning-rate schedule, cosine weight-decay
schedule, and the original EMA momentum schedule. No hierarchical loss or
early-exit behavior is enabled.

The primary configuration is `configs/p0.yml`. Its paths are relative to the
repository root and point to the four local pretraining caches plus the three
downstream files. Under the Version 1 hierarchical early-exit protocol, the
trained `context_encoder` checkpoint namespace is the frozen downstream
foundation backbone. The EMA `target_encoder` is a pretraining-only,
stop-gradient branch for that protocol.

The pinned upstream `src/pretrain.py` has a separate source-style L8 validation
path: it reads the EMA `target_encoder`, mean pools full-input tokens, and
runs scikit-learn probes. `scripts/p0_source_probe.py` preserves that path for
a P0 reproduction probe. Its target-branch metrics are not comparable to the
context-encoder L2/L4/L6/L8 early-exit results and are reported separately.

Formal Phase 5 P0 runs retain the source configuration's fixed 240-epoch
schedule over all four pretraining caches. They do not use a held-out latent
loss or early stopping. The historical latent-early-stopped runs are diagnostic
only, because their changing EMA target and stochastic mask draw do not form a
stationary model-selection objective.

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
