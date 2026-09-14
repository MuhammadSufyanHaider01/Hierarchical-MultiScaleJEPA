# Upstream LatentWave provenance

- Repository: https://github.com/AhmedEssam19/latent-wave
- Pinned commit: `330e9ef5f6eb69e728ba43b3755036cffdd086cd`
- Local source imported during Phase 0: `configs/`, `src/`, `pyproject.toml`,
  `uv.lock`, and `.python-version`
- Reference checkout used for verification: `/tmp/latent-wave-phase0`

The source was copied from the detached pinned commit. Phase 0 does not alter
model computations, masking, optimization, or loss behavior. Repository-local
dataset paths are provided separately in `configs/jepa_local.yml` so the
upstream configuration remains available for provenance comparison.

The pinned tree is the implementation reference for P0. Any compatibility
changes required because of missing optional modules or environment drift must
be recorded explicitly and must not be silently treated as upstream behavior.

Phase 1 adds a narrow local compatibility layer: the package initializers and
P0 entrypoint expose only the JEPA files that actually exist at this commit;
the unused optional collator and deferred WavesFM path are not imported. The
JEPA encoder, predictor, masking, loss, optimizer, scheduler, and EMA update
remain unchanged apart from extracting the EMA arithmetic into a testable
method.

Phase 2 adds only an optional representation-read API and frozen downstream
head wrappers around that encoder. The pinned checkout remains the reference
for any regenerated P0 results; Phase 2 must not be presented as a change to
the original LatentWave baseline.

The pinned `JEPA._get_validation_tokens_reps` reads the EMA `target_encoder`
and the pinned `src/pretrain.py` applies source-style scikit-learn probes to
its pooled L8 features. Version 1 deliberately changes only the separate
early-exit protocol to use the trained `context_encoder` at L2/L4/L6/L8. The
two evaluations are now exposed separately: `scripts/p0_source_probe.py`
preserves the pinned target-L8 probe, while `scripts/phase5_exits.py` remains
the context-encoder early-exit runner. Their metrics must not be combined.

Phase 3 adds same-scale auxiliary predictors only when the separate P1
configuration enables them. The original P0 configuration keeps the
hierarchy-disabled path and remains the minimal-deviation reference.

## Phase 0 import audit

The source tree compiles successfully. A runtime import using the existing
bootstrap environment is currently blocked before model loading because that
environment does not provide `pydantic_settings`. After that dependency is
provided, the pinned package initializers also reference absent optional files
(`jepa_simclr`, `mae`, `simclr`, `vicreg`, and related modules). These are
upstream tree defects at the pinned commit, not local dataset or model changes.
They are kept visible for the Phase 1 baseline decision; no model code was
patched in Phase 0.
