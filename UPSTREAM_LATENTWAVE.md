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

## Phase 0 import audit

The source tree compiles successfully. A runtime import using the existing
bootstrap environment is currently blocked before model loading because that
environment does not provide `pydantic_settings`. After that dependency is
provided, the pinned package initializers also reference absent optional files
(`jepa_simclr`, `mae`, `simclr`, `vicreg`, and related modules). These are
upstream tree defects at the pinned commit, not local dataset or model changes.
They are kept visible for the Phase 1 baseline decision; no model code was
patched in Phase 0.
