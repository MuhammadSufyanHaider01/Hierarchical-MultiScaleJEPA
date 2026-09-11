# Local dataset layout

The HDF5 files in this directory are local dataset artifacts and are intentionally ignored by Git.

## Pretraining

| File | HDF5 key | Shape | Role |
|---|---|---|---|
| `pretraining/spectrogram-pretrain.h5` | `spectrogram` | `(3332, 1, 224, 224)` | Spectrogram pretraining |
| `pretraining/rfp.h5` | `normalized_spectrograms` | `(5932, 1, 224, 224)` | RF fingerprinting pretraining |
| `pretraining/wifi-csi.h5` | `normalized_csi` | `(840, 3, 224, 224)` | WiFi CSI pretraining |
| `pretraining/nrpos-indoor.h5` | `normalized_csi` | `(14448, 5, 224, 224)` | Indoor 5G NR CSI pretraining |

## Downstream evaluation

| File | HDF5 key(s) | Shape | Role |
|---|---|---|---|
| `downstream/rfs.h5` | `image`, `label` | `(3624, 1, 224, 224)` | RF signal classification |
| `downstream/nrpos-outdoor.h5` | `features`, `label` | `(11628, 4, 224, 224)` | 5G NR positioning |
| `downstream/deepmimo.h5` | `sample`, `label_beam_16`, `label_los` | `(14840, 2, 32, 32)` | Beam prediction and LoS/NLoS classification |

The original source files remain one directory above this repository. The copied files are intended to be referenced through configuration, not committed to the remote repository.

## Audit

Run the read-only schema audit from the repository root:

```bash
python scripts/audit_hdf5.py
```

Use `python scripts/audit_hdf5.py --scan-finite` for a full numeric finite-value scan instead of the default first/last sample check.
