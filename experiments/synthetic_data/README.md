# Synthetic-data availability experiments

This directory is the synthetic-domain counterpart to `../original_data/`.
It mirrors the antenna/time sweep layout and has paired `latentwave_ee` (P0)
and `hierarchical` (P2) result folders for NR positioning, interference, and jamming.

The selected use case is a mobile receiver in a private 5G industrial factory
network, with co-channel interference and jamming. Baseline radio settings
come from the open Sionna-based OFDMA anomaly scenario: 3.75 GHz, 20 MHz,
15 kHz subcarrier spacing, multiple legitimate transmitters, and configurable
jammer emissions. The 16-element mobile receive array, positioning labels, and
task adapters are study extensions. These are supplementary zero-shot OOD
stress tests, not reproductions of measured LatentWave data.

The settings and deviations are recorded in
[`factory_ofdma_paper_v3.yml`](../../configs/sionna/factory_ofdma_paper_v3.yml)
and [`factory_ofdma_synthetic_availability_plan.md`](../../docs/factory_ofdma_synthetic_availability_plan.md).
The [paper](https://arxiv.org/abs/2606.02102) and [released radio
configuration](https://github.com/akdd11/ofdma-spectrum-anomalies-simulation/blob/main/src/conf/dataset_generation.yaml)
are the sources. Versioned generated data and run artifacts are kept outside
this report tree under `data/synthetic/` and `runs/synthetic_availability/`.
Superseded HDF5 payloads were removed while their manifests and provenance
remain. Fresh masters will use a new versioned output root, never overwrite
those records. The previous full-information
anchors failed their calibration gates, so curves must be reported as OOD
stress-test/class-index-transfer results. Held-out test evaluation is separate
and requires review of the selection artifacts.
