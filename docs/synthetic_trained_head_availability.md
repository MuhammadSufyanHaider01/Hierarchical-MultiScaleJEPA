# Synthetic-trained-head availability study

## Purpose and claim boundary

This is a paired synthetic-domain stress study for the frozen LatentWave P0
encoder and hierarchical P2 encoder. The simulator is **not** intended to
recreate the measured LatentWave collection. The real-data curves remain the
primary evidence; these results describe what the two fixed representations
support after their task heads are trained on one controlled synthetic
wireless environment.

The simulator scenario is the open factory/OFDMA spectrum-anomaly setting
described by Schösser et al. and its public implementation. We adapt that
setting to three explicitly synthetic tasks: factory-scene 3-D localization,
binary jammer-presence detection, and six-way jammer-mode recognition. These
are task proxies with frozen label maps, not recovered meanings of the
measured NR-positioning or interference/jamming labels.

## Locked model and data pairing

- P0 encoder: the `time_n6` original-LatentWave checkpoint.
- P2 encoder: the `time_n6` hierarchical checkpoint at epoch 29, the checkpoint
  used by its six-task downstream packet.
- Load the context-encoder branch strictly for each checkpoint. Hash both
  checkpoints and record the config, branch, and loader result. Do not load or
  update the measured-data task heads.
- Generate one maximum-information master. Every sample contains a 16-element
  received feature tensor and the same realization ID, receiver location,
  jammer-presence label, and jammer-mode label. Both encoders and every
  availability condition use the same sample IDs and features.
- Split by independent scene/signal realization ID into disjoint
  `head_train`, `head_validation`, `exit_selection`, and `test` sets. Use the
  same immutable manifest for all tasks, methods, sweeps, and diagnostics.
- The feature transform and its normalization are shared across methods and
  conditions. Fit normalization from `head_train` full-information samples
  only; apply it unchanged to every split and crop. Never normalize a
  condition independently.

## Synthetic tasks

| Study task | Target definition | Outputs / primary metric |
| --- | --- | --- |
| `nrpos-outdoor` proxy | Receiver 3-D coordinates in the factory scene, normalized per-axis to `[-1,1]` for training and decoded to scene meters for metrics | 3 / mean Euclidean distance error |
| `interference` proxy | 0 = no jammer; 1 = any jammer active | 2 / macro accuracy |
| `jamming` proxy | 0 = no jammer; 1 = barrage; 2 = deceptive; 3 = pilot; 4 = random-hop; 5 = sweep | 6 / macro accuracy |

The interference and jamming labels are generated from the same physical
event. The six class definitions and waveform parameter ranges are fixed in
the versioned simulator profile and copied into every manifest and result
packet. Class-index agreement with the measured-data heads is not asserted;
the synthetic heads are trained from scratch for these target definitions.

## Signal and availability representation

Use a fixed 16-element half-wavelength receiver array at the selected carrier.
Rasterize its received complex OFDMA capture into one common per-element
time-frequency representation. The locked master tensor is `[16, 224, 896]`:
224 frequency bins by 896 ordered time columns. The transform must be
deterministic, retain receive-power information, and be identical for every
task and method. No resizing is permitted.

- Antenna conditions: `C={1,2,4,6,8,10,12,16}`, with `Tₚ=14` and the first `C`
  elements of one recorded deterministic nested subset ordering.
- Time conditions: `Tₚ={2,4,6,8,10,12,14,18,22,28,36,44,56}` leading patch
  columns, each 16 raster columns wide. Use `C=4` for localization and `C=1`
  for the two classification tasks.
- The two sweep maxima differ. Train separate head packets for each sweep at
  that sweep's full-information condition: antenna heads at `C=16,Tₚ=14`, and
  time heads at the task's fixed `C,Tₚ=56`. For a given sweep/task, train P0
  and P2 heads on exactly the same training examples and split.
- Never evaluate the combined `C=16,Tₚ=56` input. Preserve realization IDs,
  array element indices, waveform settings, per-condition crop indices, and
  patch/token counts.

## Training, selection, and test protocol

1. Run deterministic simulator and feature-extraction unit checks, then a
   shared pilot. Check sample repeatability, finite values, feature shape,
   class support, label consistency, nesting, and split disjointness. Run
   short head-training probes to confirm each synthetic task is learnable by
   both frozen representations. Lock the formal sample count and all random
   seeds before formal generation.
2. For each P0/P2 × task × sweep × exit layer (`L2/L4/L6/L8`), freeze the
   encoder and train only an independent `ExitHead` using the full-information
   `head_train` split. Select head checkpoints and stopping epochs using only
   `head_validation`, with identical architecture, optimizer, budget, and
   seed schedule across P0/P2. Record encoder-weight hashes before and after.
3. Evaluate each full-information trained head on every nested availability
   condition using `exit_selection`; store all fixed-exit metrics. Select the
   earliest exit within the existing 5% rule, using only this split. Measure
   the selected exit's FLOPs and synchronized batch-1 latency.
4. Save selection artifacts and stop for manual review. Test evaluation is a
   separate command; only after approval, evaluate every fixed exit and the
   validation-selected exit on the locked `test` IDs without rerunning any
   selector or training procedure.
5. Use paired bootstrap resampling of realization IDs for uncertainty
   intervals and P0-vs-P2 differences. Report performance curves, selected
   exits, selection frequencies from repeated head seeds, L8 gaps, compute,
   and provenance. Include calibration / learnability results and any
   precision shortfall. Do not tune generation or training against test
   outcomes.

## Required diagnostics

- **Condition-specific head diagnostic:** separately train heads on each
  condition's `head_train` view, with the same split and optimization budget.
  Keep this distinct from the primary full-information-trained-head sweep;
  it indicates whether a trend is caused by head/input mismatch.
- **Scratch-encoder diagnostic:** for at least one predeclared representative
  task/axis, train a randomly initialized encoder and head on the same
  synthetic training data under an explicitly matched budget. This diagnostic
  is needed before attributing a persistent synthetic-domain failure to
  pretrained representations rather than data or protocol.
- Do not claim that the synthetic curves should match measured-data trends.
  Agreement is supporting evidence; disagreement is evidence about transfer
  and representation behavior in this simulator domain, not proof of a model
  defect by itself.

## Reproducibility and output tree

Generated tensors and manifests live in an ignored, versioned
`data/synthetic_trained_heads/factory_ofdma_v1/` root. Training and evaluation
packets live under `runs/synthetic_trained_heads/factory_ofdma_v1/`. Result
packets mirror the original structure under
`experiments/synthetic_data/vary_antenna/<task>/` and
`experiments/synthetic_data/vary_time_samples/<task>/`.

Each method subfolder (`latentwave_ee/`, `hierarchical/`) contains its immutable
JSON metrics, CSV/Markdown tables, provenance README, and individual PNG/PDF
plot. Each task root contains the paired two-line PNG/PDF comparison, paired
CSV/Markdown table, and README. Top-level synthetic-study documentation must
state the synthetic-domain caveat and link the exact generator profile,
manifest, and checkpoint hashes. Do not overwrite or move anything in
`experiments/original_data/`.
