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
setting to three explicitly synthetic tasks: factory-floor localization at a
fixed receiver height (not 3-D localization),
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
- The 512-sample pilot uses disjoint `head_train` (70%), `head_validation`
  (15%), and `exit_selection` (15%) partitions; for pilot calibration only,
  `exit_selection` is treated as the anchor holdout and no exit selector is run.
- Formal data use disjoint `head_train` (60%), `head_validation` (10%),
  `calibration` (10%), `exit_selection` (10%), and `test` (10%) partitions.
  One immutable manifest is shared by tasks, methods, sweeps, and diagnostics.
- The feature transform and its normalization are shared across methods and
  conditions. Fit normalization from `head_train` full-information samples
  only; apply it unchanged to every split and crop. Never normalize a
  condition independently.

## Synthetic tasks

| Study task | Target definition | Outputs / primary metric |
| --- | --- | --- |
| `nrpos-outdoor` proxy | Factory-floor receiver coordinates `(x,y,z)` with `z=1.5 m` fixed; the two varying axes are normalized to `[-1,1]` for training and decoded to scene meters | 3 outputs / mean Euclidean distance error |
| `interference` proxy | 0 = no jammer; 1 = any jammer active | 2 / macro accuracy |
| `jamming` proxy | 0 = no jammer; 1 = barrage; 2 = deceptive; 3 = pilot; 4 = random-hop; 5 = sweep | 6 / macro accuracy |

The interference and jamming labels are generated from the same physical
event. The six class definitions and waveform parameter ranges are fixed in
the versioned simulator profile and copied into every manifest and result
packet. Class-index agreement with the measured-data heads is not asserted;
the synthetic heads are trained from scratch for these target definitions.

## Pilot v1 audit and v2 localization correction

The first v1 pilot is retained only as diagnostic provenance. Its legitimate
transmitter locations were redrawn independently on every realization, which
removed a stable reference for the absolute factory-floor coordinate target.
The P0 full-information antenna anchor therefore had 17.39 m error versus the
16.54 m training-centroid baseline, with a paired 95% improvement interval of
[-1.89 m, 0.21 m]; that is a failed learnability gate, not a result to carry
into the paired study. No formal v1 data or test result was generated.

The active v2 profile fixes four access points at known factory-floor corner
coordinates and fixes each access point's OFDMA resource-block assignment.
This is an explicit stable-anchor localization use case. All jammer classes,
receiver realizations, availability crops, and model comparisons share the
same v2 master. The v1 pilot remains untouched under its separate versioned
root; v2 will be accepted only if its pilot gates pass.

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

1. Run deterministic simulator and feature-extraction unit checks, then one
   shared pilot. Pilot partitions are `head_train` (70%), `head_validation`
   (15%), and an anchor holdout (the pilot manifest's `exit_selection`, 15%);
   no pilot exit selector or test evaluation is run. Require full-information
   L8 to beat chance (classification) or the training-centroid predictor
   (localization) by a positive 95% bootstrap lower bound for both methods and
   both sweeps. Lock the common formal count from the least-precise pilot
   anchors before formal generation.
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
5. Test stage requires a separate `review_approval.json` that names the reviewer,
   sets `approved: true`, and binds the exact selection artifact SHA-256. Use
   paired bootstrap resampling of realization IDs for uncertainty intervals
   and P0-vs-P2 differences. Report performance curves, selected
   exits, selection frequencies from repeated head seeds, L8 gaps, compute,
   and provenance. Include calibration / learnability results and any
   precision shortfall. Do not tune generation or training against test
   outcomes.

## Execution commands

Use the isolated `.venv-synthetic-study` for Sionna generation and the paired
model runner. Set `CUDA_VISIBLE_DEVICES=` for simulator/model steps when the
accelerator is occupied. The pilot and formal data are each one shared master;
do not make separate antenna/time files.

```bash
# Shared pilot; no test partition.
CUDA_VISIBLE_DEVICES= .venv-synthetic-study/bin/python \
  scripts/generate_synthetic_trained_head_master.py --mode pilot --count 512 --seed 341903

# For each task (nrpos-outdoor, interference, jamming), each axis (antenna,
# time), and each method (p0, p2), train the pilot heads and run the full-info
# calibration anchor. Complete all 12 train/anchor pairs before count planning.
.venv-synthetic-study/bin/python scripts/synthetic_trained_head_experiment.py \
  --stage train --mode pilot --master data/synthetic_trained_heads/factory_ofdma_v2/pilot_n512_seed341903/master.h5 \
  --task nrpos-outdoor --axis antenna --method both --device cpu
.venv-synthetic-study/bin/python scripts/synthetic_trained_head_experiment.py \
  --stage anchor --mode pilot --master data/synthetic_trained_heads/factory_ofdma_v2/pilot_n512_seed341903/master.h5 \
  --task nrpos-outdoor --axis antenna --method both --device cpu

# After all pilot anchors complete, the planner locks a count for tasks whose
# four paired gates passed and marks blocked tasks without preventing the others.
.venv-synthetic-study/bin/python scripts/plan_synthetic_study_count.py \
  --pilot-run-root runs/synthetic_trained_heads/factory_ofdma_v2/pilot \
  --master data/synthetic_trained_heads/factory_ofdma_v2/pilot_n512_seed341903/master.h5 \
  --output data/synthetic_trained_heads/factory_ofdma_v2/count_plan_test_fraction_v2.json
# The test-fraction-aware lock is 7,500 total rows (the shared source cap);
# its projected test-CI shortfall is stored in the count plan.
CUDA_VISIBLE_DEVICES= .venv-synthetic-study/bin/python scripts/generate_synthetic_trained_head_master.py \
  --mode formal --count-plan data/synthetic_trained_heads/factory_ofdma_v2/count_plan_test_fraction_v2.json \
  --seed 341902 --output data/synthetic_trained_heads/factory_ofdma_v2/formal_n7500_seed341902_countplan_v2

# Audit the completed master and split provenance before training any heads.
.venv-synthetic-study/bin/python scripts/audit_synthetic_trained_head_master.py \
  --master data/synthetic_trained_heads/factory_ofdma_v2/formal_n7500_seed341902_countplan_v2/master.h5 \
  --count-plan data/synthetic_trained_heads/factory_ofdma_v2/count_plan_test_fraction_v2.json

# Train/anchor all formal heads first. Selection requires passed formal anchors
# for both methods and both sweeps of that task.
.venv-synthetic-study/bin/python scripts/synthetic_trained_head_experiment.py \
  --stage select --mode formal --master <formal-master.h5> --task nrpos-outdoor --axis antenna --method both --device cuda

# Selection report only. It never reads test samples.
.venv-synthetic-study/bin/python scripts/report_synthetic_trained_head_availability.py \
  --mode formal --task nrpos-outdoor --axis antenna

# After primary formal heads and calibration anchors exist, the
# condition-specific diagnostic trains a separate head quartet for one exact
# availability level, then selects using only the same formal exit_selection IDs.
.venv-synthetic-study/bin/python scripts/synthetic_trained_head_experiment.py \
  --stage train-condition --mode formal --master <formal-master.h5> \
  --task nrpos-outdoor --axis antenna --condition-level 4 --method both --device cuda
.venv-synthetic-study/bin/python scripts/synthetic_trained_head_experiment.py \
  --stage select --mode formal --master <formal-master.h5> \
  --task nrpos-outdoor --axis antenna --condition-level 4 --method both --device cuda
```

Held-out test is intentionally a separate stage. After reviewing a task's P0
and P2 `selection.json` files and paired report, create a
`review_approval.json` in each method packet with `approved: true`, a non-empty
`reviewer`, and `selection_sha256` equal to that method's exact selection JSON
hash; only then run `--stage test`. The test runner does not recompute or change
the selected exit; it stores fixed-exit predictions bound by SHA-256 for paired
uncertainty reporting. After both approved test packets are complete, render a
separate `test/` report (the validation report remains immutable):

```bash
.venv-synthetic-study/bin/python scripts/report_synthetic_trained_head_availability.py \
  --mode formal --include-test --task nrpos-outdoor --axis antenna
```

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
`data/synthetic_trained_heads/factory_ofdma_v2/` root. Training and evaluation
packets live under `runs/synthetic_trained_heads/factory_ofdma_v2/`. Result
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
