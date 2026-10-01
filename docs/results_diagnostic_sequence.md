# Diagnostic sequence for the masking and hierarchy results

## Purpose

This document turns the current results review into an ordered set of checks
and experiments. The goal is to find out whether the small and task-dependent
differences between P1, P2, and P2-Reverse come from the learned objectives,
mask geometry, checkpoint horizon, downstream evaluation, or ordinary
single-seed variation.

The current results are a useful pilot, not evidence that one method wins
reliably. Across the four reported conditions, P2's mean classification
accuracy is only about 0.31 percentage points above P1. P2 leads on the
multi-block and time aggregates, is slightly ahead on frequency, and trails
P1 on region. NR positioning should be compared separately: lower error is
better, and P2 has the lowest four-condition mean among the local methods.
Task-level results vary substantially, so aggregate scores must not replace
per-task reporting.

The numerical source of truth is the per-mask CSV/manifest and tables in
`reports/masking_study/`, plus
`exports/peer_results_validation_best/downstream_validation_best_results.csv`.
The CSV is a convenience export; imported P2-Reverse entries have weaker
provenance than the locally produced methods.

## Findings that motivate the sequence

1. **Mask count does not equal matched masking difficulty.** The local patch
   grid is 14 by 14. Frequency and time spanning masks make 2 by 14 or 14 by 2
   blocks (28 patches per block). Region masking samples rectangular shapes,
   often around 5 by 6 through 8 by 5. Prediction blocks can overlap each
   other. The random region profile can also yield a smaller context field
   than the spanning profiles. The six-block setting therefore changes target
   size, unique masked coverage, context available to the encoder, and target
   overlap together.
2. **Region P2 has an additional L4 geometry change.** The current partition
   makes exact local 2 by 2 regions for even-sided blocks. If a dimension is
   odd, it makes up to four larger, balanced quadrant regions. Common region
   masks are often odd-sided, while a 2 by 14 spanning block gives seven small
   regions. This is a plausible explanation for part of the region result,
   but not a proven cause. P2 also trails P1 at L8 under region masking.
3. **Checkpoint horizons differ.** The reported P1 heads use 60 completed
   pretraining epochs for the completed region, time, and frequency runs. The
   reported P2 checkpoint is at 45 completed epochs for region and 55 for time
   and frequency. The region P1 epoch-44 checkpoint exists, enabling a matched
   45-epoch head comparison without new pretraining.
4. **One pretraining seed cannot estimate run-to-run variation.** Repeated
   head training can estimate head-seed variation conditional on a fixed
   representation, but only repeated pretraining seeds measure representation
   variability.
5. **Some task scores have limited resolution or a different generalization
   meaning.** HAS has 120 balanced test examples, so one changed prediction
   corresponds to about 0.83 percentage points in macro accuracy. The current
   DeepMIMO split has all six scenarios represented in both train and test;
   it measures new samples from known scenarios, not unseen-scenario
   generalization.
6. **The conditions are not all independent masking geometries.** The
   multi-block and frequency conditions use the same spanning-frequency mask
   geometry in this implementation, although their historical checkpoints
   and training histories differ. Do not count them as independent geometry
   replications. P2-Reverse was imported; its exact split indices and complete
   region pretraining provenance are not present locally.
7. **The present task scores do not establish frequency-versus-time layer
   specialization.** Task names alone do not show which signal dimension
   drives performance. That requires controlled perturbations and, for
   transfer claims, splits grouped by capture/session/scenario where the data
   supports them.

## Ordered work plan

### Step 1 — Freeze the result inventory

Create one audit ledger with one row per reported method, mask, and
pretraining checkpoint. Record:

- manuscript method name and local run ID;
- mask profile, patch grid, block count, block shape/range, and realized
  prediction/context mask statistics if available;
- pretraining seed, completed epoch, checkpoint path and checksum;
- precision, attention implementation, effective batch size, loader profile,
  and relevant configuration checksum;
- downstream head seed, task suite, split manifest and checksum;
- validation-selected exit and the fixed-exit metrics;
- test metric provenance, benchmark provenance, and whether the artifact was
  imported.

Verify that local B0/P0/P1/P2 comparisons use the same persisted six-task
80/10/10 split and task mappings. Mark P2-Reverse as exploratory wherever its
split identity or training provenance cannot be verified. Record the
multi-block/frequency geometry duplication explicitly. Do not combine
incompatible source-style full-depth probe results with frozen-head metrics.

**Deliverable:** a provenance table and a short list of unresolved metadata.
Do not launch training for this step.

### Step 2 — Audit realized mask and target-region geometry

Run a deterministic sampler audit for each implemented profile using the
same collator settings as formal training. Cover enough draws to describe the
distribution, and test the real training batch size as well as batch size one.
For every draw, measure:

- per-block target token count;
- unique target-token union and fraction of the 196-patch grid;
- pairwise and total prediction-block overlap;
- visible context token count and the context-to-target ratio;
- target-region count and membership-size distribution at each supervised
  layer, especially L4;
- batch-level truncation/padding behavior and whether it changes the mask
  distribution.

Check invariants: prediction blocks are wholly hidden from the context;
region membership is contiguous and complete; every target token appears in
the intended target region(s); pooled targets and predictor queries have
matching axes and counts. Save the random seed, config checksum, summary
statistics, and several synthetic examples.

**Deliverable:** a geometry table comparing masks on realized coverage and
context, not only the configured `N_m` value. Do not alter training masks in
this step.

### Step 3 — Audit split independence and define the intended generalization

For each task, inspect available capture/session/source/scenario metadata and
check whether groups cross train, validation, and test. A unique `source_file`
per row is not proof that the underlying captures are independent; document
what the field actually identifies. Jamming and interference currently lack
the same obvious group identifiers, so record that limitation rather than
assuming independence.

For DeepMIMO, keep the current random-sample split labeled as known-scenario
evaluation. If the research claim includes transfer to unseen environments,
create a separate scenario-held-out split. Where capture/session IDs exist,
create group-held-out splits for those tasks. Keep the current result packet
intact and distinguish any new split from it.

**Deliverable:** a split-independence report with group overlap counts and a
clear statement of the generalization each split tests.

### Step 4 — Run a matched-horizon region P1/P2 head comparison

Use the existing P1 region epoch-44 checkpoint and the reported P2 region
epoch-44 checkpoint. Train independent frozen heads for all six tasks at
L2/L4/L6/L8 using identical head configuration, head seed(s), preprocessing,
and the same saved train/validation indices. Do not retrain either backbone
for this comparison.

Report every fixed exit as well as the validation-selected exit. Choose exits
from validation only. The existing test set has already been examined, so
this comparison is diagnostic on the existing split; do not present its test
results as a fresh confirmatory evaluation. If existing scripts can resume
head seeds deterministically, first verify that the seed is applied to head
initialization and data order in both runs.

**Deliverable:** paired P1-versus-P2 epoch-44 validation table, per-task and
per-exit deltas, and head-training logs. This step tests whether unequal
pretraining exposure plausibly explains the reported region gap.

### Step 5 — Estimate downstream-head variability

For the fixed P1/P2 region checkpoints, repeat head training with at least
three distinct head seeds, keeping data splits and all other settings fixed.
Use the same set of seeds for both methods. Report mean and standard
deviation across head seeds and paired method differences. Treat this as
conditional head variability; it is not a substitute for pretraining seeds.

Pay particular attention to HAS, beam prediction, and any selected-exit
changes. Report class-wise results and confusion matrices for classification,
and error distributions for NR positioning, where saved predictions permit.

**Deliverable:** head-seed variability table and per-task plots. If the
observed P1/P2 gap is smaller than this variation, describe the methods as
similar on that task under this protocol.

### Step 6 — Test the checkpoint-horizon explanation

Only if Step 4 leaves a plausible region P2 undertraining explanation, resume
region P2 from its epoch-44 checkpoint to the same 60-epoch horizon already
available for region P1. Preserve optimizer, scheduler, precision, loader,
mask, and objective state when resuming. Verify checkpoint state before
launch. Then train heads using the exact Step 4 protocol and compare 60-epoch
P1 and P2 validation performance.

Do not choose the continuation checkpoint by test performance. Because the
current test results have already been inspected, use a future group-held-out
test split for confirmatory claims if one can be constructed; otherwise label
the comparison as additional analysis on the existing split.

**Deliverable:** paired 45- and 60-epoch validation curves/results. If P2
closes the gap with longer training, report horizon sensitivity; if it does
not, checkpoint horizon is less likely to be the main explanation.

### Step 7 — Isolate the region L4 partition rule

If the mask audit confirms that region P2 frequently receives odd-sided
blocks and the matched-horizon gap remains, implement a controlled L4
partition ablation. Define and test a partition rule with consistent
region-count and membership semantics across even and odd block shapes.
Compare it with the current parity-dependent rule while holding the mask
draws, P2 objective weights, all other layers, training profile, seed, and
training horizon fixed.

Before full training, unit-test hand-computable shapes (including 2 by 14,
5 by 6, 6 by 5, 6 by 6, and 7 by 5). Verify full target coverage, no visible
tokens, stable region counts, correct pooling, and matching predictor/target
shapes. This is a new method ablation and requires a distinct run ID; do not
overwrite the current P2 result.

**Deliverable:** synthetic geometry tests followed, only if they pass review,
by a targeted pretraining and downstream comparison.

### Step 8 — Directly test time/frequency specialization

Design paired evaluation perturbations on the same held-out examples. For
each task and exit, compare the unmodified input with controlled variants
such as temporal-order shuffling and frequency-bin shuffling, using
label-preserving transformations only where the task definition permits.
Document which transformations preserve labels; do not assume they do.
Measure the change in task metric by exit, with confidence intervals from
paired resampling of examples or independent groups.

This tests sensitivity to time and frequency structure more directly than
assigning a semantic category to each task. Add representation-similarity
analyses only as supporting evidence, not as a substitute for task results.

**Deliverable:** perturbation protocol, label-preservation rationale, and
per-task/per-exit sensitivity figure. Use group-held-out data where available.

### Step 9 — Run confirmatory pretraining seeds

If the controlled checks support a real P1/P2 difference worth claiming,
repeat pretraining with at least three seeds for the distinct geometries.
Use matched seed sets, data order policy, 60-epoch horizon, precision,
attention, effective batch size, loader profile, downstream splits, head
seeds, and validation selection. Log realized mask statistics per run.

Treat multi-block and frequency as one geometry family unless a deliberately
different profile is introduced. Add the fixed-75% random condition only
when its configuration, mask audit, and complete B0/P1/P2 run sequence are
ready. Keep P2-Reverse exploratory unless it can be reproduced under the
same code, seeds, splits, and training protocol.

**Deliverable:** mean and standard deviation across pretraining seeds,
paired per-task differences, and masking-by-method interaction plots. Keep
NR error separate from accuracy aggregates.

### Step 10 — Update manuscript-facing conclusions

Regenerate tables and figures from the audited source artifacts. Show:

- per-task performance and fixed-exit curves;
- selected exits chosen from validation;
- inference FLOPs and synchronized latency in separate panels;
- pretraining horizon/profile and mask geometry;
- uncertainty across seeds when available;
- which evaluations are sample-wise, group-held-out, or
  scenario-held-out;
- imported or otherwise incomplete-provenance results as exploratory.

State a broad P2 superiority claim only if it is consistent across matched
conditions and supported by repeated pretraining seeds. If the result remains
task- or mask-specific, report that interaction directly. If differences are
within uncertainty, report that the methods are comparable under the tested
protocol rather than implying an unobserved separation.

## Execution order and gates

| Order | Work | GPU required | Proceed when |
|---:|---|---|---|
| 1 | Result inventory and provenance ledger | No | Every plotted value maps to a run, checkpoint, split, and profile |
| 2 | Realized mask and L4 region audit | No | Coverage, context, overlap, and target-region distributions are summarized |
| 3 | Split independence audit | No | Generalization scope and known group overlap are documented |
| 4 | Region P1/P2 epoch-44 frozen-head comparison | Yes | Same checkpoint horizon, split, head settings, and validation-only selection are verified |
| 5 | Repeated downstream head seeds | Yes | Paired head-seed variability is reported |
| 6 | Region P2 continuation to epoch 59 | Yes | Step 4 suggests horizon may explain the gap |
| 7 | L4 partition ablation | Yes | Steps 2 and 4 support the geometry hypothesis and synthetic tests pass |
| 8 | Time/frequency perturbation evaluation | Usually | Label-preserving perturbations and evaluation split are defined |
| 9 | Multi-seed pretraining | Yes | Earlier checks identify a claim worth confirming and resources are approved |
| 10 | Final report refresh | No | Provenance and uncertainty are represented accurately |

GPU jobs remain individual runs with an explicit configuration, artifact
path, expected duration, and completion check. No step in this document
authorizes an automatic training launch.

## Interpretation guardrails

- A higher P2 aggregate from one seed does not establish a robust P2 effect.
- A difference in validation-selected exits can reflect head noise or
  validation selection, so always show fixed-exit results too.
- Test results must not be used to choose the checkpoint, exit, mask rule, or
  stopping point.
- Masking changes can alter target coverage and visible context; compare the
  realized distributions before attributing outcomes to semantic mask names.
- A task called “time-related” or “frequency-related” is not evidence of
  layer specialization. Use controlled perturbations and suitable grouped
  splits.
- A lack of a large performance gap is a valid result. It may indicate that
  the shared L8 anchor, frozen-head protocol, task difficulty, or current
  data scale limits the measurable effect of the auxiliary hierarchy.
