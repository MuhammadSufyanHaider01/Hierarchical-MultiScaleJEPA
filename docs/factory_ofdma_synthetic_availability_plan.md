# Factory-OFDMA synthetic availability study

## Decision and scope

Use the open Sionna factory/OFDMA anomaly scenario as the primary setting:
3.75 GHz, 20 MHz, a 40 x 40 x 5 m concrete hall, six metal production-line
obstacles, 3--10 legitimate transmitters, and an optional jammer. The published
simulation framework and its arXiv paper are the source for those baseline
values. The study is an independent synthetic-domain stress test, not an
attempt to recreate the physically collected LatentWave data.

Use S-ICDF only as a secondary design reference for coherent multi-element
capture and receiver motion. Its GNSS-L1 carrier and signal families are not
copied into the factory-5G baseline. The 16-element array, mobile-UE route,
availability schedule, task-label adapters, and position target are explicit
study extensions; the array uses half-wavelength spacing at 3.75 GHz rather
than copying S-ICDF's spacing at a different carrier.


The first hand-built v1 pilots did not implement all values in the released paper configuration (notably the 3--10 transmitter draw, jammer probability, impairment/noise settings, directional jammer pattern, pilot/resource allocation behavior, and exact paper jammer classes). Their full-information frozen-head anchors failed for both methods on NR positioning, interference, and jamming. Their HDF5 payloads have been removed; split manifests, provenance, and failed-anchor records remain as an audit trail. These pilots must not be used as the paper-config study results.


The current versioned registry is `configs/sionna/factory_ofdma_paper_v3.yml`.
It separates published parameters from study extensions. The P0/P2 pilot
anchors failed, so no matched-task transfer claim is permitted; formal
generation is permitted only as the explicitly labeled OOD stress test
requested for the supplementary study.
The released radio configuration at https://github.com/akdd11/ofdma-spectrum-anomalies-simulation/blob/main/src/conf/dataset_generation.yaml specifies 3--10 legitimate transmitters, 0.5 jammer probability, 10 dBm legitimate transmit power, 0--30 dBm jammer power in 5 dB steps, 20 MHz bandwidth, 15 kHz subcarrier spacing, five 14-symbol slots, 2048 FFT, 144-sample cyclic prefix, a 10 dB sensing-unit noise figure, 15 dB additional impairments, 100,000 rays per source, reflection depth four, and diffraction enabled. The 21 fixed sensing-unit coordinates are in a separate file in that repository. Its simulation environment pins Sionna 1.2.2; our adapter uses 2.1.x and is therefore a versioned implementation, not a bit-for-bit reproduction.

The added 16-element receiver array, receiver-coordinate target, nested
antenna/time views, and adapters to our frozen task outputs are extensions.
The formal generator uses one independent receiver/signal realization per
sample rather than repeated snapshots of a route. The published 21 fixed
sensing-unit locations are not used as 16 antenna elements. Sionna 2.1.x is used instead of the released repository's Sionna 1.2.2
environment. Legitimate sources are traced with isotropic patterns; jammer
sources use a separate TR 38.901 directional-pattern trace. Local interference label meanings and equivalence between
the paper's five jammer classes and the frozen six-class jamming head are
unverified. Reports must identify results as assumed class-index agreement or
out-of-domain stress-test metrics, not recovered source-task accuracy.

## Tasks and models

Evaluate only the frozen P0 LatentWave+EE and P2 Hierarchical MultiScaleJEPA
checkpoints and their matching L2/L4/L6/L8 heads for:

- `nrpos-outdoor`: predict the generated receiver's 3-D position in the
  factory scene. This is derived indoor localization, not cellular NR
  positioning and not the original NR-positioning dataset.
- `interference`: two synthetic operating conditions, preserving output IDs
  0/1. The local source meanings for those IDs are undocumented, so results
  must be called class-index transfer unless the mapping is recovered.
- `jamming`: six explicitly generated signal modes aligned by index with the
  frozen head's output IDs: DME, narrow-band, no-jam, AM, chirp, and FM. Their
  mathematical definitions and parameter ranges must be locked in the task
  adapter; they are not claimed identical to the measured waveforms.

The frozen models accept a variable channel dimension by embedding each
channel as a separate image and pooling the resulting tokens. Nonetheless,
channel count and feature scaling differ from head-training distributions;
these results measure zero-shot synthetic-domain behavior, not measured-domain
performance. Position metrics must use the synthetic scene's coordinate bounds
and report meters in that scene, not the original source dataset's nominal
bounds.

## Availability protocol

Generate one maximum-information feature tensor per realization, then derive
both sweeps as read-only nested views:

- Antennas: `C={1,2,4,6,8,10,12,16}` from a fixed 16-element aperture.
- Antenna sweep fixes `T_p=14`; time sweep fixes `C=4` for NR positioning and `C=1` for interference and jamming. Each evaluated encoder condition is capped at 3,136 tokens.
- Time: `T_p={2,4,6,8,10,12,14,18,22,28,36,44,56}` leading patch columns,
  where each patch is 16 feature columns.
- Master shape: `[16,224,896]`; no condition-specific resizing or
  normalization. Do not evaluate the combined `C=16,T_p=56` condition.
- Antenna and time conditions share identical realization IDs and labels.
- Split by realization ID; the current plugin emits one observation per
  independently seeded receiver/signal realization, not route snapshots.
  Preserve seeds, scene draws, array element indices, waveform parameters,
  simulator/config/code hashes, and source checkpoint provenance.

For every sweep, P0 and P2 use the same master and manifest. Selection uses
only the selection/validation split and the existing earliest-exit-within-5%-of
best rule. Measure FLOPs and synchronized batch-1 latency for the selected exit.
Held-out test evaluation is a separate command after artifact review, and it
never reruns selection.

## Execution sequence

1. Check the frozen source packets and head files; verify all three task output
  widths, feature shapes, regression coordinates, and transforms. Keep all
  source models/heads and measured results immutable.
2. Implement one reusable Sionna RT factory scene and explicit transmitter,
  receiver-array, OFDMA, jammer, and mobility configuration. Confirm the
  published configuration values against the paper/released code; label every
  deviation as a study extension.
3. Implement task adapters and verify exact tensor shape, finite feature
  values, labels, normalized coordinates, sample IDs, and deterministic seeds.
  Do not train or fine-tune any model/head.
4. Implement a new paper-config-aligned scenario version and its physical transmitter, jammer, antenna-pattern, pilot, noise/impairment, and mobility adapters. Do not overwrite v1 artifacts. Generate a small shared pilot (first 32, then 256--512 after profiling).
  Run both P0/P2 antenna and time full-information anchors on the same pilot.
  Review class balance, feature statistics, model outputs, route leakage, and
  GPU/disk budgets before locking formal counts.
5. Lock counts from pilot-only precision. Generate one immutable formal master
  per task and realization-ID split manifest. Run validation selection for
  each task/sweep/method, review packets, and only then run held-out tests.
6. Save per-method packets and individual method plots, plus one paired
  comparison plot at each task/sweep root. Keep the result tree mirrored under
  `experiments/original_data/` and `experiments/synthetic_data/`.

## Output tree

```text
experiments/
  original_data/
    vary_antenna/<task>/{latentwave_ee,hierarchical}/...
    vary_time_samples/<task>/{latentwave_ee,hierarchical}/...
  synthetic_data/
    vary_antenna/<task>/{latentwave_ee,hierarchical}/...
    vary_time_samples/<task>/{latentwave_ee,hierarchical}/...
```

Each task root will also hold the single paired plot (PNG/PDF), comparison CSV,
Markdown table, and README with dataset, generator, model, and split provenance.
Synthetic outputs are to remain clearly separated from the original measured
results in every report and figure caption.

## Paper-v3 pilot outcomes and locked counts

Each task has a shared 256-realization pilot. Pilot anchors evaluate the full-information L8 condition on the calibration IDs only; no validation selector or test split was used. All six source-task improvement gates failed, so the results below are strictly synthetic-domain stress-test measurements.

| Task | Sweep | P0 L8 metric | P2 L8 metric | Gate |
|---|---|---:|---:|---|
| NR positioning proxy | Antenna | 17.590 m | 20.100 m | Failed; centroid baseline 14.244 m |
| NR positioning proxy | Time | 17.267 m | 20.285 m | Failed; centroid baseline 14.244 m |
| Interference proxy | Antenna | 0.481 macro accuracy | 0.500 | Failed; chance 0.500 |
| Interference proxy | Time | 0.346 macro accuracy | 0.442 | Failed; chance 0.500 |
| Jamming proxy | Antenna | 0.171 macro accuracy | 0.168 | Failed; chance 0.167 |
| Jamming proxy | Time | 0.223 macro accuracy | 0.130 | Failed; chance 0.167 |

Pilot-only precision planning locked the formal totals before selection: NR 7,797 (cap 11,628), interference 1,752 (cap 14,000), and jamming 967 (cap 7,500). These counts project the requested 95% test-interval half-widths under the planner’s inverse-square-root precision assumption; the eventual test intervals must still be reported. The low full-information results are an explicit warning that availability curves may primarily reflect OOD failure rather than graceful information degradation.

The NR pilot was generated before a metadata-only correction changed its per-sample jammer-pattern description from an isotropic approximation to the separate TR 38.901 jammer trace actually used by the simulator. The pilot signal generation itself was unchanged. The correction is reflected in the v3 generator and all subsequent formal records; retain this caveat with the NR pilot provenance.

## Calibration gates and stress-test exception

The formal source-task transfer gates did not pass on the pilot. This blocks
claims that synthetic performance represents measured NR positioning,
interference, or jamming, and it blocks treating the synthetic metric as
source-task accuracy. Per the user's revised objective, the paper-config data
may nevertheless be generated and evaluated as a zero-shot synthetic-domain
stress test. The explicit `--allow-uncalibrated-stress-test` selection flag is
required whenever a formal calibration anchor fails. Such output must retain
the failed anchor status and scope warning. The following checks remain
acceptance requirements for the stress-test implementation itself:

1. The Sionna RT implementation reproduces the chosen scene, OFDMA timing, and
   multi-element channel shape in deterministic tests.
2. Every classification adapter freezes one physically explicit label map;
   the interference result is not misrepresented as having recovered source
   class meanings.
3. The position adapter's coordinate bounds and model-head denormalization are
   verified. For this synthetic task use factory-scene meters.
4. The full-information controls can be evaluated by both P0 and P2 without
   shape errors, non-finite values, gradients, or checkpoint changes.
5. A paired pilot shows usable class support and storage/compute estimates are
   acceptable; the frozen-head anchor may still fail, in which case all
   downstream results remain stress-test-only.

If frozen-head semantics are incompatible with a generated label ontology, do
not silently change class labels, retrain heads, or claim task accuracy. Record
that task as blocked for frozen-head transfer and surface the mismatch.

## Rerun after payload cleanup

The earlier large HDF5 masters were removed; the small manifests, hashes,
pilot-anchor records, and locked sample-count plans remain as provenance. The
old `factory_ofdma_paper_v3` output paths are therefore historical and must not
be overwritten. Recreate the formal masters under a fresh write-once root, for
example `data/synthetic/factory_ofdma_paper_v3_regen_20260924/`, and keep
selection/anchor records under the correspondingly versioned
`runs/synthetic_availability/` root. Reuse only the already locked counts
(NR 7,797; interference 1,752; jamming 967); do not re-estimate counts from
selection or test data. Recompute formal calibration anchors on each regenerated
master because the old anchors bind to deleted master hashes; their classification
gate also used ordinary accuracy rather than the planned macro-accuracy lower
bound, so those old pass/fail decisions must not be reused.

For each task, run the formal generator with its matching
`runs/synthetic_availability/factory_ofdma_paper_v3_precision/<task>/sample_count_plan.json`,
the precision root as `--pilot-anchors-root`, and the fresh `--data-root`.
Then run `factory_ofdma_paper_v3_exits.py anchor` for antenna and time. Since
the pilot gates failed, run selection separately for P0 and P2 with the fresh
matching formal anchor and the explicit `--allow-uncalibrated-stress-test`
flag. Do not run test evaluation until both selection packets for that task
and sweep have been manually reviewed. Finally, call
`report_factory_ofdma_paper_v2.py` with the paired P0/P2 selection packets to
write method-specific packets/plots and the single two-line comparison figure
under `experiments/synthetic_data/<sweep>/<task>/`.
