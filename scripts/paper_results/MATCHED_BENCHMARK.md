# Matched exit benchmark

The reporting build never runs a benchmark. This workflow requires a CUDA GPU,
repository dependencies, a clean committed checkout, the exact P0 and P2
multi-block pretrained checkpoints, and the original seed-0 evaluation packets.
It measures 96 fixed encoder/head/task/exit combinations without training.

Obtain the four original `exit_metrics.json` files listed in the checked-in
`reports/early_exit_optimization/multi_block_n6/residual_dynamic_fusion_multiseed/manifest.json`
for head seed 0. Place copies in a private input directory with these names:

- `p0_current.json` and `p2_current.json`
- `p0_single_gelu.json` and `p2_single_gelu.json`

The command compares each file's SHA-256 to the study manifest and checks that
all four independently recorded task shapes and output dimensions agree. It
also checks each pretrained checkpoint SHA-256. No private dataset values are
needed: random full-shape inputs are generated from the authenticated specs.
The original packets are required because the summaries in this checkout do
not independently establish all six shapes and dimensions.

On the GPU machine, from a clean committed checkout, run:

```sh
python scripts/matched_exit_benchmark.py \
  --p0-checkpoint /private/P0/last.ckpt \
  --p2-checkpoint /private/P2/last.ckpt \
  --packets-dir /private/paper-packets \
  --output /private/matched-exit-benchmark.json
python scripts/paper_results/build_all.py \
  --matched-benchmark /private/matched-exit-benchmark.json
```

Defaults are one GPU, batch 1, float32, 50 warm-up and 200 measured
iterations. GPU timing is synchronized. The output records the model and
runtime settings and the FLOP operators counted; unregistered operations are
not included. Do not compare latency measured under different protocols.

The default heads use initialized weights. For verified trained fusion heads,
pass `--head-root /private/fusion-heads` and place the seed-0 files at
`<root>/<method>/<task>/L<exit>.pt`; their hashes and identities must match
the authenticated fusion packets. Historical baseline packets do not provide
head hashes, so baseline heads remain initialized: a file without a verified
identity must not be claimed as the original evaluated head. The architecture
has a fixed execution path, and the script checks the actual block and fusion
inputs at every exit. Trained weights are not needed for architecture FLOPs,
but latency remains specific to the recorded GPU and execution setup.

The report rejects incomplete, inconsistent, duplicate, dirty-revision, or
mismatched measurements. It produces fixed-exit benchmark data and a
validation-selected cost table only after all 96 entries validate. Selected
cost is averaged over seven head runs and is not per-sample dynamic stopping.
Existing baseline cost reports remain provisional until a compatible benchmark
is provided. Head-seed variation is conditional on one pretrained checkpoint;
matched-capacity controls and independent pretraining seeds remain future work.
