# P0/P1/P2 checkpoint bundle

The local archive is retained at
`pretraining_checkpoints/archive/p0_p1_p2_checkpoints.tar.zst`. It is not
committed because it exceeds GitHub's 100 MB per-file limit; Git LFS is not
configured in this repository. Its container checksum is recorded in
`ARCHIVE_SHA256SUMS`, while `SHA256SUMS` verifies the checkpoint files inside
the archive.

This archive contains the available seed-0 pretraining checkpoints for the four masking conditions. There are 12 logical method/condition slots and 11 unique checkpoint files: the M-block report explicitly reuses the frequency P1 checkpoint, so it is included once and mapped to both slots below.

| Masking | Method | Archive path | Source note |
|---|---|---|---|
| Frequency Nₘ=6 | P0 | `checkpoints/frequency_n6/P0/last.ckpt` | Formal frequency-mask run |
| Frequency Nₘ=6 | P1 | `checkpoints/frequency_n6/P1/last.ckpt` | Formal frequency-mask run; also used for M-block P1 per report manifest |
| Frequency Nₘ=6 | P2 | `checkpoints/frequency_n6/P2/last.ckpt` | Formal frequency-mask run |
| Time Nₘ=6 | P0 | `checkpoints/time_n6/P0/last.ckpt` | Formal time-mask run |
| Time Nₘ=6 | P1 | `checkpoints/time_n6/P1/last.ckpt` | Formal time-mask run |
| Time Nₘ=6 | P2 | `checkpoints/time_n6/P2/last.ckpt` | Formal time-mask run |
| Region Nₘ=6 | P0 | `checkpoints/region_n6/P0/last.ckpt` | Formal region-mask run; source is the sibling `MultiScaleJEPA` checkout |
| Region Nₘ=6 | P1 | `checkpoints/region_n6/P1/last.ckpt` | Formal region-mask run |
| Region Nₘ=6 | P2 | `checkpoints/region_n6/P2/last.ckpt` | Formal region-mask run |
| M-block Nₘ=6 | P0 | `checkpoints/multi_block_n6/P0/last.ckpt` | Recovered from `phase5_formal`; its run metadata records a 240-epoch maximum, while saved checkpoints reach epoch 64. This is not the missing report-path file, so verify compatibility before using it as an exact report reproduction checkpoint. |
| M-block Nₘ=6 | P1 | Shared file: `checkpoints/frequency_n6/P1/last.ckpt` | The M-block report manifest points to the frequency P1 pretraining checkpoint. |
| M-block Nₘ=6 | P2 | `checkpoints/multi_block_n6/P2/last.ckpt` | Recovered from `phase5_formal`; metadata records M-block masking and 60 epochs. |

All checkpoint files are named `last.ckpt` in the archive for convenience. See `SHA256SUMS` for integrity verification. The archive contains checkpoints only; no downstream heads or inference benchmark artifacts are included.
