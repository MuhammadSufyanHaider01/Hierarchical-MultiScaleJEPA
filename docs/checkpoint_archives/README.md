# Local checkpoint archives

Large checkpoint archives are retained locally under
`pretraining_checkpoints/archive/` and are intentionally omitted from Git. Each
archive exceeds GitHub's 100 MB per-file limit, and Git LFS is not configured
for this repository. `ARCHIVE_SHA256SUMS` records the hashes of both archive
containers. The P0/P1/P2 bundle's extracted checkpoint hashes and contents are
documented in [`p0_p1_p2/README.md`](p0_p1_p2/README.md).

The second local archive, `P2_reverse_multi_block_n6_last_checkpoint.zip`,
contains a single `last.ckpt` file for the P2-Reverse M-block run.
