# Imported P2-Reverse artifacts

This directory contains only the P2-Reverse result packets imported from
`p2_reverse_imports_90c4793.tar.gz` (SHA-256
`83aeed0d9292cf89be5f1ed63e3b897bfce5dee3b41a050cba2bdf68c5c38ad6`).
The claimed source revision is `90c4793ac9ce6a174bb39138c7fc324e742a9147`.

Included per masking condition are downstream held-out metrics, experiment
metadata, summaries, and—where supplied—batch-1 CUDA benchmark packets. No
checkpoints are imported. The M-block benchmark packet was separately attached
and validated (SHA-256 `078d2715eed558cc2b6e7262e292a88517c01b07f261f6e2709dff40f61dbee7`).

Each packet references a persisted split manifest on the source machine, but
the split-index JSON itself was not included in the archive. The report
generator therefore verifies the task suite and test-evaluation protocol, but
marks exact split-index identity as not locally re-verified.
