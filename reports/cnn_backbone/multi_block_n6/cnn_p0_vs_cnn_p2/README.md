# M-block CNN comparison — seven-head-seed update

The authoritative multi-seed comparison for CNN C0 and C2 is now in [README_multiseed.md](README_multiseed.md). It reports seven independently initialized downstream-head seeds (0–6), reuses the existing seven-seed B0/P0/P2 results, and includes validation-selected held-out performance, fixed-exit results, inference summaries, provenance, and split-comparability caveats.

Open `multiseed_validation_selected.csv`, `multiseed_fixed_exits.csv`, and `multiseed_selected_inference.csv` for the tables; the corresponding `multiseed_*.png` and PDF files are the updated comparison figures.

The original single-seed packet is retained for traceability in [README_seed000.md](README_seed000.md) and its original unprefixed files; do not use those single-seed files as the multi-seed result.
