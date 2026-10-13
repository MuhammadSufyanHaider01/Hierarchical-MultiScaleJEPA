# Two-stage early-exit optimization

P0/P2 on `multi_block_n6`, downstream head seeds 0–6. Validation selects the reported exit and test evaluates it afterward. Classification gains are new minus current; positioning gains are current error minus new error. Aggregates use mean and sample SD across seven seeds. Fusion-Teacher KD deploys the single-representation student; its teacher is training-only.

## Layer-wise algorithm comparison

`algorithm_layerwise_performance.png` and `.pdf` show test performance at L2/L4/L6/L8 for the P2 method (Proposed Hierarchical Multi-Scale JEPA) across all six tasks. Each algorithm is shown as a solid line; points show mean test performance across seven head seeds.
