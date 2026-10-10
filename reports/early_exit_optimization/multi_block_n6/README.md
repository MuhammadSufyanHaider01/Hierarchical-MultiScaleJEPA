# Two-stage early-exit optimization

P0/P2 on `multi_block_n6`, downstream head seeds 0–6. Validation selects the reported exit and test evaluates it afterward. Classification gains are new minus current; positioning gains are current error minus new error. Aggregates use mean and sample SD across seven seeds. Fusion-Teacher KD deploys the single-representation student; its teacher is training-only.
