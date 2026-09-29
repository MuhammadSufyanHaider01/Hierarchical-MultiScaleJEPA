# NR Positioning (Factory-Scene Localization Proxy) — strict best-layer antenna diagnostic

Synthetic-data validation/selection split results. Strict argbest is selected per condition from L2/L4/L6/L8; the displayed metric is from the same selection split used to select the exit and is therefore descriptive, not an unbiased held-out estimate. The established 5%-rule selected exit is included separately. Test data were not used.

| Level | Method | Head strategy | Strict best exit / metric | Official 5% exit / metric |
|---:|---|---|---:|---:|
| 1 | LatentWave + EE | Condition-specific heads | L8 / 8.86009 | L2 / 9.28843 |
| 1 | LatentWave + EE | Full-information-trained heads | L8 / 9.32376 | L2 / 9.78313 |
| 1 | Proposed hierarchical approach | Condition-specific heads | L6 / 8.97277 | L4 / 8.97339 |
| 1 | Proposed hierarchical approach | Full-information-trained heads | L6 / 9.49621 | L2 / 9.69829 |
| 2 | LatentWave + EE | Condition-specific heads | L6 / 8.42407 | L4 / 8.59494 |
| 2 | LatentWave + EE | Full-information-trained heads | L6 / 8.42437 | L4 / 8.64068 |
| 2 | Proposed hierarchical approach | Condition-specific heads | L6 / 8.59898 | L2 / 8.93495 |
| 2 | Proposed hierarchical approach | Full-information-trained heads | L6 / 8.79152 | L2 / 9.13176 |
| 4 | LatentWave + EE | Condition-specific heads | L6 / 7.84083 | L4 / 8.00901 |
| 4 | LatentWave + EE | Full-information-trained heads | L6 / 7.79701 | L4 / 7.95078 |
| 4 | Proposed hierarchical approach | Condition-specific heads | L6 / 8.0848 | L4 / 8.13196 |
| 4 | Proposed hierarchical approach | Full-information-trained heads | L6 / 8.09527 | L2 / 8.38973 |
| 6 | LatentWave + EE | Condition-specific heads | L6 / 7.70166 | L4 / 7.82179 |
| 6 | LatentWave + EE | Full-information-trained heads | L6 / 7.62639 | L4 / 7.79615 |
| 6 | Proposed hierarchical approach | Condition-specific heads | L6 / 7.93896 | L2 / 8.27446 |
| 6 | Proposed hierarchical approach | Full-information-trained heads | L6 / 7.94844 | L2 / 8.28184 |
| 8 | LatentWave + EE | Condition-specific heads | L6 / 7.55529 | L4 / 7.63189 |
| 8 | LatentWave + EE | Full-information-trained heads | L6 / 7.46466 | L4 / 7.67431 |
| 8 | Proposed hierarchical approach | Condition-specific heads | L6 / 7.71195 | L4 / 7.94973 |
| 8 | Proposed hierarchical approach | Full-information-trained heads | L6 / 7.82667 | L2 / 8.15063 |
| 10 | LatentWave + EE | Condition-specific heads | L6 / 7.54907 | L4 / 7.68222 |
| 10 | LatentWave + EE | Full-information-trained heads | L6 / 7.46973 | L4 / 7.71734 |
| 10 | Proposed hierarchical approach | Condition-specific heads | L6 / 7.78748 | L2 / 8.10487 |
| 10 | Proposed hierarchical approach | Full-information-trained heads | L6 / 7.7954 | L2 / 8.1047 |
| 12 | LatentWave + EE | Condition-specific heads | L6 / 7.54333 | L4 / 7.6958 |
| 12 | LatentWave + EE | Full-information-trained heads | L6 / 7.46548 | L4 / 7.71892 |
| 12 | Proposed hierarchical approach | Condition-specific heads | L6 / 7.80829 | L2 / 8.07115 |
| 12 | Proposed hierarchical approach | Full-information-trained heads | L6 / 7.8247 | L2 / 8.07975 |
| 16 | LatentWave + EE | Condition-specific heads | L6 / 7.42561 | L4 / 7.6564 |
| 16 | LatentWave + EE | Full-information-trained heads | L6 / 7.42561 | L4 / 7.6564 |
| 16 | Proposed hierarchical approach | Condition-specific heads | L6 / 7.74381 | L2 / 7.98023 |
| 16 | Proposed hierarchical approach | Full-information-trained heads | L6 / 7.74381 | L2 / 7.98023 |
