# Jamming (OFDMA Jammer-Mode Proxy) — strict best-layer antenna diagnostic

Synthetic-data validation/selection split results. Strict argbest is selected per condition from L2/L4/L6/L8; the displayed metric is from the same selection split used to select the exit and is therefore descriptive, not an unbiased held-out estimate. The established 5%-rule selected exit is included separately. Test data were not used.

| Level | Method | Head strategy | Strict best exit / metric | Official 5% exit / metric |
|---:|---|---|---:|---:|
| 1 | LatentWave + EE | Condition-specific heads | L4 / 0.484 | L4 / 0.484 |
| 1 | LatentWave + EE | Full-information-trained heads | L4 / 0.476889 | L2 / 0.468889 |
| 1 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.472 | L2 / 0.450222 |
| 1 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.473778 | L2 / 0.460889 |
| 2 | LatentWave + EE | Condition-specific heads | L4 / 0.488889 | L4 / 0.488889 |
| 2 | LatentWave + EE | Full-information-trained heads | L4 / 0.496889 | L2 / 0.483111 |
| 2 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.492444 | L4 / 0.492444 |
| 2 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.497778 | L2 / 0.485333 |
| 4 | LatentWave + EE | Condition-specific heads | L4 / 0.504 | L2 / 0.481778 |
| 4 | LatentWave + EE | Full-information-trained heads | L4 / 0.503111 | L2 / 0.490222 |
| 4 | Proposed hierarchical approach | Condition-specific heads | L6 / 0.495111 | L2 / 0.488 |
| 4 | Proposed hierarchical approach | Full-information-trained heads | L6 / 0.512 | L4 / 0.507556 |
| 6 | LatentWave + EE | Condition-specific heads | L4 / 0.499556 | L2 / 0.495111 |
| 6 | LatentWave + EE | Full-information-trained heads | L4 / 0.501778 | L2 / 0.480444 |
| 6 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.503111 | L2 / 0.484889 |
| 6 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.514222 | L2 / 0.490667 |
| 8 | LatentWave + EE | Condition-specific heads | L4 / 0.508444 | L4 / 0.508444 |
| 8 | LatentWave + EE | Full-information-trained heads | L8 / 0.503556 | L2 / 0.488444 |
| 8 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.517778 | L2 / 0.492889 |
| 8 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.517778 | L4 / 0.517778 |
| 10 | LatentWave + EE | Condition-specific heads | L6 / 0.515556 | L2 / 0.492 |
| 10 | LatentWave + EE | Full-information-trained heads | L8 / 0.511111 | L2 / 0.492889 |
| 10 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.521333 | L2 / 0.502667 |
| 10 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.534667 | L4 / 0.534667 |
| 12 | LatentWave + EE | Condition-specific heads | L4 / 0.511111 | L2 / 0.487556 |
| 12 | LatentWave + EE | Full-information-trained heads | L8 / 0.506222 | L2 / 0.494667 |
| 12 | Proposed hierarchical approach | Condition-specific heads | L6 / 0.540444 | L4 / 0.516 |
| 12 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.533333 | L4 / 0.533333 |
| 16 | LatentWave + EE | Condition-specific heads | L8 / 0.510667 | L2 / 0.494667 |
| 16 | LatentWave + EE | Full-information-trained heads | L8 / 0.510667 | L2 / 0.494667 |
| 16 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.532444 | L4 / 0.532444 |
| 16 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.532444 | L4 / 0.532444 |
