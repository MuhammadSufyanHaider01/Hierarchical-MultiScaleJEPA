# Interference (OFDMA Jammer-Presence Proxy) — strict best-layer time diagnostic

Synthetic-data validation/selection split results. Strict argbest is selected per condition from L2/L4/L6/L8; the displayed metric is from the same selection split used to select the exit and is therefore descriptive, not an unbiased held-out estimate. The established 5%-rule selected exit is included separately. Test data were not used.

| Level | Method | Head strategy | Strict best exit / metric | Official 5% exit / metric |
|---:|---|---|---:|---:|
| 2 | LatentWave + EE | Condition-specific heads | L4 / 0.661333 | L2 / 0.632 |
| 2 | LatentWave + EE | Full-information-trained heads | L4 / 0.636 | L2 / 0.609333 |
| 2 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.66 | L2 / 0.654667 |
| 2 | Proposed hierarchical approach | Full-information-trained heads | L2 / 0.617333 | L2 / 0.617333 |
| 4 | LatentWave + EE | Condition-specific heads | L6 / 0.666667 | L2 / 0.656 |
| 4 | LatentWave + EE | Full-information-trained heads | L4 / 0.64 | L4 / 0.64 |
| 4 | Proposed hierarchical approach | Condition-specific heads | L2 / 0.666667 | L2 / 0.666667 |
| 4 | Proposed hierarchical approach | Full-information-trained heads | L2 / 0.64 | L2 / 0.64 |
| 6 | LatentWave + EE | Condition-specific heads | L6 / 0.658667 | L2 / 0.652 |
| 6 | LatentWave + EE | Full-information-trained heads | L4 / 0.632 | L2 / 0.617333 |
| 6 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.673333 | L2 / 0.669333 |
| 6 | Proposed hierarchical approach | Full-information-trained heads | L8 / 0.632 | L2 / 0.629333 |
| 8 | LatentWave + EE | Condition-specific heads | L6 / 0.669333 | L2 / 0.646667 |
| 8 | LatentWave + EE | Full-information-trained heads | L4 / 0.636 | L2 / 0.614667 |
| 8 | Proposed hierarchical approach | Condition-specific heads | L2 / 0.668 | L2 / 0.668 |
| 8 | Proposed hierarchical approach | Full-information-trained heads | L8 / 0.634667 | L2 / 0.618667 |
| 10 | LatentWave + EE | Condition-specific heads | L8 / 0.664 | L2 / 0.650667 |
| 10 | LatentWave + EE | Full-information-trained heads | L6 / 0.632 | L2 / 0.617333 |
| 10 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.677333 | L2 / 0.662667 |
| 10 | Proposed hierarchical approach | Full-information-trained heads | L8 / 0.64 | L2 / 0.609333 |
| 12 | LatentWave + EE | Condition-specific heads | L8 / 0.68 | L2 / 0.658667 |
| 12 | LatentWave + EE | Full-information-trained heads | L4 / 0.652 | L2 / 0.629333 |
| 12 | Proposed hierarchical approach | Condition-specific heads | L6 / 0.696 | L2 / 0.669333 |
| 12 | Proposed hierarchical approach | Full-information-trained heads | L8 / 0.653333 | L4 / 0.645333 |
| 14 | LatentWave + EE | Condition-specific heads | L2 / 0.690667 | L2 / 0.690667 |
| 14 | LatentWave + EE | Full-information-trained heads | L4 / 0.665333 | L2 / 0.658667 |
| 14 | Proposed hierarchical approach | Condition-specific heads | L6 / 0.698667 | L2 / 0.674667 |
| 14 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.661333 | L4 / 0.661333 |
| 18 | LatentWave + EE | Condition-specific heads | L8 / 0.697333 | L2 / 0.672 |
| 18 | LatentWave + EE | Full-information-trained heads | L4 / 0.672 | L2 / 0.66 |
| 18 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.702667 | L2 / 0.684 |
| 18 | Proposed hierarchical approach | Full-information-trained heads | L8 / 0.650667 | L2 / 0.637333 |
| 22 | LatentWave + EE | Condition-specific heads | L4 / 0.706667 | L2 / 0.692 |
| 22 | LatentWave + EE | Full-information-trained heads | L4 / 0.681333 | L2 / 0.669333 |
| 22 | Proposed hierarchical approach | Condition-specific heads | L2 / 0.704 | L2 / 0.704 |
| 22 | Proposed hierarchical approach | Full-information-trained heads | L8 / 0.670667 | L2 / 0.649333 |
| 28 | LatentWave + EE | Condition-specific heads | L4 / 0.737333 | L2 / 0.726667 |
| 28 | LatentWave + EE | Full-information-trained heads | L8 / 0.725333 | L2 / 0.714667 |
| 28 | Proposed hierarchical approach | Condition-specific heads | L8 / 0.734667 | L2 / 0.725333 |
| 28 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.725333 | L2 / 0.710667 |
| 36 | LatentWave + EE | Condition-specific heads | L4 / 0.749333 | L2 / 0.734667 |
| 36 | LatentWave + EE | Full-information-trained heads | L8 / 0.742667 | L2 / 0.734667 |
| 36 | Proposed hierarchical approach | Condition-specific heads | L8 / 0.748 | L2 / 0.736 |
| 36 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.74 | L2 / 0.721333 |
| 44 | LatentWave + EE | Condition-specific heads | L8 / 0.744 | L2 / 0.729333 |
| 44 | LatentWave + EE | Full-information-trained heads | L8 / 0.744 | L2 / 0.730667 |
| 44 | Proposed hierarchical approach | Condition-specific heads | L6 / 0.745333 | L2 / 0.741333 |
| 44 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.745333 | L2 / 0.737333 |
| 56 | LatentWave + EE | Condition-specific heads | L8 / 0.746667 | L2 / 0.728 |
| 56 | LatentWave + EE | Full-information-trained heads | L8 / 0.746667 | L2 / 0.728 |
| 56 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.745333 | L2 / 0.725333 |
| 56 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.745333 | L2 / 0.725333 |
