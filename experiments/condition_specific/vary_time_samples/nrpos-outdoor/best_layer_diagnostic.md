# NR Positioning (Factory-Scene Localization Proxy) — strict best-layer time diagnostic

Synthetic-data validation/selection split results. Strict argbest is selected per condition from L2/L4/L6/L8; the displayed metric is from the same selection split used to select the exit and is therefore descriptive, not an unbiased held-out estimate. The established 5%-rule selected exit is included separately. Test data were not used.

| Level | Method | Head strategy | Strict best exit / metric | Official 5% exit / metric |
|---:|---|---|---:|---:|
| 2 | LatentWave + EE | Condition-specific heads | L8 / 9.13258 | L4 / 9.25963 |
| 2 | LatentWave + EE | Full-information-trained heads | L6 / 11.5974 | L6 / 11.5974 |
| 2 | Proposed hierarchical approach | Condition-specific heads | L6 / 9.2428 | L2 / 9.66103 |
| 2 | Proposed hierarchical approach | Full-information-trained heads | L4 / 12.2915 | L4 / 12.2915 |
| 4 | LatentWave + EE | Condition-specific heads | L6 / 8.91971 | L4 / 9.1594 |
| 4 | LatentWave + EE | Full-information-trained heads | L6 / 11.3001 | L6 / 11.3001 |
| 4 | Proposed hierarchical approach | Condition-specific heads | L6 / 9.07371 | L2 / 9.51497 |
| 4 | Proposed hierarchical approach | Full-information-trained heads | L4 / 11.2548 | L4 / 11.2548 |
| 6 | LatentWave + EE | Condition-specific heads | L6 / 8.9791 | L2 / 9.31195 |
| 6 | LatentWave + EE | Full-information-trained heads | L6 / 11.1861 | L6 / 11.1861 |
| 6 | Proposed hierarchical approach | Condition-specific heads | L6 / 9.01887 | L2 / 9.42832 |
| 6 | Proposed hierarchical approach | Full-information-trained heads | L4 / 11.001 | L4 / 11.001 |
| 8 | LatentWave + EE | Condition-specific heads | L6 / 8.86434 | L2 / 9.26655 |
| 8 | LatentWave + EE | Full-information-trained heads | L6 / 11 | L6 / 11 |
| 8 | Proposed hierarchical approach | Condition-specific heads | L4 / 8.95605 | L2 / 9.33965 |
| 8 | Proposed hierarchical approach | Full-information-trained heads | L4 / 11.0917 | L4 / 11.0917 |
| 10 | LatentWave + EE | Condition-specific heads | L6 / 8.83736 | L4 / 8.98816 |
| 10 | LatentWave + EE | Full-information-trained heads | L6 / 10.8253 | L6 / 10.8253 |
| 10 | Proposed hierarchical approach | Condition-specific heads | L4 / 8.92225 | L2 / 9.33826 |
| 10 | Proposed hierarchical approach | Full-information-trained heads | L6 / 11.4115 | L2 / 11.6976 |
| 12 | LatentWave + EE | Condition-specific heads | L8 / 8.03688 | L4 / 8.36099 |
| 12 | LatentWave + EE | Full-information-trained heads | L6 / 9.20593 | L6 / 9.20593 |
| 12 | Proposed hierarchical approach | Condition-specific heads | L4 / 8.38179 | L4 / 8.38179 |
| 12 | Proposed hierarchical approach | Full-information-trained heads | L8 / 9.96727 | L2 / 10.3197 |
| 14 | LatentWave + EE | Condition-specific heads | L6 / 7.89624 | L4 / 7.99843 |
| 14 | LatentWave + EE | Full-information-trained heads | L6 / 8.79329 | L6 / 8.79329 |
| 14 | Proposed hierarchical approach | Condition-specific heads | L6 / 8.09982 | L2 / 8.49813 |
| 14 | Proposed hierarchical approach | Full-information-trained heads | L6 / 9.50662 | L2 / 9.67184 |
| 18 | LatentWave + EE | Condition-specific heads | L6 / 7.68696 | L4 / 7.73516 |
| 18 | LatentWave + EE | Full-information-trained heads | L6 / 8.59669 | L6 / 8.59669 |
| 18 | Proposed hierarchical approach | Condition-specific heads | L4 / 7.81622 | L2 / 8.09143 |
| 18 | Proposed hierarchical approach | Full-information-trained heads | L2 / 9.25746 | L2 / 9.25746 |
| 22 | LatentWave + EE | Condition-specific heads | L4 / 7.59922 | L4 / 7.59922 |
| 22 | LatentWave + EE | Full-information-trained heads | L6 / 8.31812 | L6 / 8.31812 |
| 22 | Proposed hierarchical approach | Condition-specific heads | L4 / 7.78768 | L4 / 7.78768 |
| 22 | Proposed hierarchical approach | Full-information-trained heads | L2 / 9.29789 | L2 / 9.29789 |
| 28 | LatentWave + EE | Condition-specific heads | L6 / 7.21977 | L4 / 7.31617 |
| 28 | LatentWave + EE | Full-information-trained heads | L6 / 7.69751 | L6 / 7.69751 |
| 28 | Proposed hierarchical approach | Condition-specific heads | L6 / 7.4655 | L4 / 7.49806 |
| 28 | Proposed hierarchical approach | Full-information-trained heads | L6 / 7.7191 | L4 / 7.99733 |
| 36 | LatentWave + EE | Condition-specific heads | L8 / 6.86361 | L4 / 7.10834 |
| 36 | LatentWave + EE | Full-information-trained heads | L8 / 7.184 | L6 / 7.22406 |
| 36 | Proposed hierarchical approach | Condition-specific heads | L6 / 6.94433 | L4 / 7.09348 |
| 36 | Proposed hierarchical approach | Full-information-trained heads | L6 / 7.26993 | L4 / 7.33261 |
| 44 | LatentWave + EE | Condition-specific heads | L8 / 6.60295 | L4 / 6.90558 |
| 44 | LatentWave + EE | Full-information-trained heads | L8 / 6.67974 | L6 / 6.85393 |
| 44 | Proposed hierarchical approach | Condition-specific heads | L6 / 6.75785 | L4 / 6.8872 |
| 44 | Proposed hierarchical approach | Full-information-trained heads | L6 / 6.80579 | L4 / 6.88489 |
| 56 | LatentWave + EE | Condition-specific heads | L8 / 6.16421 | L6 / 6.23668 |
| 56 | LatentWave + EE | Full-information-trained heads | L8 / 6.12252 | L6 / 6.26778 |
| 56 | Proposed hierarchical approach | Condition-specific heads | L6 / 6.20987 | L4 / 6.33735 |
| 56 | Proposed hierarchical approach | Full-information-trained heads | L6 / 6.21545 | L4 / 6.34226 |
