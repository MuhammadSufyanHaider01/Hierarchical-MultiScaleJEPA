# Jamming (OFDMA Jammer-Mode Proxy) — strict best-layer time diagnostic

Synthetic-data validation/selection split results. Strict argbest is selected per condition from L2/L4/L6/L8; the displayed metric is from the same selection split used to select the exit and is therefore descriptive, not an unbiased held-out estimate. The established 5%-rule selected exit is included separately. Test data were not used.

| Level | Method | Head strategy | Strict best exit / metric | Official 5% exit / metric |
|---:|---|---|---:|---:|
| 2 | LatentWave + EE | Condition-specific heads | L2 / 0.436444 | L2 / 0.436444 |
| 2 | LatentWave + EE | Full-information-trained heads | L8 / 0.302222 | L4 / 0.289333 |
| 2 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.427556 | L2 / 0.424889 |
| 2 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.301333 | L2 / 0.286667 |
| 4 | LatentWave + EE | Condition-specific heads | L2 / 0.445333 | L2 / 0.445333 |
| 4 | LatentWave + EE | Full-information-trained heads | L8 / 0.316889 | L8 / 0.316889 |
| 4 | Proposed hierarchical approach | Condition-specific heads | L6 / 0.450667 | L2 / 0.436889 |
| 4 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.318222 | L4 / 0.318222 |
| 6 | LatentWave + EE | Condition-specific heads | L6 / 0.451111 | L4 / 0.436444 |
| 6 | LatentWave + EE | Full-information-trained heads | L8 / 0.331556 | L8 / 0.331556 |
| 6 | Proposed hierarchical approach | Condition-specific heads | L6 / 0.448 | L2 / 0.440889 |
| 6 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.318667 | L2 / 0.316444 |
| 8 | LatentWave + EE | Condition-specific heads | L6 / 0.469778 | L6 / 0.469778 |
| 8 | LatentWave + EE | Full-information-trained heads | L8 / 0.343556 | L2 / 0.326667 |
| 8 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.456 | L2 / 0.434222 |
| 8 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.313778 | L2 / 0.312444 |
| 10 | LatentWave + EE | Condition-specific heads | L6 / 0.461333 | L4 / 0.441778 |
| 10 | LatentWave + EE | Full-information-trained heads | L8 / 0.357333 | L8 / 0.357333 |
| 10 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.457333 | L4 / 0.457333 |
| 10 | Proposed hierarchical approach | Full-information-trained heads | L2 / 0.324889 | L2 / 0.324889 |
| 12 | LatentWave + EE | Condition-specific heads | L6 / 0.474667 | L4 / 0.454222 |
| 12 | LatentWave + EE | Full-information-trained heads | L8 / 0.374667 | L4 / 0.356889 |
| 12 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.459556 | L2 / 0.441778 |
| 12 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.350222 | L2 / 0.341778 |
| 14 | LatentWave + EE | Condition-specific heads | L4 / 0.472444 | L4 / 0.472444 |
| 14 | LatentWave + EE | Full-information-trained heads | L8 / 0.406222 | L2 / 0.392 |
| 14 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.471556 | L4 / 0.471556 |
| 14 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.372 | L2 / 0.355556 |
| 18 | LatentWave + EE | Condition-specific heads | L8 / 0.476444 | L2 / 0.454222 |
| 18 | LatentWave + EE | Full-information-trained heads | L6 / 0.435111 | L2 / 0.420444 |
| 18 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.48 | L2 / 0.46 |
| 18 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.396444 | L4 / 0.396444 |
| 22 | LatentWave + EE | Condition-specific heads | L6 / 0.471111 | L2 / 0.466222 |
| 22 | LatentWave + EE | Full-information-trained heads | L6 / 0.453333 | L4 / 0.443556 |
| 22 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.466667 | L2 / 0.463556 |
| 22 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.421778 | L4 / 0.421778 |
| 28 | LatentWave + EE | Condition-specific heads | L8 / 0.540444 | L4 / 0.533778 |
| 28 | LatentWave + EE | Full-information-trained heads | L4 / 0.539111 | L2 / 0.518667 |
| 28 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.555556 | L4 / 0.555556 |
| 28 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.528889 | L4 / 0.528889 |
| 36 | LatentWave + EE | Condition-specific heads | L6 / 0.558222 | L4 / 0.554222 |
| 36 | LatentWave + EE | Full-information-trained heads | L6 / 0.547111 | L2 / 0.532444 |
| 36 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.559556 | L4 / 0.559556 |
| 36 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.574222 | L4 / 0.574222 |
| 44 | LatentWave + EE | Condition-specific heads | L6 / 0.555111 | L4 / 0.541333 |
| 44 | LatentWave + EE | Full-information-trained heads | L8 / 0.556 | L2 / 0.528889 |
| 44 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.559111 | L4 / 0.559111 |
| 44 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.570222 | L4 / 0.570222 |
| 56 | LatentWave + EE | Condition-specific heads | L6 / 0.549778 | L2 / 0.528 |
| 56 | LatentWave + EE | Full-information-trained heads | L6 / 0.549778 | L2 / 0.528 |
| 56 | Proposed hierarchical approach | Condition-specific heads | L4 / 0.551556 | L4 / 0.551556 |
| 56 | Proposed hierarchical approach | Full-information-trained heads | L4 / 0.551556 | L4 / 0.551556 |
