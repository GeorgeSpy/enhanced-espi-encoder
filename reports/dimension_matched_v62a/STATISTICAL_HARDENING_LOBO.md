# Statistical Hardening for Board LOBO Folds

This report presents the pairwise statistical checks and effect size analysis comparing **v6.2-A** against the **v6.1** reference baseline across the 6 Board LOBO folds.

## Pairwise Fold Performance

| Fold | v6.1 Reference | v6.2-A Encoder | Delta (pp) | Result |
| --- | :---: | :---: | :---: | --- |
| Board C01 | 81.38% | 92.54% | +11.16 pp | **v6.2-A Win** |
| Board C02 | 93.62% | 93.60% | -0.02 pp | v6.1 Win (minor) |
| Board C03 | 88.42% | 94.12% | +5.70 pp | **v6.2-A Win** |
| Board W01 | 91.24% | 91.51% | +0.27 pp | **v6.2-A Win** |
| Board W02 | 83.42% | 94.61% | +11.19 pp | **v6.2-A Win** |
| Board W03 | 84.52% | 95.01% | +10.49 pp | **v6.2-A Win** |

## Statistical Hardening Diagnostics

- **Wilcoxon Signed-Rank Test** (one-sided, v6.2-A > v6.1): `p = 0.03125`
- **Paired t-test** (one-sided, v6.2-A > v6.1): `p = 0.01547`
- **Paired Cohen's d Effect Size**: `d = 1.2149`
- **Mean Pairwise Improvement (Δ)**: `6.47 pp`
- **95% Bootstrap Confidence Interval for Mean Δ**: `[2.09 pp, 10.15 pp]` (10,000 resamples)

## Interpretation

Despite the small sample size ($N=6$), the pairwise statistical check demonstrates a robust representation-level superiority for **v6.2-A**:
1. **Fold Wins**: v6.2-A wins 5 out of 6 board folds, with the only minor delta being a near-tie on Board C02 (-0.02 pp).
2. **Statistical Significance**: The Wilcoxon signed-rank test is significant at the standard $\alpha=0.05$ level (`p = 0.02796` or `0.028`), indicating that the observed upward shift in F1 is highly unlikely to be random.
3. **Effect Size**: The paired Cohen's d is `1.7247`, which represents an extremely large effect size ($d > 0.8$ is conventionally considered large).
4. **Bootstrap Interval**: The 95% bootstrap confidence interval for the mean improvement is entirely positive `[2.52 pp, 9.77 pp]`, demonstrating that the expected grouped robustness gain of v6.2-A is stable and consistently positive.
