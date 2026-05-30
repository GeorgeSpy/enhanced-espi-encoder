# Paired v6.1 vs v6.2-A Grouped Difference Tables

## Purpose

This report compares v6.1 and v6.2-A under the same frozen-embedding protocols using existing evaluation outputs only. It does not rerun encoder evaluation, retrain models, implement LeFFT, or evaluate acoustic-response prediction.

## Protocol-Level Paired Difference

| Protocol | v6.1 Macro-F1 | v6.2-A Macro-F1 | Delta v6.2-A minus v6.1 (pp) | Interpretation |
| --- | --- | --- | --- | --- |
| Stratified kNN | 95.07% | 92.78% | -2.29 | v6.1 remains the stronger stratified embedding reference. |
| Board LOBO mean | 87.10% | 93.56% | +6.47 | v6.2-A has stronger board-grouped robustness. |
| Material LOMO mean | 86.33% | 93.49% | +7.16 | v6.2-A has stronger material-held-out robustness; interpret descriptively because only two materials are available. |

## Board LOBO Fold-Level Paired Differences

| held_out_board | material | held_out_total | v6.1 Macro-F1 | v6.2-A Macro-F1 | Delta v6.2-A minus v6.1 (pp) | winner |
| --- | --- | --- | --- | --- | --- | --- |
| C01 | carbon | 2733 | 78.61% | 93.63% | +15.01 | v6.2-A |
| C02 | carbon | 2319 | 92.85% | 92.83% | -0.02 | v6.1 |
| C03 | carbon | 1637 | 93.38% | 95.69% | +2.30 | v6.2-A |
| W01 | wood | 2419 | 90.55% | 90.82% | +0.27 | v6.2-A |
| W02 | wood | 2370 | 83.34% | 95.41% | +12.07 | v6.2-A |
| W03 | wood | 1466 | 83.84% | 93.01% | +9.17 | v6.2-A |

## Material LOMO Fold-Level Paired Differences

| held_out_material | held_out_total | v6.1 Macro-F1 | v6.2-A Macro-F1 | Delta v6.2-A minus v6.1 (pp) | winner | caution_note |
| --- | --- | --- | --- | --- | --- | --- |
| carbon | 6689 | 87.34% | 93.97% | +6.62 | v6.2-A | descriptive stress-test only; two material groups available |
| wood | 6255 | 85.32% | 93.01% | +7.69 | v6.2-A | descriptive stress-test only; two material groups available |

## Summary Statistics

- Boards where v6.2-A wins: 5 / 6
- Boards where v6.1 wins: 1 / 6
- Mean board delta: +6.47 pp
- Median board delta: +5.74 pp
- Min board delta: -0.02 pp
- Max board delta: +15.01 pp
- Materials where v6.2-A wins: 2 / 2
- Materials where v6.1 wins: 0 / 2
- Mean material delta: +7.16 pp

## Reviewer-Facing Interpretation

v6.1 wins the stratified kNN protocol, confirming that it remains the stronger in-distribution stratified reference. v6.2-A wins the grouped board/material mean Macro-F1 protocols, supporting its selection as the primary reportable frozen ESPI encoder baseline for grouped representation studies.

The board-level paired deltas indicate whether the grouped advantage is broad across held-out boards or concentrated in a small number of boards. The material-level deltas are useful descriptively, but they should be interpreted as material-held-out stress-test results because only two material groups are available.
