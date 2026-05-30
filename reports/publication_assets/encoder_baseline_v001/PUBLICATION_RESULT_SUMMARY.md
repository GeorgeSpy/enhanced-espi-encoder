# Publication Result Summary - Encoder Baseline v001

## Source
- Evaluation directory: `reports\encoder_baselines\eval_v002`
- Bootstrap enabled: `True`
- Generated UTC: `2026-05-26T14:45:50.472804+00:00`

## Main result
- v6.1 wins stratified kNN Macro-F1: 95.07%.
- v6.2-A wins board LOBO Macro-F1: 93.56%.
- v6.2-A wins material LOMO Macro-F1: 93.49%.
- v6.2-A remains the primary reportable frozen ESPI encoder baseline.

## Interpretation boundary
- v6.1 is the strongest stratified retrieval baseline, but v6.2-A is more robust under grouped board/material exclusion.
- Random and ImageNet ResNet-18 controls show that stratified texture features alone are insufficient for grouped robustness.
- Hierarchical v6.2 phase2 is technically valid but not superior in this frozen embedding evaluation.
- No acoustic-response predictive value is claimed.
- No validated Physics-Aligned Encoder claim is made.
- LeFFT remains a future/ablation direction, not part of the current core claim.
