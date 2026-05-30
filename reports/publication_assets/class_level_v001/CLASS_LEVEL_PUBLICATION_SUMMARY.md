# Class-Level Publication Summary

## Main findings
- The dataset is strongly imbalanced, with `higher` contributing `10115` of `12944` samples (78.14%).
- v6.2-A improves grouped F1 over v6.1 across all five classes.
- The largest improvements occur for `2_1` (13.43 pp) and `1_2` (12.21 pp).
- This supports the claim that v6.2-A improves grouped class stability, not only aggregate Macro-F1.
- Zero-recall cases must be interpreted by encoder and evaluation mode, because they are not uniformly distributed across models.

## Zero-recall concentration
- `ImageNet ResNet-18`: `20` zero-recall rows.
- `Random ResNet-18`: `18` zero-recall rows.
- `hierarchical v6.2 phase2`: `9` zero-recall rows.

## Claim boundary
- No acoustic-response predictive value is claimed.
- No validated Physics-Aligned Encoder claim is made.
- LeFFT is not part of this analysis.
