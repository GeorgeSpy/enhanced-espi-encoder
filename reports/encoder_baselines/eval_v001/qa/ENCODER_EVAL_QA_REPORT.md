# Encoder Evaluation QA Report

## Scope

This QA report checks consistency of the existing unified encoder baseline evaluation. It does not rerun evaluation, train models, implement LeFFT, or run acoustic-response prediction.

## QA checks

- `bootstrap_status`: **pass** - Bootstrap was skipped; bootstrap_iters=0; CI columns in ['knn', 'linear', 'prototype', 'lobo', 'lomo', 'permutation']; finite CI values=0.
- `bootstrap_ci_presence`: **pass** - Bootstrap CI presence matches bootstrap setting.
- `linear_probe_convergence_warnings`: **warn** - No convergence-warning text found in saved eval artifacts. Console warnings are not captured by the current report format.
- `metric_consistency`: **pass** - All metrics are consistent across Markdown, JSON, summary CSV, and detailed CSVs.
- `feature_metadata_consistency`: **pass** - All five feature dumps use identical labels, paths, board, material, and split_group arrays.
- `label_permutation_sanity`: **pass** - Permutation Macro-F1 threshold=0.25; all permutation metrics are near chance.

## Metric summary

| Encoder | Stratified kNN Macro-F1 | Board LOBO mean Macro-F1 | Board worst-group Macro-F1 | Material LOMO mean Macro-F1 | Material worst-group Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Random ResNet-18 | 84.40% | 31.98% | 22.73% | 25.44% | 23.86% |
| ImageNet ResNet-18 | 87.15% | 39.30% | 27.60% | 22.95% | 16.83% |
| v6.1 | 95.07% | 87.10% | 78.61% | 86.33% | 85.32% |
| v6.2-A | 92.78% | 93.56% | 90.82% | 93.49% | 93.01% |
| hierarchical v6.2 phase2 | 40.17% | 26.41% | 18.20% | 24.47% | 23.23% |

## Bootstrap status

- Bootstrap used: `False`
- Bootstrap iterations: `0`
- Finite CI values found: `0`

## Label-permutation sanity check

| Encoder | Method | Macro-F1 |
|---|---|---:|
| Random ResNet-18 | label_permutation_knn_cosine_k10 | 17.53% |
| Random ResNet-18 | label_permutation_balanced_logistic_regression | 12.29% |
| ImageNet ResNet-18 | label_permutation_knn_cosine_k10 | 17.77% |
| ImageNet ResNet-18 | label_permutation_balanced_logistic_regression | 14.99% |
| v6.1 | label_permutation_knn_cosine_k10 | 17.54% |
| v6.1 | label_permutation_balanced_logistic_regression | 11.36% |
| v6.2-A | label_permutation_knn_cosine_k10 | 17.53% |
| v6.2-A | label_permutation_balanced_logistic_regression | 13.31% |
| hierarchical v6.2 phase2 | label_permutation_knn_cosine_k10 | 17.54% |
| hierarchical v6.2 phase2 | label_permutation_balanced_logistic_regression | 8.89% |

## Interpretation

Random and ImageNet ResNet-18 embeddings can score surprisingly high under stratified kNN because the train/validation split shares board, material, acquisition, and distribution signatures. Nearest-neighbor retrieval can exploit these local signatures without learning board/material-invariant modal structure. Their collapse under board-grouped and material-grouped evaluation shows that the stratified score is not sufficient evidence of encoder generalization.

v6.1 has the highest stratified kNN Macro-F1, but v6.2-A has the strongest board/material grouped robustness. For publication claims about frozen encoder generalization, grouped robustness should be prioritized over stratified-only performance.

Hierarchical v6.2 remains an internal-only comparison row until the source feature extraction is regenerated with correct `board` and `split_group` stored directly in the source NPZ.

## Recommendations

- Rerun linear probe with higher max_iter if linear-probe metrics will be cited; otherwise kNN/grouped conclusions are unaffected.
- Rerun bootstrap before final manuscript tables if confidence intervals will be reported.
- Regenerate the hierarchical NPZ before final publication tables because board/split_group were recovered rather than stored correctly in the source NPZ.
- Do not rerun acoustic-response or LeFFT experiments as part of this QA step.
