# Unified Encoder Baseline Evaluation

## Scope

This report evaluates normalized encoder feature dumps under a shared frozen-representation protocol. It does not run model inference, train or fine-tune CNNs, implement LeFFT, or claim acoustic-response predictive value.

## Main comparison

| Encoder | Dim | Best kNN Macro-F1 | Linear Macro-F1 | Prototype Macro-F1 | Best LOBO mean Macro-F1 | Best LOMO mean Macro-F1 | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| Random ResNet-18 | 512 | 84.40% | 84.43% | 29.68% | 31.98% | 25.44% | usable |
| ImageNet ResNet-18 | 512 | 87.15% | 81.11% | 31.84% | 39.30% | 22.95% | usable |
| v6.1 | 512 | 95.07% | 92.95% | 65.95% | 87.10% | 86.33% | usable |
| v6.2-A | 1280 | 92.78% | 92.84% | 69.87% | 93.56% | 93.49% | usable |
| hierarchical v6.2 phase2 | 512 | 40.17% | 29.28% | 26.55% | 26.41% | 24.47% | internal-only until regenerated |

## Decision boundary

- v6.2-A is expected to remain the primary reportable frozen ESPI encoder if it outperforms random ResNet-18, ImageNet ResNet-18, v6.1, and hierarchical v6.2 under this matched protocol.
- The best stratified kNN result in this run is `v6.1` with Macro-F1 `95.07%`.
- Primary encoder selection should prioritize grouped board/material robustness over stratified-only performance.
- Hierarchical v6.2 is internal-only for this table unless regenerated with correct `board` and `split_group` stored in the source NPZ.
- This evaluation does not support a validated Physics-Aligned Encoder claim.
- This evaluation does not support any acoustic-response predictive-value claim.

## Protocol

- Stratified train/validation: kNN cosine for k = 1, 3, 5, 10, 20; nearest class prototype; balanced logistic-regression linear probe.
- Grouped evaluation: board-grouped LOBO-style and material-grouped LOMO-style using kNN k=10, prototype, and balanced logistic regression.
- Sanity check: label-permutation test for kNN k=10 and linear probe.
- Bootstrap 95% CI for stratified Macro-F1: `False`.

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

## Figures

- `reports\encoder_baselines\eval_v001\stratified_macro_f1_barplot.svg`
- `reports\encoder_baselines\eval_v001\grouped_macro_f1_barplot.svg`
