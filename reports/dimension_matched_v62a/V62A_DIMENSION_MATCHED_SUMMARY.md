# v6.2-A Dimension-Matched PCA Evaluation

Fold-local PCA was fitted only on each training/reference fold and applied to the corresponding held-out fold.

- Input feature file: `outputs\encoder_features_normalized_v001\features_v62a_epoch25.normalized.npz`
- Input shape: `(12944, 1280)`
- kNN: cosine, fixed k=10, distance-weighted

## Summary

| protocol | n_components | k | n_folds | macro_f1_mean | macro_f1_std | balanced_accuracy_mean | accuracy_mean | weighted_f1_mean | explained_variance_ratio_sum_mean | worst_group | worst_group_macro_f1 | best_group | best_group_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| board_lobo | 128 | 10 | 6 | 0.936540 | 0.016607 | 0.927403 | 0.970126 | 0.970063 | 0.995696 | W01 | 0.908154 | C03 | 0.957840 |
| board_lobo | 256 | 10 | 6 | 0.935850 | 0.016462 | 0.927472 | 0.969738 | 0.969694 | 0.997919 | W01 | 0.907961 | C03 | 0.956879 |
| board_lobo | 512 | 10 | 6 | 0.935760 | 0.016614 | 0.927454 | 0.969669 | 0.969629 | 0.999335 | W01 | 0.907423 | C03 | 0.956879 |
| material_lomo | 128 | 10 | 2 | 0.935141 | 0.004152 | 0.926697 | 0.969150 | 0.969122 | 0.996101 | wood | 0.930989 | carbon | 0.939294 |
| material_lomo | 256 | 10 | 2 | 0.934915 | 0.004964 | 0.927416 | 0.968980 | 0.968971 | 0.998179 | wood | 0.929951 | carbon | 0.939879 |
| material_lomo | 512 | 10 | 2 | 0.935060 | 0.005138 | 0.927697 | 0.969055 | 0.969045 | 0.999449 | wood | 0.929922 | carbon | 0.940198 |
| stratified | 128 | 10 | 1 | 0.932760 | 0.000000 | 0.924211 | 0.971418 | 0.971251 | 0.995613 | validation | 0.932760 | validation | 0.932760 |
| stratified | 256 | 10 | 1 | 0.933540 | 0.000000 | 0.924309 | 0.971804 | 0.971631 | 0.997869 | validation | 0.933540 | validation | 0.933540 |
| stratified | 512 | 10 | 1 | 0.933540 | 0.000000 | 0.924309 | 0.971804 | 0.971631 | 0.999316 | validation | 0.933540 | validation | 0.933540 |

## Files

- `reports\dimension_matched_v62a\v62a_dimension_matched_fold_results.csv`
- `reports\dimension_matched_v62a\v62a_dimension_matched_summary.csv`
- `reports\dimension_matched_v62a\v62a_dimension_matched_summary.json`
