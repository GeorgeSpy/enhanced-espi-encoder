# Frequency-Overlap Pair Analysis

This analysis isolates the target modal pair and evaluates whether image-derived embeddings help when frequency information is ambiguous.

- Labels: `[2, 3]` / `['2', '3']`
- Overlap only: `True`
- Selected samples: `160`
- Frequency overlap range: `550.0000` to `556.0000` Hz

## Frequency ranges by class

| label | label_name | freq_min | freq_max | n   |
| ----- | ---------- | -------- | -------- | --- |
| 2     | 2          | 500.0000 | 556.0000 | 738 |
| 3     | 3          | 550.0000 | 605.0000 | 769 |

## Overall metrics

| accuracy | balanced_accuracy | macro_f1 | weighted_f1 | protocol   | family | feature_kind | n  |
| -------- | ----------------- | -------- | ----------- | ---------- | ------ | ------------ | -- |
| 0.9767   | 0.9750            | 0.9765   | 0.9767      | stratified | lr     | embed        | 43 |
| 0.5349   | 0.5457            | 0.5285   | 0.5247      | stratified | lr     | freq         | 43 |
| 0.9767   | 0.9750            | 0.9765   | 0.9767      | stratified | lr     | fusion       | 43 |
| 0.9535   | 0.9500            | 0.9529   | 0.9532      | stratified | rf     | embed        | 43 |
| 0.7674   | 0.7761            | 0.7663   | 0.7652      | stratified | rf     | freq         | 43 |
| 0.9302   | 0.9283            | 0.9296   | 0.9301      | stratified | rf     | fusion       | 43 |

## Fold metrics

| accuracy | balanced_accuracy | macro_f1 | weighted_f1 | protocol   | family | feature_kind | heldout_group | n_train | n_test |
| -------- | ----------------- | -------- | ----------- | ---------- | ------ | ------------ | ------------- | ------- | ------ |
| 0.9767   | 0.9750            | 0.9765   | 0.9767      | stratified | lr     | embed        | validation    | 117     | 43     |
| 0.5349   | 0.5457            | 0.5285   | 0.5247      | stratified | lr     | freq         | validation    | 117     | 43     |
| 0.9767   | 0.9750            | 0.9765   | 0.9767      | stratified | lr     | fusion       | validation    | 117     | 43     |
| 0.9535   | 0.9500            | 0.9529   | 0.9532      | stratified | rf     | embed        | validation    | 117     | 43     |
| 0.7674   | 0.7761            | 0.7663   | 0.7652      | stratified | rf     | freq         | validation    | 117     | 43     |
| 0.9302   | 0.9283            | 0.9296   | 0.9301      | stratified | rf     | fusion       | validation    | 117     | 43     |

## Output files

- `reports\frequency_overlap_pair_v62a\overlap_only\overlap_pair_overall_metrics.csv`
- `reports\frequency_overlap_pair_v62a\overlap_only\overlap_pair_fold_metrics.csv`
- `reports\frequency_overlap_pair_v62a\overlap_only\overlap_pair_confusion_counts.csv`
- `reports\frequency_overlap_pair_v62a\overlap_only\overlap_pair_key_numbers.json`
