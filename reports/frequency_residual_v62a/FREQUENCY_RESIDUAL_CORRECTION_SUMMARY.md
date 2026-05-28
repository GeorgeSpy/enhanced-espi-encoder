# Frequency-Residual / Error-Correction Analysis

This analysis compares `frequency_hz`-only prediction against `frequency_hz + v6.2-A embedding` fusion.

The central question is whether image-derived v6.2-A embeddings correct errors made by the frequency-only metadata baseline.

- Feature file: `outputs\encoder_features_normalized_v001\features_v62a_epoch25.normalized.npz`
- Input embedding shape: `(12944, 1280)`
- Families: `['lr']`
- Protocols: `['stratified', 'board_lobo', 'material_lomo']`

## Overall Metrics

| accuracy | balanced_accuracy | macro_f1 | weighted_f1 | family | protocol      | feature_kind | n     |
| -------- | ----------------- | -------- | ----------- | ------ | ------------- | ------------ | ----- |
| 0.9587   | 0.9351            | 0.9106   | 0.9597      | lr     | board_lobo    | embed        | 12944 |
| 0.9924   | 0.9739            | 0.9737   | 0.9924      | lr     | board_lobo    | freq         | 12944 |
| 0.9774   | 0.9534            | 0.9423   | 0.9778      | lr     | board_lobo    | fusion       | 12944 |
| 0.9576   | 0.9308            | 0.9082   | 0.9585      | lr     | material_lomo | embed        | 12944 |
| 0.9833   | 0.9429            | 0.9427   | 0.9833      | lr     | material_lomo | freq         | 12944 |
| 0.9737   | 0.9458            | 0.9370   | 0.9741      | lr     | material_lomo | fusion       | 12944 |
| 0.9683   | 0.9384            | 0.9284   | 0.9685      | lr     | stratified    | embed        | 2589  |
| 0.9911   | 0.9696            | 0.9694   | 0.9911      | lr     | stratified    | freq         | 2589  |
| 0.9884   | 0.9669            | 0.9667   | 0.9884      | lr     | stratified    | fusion       | 2589  |

## Frequency-only vs Frequency+Embedding Correction Summary

| family | protocol      | n     | freq_errors | fusion_errors | corrected_by_fusion | degraded_by_fusion | both_wrong | both_correct | net_corrections | correction_rate_of_freq_errors | degradation_rate_of_freq_correct | freq_macro_f1 | fusion_macro_f1 | delta_macro_f1_pp |
| ------ | ------------- | ----- | ----------- | ------------- | ------------------- | ------------------ | ---------- | ------------ | --------------- | ------------------------------ | -------------------------------- | ------------- | --------------- | ----------------- |
| lr     | board_lobo    | 12944 | 99          | 293           | 83                  | 277                | 16         | 12568        | -194            | 0.8384                         | 0.0216                           | 0.9737        | 0.9423          | -3.1389           |
| lr     | material_lomo | 12944 | 216         | 340           | 186                 | 310                | 30         | 12418        | -124            | 0.8611                         | 0.0244                           | 0.9427        | 0.9370          | -0.5639           |
| lr     | stratified    | 2589  | 23          | 30            | 20                  | 27                 | 3          | 2539         | -7              | 0.8696                         | 0.0105                           | 0.9694        | 0.9667          | -0.2712           |

## Per-Class Corrections

| family | protocol      | class_id | class_name | support | freq_errors | fusion_errors | corrected_by_fusion | degraded_by_fusion | net_corrections | correction_rate_of_freq_errors | freq_class_f1 | fusion_class_f1 | delta_class_f1_pp |
| ------ | ------------- | -------- | ---------- | ------- | ----------- | ------------- | ------------------- | ------------------ | --------------- | ------------------------------ | ------------- | --------------- | ----------------- |
| lr     | board_lobo    | 4        | 4          | 10115   | 0           | 133           | 0                   | 133                | -133            | 0.0000                         | 1.0000        | 0.9920          | -0.7951           |
| lr     | board_lobo    | 0        | 0          | 653     | 0           | 17            | 0                   | 17                 | -17             | 0.0000                         | 1.0000        | 0.9747          | -2.5287           |
| lr     | board_lobo    | 1        | 1          | 669     | 0           | 22            | 0                   | 22                 | -22             | 0.0000                         | 1.0000        | 0.9729          | -2.7068           |
| lr     | board_lobo    | 2        | 2          | 738     | 30          | 63            | 25                  | 58                 | -33             | 0.8333                         | 0.9347        | 0.8887          | -4.5911           |
| lr     | board_lobo    | 3        | 3          | 769     | 69          | 58            | 58                  | 47                 | 11              | 0.8406                         | 0.9340        | 0.8832          | -5.0726           |
| lr     | material_lomo | 3        | 3          | 769     | 130         | 69            | 111                 | 50                 | 61              | 0.8538                         | 0.8554        | 0.9026          | 4.7222            |
| lr     | material_lomo | 2        | 2          | 738     | 86          | 59            | 75                  | 48                 | 27              | 0.8721                         | 0.8579        | 0.8835          | 2.5645            |
| lr     | material_lomo | 4        | 4          | 10115   | 0           | 155           | 0                   | 155                | -155            | 0.0000                         | 1.0000        | 0.9885          | -1.1463           |
| lr     | material_lomo | 1        | 1          | 669     | 0           | 41            | 0                   | 41                 | -41             | 0.0000                         | 1.0000        | 0.9632          | -3.6810           |
| lr     | material_lomo | 0        | 0          | 653     | 0           | 16            | 0                   | 16                 | -16             | 0.0000                         | 1.0000        | 0.9472          | -5.2788           |
| lr     | stratified    | 3        | 3          | 154     | 14          | 10            | 12                  | 8                  | 4               | 0.8571                         | 0.9241        | 0.9351          | 1.0973            |
| lr     | stratified    | 2        | 2          | 147     | 9           | 11            | 8                   | 10                 | -2              | 0.8889                         | 0.9231        | 0.9315          | 0.8430            |
| lr     | stratified    | 4        | 4          | 2023    | 0           | 6             | 0                   | 6                  | -6              | 0.0000                         | 1.0000        | 0.9970          | -0.2966           |
| lr     | stratified    | 0        | 0          | 131     | 0           | 1             | 0                   | 1                  | -1              | 0.0000                         | 1.0000        | 0.9886          | -1.1407           |
| lr     | stratified    | 1        | 1          | 134     | 0           | 2             | 0                   | 2                  | -2              | 0.0000                         | 1.0000        | 0.9814          | -1.8587           |

## Most Frequent Corrected Error Pairs

| family | protocol      | true_id | true_name | freq_pred_id | freq_pred_name | fusion_pred_id | fusion_pred_name | count |
| ------ | ------------- | ------- | --------- | ------------ | -------------- | -------------- | ---------------- | ----- |
| lr     | material_lomo | 3       | 3         | 2            | 2              | 3              | 3                | 111   |
| lr     | material_lomo | 2       | 2         | 3            | 3              | 2              | 2                | 75    |
| lr     | board_lobo    | 3       | 3         | 2            | 2              | 3              | 3                | 58    |
| lr     | board_lobo    | 2       | 2         | 3            | 3              | 2              | 2                | 25    |
| lr     | stratified    | 3       | 3         | 2            | 2              | 3              | 3                | 12    |
| lr     | stratified    | 2       | 2         | 3            | 3              | 2              | 2                | 8     |

## Output Files

- `reports\frequency_residual_v62a\frequency_residual_fold_metrics.csv`
- `reports\frequency_residual_v62a\frequency_residual_overall_metrics.csv`
- `reports\frequency_residual_v62a\frequency_residual_correction_summary.csv`
- `reports\frequency_residual_v62a\frequency_residual_per_class_corrections.csv`
- `reports\frequency_residual_v62a\frequency_residual_corrected_error_pairs.csv`
- `reports\frequency_residual_v62a\frequency_residual_key_numbers.json`
