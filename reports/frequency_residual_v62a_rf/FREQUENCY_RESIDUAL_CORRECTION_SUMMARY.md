# Frequency-Residual / Error-Correction Analysis

This analysis compares `frequency_hz`-only prediction against `frequency_hz + v6.2-A embedding` fusion.

The central question is whether image-derived v6.2-A embeddings correct errors made by the frequency-only metadata baseline.

- Feature file: `outputs\encoder_features_normalized_v001\features_v62a_epoch25.normalized.npz`
- Input embedding shape: `(12944, 1280)`
- Families: `['rf']`
- Protocols: `['stratified', 'board_lobo', 'material_lomo']`

## Overall Metrics

| accuracy | balanced_accuracy | macro_f1 | weighted_f1 | family | protocol      | feature_kind | n     |
| -------- | ----------------- | -------- | ----------- | ------ | ------------- | ------------ | ----- |
| 0.9756   | 0.9249            | 0.9466   | 0.9752      | rf     | board_lobo    | embed        | 12944 |
| 0.9915   | 0.9706            | 0.9707   | 0.9915      | rf     | board_lobo    | freq         | 12944 |
| 0.9777   | 0.9284            | 0.9508   | 0.9773      | rf     | board_lobo    | fusion       | 12944 |
| 0.9727   | 0.9141            | 0.9401   | 0.9722      | rf     | material_lomo | embed        | 12944 |
| 0.9833   | 0.9429            | 0.9427   | 0.9833      | rf     | material_lomo | freq         | 12944 |
| 0.9759   | 0.9227            | 0.9458   | 0.9755      | rf     | material_lomo | fusion       | 12944 |
| 0.9726   | 0.9130            | 0.9356   | 0.9721      | rf     | stratified    | embed        | 2589  |
| 0.9961   | 0.9865            | 0.9867   | 0.9961      | rf     | stratified    | freq         | 2589  |
| 0.9741   | 0.9157            | 0.9405   | 0.9736      | rf     | stratified    | fusion       | 2589  |

## Frequency-only vs Frequency+Embedding Correction Summary

| family | protocol      | n     | freq_errors | fusion_errors | corrected_by_fusion | degraded_by_fusion | both_wrong | both_correct | net_corrections | correction_rate_of_freq_errors | degradation_rate_of_freq_correct | freq_macro_f1 | fusion_macro_f1 | delta_macro_f1_pp |
| ------ | ------------- | ----- | ----------- | ------------- | ------------------- | ------------------ | ---------- | ------------ | --------------- | ------------------------------ | -------------------------------- | ------------- | --------------- | ----------------- |
| rf     | board_lobo    | 12944 | 110         | 289           | 89                  | 268                | 21         | 12566        | -179            | 0.8091                         | 0.0209                           | 0.9707        | 0.9508          | -1.9962           |
| rf     | material_lomo | 12944 | 216         | 312           | 176                 | 272                | 40         | 12456        | -96             | 0.8148                         | 0.0214                           | 0.9427        | 0.9458          | 0.3113            |
| rf     | stratified    | 2589  | 10          | 67            | 9                   | 66                 | 1          | 2513         | -57             | 0.9000                         | 0.0256                           | 0.9867        | 0.9405          | -4.6181           |

## Per-Class Corrections

| family | protocol      | class_id | class_name | support | freq_errors | fusion_errors | corrected_by_fusion | degraded_by_fusion | net_corrections | correction_rate_of_freq_errors | freq_class_f1 | fusion_class_f1 | delta_class_f1_pp |
| ------ | ------------- | -------- | ---------- | ------- | ----------- | ------------- | ------------------- | ------------------ | --------------- | ------------------------------ | ------------- | --------------- | ----------------- |
| rf     | board_lobo    | 2        | 2          | 738     | 75          | 80            | 58                  | 63                 | -5              | 0.7733                         | 0.9234        | 0.9255          | 0.2059            |
| rf     | board_lobo    | 3        | 3          | 769     | 35          | 80            | 31                  | 76                 | -45             | 0.8857                         | 0.9303        | 0.9273          | -0.2970           |
| rf     | board_lobo    | 4        | 4          | 10115   | 0           | 35            | 0                   | 35                 | -35             | 0.0000                         | 1.0000        | 0.9876          | -1.2394           |
| rf     | board_lobo    | 1        | 1          | 669     | 0           | 49            | 0                   | 49                 | -49             | 0.0000                         | 1.0000        | 0.9583          | -4.1731           |
| rf     | board_lobo    | 0        | 0          | 653     | 0           | 45            | 0                   | 45                 | -45             | 0.0000                         | 1.0000        | 0.9552          | -4.4776           |
| rf     | material_lomo | 3        | 3          | 769     | 130         | 86            | 111                 | 67                 | 44              | 0.8538                         | 0.8554        | 0.9248          | 6.9426            |
| rf     | material_lomo | 2        | 2          | 738     | 86          | 86            | 65                  | 65                 | 0               | 0.7558                         | 0.8579        | 0.9177          | 5.9769            |
| rf     | material_lomo | 4        | 4          | 10115   | 0           | 38            | 0                   | 38                 | -38             | 0.0000                         | 1.0000        | 0.9869          | -1.3075           |
| rf     | material_lomo | 1        | 1          | 669     | 0           | 49            | 0                   | 49                 | -49             | 0.0000                         | 1.0000        | 0.9583          | -4.1731           |
| rf     | material_lomo | 0        | 0          | 653     | 0           | 53            | 0                   | 53                 | -53             | 0.0000                         | 1.0000        | 0.9412          | -5.8824           |
| rf     | stratified    | 4        | 4          | 2023    | 0           | 7             | 0                   | 7                  | -7              | 0.0000                         | 1.0000        | 0.9865          | -1.3457           |
| rf     | stratified    | 1        | 1          | 134     | 0           | 9             | 0                   | 9                  | -9              | 0.0000                         | 1.0000        | 0.9615          | -3.8462           |
| rf     | stratified    | 0        | 0          | 131     | 0           | 12            | 0                   | 12                 | -12             | 0.0000                         | 1.0000        | 0.9444          | -5.5556           |
| rf     | stratified    | 2        | 2          | 147     | 8           | 19            | 8                   | 19                 | -11             | 1.0000                         | 0.9653        | 0.9046          | -6.0684           |
| rf     | stratified    | 3        | 3          | 154     | 2           | 20            | 1                   | 19                 | -18             | 0.5000                         | 0.9682        | 0.9054          | -6.2747           |

## Most Frequent Corrected Error Pairs

| family | protocol      | true_id | true_name | freq_pred_id | freq_pred_name | fusion_pred_id | fusion_pred_name | count |
| ------ | ------------- | ------- | --------- | ------------ | -------------- | -------------- | ---------------- | ----- |
| rf     | material_lomo | 3       | 3         | 2            | 2              | 3              | 3                | 111   |
| rf     | material_lomo | 2       | 2         | 3            | 3              | 2              | 2                | 65    |
| rf     | board_lobo    | 2       | 2         | 3            | 3              | 2              | 2                | 58    |
| rf     | board_lobo    | 3       | 3         | 2            | 2              | 3              | 3                | 31    |
| rf     | stratified    | 2       | 2         | 3            | 3              | 2              | 2                | 8     |
| rf     | stratified    | 3       | 3         | 2            | 2              | 3              | 3                | 1     |

## Output Files

- `reports\frequency_residual_v62a_rf\frequency_residual_fold_metrics.csv`
- `reports\frequency_residual_v62a_rf\frequency_residual_overall_metrics.csv`
- `reports\frequency_residual_v62a_rf\frequency_residual_correction_summary.csv`
- `reports\frequency_residual_v62a_rf\frequency_residual_per_class_corrections.csv`
- `reports\frequency_residual_v62a_rf\frequency_residual_corrected_error_pairs.csv`
- `reports\frequency_residual_v62a_rf\frequency_residual_key_numbers.json`
