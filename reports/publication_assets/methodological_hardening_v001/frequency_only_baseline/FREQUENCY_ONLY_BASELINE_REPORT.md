# Frequency-Only Baseline Report

## Purpose

This report evaluates whether `frequency_hz` alone predicts the five ESPI modal labels under the same stratified, board-grouped, and material-grouped protocols used for frozen encoder evaluation. This is a metadata leakage/control baseline, not the main model.

## Models

- `frequency_logistic_regression`: Balanced logistic regression on standardized frequency_hz. Parameters: `{"class_weight": "balanced", "max_iter": 5000, "random_state": 42}`
- `frequency_random_forest`: Random forest on raw frequency_hz. Parameters: `{"n_estimators": 300, "class_weight": "balanced_subsample", "min_samples_leaf": 2, "random_state": 42}`
- `frequency_bin_majority_diagnostic`: Diagnostic majority-class frequency-bin baseline fitted on reference frequency_hz only. Parameters: `{"n_bins": 20, "diagnostic": true}`

## Summary Metrics

| protocol | method | macro_f1 | mean_macro_f1 | worst_group | worst_group_macro_f1 | best_group | best_group_macro_f1 | n_groups |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stratified_train_val | frequency_logistic_regression | 0.9694338664635694 |  |  |  |  |  |  |
| stratified_train_val | frequency_random_forest | 0.9866861288039631 |  |  |  |  |  |  |
| stratified_train_val | frequency_bin_majority_diagnostic | 0.8791968642935478 |  |  |  |  |  |  |
| board_grouped | frequency_bin_majority_diagnostic |  | 0.8941244107953176 | C01 | 0.7762836329085137 | W01 | 0.9913743751869099 | 6 |
| board_grouped | frequency_logistic_regression |  | 0.9748609579888732 | W02 | 0.9549541161012653 | C01 | 1.0 | 6 |
| board_grouped | frequency_random_forest |  | 0.9671234211089278 | C01 | 0.9234618488855777 | W03 | 1.0 | 6 |
| material_grouped | frequency_bin_majority_diagnostic |  | 0.9040121874471116 | carbon | 0.8404971398264827 | wood | 0.9675272350677405 | 2 |
| material_grouped | frequency_logistic_regression |  | 0.9408561364311898 | wood | 0.9375317788915524 | carbon | 0.9441804939708274 | 2 |
| material_grouped | frequency_random_forest |  | 0.9408561364311898 | wood | 0.9375317788915524 | carbon | 0.9441804939708274 | 2 |

## Board LOBO-Style Frequency-Only Results

| protocol | method | held_out_group | accuracy | macro_recall | macro_f1 | n_test | stress_test_note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| board_grouped | frequency_logistic_regression | C01 | 1.0 | 1.0 | 1.0 | 2733 | none |
| board_grouped | frequency_random_forest | C01 | 0.9915843395536041 | 0.9192982456140351 | 0.9234618488855777 | 2733 | none |
| board_grouped | frequency_bin_majority_diagnostic | C01 | 0.9608488840102452 | 0.8073964987410536 | 0.7762836329085137 | 2733 | none |
| board_grouped | frequency_logistic_regression | C02 | 0.9918068132815869 | 0.973049645390071 | 0.9741399193870677 | 2319 | none |
| board_grouped | frequency_random_forest | C02 | 0.9831824062095731 | 0.9446808510638298 | 0.945804917287448 | 2319 | none |
| board_grouped | frequency_bin_majority_diagnostic | C02 | 0.9383354894351014 | 0.7969436270239225 | 0.8302761614591775 | 2319 | none |
| board_grouped | frequency_logistic_regression | C03 | 0.9932803909590715 | 0.9775510204081632 | 0.9791285162713734 | 1637 | none |
| board_grouped | frequency_random_forest | C03 | 0.9920586438607208 | 0.9734693877551021 | 0.9752660339373023 | 1637 | none |
| board_grouped | frequency_bin_majority_diagnostic | C03 | 0.9346365302382407 | 0.790262633119776 | 0.8279158919655007 | 1637 | none |
| board_grouped | frequency_logistic_regression | W01 | 0.9921455146754857 | 0.9794594594594596 | 0.9789861216134674 | 2419 | none |
| board_grouped | frequency_random_forest | W01 | 0.9925589086399339 | 0.9805405405405405 | 0.9800953079178886 | 2419 | none |
| board_grouped | frequency_bin_majority_diagnostic | W01 | 0.9942124844977264 | 0.9848648648648648 | 0.9913743751869099 | 2419 | none |
| board_grouped | frequency_logistic_regression | W02 | 0.9852320675105485 | 0.9585798816568047 | 0.9549541161012653 | 2370 | none |
| board_grouped | frequency_random_forest | W02 | 0.9928270042194093 | 0.9798816568047337 | 0.9781124186253507 | 2370 | none |
| board_grouped | frequency_bin_majority_diagnostic | W02 | 0.979746835443038 | 0.9404950412534377 | 0.9586984602758302 | 2370 | none |
| board_grouped | frequency_logistic_regression | W03 | 0.9897680763983628 | 0.953125 | 0.9619570745600651 | 1466 | none |
| board_grouped | frequency_random_forest | W03 | 1.0 | 1.0 | 1.0 | 1466 | none |
| board_grouped | frequency_bin_majority_diagnostic | W03 | 0.9924965893587995 | 0.965625 | 0.980197942975973 | 1466 | none |

## Material LOMO-Style Frequency-Only Results

| protocol | method | held_out_group | accuracy | macro_recall | macro_f1 | n_test | stress_test_note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| material_grouped | frequency_logistic_regression | carbon | 0.987143070713111 | 0.9418918918918919 | 0.9441804939708274 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | frequency_random_forest | carbon | 0.987143070713111 | 0.9418918918918919 | 0.9441804939708274 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | frequency_bin_majority_diagnostic | carbon | 0.9611302137838242 | 0.8606921574312878 | 0.8404971398264827 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | frequency_logistic_regression | wood | 0.9792166266986411 | 0.937799043062201 | 0.9375317788915524 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | frequency_random_forest | wood | 0.9792166266986411 | 0.937799043062201 | 0.9375317788915524 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | frequency_bin_majority_diagnostic | wood | 0.988968824940048 | 0.9669856459330143 | 0.9675272350677405 | 6255 | descriptive stress-test only; two material groups available |

## Comparison Against Existing Encoder Baselines

| protocol | frequency_method | frequency_macro_f1 | comparison_encoder | comparison_macro_f1 | delta_frequency_minus_encoder_pp |
| --- | --- | --- | --- | --- | --- |
| stratified_train_val | frequency_random_forest | 0.9866861288039631 | Random ResNet-18 | 0.8439863653774754 | 14.269976342648771 |
| stratified_train_val | frequency_random_forest | 0.9866861288039631 | ImageNet ResNet-18 | 0.8715130509203878 | 11.517307788357535 |
| stratified_train_val | frequency_random_forest | 0.9866861288039631 | v6.1 | 0.9506911900286517 | 3.599493877531146 |
| stratified_train_val | frequency_random_forest | 0.9866861288039631 | v6.2-A | 0.927755752028969 | 5.893037677499413 |
| board_grouped | frequency_logistic_regression | 0.9748609579888732 | Random ResNet-18 | 0.319913304491168 | 65.49476534977052 |
| board_grouped | frequency_logistic_regression | 0.9748609579888732 | ImageNet ResNet-18 | 0.3930272194094586 | 58.183373857941454 |
| board_grouped | frequency_logistic_regression | 0.9748609579888732 | v6.1 | 0.8709697195504678 | 10.389123843840542 |
| board_grouped | frequency_logistic_regression | 0.9748609579888732 | v6.2-A | 0.9356408210770706 | 3.9220136911802594 |
| material_grouped | frequency_logistic_regression | 0.9408561364311898 | Random ResNet-18 | 0.2543697950088857 | 68.6486341422304 |
| material_grouped | frequency_logistic_regression | 0.9408561364311898 | ImageNet ResNet-18 | 0.229527070800846 | 71.13290656303438 |
| material_grouped | frequency_logistic_regression | 0.9408561364311898 | v6.1 | 0.8632932432336362 | 7.756289319755361 |
| material_grouped | frequency_logistic_regression | 0.9408561364311898 | v6.2-A | 0.9348717330088938 | 0.5984403422296003 |

## Reviewer-Facing Interpretation

- Best frequency-only stratified Macro-F1: 98.67%.
- Best frequency-only Board LOBO mean Macro-F1: 97.49%.
- Best frequency-only Material LOMO mean Macro-F1: 94.09%.
- v6.2-A Board LOBO mean Macro-F1: 93.56%.
- v6.2-A Material LOMO mean Macro-F1: 93.49%.
- Interpretation: Frequency metadata is a strong predictor and should be discussed as carrying substantial modal information.
- Material LOMO remains a descriptive stress test because only two material groups are available.
- This control does not use ESPI image-derived embeddings and does not modify the central frozen encoder claim.
