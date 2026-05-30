# kNN Protocol Hardening Report

## Purpose

This report separates fixed-k kNN results from best-k diagnostic selection using existing eval_v002 outputs only. No encoder evaluation is rerun.

## Fixed-k Summary

| protocol | encoder_name | k | macro_f1 | mean_macro_f1 | worst_group | worst_group_macro_f1 | best_group | best_group_macro_f1 | availability_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stratified_train_val | Random ResNet-18 | 1 | 0.8385579823143583 |  |  |  |  |  | available |
| stratified_train_val | Random ResNet-18 | 5 | 0.8439863653774754 |  |  |  |  |  | available |
| stratified_train_val | Random ResNet-18 | 10 | 0.8114196118916214 |  |  |  |  |  | available |
| stratified_train_val | ImageNet ResNet-18 | 1 | 0.870694708357586 |  |  |  |  |  | available |
| stratified_train_val | ImageNet ResNet-18 | 5 | 0.8715130509203878 |  |  |  |  |  | available |
| stratified_train_val | ImageNet ResNet-18 | 10 | 0.8434203551215631 |  |  |  |  |  | available |
| stratified_train_val | v6.1 | 1 | 0.9485004728299975 |  |  |  |  |  | available |
| stratified_train_val | v6.1 | 5 | 0.9506911900286517 |  |  |  |  |  | available |
| stratified_train_val | v6.1 | 10 | 0.936526975749645 |  |  |  |  |  | available |
| stratified_train_val | v6.2-A | 1 | 0.9174939833077506 |  |  |  |  |  | available |
| stratified_train_val | v6.2-A | 5 | 0.9232123506049185 |  |  |  |  |  | available |
| stratified_train_val | v6.2-A | 10 | 0.927755752028969 |  |  |  |  |  | available |
| stratified_train_val | hierarchical v6.2 phase2 | 1 | 0.4017060761922714 |  |  |  |  |  | available |
| stratified_train_val | hierarchical v6.2 phase2 | 5 | 0.397284660935126 |  |  |  |  |  | available |
| stratified_train_val | hierarchical v6.2 phase2 | 10 | 0.3780477751043327 |  |  |  |  |  | available |
| board_grouped | hierarchical v6.2 phase2 | 10 |  | 0.24987349086129806 | C02 | 0.2042277499082381 | W03 | 0.3189077692411215 | grouped kNN available only for this k in eval_v002 |
| board_grouped | ImageNet ResNet-18 | 10 |  | 0.3930272194094586 | C01 | 0.27603304971052744 | W01 | 0.5375053235213734 | grouped kNN available only for this k in eval_v002 |
| board_grouped | Random ResNet-18 | 10 |  | 0.2617083239705258 | W02 | 0.198524633624543 | C01 | 0.3378849152699632 | grouped kNN available only for this k in eval_v002 |
| board_grouped | v6.1 | 10 |  | 0.8709697195504678 | C01 | 0.7861348174101798 | C03 | 0.9338313540169543 | grouped kNN available only for this k in eval_v002 |
| board_grouped | v6.2-A | 10 |  | 0.9356408210770706 | W01 | 0.9081748250444873 | C03 | 0.9568787325554176 | grouped kNN available only for this k in eval_v002 |
| material_grouped | hierarchical v6.2 phase2 | 10 |  | 0.24469544501204776 | wood | 0.23229839588711307 | carbon | 0.25709249413698243 | grouped kNN available only for this k in eval_v002 |
| material_grouped | ImageNet ResNet-18 | 10 |  | 0.22515158572778654 | carbon | 0.17244212434113235 | wood | 0.27786104711444076 | grouped kNN available only for this k in eval_v002 |
| material_grouped | Random ResNet-18 | 10 |  | 0.2459369818476293 | carbon | 0.23465212417866338 | wood | 0.2572218395165952 | grouped kNN available only for this k in eval_v002 |
| material_grouped | v6.1 | 10 |  | 0.8632932432336362 | wood | 0.8531773282001325 | carbon | 0.87340915826714 | grouped kNN available only for this k in eval_v002 |
| material_grouped | v6.2-A | 10 |  | 0.9348717330088938 | wood | 0.9300893427639704 | carbon | 0.9396541232538173 | grouped kNN available only for this k in eval_v002 |

## Best-k Stratified Diagnostic

| encoder_name | best_k | best_stratified_macro_f1 | fixed_k10_macro_f1 | difference_best_minus_fixed_k10_pp |
| --- | --- | --- | --- | --- |
| hierarchical v6.2 phase2 | 1 | 0.4017060761922714 | 0.3780477751043327 | 2.3658301087938707 |
| ImageNet ResNet-18 | 5 | 0.8715130509203878 | 0.8434203551215631 | 2.8092695798824674 |
| Random ResNet-18 | 5 | 0.8439863653774754 | 0.8114196118916214 | 3.2566753485854028 |
| v6.1 | 5 | 0.9506911900286517 | 0.936526975749645 | 1.4164214279006604 |
| v6.2-A | 10 | 0.927755752028969 | 0.927755752028969 | 0.0 |

## Grouped kNN Availability

- Stratified train-to-val kNN values available: `[1, 3, 5, 10, 20]`.
- Board LOBO grouped kNN values available: `[10]`.
- Material LOMO grouped kNN values available: `[10]`.
- eval_v002 stores grouped LOBO/LOMO kNN rows only for k=10; fixed-k sensitivity across k=1,3,5,20 is available for stratified train-to-val only.

## Board LOBO kNN Summary

| protocol | encoder_name | k | mean_macro_f1 | worst_group | worst_group_macro_f1 | best_group | best_group_macro_f1 | n_groups |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| board_grouped | hierarchical v6.2 phase2 | 10 | 0.24987349086129806 | C02 | 0.2042277499082381 | W03 | 0.3189077692411215 | 6 |
| board_grouped | ImageNet ResNet-18 | 10 | 0.3930272194094586 | C01 | 0.27603304971052744 | W01 | 0.5375053235213734 | 6 |
| board_grouped | Random ResNet-18 | 10 | 0.2617083239705258 | W02 | 0.198524633624543 | C01 | 0.3378849152699632 | 6 |
| board_grouped | v6.1 | 10 | 0.8709697195504678 | C01 | 0.7861348174101798 | C03 | 0.9338313540169543 | 6 |
| board_grouped | v6.2-A | 10 | 0.9356408210770706 | W01 | 0.9081748250444873 | C03 | 0.9568787325554176 | 6 |

## Material LOMO kNN Summary

| protocol | encoder_name | k | mean_macro_f1 | worst_group | worst_group_macro_f1 | best_group | best_group_macro_f1 | n_groups |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| material_grouped | hierarchical v6.2 phase2 | 10 | 0.24469544501204776 | wood | 0.23229839588711307 | carbon | 0.25709249413698243 | 2 |
| material_grouped | ImageNet ResNet-18 | 10 | 0.22515158572778654 | carbon | 0.17244212434113235 | wood | 0.27786104711444076 | 2 |
| material_grouped | Random ResNet-18 | 10 | 0.2459369818476293 | carbon | 0.23465212417866338 | wood | 0.2572218395165952 | 2 |
| material_grouped | v6.1 | 10 | 0.8632932432336362 | wood | 0.8531773282001325 | carbon | 0.87340915826714 | 2 |
| material_grouped | v6.2-A | 10 | 0.9348717330088938 | wood | 0.9300893427639704 | carbon | 0.9396541232538173 | 2 |

## Reviewer-Facing Interpretation

- Recommended primary k for manuscript reporting: `k=10`.
- Rationale: k=10 is the development/default grouped kNN setting in eval_v002 and avoids selecting k solely by maximum stratified score.
- At fixed k=10, v6.1 remains above v6.2-A under stratified train-to-val evaluation (93.65% vs 92.78%).
- At grouped k=10, v6.2-A is strongest for Board LOBO (93.56%), and v6.2-A is strongest for Material LOMO (93.49%).
- Grouped LOBO/LOMO by-k sensitivity is not available in eval_v002 beyond k=10. Therefore, grouped kNN metrics should be treated as fixed-k locked summaries, while best-k selection is used only as a stratified diagnostic sensitivity analysis.
- Best-k results should be labeled as diagnostic sensitivity analysis, not as the primary manuscript protocol.
- Since the grouped conclusion uses the locked k=10 grouped rows, the v6.2-A grouped robustness conclusion is not based on post-hoc selection among grouped k values.