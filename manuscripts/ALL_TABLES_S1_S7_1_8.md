# Combined Tables 1-8 and S1-S7

## SUPPLEMENTARY_TABLE_S1_BOARD_LEVEL_SUPPORT.md

| board | material | distribution_group_count | distribution_groups | class_0 | class_1 | class_2 | class_3 | class_4 | total_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C01 | carbon | 2 | C01_ESPI_90db;C01_ESPI_90db-Averaged | 129 | 170 | 57 | 77 | 2300 | 2733 |
| C02 | carbon | 2 | C02_ESPI_90db;C02_ESPI_90db-Averaged | 134 | 112 | 141 | 157 | 1775 | 2319 |
| C03 | carbon | 2 | C03_ESPI_90db;C03_ESPI_90db-Averaged | 110 | 74 | 98 | 117 | 1238 | 1637 |
| W01 | wood | 3 | W01_ESPI_90db;W01_ESPI_90db-Averaged;W01_ESPI_90db-PseudoNoisy_MATCH_v25 | 116 | 107 | 177 | 185 | 1834 | 2419 |
| W02 | wood | 2 | W02_ESPI_90db;W02_ESPI_90db-Averaged | 74 | 113 | 142 | 169 | 1872 | 2370 |
| W03 | wood | 2 | W03_ESPI_90db;W03_ESPI_90db-Averaged | 90 | 93 | 123 | 64 | 1096 | 1466 |


---

## SUPPLEMENTARY_TABLE_S2_MATERIAL_LEVEL_SUPPORT.md

| material | class_0 | class_1 | class_2 | class_3 | class_4 | total_samples | number_of_boards | boards |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| carbon | 373 | 356 | 296 | 351 | 5313 | 6689 | 3 | C01,C02,C03 |
| wood | 280 | 313 | 442 | 418 | 4802 | 6255 | 3 | W01,W02,W03 |


---

## SUPPLEMENTARY_TABLE_S3_CLASS_BY_BOARD_SUPPORT.md

| board | material | class_0 | class_1 | class_2 | class_3 | class_4 | total | missing_classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C01 | carbon | 129 | 170 | 57 | 77 | 2300 | 2733 | none |
| C02 | carbon | 134 | 112 | 141 | 157 | 1775 | 2319 | none |
| C03 | carbon | 110 | 74 | 98 | 117 | 1238 | 1637 | none |
| W01 | wood | 116 | 107 | 177 | 185 | 1834 | 2419 | none |
| W02 | wood | 74 | 113 | 142 | 169 | 1872 | 2370 | none |
| W03 | wood | 90 | 93 | 123 | 64 | 1096 | 1466 | none |


---

## SUPPLEMENTARY_TABLE_S4_CLASS_BY_MATERIAL_SUPPORT.md

| material | class_0 | class_1 | class_2 | class_3 | class_4 | total | missing_classes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| carbon | 373 | 356 | 296 | 351 | 5313 | 6689 | none |
| wood | 280 | 313 | 442 | 418 | 4802 | 6255 | none |


---

## SUPPLEMENTARY_TABLE_S5_PAIRED_BOARD_AND_MATERIAL_DELTAS_V61_V62A.md

| Protocol | Held-out group | Material | Held-out total | v6.1 Macro-F1 | v6.2-A Macro-F1 | Delta v6.2-A minus v6.1 (pp) | Winner | Caution note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Board LOBO | C01 | carbon | 2733 | 78.61% | 93.63% | 15.01 | v6.2-A | paired board fold |
| Board LOBO | C02 | carbon | 2319 | 92.85% | 92.83% | -0.02 | v6.1 | paired board fold |
| Board LOBO | C03 | carbon | 1637 | 93.38% | 95.69% | 2.3 | v6.2-A | paired board fold |
| Board LOBO | W01 | wood | 2419 | 90.55% | 90.82% | 0.27 | v6.2-A | paired board fold |
| Board LOBO | W02 | wood | 2370 | 83.34% | 95.41% | 12.07 | v6.2-A | paired board fold |
| Board LOBO | W03 | wood | 1466 | 83.84% | 93.01% | 9.17 | v6.2-A | paired board fold |
| Material LOMO | carbon | carbon | 6689 | 87.34% | 93.97% | 6.62 | v6.2-A | descriptive stress-test only; two material groups available |
| Material LOMO | wood | wood | 6255 | 85.32% | 93.01% | 7.69 | v6.2-A | descriptive stress-test only; two material groups available |


---

## SUPPLEMENTARY_TABLE_S6_FREQUENCY_FUSION_PER_CLASS_CHANGES.md

| Protocol | Class ID | Class name | Frequency only | v6.2-A embedding-only | Frequency + v6.2-A | Fusion minus frequency-only | Fusion minus embedding-only |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stratified_train_val | 0 | 1_1H | 100.00% | 94.16% | 98.86% | -1.14 pp | +4.70 pp |
| stratified_train_val | 1 | 1_1T | 100.00% | 95.79% | 98.14% | -1.86 pp | +2.36 pp |
| stratified_train_val | 2 | 1_2 | 96.53% | 89.20% | 93.15% | -3.38 pp | +3.95 pp |
| stratified_train_val | 3 | 2_1 | 96.82% | 89.63% | 93.85% | -2.96 pp | +4.22 pp |
| stratified_train_val | 4 | higher | 100.00% | 98.63% | 99.73% | -0.27 pp | +1.10 pp |
| board_grouped | 0 | 1_1H | 100.00% | 94.22% | 95.13% | -4.87 pp | +0.91 pp |
| board_grouped | 1 | 1_1T | 100.00% | 95.26% | 95.53% | -4.47 pp | +0.27 pp |
| board_grouped | 2 | 1_2 | 94.17% | 91.23% | 92.26% | -1.91 pp | +1.04 pp |
| board_grouped | 3 | 2_1 | 93.26% | 92.12% | 92.29% | -0.97 pp | +0.17 pp |
| board_grouped | 4 | higher | 100.00% | 98.58% | 98.72% | -1.28 pp | +0.14 pp |
| material_grouped | 0 | 1_1H | 100.00% | 94.10% | 94.56% | -5.44 pp | +0.47 pp |
| material_grouped | 1 | 1_1T | 100.00% | 95.14% | 95.61% | -4.39 pp | +0.47 pp |
| material_grouped | 2 | 1_2 | 85.09% | 91.09% | 91.52% | +6.43 pp | +0.43 pp |
| material_grouped | 3 | 2_1 | 85.34% | 91.42% | 92.07% | +6.73 pp | +0.65 pp |
| material_grouped | 4 | higher | 100.00% | 98.53% | 98.73% | -1.27 pp | +0.21 pp |


---

## SUPPLEMENTARY_TABLE_S7_SPECTRAL_DESCRIPTOR_CONTROL.md

| Condition | Stratified Macro-F1 | Board LOBO Macro-F1 | Material LOMO Macro-F1 | Delta vs frequency-only | Delta vs v6.2-A only | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| Frequency-only | 98.67% | 97.49% | 94.09% | reference | Stratified +5.19 pp, Board +3.20 pp, Material +0.03 pp | Metadata-only control |
| Spectral descriptors only | 52.92% | 22.77% | 19.72% | Stratified -45.75 pp, Board -74.71 pp, Material -74.37 pp | Stratified -40.56 pp, Board -71.51 pp, Material -74.33 pp | Deterministic spectral / LeFFT-inspired descriptors |
| v6.2-A embedding-only | 93.48% | 94.28% | 94.05% | Stratified -5.19 pp, Board -3.20 pp, Material -0.03 pp | reference | Image-derived frozen encoder reference |
| Frequency + spectral descriptors | 98.53% | 93.46% | 73.33% | Stratified -0.13 pp, Board -4.03 pp, Material -20.76 pp | Stratified +5.05 pp, Board -0.82 pp, Material -20.73 pp | Metadata plus deterministic spectral descriptors |
| v6.2-A + spectral descriptors | 93.48% | 94.35% | 94.00% | Stratified -5.19 pp, Board -3.13 pp, Material -0.08 pp | Stratified -0.00 pp, Board +0.07 pp, Material -0.05 pp | Image embedding plus deterministic spectral descriptors |
| Frequency + v6.2-A | 96.75% | 94.79% | 94.50% | Stratified -1.92 pp, Board -2.70 pp, Material +0.41 pp | Stratified +3.27 pp, Board +0.50 pp, Material +0.45 pp | Current strongest metadata/image fusion reference |
| Frequency + v6.2-A + spectral descriptors | 96.41% | 94.97% | 94.64% | Stratified -2.26 pp, Board -2.51 pp, Material +0.55 pp | Stratified +2.93 pp, Board +0.69 pp, Material +0.58 pp | Full deterministic spectral control fusion |


---

## TABLE_1_ENCODER_BASELINE_SUMMARY_FIXED_K10.md

| Encoder | Role | Embedding dimension | Primary stratified kNN Macro-F1 at fixed k=10 | Board LOBO Macro-F1 at fixed k=10 | Material LOMO Macro-F1 at fixed k=10 | Main interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| Random ResNet-18 | Random generic visual-control encoder | 512 | 81.14% | 26.17% | 24.59% | Non-trivial stratified texture baseline but weak grouped robustness. |
| ImageNet ResNet-18 | ImageNet generic visual-control encoder | 512 | 84.34% | 39.30% | 22.52% | Generic pretrained features remain insufficient under grouped evaluation. |
| v6.1 | Reference ESPI-specific frozen ResNet18-style baseline | 512 | 93.65% | 87.10% | 86.33% | Strong stratified ESPI-specific representation, weaker grouped robustness than v6.2-A. |
| v6.2-A | Primary image-derived frozen ESPI encoder baseline | 1280 | 92.78% | 93.56% | 93.49% | Strongest image-derived frozen ESPI encoder under grouped board/material evaluation. |
| hierarchical v6.2 phase2 | Controlled physics-aware architecture comparison | 512 | 37.80% | 24.99% | 24.47% | Technically valid but not superior; internal comparison only pending metadata regeneration. |


---

## TABLE_2_GROUPED_ROBUSTNESS_WITH_CI.md

| Encoder | Board LOBO Macro-F1 | Board LOBO 95% CI if available | Material LOMO Macro-F1 | Material LOMO 95% CI if available | Worst board | Worst board Macro-F1 | Worst material | Worst material Macro-F1 | Caution note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random ResNet-18 | 26.17% | not available for fixed k=10 | 24.59% | not available for fixed k=10 | W02 | 19.85% | carbon | 23.47% | Material LOMO is descriptive because only two material groups are available. |
| ImageNet ResNet-18 | 39.30% | [33.16%, 46.36%] | 22.52% | not available for fixed k=10 | C01 | 27.60% | carbon | 17.24% | Material LOMO is descriptive because only two material groups are available. |
| v6.1 | 87.10% | [82.73%, 91.23%] | 86.33% | [85.32%, 87.34%] | C01 | 78.61% | wood | 85.32% | Material LOMO is descriptive because only two material groups are available. |
| v6.2-A | 93.56% | [92.36%, 94.67%] | 93.49% | [93.01%, 93.97%] | W01 | 90.82% | wood | 93.01% | Material LOMO is descriptive because only two material groups are available. |
| hierarchical v6.2 phase2 | 24.99% | not available for fixed k=10 | 24.47% | [23.23%, 25.71%] | C02 | 20.42% | wood | 23.23% | Material LOMO is descriptive because only two material groups are available. Hierarchical row is internal-only pending source metadata regeneration. |


---

## TABLE_3_CONTROLLED_ARCHITECTURE_COMPARISON.md

| Encoder | Architecture role | Embedding point | Embedding dimension | Stratified fixed k=10 Macro-F1 | Board LOBO Macro-F1 | Material LOMO Macro-F1 | Publication decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v6.1 | Reference frozen ResNet18-style ESPI baseline | avgpool_pre_classifier | 512 | 93.65% | 87.10% | 86.33% | Retain as strongest stratified ESPI-specific reference. |
| v6.2-A | Official reportable image-derived frozen encoder baseline | MCDropoutClassifier.global_pool | 1280 | 92.78% | 93.56% | 93.49% | Use as primary reportable image-derived frozen ESPI encoder baseline. |
| hierarchical v6.2 phase2 | Physics-aware hierarchical phase2 expert comparison branch | z_expert_prelogit / z_arcface_input | 512 | 37.80% | 24.99% | 24.47% | Keep internal/controlled; not final-table ready until source metadata are regenerated. |


---

## TABLE_4_CLASS_SUPPORT_AND_IMBALANCE.md

| Class ID | Class name | Total support | Support percentage | Notes on imbalance |
| --- | --- | --- | --- | --- |
| 0 | 1_1H | 653 | 5.04% | Minority class; grouped robustness is sensitive to errors. |
| 1 | 1_1T | 669 | 5.17% | Minority class; grouped robustness is sensitive to errors. |
| 2 | 1_2 | 738 | 5.70% | Minority class; grouped robustness is sensitive to errors. |
| 3 | 2_1 | 769 | 5.94% | Minority class; grouped robustness is sensitive to errors. |
| 4 | higher | 10115 | 78.14% | Dominant majority class. |


---

## TABLE_5_V62A_VS_V61_PER_CLASS_GROUPED_F1.md

| Class name | v6.1 grouped F1 | v6.2-A grouped F1 | Delta pp | Interpretation |
| --- | --- | --- | --- | --- |
| 2_1 | 77.88% | 91.32% | 13.43 pp | Large grouped stability gain for a difficult modal class. |
| 1_2 | 78.35% | 90.57% | 12.21 pp | Large grouped stability gain for a difficult modal class. |
| 1_1H | 89.11% | 92.65% | 3.53 pp | Moderate grouped stability gain. |
| 1_1T | 92.97% | 94.46% | 1.49 pp | Small positive grouped stability gain. |
| higher | 97.63% | 98.26% | 0.63 pp | Small positive grouped stability gain. |


---

## TABLE_6_ZERO_RECALL_BREAKDOWN.md

| Encoder | Zero-recall rows | Main affected protocol or condition if available | Interpretation |
| --- | --- | --- | --- |
| Random ResNet-18 | 18 | board_grouped, material_grouped, material_grouped_aggregate | Zero-recall cases indicate unstable grouped class recovery for this baseline. |
| ImageNet ResNet-18 | 20 | board_grouped, material_grouped, material_grouped_aggregate | Zero-recall cases indicate unstable grouped class recovery for this baseline. |
| v6.1 | 0 | none | No zero-recall rows in the generated class-level summary. |
| v6.2-A | 0 | none | No zero-recall rows in the generated class-level summary. |
| hierarchical v6.2 phase2 | 9 | board_grouped, material_grouped | Zero-recall cases indicate unstable grouped class recovery for this baseline. |


---

## TABLE_7_FREQUENCY_INFORMATION_BUDGET.md

| Condition | Stratified Macro-F1 | Board LOBO Macro-F1 | Material LOMO Macro-F1 | Delta vs frequency-only | Delta vs embedding-only | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| frequency-only | 98.67% | 97.49% | 94.09% | reference | not applicable | Frequency_hz is the dominant metadata-only predictor for the current label protocol. |
| v6.2-A embedding-only | 93.48% | 94.28% | 94.05% | Stratified -5.19 pp, Board -3.20 pp, Material -0.03 pp | reference | Image-derived v6.2-A embeddings remain the strongest frozen ESPI encoder baseline. |
| frequency + v6.2-A | 96.75% | 94.79% | 94.50% | Stratified -1.92 pp, Board -2.70 pp, Material +0.41 pp | Stratified +3.27 pp, Board +0.50 pp, Material +0.45 pp | Fusion improves over embedding-only but does not clearly exceed frequency-only except in Material LOMO. |


---

## TABLE_8_FIXED_K_VS_BEST_K_DIAGNOSTIC.md

| Encoder | Fixed k | Fixed-k stratified Macro-F1 | Best k | Best-k stratified Macro-F1 | Best-k minus fixed-k delta | Manuscript use |
| --- | --- | --- | --- | --- | --- | --- |
| Random ResNet-18 | 10 | 81.14% | 5 | 84.40% | +3.26 pp | Fixed k=10 is primary; best-k is diagnostic only. |
| ImageNet ResNet-18 | 10 | 84.34% | 5 | 87.15% | +2.81 pp | Fixed k=10 is primary; best-k is diagnostic only. |
| v6.1 | 10 | 93.65% | 5 | 95.07% | +1.42 pp | Fixed k=10 is primary; best-k is diagnostic only. |
| v6.2-A | 10 | 92.78% | 10 | 92.78% | 0.00 pp | Fixed k=10 is primary; best-k is diagnostic only. |
| hierarchical v6.2 phase2 | 10 | 37.80% | 1 | 40.17% | +2.37 pp | Fixed k=10 is primary; best-k is diagnostic only. |


---

## TABLE_INDEX.md

# Final Manuscript Table Index

All tables are generated from locked evaluation, class-diagnostic, and methodological-hardening artifacts. No experiments are rerun by this script.

| Table | Output Markdown | Output CSV | Source artifact paths | Notes |
| --- | --- | --- | --- | --- |
| Main text Table 1 | reports\publication_assets\final_tables_v001\TABLE_1_ENCODER_BASELINE_SUMMARY_FIXED_K10.md | reports\publication_assets\final_tables_v001\TABLE_1_ENCODER_BASELINE_SUMMARY_FIXED_K10.csv | reports/encoder_baselines/eval_v002/encoder_baseline_summary.csv; reports/publication_assets/methodological_hardening_v001/knn_protocol/knn_fixed_k_summary.csv | Generated from locked artifacts; no experiments rerun. |
| Main text Table 2 | reports\publication_assets\final_tables_v001\TABLE_2_GROUPED_ROBUSTNESS_WITH_CI.md | reports\publication_assets\final_tables_v001\TABLE_2_GROUPED_ROBUSTNESS_WITH_CI.csv | reports/encoder_baselines/eval_v002/encoder_baseline_summary.csv; reports/publication_assets/methodological_hardening_v001/knn_protocol/knn_fixed_k_summary.csv | Generated from locked artifacts; no experiments rerun. |
| Main text Table 3 | reports\publication_assets\final_tables_v001\TABLE_3_CONTROLLED_ARCHITECTURE_COMPARISON.md | reports\publication_assets\final_tables_v001\TABLE_3_CONTROLLED_ARCHITECTURE_COMPARISON.csv | reports/encoder_baselines/eval_v002/encoder_baseline_summary.csv; reports/publication_assets/methodological_hardening_v001/knn_protocol/knn_fixed_k_summary.csv | Generated from locked artifacts; no experiments rerun. |
| Main text Table 4 | reports\publication_assets\final_tables_v001\TABLE_4_CLASS_SUPPORT_AND_IMBALANCE.md | reports\publication_assets\final_tables_v001\TABLE_4_CLASS_SUPPORT_AND_IMBALANCE.csv | reports/publication_assets/class_level_v001/TABLE_4_CLASS_SUPPORT.csv | Generated from locked artifacts; no experiments rerun. |
| Main text Table 5 | reports\publication_assets\final_tables_v001\TABLE_5_V62A_VS_V61_PER_CLASS_GROUPED_F1.md | reports\publication_assets\final_tables_v001\TABLE_5_V62A_VS_V61_PER_CLASS_GROUPED_F1.csv | reports/publication_assets/class_level_v001/TABLE_5_V62A_VS_V61_GROUPED_PER_CLASS_F1.csv | Generated from locked artifacts; no experiments rerun. |
| Main text Table 6 | reports\publication_assets\final_tables_v001\TABLE_6_ZERO_RECALL_BREAKDOWN.md | reports\publication_assets\final_tables_v001\TABLE_6_ZERO_RECALL_BREAKDOWN.csv | reports/publication_assets/class_level_v001/TABLE_6_ZERO_RECALL_BREAKDOWN.csv | Generated from locked artifacts; no experiments rerun. |
| Main text Table 7 | reports\publication_assets\final_tables_v001\TABLE_7_FREQUENCY_INFORMATION_BUDGET.md | reports\publication_assets\final_tables_v001\TABLE_7_FREQUENCY_INFORMATION_BUDGET.csv | reports/publication_assets/methodological_hardening_v001/frequency_embedding_fusion/frequency_embedding_fusion_summary.csv | Generated from locked artifacts; no experiments rerun. |
| Main text Table 8 | reports\publication_assets\final_tables_v001\TABLE_8_FIXED_K_VS_BEST_K_DIAGNOSTIC.md | reports\publication_assets\final_tables_v001\TABLE_8_FIXED_K_VS_BEST_K_DIAGNOSTIC.csv | reports/publication_assets/methodological_hardening_v001/knn_protocol/knn_best_k_diagnostic.csv | Generated from locked artifacts; no experiments rerun. |
| Supplementary Table S1 | reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S1_BOARD_LEVEL_SUPPORT.md | reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S1_BOARD_LEVEL_SUPPORT.csv | reports/publication_assets/methodological_hardening_v001/dataset_composition_by_board.csv | Generated from locked artifacts; no experiments rerun. |
| Supplementary Table S2 | reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S2_MATERIAL_LEVEL_SUPPORT.md | reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S2_MATERIAL_LEVEL_SUPPORT.csv | reports/publication_assets/methodological_hardening_v001/dataset_composition_by_material.csv | Generated from locked artifacts; no experiments rerun. |
| Supplementary Table S3 | reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S3_CLASS_BY_BOARD_SUPPORT.md | reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S3_CLASS_BY_BOARD_SUPPORT.csv | reports/publication_assets/methodological_hardening_v001/class_by_board_support.csv | Generated from locked artifacts; no experiments rerun. |
| Supplementary Table S4 | reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S4_CLASS_BY_MATERIAL_SUPPORT.md | reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S4_CLASS_BY_MATERIAL_SUPPORT.csv | reports/publication_assets/methodological_hardening_v001/class_by_material_support.csv | Generated from locked artifacts; no experiments rerun. |
| Supplementary Table S5 | reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S5_PAIRED_BOARD_AND_MATERIAL_DELTAS_V61_V62A.md | reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S5_PAIRED_BOARD_AND_MATERIAL_DELTAS_V61_V62A.csv | reports/publication_assets/methodological_hardening_v001/paired_board_lobo_delta_v61_v62a.csv; reports/publication_assets/methodological_hardening_v001/paired_material_lomo_delta_v61_v62a.csv | Generated from locked artifacts; no experiments rerun. |
| Supplementary Table S6 | reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S6_FREQUENCY_FUSION_PER_CLASS_CHANGES.md | reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S6_FREQUENCY_FUSION_PER_CLASS_CHANGES.csv | reports/publication_assets/methodological_hardening_v001/frequency_embedding_fusion/frequency_embedding_fusion_per_class.csv | Generated from locked artifacts; no experiments rerun. |
| Supplementary Table S7 | reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S7_SPECTRAL_DESCRIPTOR_CONTROL.md | reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S7_SPECTRAL_DESCRIPTOR_CONTROL.csv | reports/publication_assets/methodological_hardening_v001/lefft_descriptor_ablation/eval_v001/spectral_descriptor_ablation_key_numbers.json | Generated from locked spectral descriptor ablation artifacts; no experiments rerun. |


---

