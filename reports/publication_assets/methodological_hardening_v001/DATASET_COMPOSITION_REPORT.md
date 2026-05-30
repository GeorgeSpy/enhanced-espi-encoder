# Dataset Composition and Grouped-Fold Support Audit

## Purpose

This report documents dataset composition and grouped-fold support for the frozen ESPI encoder evaluation. It is generated from existing manifest/feature metadata and does not rerun encoder evaluation, feature extraction, image loading, LeFFT analysis, or acoustic-response prediction.

## Key Numbers

- Total samples: 12944
- Number of boards: 6
- Boards: C01, C02, C03, W01, W02, W03
- Number of materials: 2
- Materials: carbon, wood
- Number of distribution groups: 13

## Class Distribution

| class_id | label_name | support | percentage |
| --- | --- | --- | --- |
| 0 | 1_1H | 653 | 5.04% |
| 1 | 1_1T | 669 | 5.17% |
| 2 | 1_2 | 738 | 5.70% |
| 3 | 2_1 | 769 | 5.94% |
| 4 | higher | 10115 | 78.14% |

## Supplementary Table Mapping

| Supplementary item | Generated CSV | Description |
| --- | --- | --- |
| Supplementary Table S1 | `dataset_composition_by_board.csv` | Board-level support with material, distribution groups, class counts, and total samples. |
| Supplementary Table S2 | `dataset_composition_by_material.csv` | Material-level support with class counts, total samples, and number of boards. |
| Supplementary Table S3 | `class_by_board_support.csv` | Class-by-board support and missing-class indicators. |
| Supplementary Table S4 | `class_by_material_support.csv` | Class-by-material support and missing-class indicators. |

## Board-Level Support Preview

| board | material | distribution_group_count | distribution_groups | class_0 | class_1 | class_2 | class_3 | class_4 | total_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C01 | carbon | 2 | C01_ESPI_90db;C01_ESPI_90db-Averaged | 129 | 170 | 57 | 77 | 2300 | 2733 |
| C02 | carbon | 2 | C02_ESPI_90db;C02_ESPI_90db-Averaged | 134 | 112 | 141 | 157 | 1775 | 2319 |
| C03 | carbon | 2 | C03_ESPI_90db;C03_ESPI_90db-Averaged | 110 | 74 | 98 | 117 | 1238 | 1637 |
| W01 | wood | 3 | W01_ESPI_90db;W01_ESPI_90db-Averaged;W01_ESPI_90db-PseudoNoisy_MATCH_v25 | 116 | 107 | 177 | 185 | 1834 | 2419 |
| W02 | wood | 2 | W02_ESPI_90db;W02_ESPI_90db-Averaged | 74 | 113 | 142 | 169 | 1872 | 2370 |
| W03 | wood | 2 | W03_ESPI_90db;W03_ESPI_90db-Averaged | 90 | 93 | 123 | 64 | 1096 | 1466 |

## Material-Level Support Preview

| material | class_0 | class_1 | class_2 | class_3 | class_4 | total_samples | number_of_boards | boards |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| carbon | 373 | 356 | 296 | 351 | 5313 | 6689 | 3 | C01,C02,C03 |
| wood | 280 | 313 | 442 | 418 | 4802 | 6255 | 3 | W01,W02,W03 |

## Grouped-Fold Interpretation

- All boards contain all classes: yes
- All materials contain all classes: yes
- Board LOBO folds contain missing held-out classes: no
- Board LOBO folds contain missing reference classes: no
- Material LOMO folds contain missing held-out classes: no
- Material LOMO folds contain missing reference classes: no

## Reviewer-Facing Interpretation

The dataset contains all five modal classes in every board group and every material group. This supports Macro-F1 reporting for board- and material-held-out frozen-embedding evaluation because no held-out fold lacks an entire class and no reference fold lacks an entire class.

Board LOBO-style evaluation has six held-out board groups and should be interpreted as a grouped robustness stress test over board variation. Material LOMO-style evaluation has two material groups (`carbon` and `wood`). Because there are only two material groups, Material LOMO should be interpreted as a material-held-out stress test rather than a broad inferential estimate over a large population of material domains.

The grouped metrics should therefore be described as frozen-embedding grouped robustness tests. They support the manuscript claim that v6.2-A is the primary reportable frozen ESPI encoder baseline under grouped board/material evaluation, while avoiding claims of full retrained CNN LOBO/LOMO generalization.

## Caution Flags

- Low support threshold used for fold caution flags: 30 samples per class.
- Board LOBO folds with caution flags: 0
- Material LOMO folds with caution flags: 2
- Material LOMO folds are flagged for `small_number_of_material_groups` because only two material groups are available.
