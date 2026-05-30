# v003 Submission Readiness Audit

## Inputs

- Manuscript: `manuscripts\FROZEN_REPRESENTATION_AUDITS_DRAFT_v003_OLE_style.md`
- Tables directory: `reports\publication_assets\final_tables_v001`
- Hardening directory: `reports\publication_assets\methodological_hardening_v001`

## Readiness Summary

- Ready for journal formatting: `no`
- Ready for reference completion: `no`
- Ready for figure generation: `yes`
- Checks passed/warned/failed: `47` / `3` / `1`

## Required Fixes Before Formatting

| section | check | status | evidence | recommendation |
| --- | --- | --- | --- | --- |
| Citations and bibliography | Reference list exists | fail | missing | Add a References section in the target journal style. |

## Optional Improvements

| section | check | status | evidence | recommendation |
| --- | --- | --- | --- | --- |
| Manuscript structure | Related Work section | warn | not present | Add a Related Work section before journal submission unless the target venue accepts a combined Introduction/Methods framing. |
| Citations and bibliography | Minimum citation blocks needed | warn | ESPI/speckle interferometry; fringe analysis/phase retrieval; deep learning for ESPI/optical metrology; frozen encoder/representation learning; grouped OOD/distribution-shift evaluation; frequency-control/leakage controls | Prepare citation blocks before journal formatting. |
| Abstract readiness | Metric density | warn | 7 metric tokens | Consider compressing the abstract by retaining only fixed-k stratified, grouped v6.2-A, and frequency-only summary values. |

## Detailed Checks

| section | check | status | evidence | recommendation |
| --- | --- | --- | --- | --- |
| Manuscript structure | Title | pass | present |  |
| Manuscript structure | Abstract | pass | present |  |
| Manuscript structure | Introduction | pass | present |  |
| Manuscript structure | Methods | pass | present |  |
| Manuscript structure | Results | pass | present |  |
| Manuscript structure | Discussion | pass | present |  |
| Manuscript structure | Limitations/Future Work | pass | present |  |
| Manuscript structure | Conclusion | pass | present |  |
| Manuscript structure | Data/Code/Reproducibility Availability | pass | present |  |
| Manuscript structure | Tables/Figures list | pass | present |  |
| Manuscript structure | Related Work section | warn | not present | Add a Related Work section before journal submission unless the target venue accepts a combined Introduction/Methods framing. |
| Claim boundary | v6.2-A as strongest image-derived frozen ESPI encoder baseline | pass | phrase present |  |
| Claim boundary | frequency_hz as dominant metadata-only predictor | pass | frequency dominance stated |  |
| Claim boundary | positive acoustic-response prediction claim | pass | absent or explicitly negated/scope-limited |  |
| Claim boundary | positive validated Physics-Aligned Encoder claim | pass | absent or explicitly negated/scope-limited |  |
| Claim boundary | positive LeFFT superiority claim | pass | absent or explicitly negated/scope-limited |  |
| Claim boundary | positive trained LeFFT model claim | pass | absent or explicitly negated/scope-limited |  |
| Claim boundary | spectral descriptors supplementary/control-only | pass | supplementary/control wording present |  |
| Tables | Table 1 reference and file | pass | reference=True; file_exists=True; path=reports\publication_assets\final_tables_v001\TABLE_1_ENCODER_BASELINE_SUMMARY_FIXED_K10.md |  |
| Tables | Table 2 reference and file | pass | reference=True; file_exists=True; path=reports\publication_assets\final_tables_v001\TABLE_2_GROUPED_ROBUSTNESS_WITH_CI.md |  |
| Tables | Table 3 reference and file | pass | reference=True; file_exists=True; path=reports\publication_assets\final_tables_v001\TABLE_3_CONTROLLED_ARCHITECTURE_COMPARISON.md |  |
| Tables | Table 4 reference and file | pass | reference=True; file_exists=True; path=reports\publication_assets\final_tables_v001\TABLE_4_CLASS_SUPPORT_AND_IMBALANCE.md |  |
| Tables | Table 5 reference and file | pass | reference=True; file_exists=True; path=reports\publication_assets\final_tables_v001\TABLE_5_V62A_VS_V61_PER_CLASS_GROUPED_F1.md |  |
| Tables | Table 6 reference and file | pass | reference=True; file_exists=True; path=reports\publication_assets\final_tables_v001\TABLE_6_ZERO_RECALL_BREAKDOWN.md |  |
| Tables | Table 7 reference and file | pass | reference=True; file_exists=True; path=reports\publication_assets\final_tables_v001\TABLE_7_FREQUENCY_INFORMATION_BUDGET.md |  |
| Tables | Table 8 reference and file | pass | reference=True; file_exists=True; path=reports\publication_assets\final_tables_v001\TABLE_8_FIXED_K_VS_BEST_K_DIAGNOSTIC.md |  |
| Tables | Supplementary Table S1 reference and file | pass | reference=True; file_exists=True; path=reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S1_BOARD_LEVEL_SUPPORT.md |  |
| Tables | Supplementary Table S2 reference and file | pass | reference=True; file_exists=True; path=reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S2_MATERIAL_LEVEL_SUPPORT.md |  |
| Tables | Supplementary Table S3 reference and file | pass | reference=True; file_exists=True; path=reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S3_CLASS_BY_BOARD_SUPPORT.md |  |
| Tables | Supplementary Table S4 reference and file | pass | reference=True; file_exists=True; path=reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S4_CLASS_BY_MATERIAL_SUPPORT.md |  |
| Tables | Supplementary Table S5 reference and file | pass | reference=True; file_exists=True; path=reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S5_PAIRED_BOARD_AND_MATERIAL_DELTAS_V61_V62A.md |  |
| Tables | Supplementary Table S6 reference and file | pass | reference=True; file_exists=True; path=reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S6_FREQUENCY_FUSION_PER_CLASS_CHANGES.md |  |
| Tables | Supplementary Table S7 reference and file | pass | reference=True; file_exists=True; path=reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S7_SPECTRAL_DESCRIPTOR_CONTROL.md |  |
| Tables | Unresolved [Table X]/[Figure X] placeholders | pass | none |  |
| Figures | Figure 1 reference and file | pass | reference=True; existing_files=['reports/publication_assets/encoder_baseline_v001/FIG_1_STRATIFIED_VS_GROUPED_MACRO_F1.svg'] |  |
| Figures | Figure 2 reference and file | pass | reference=True; existing_files=['reports/publication_assets/encoder_baseline_v001/FIG_2_GROUPED_ROBUSTNESS_GAIN.svg'] |  |
| Figures | Figure 3 reference and file | pass | reference=True; existing_files=['reports/publication_assets/encoder_baseline_v001/FIG_3_ENCODER_DECISION_FLOW.svg'] |  |
| Figures | Figure 4 reference and file | pass | reference=True; existing_files=['reports/publication_assets/class_level_v001/FIG_4_CLASS_SUPPORT_IMBALANCE.svg'] |  |
| Figures | Figure 5 reference and file | pass | reference=True; existing_files=['reports/publication_assets/class_level_v001/FIG_5_V62A_VS_V61_PER_CLASS_GROUPED_F1.svg'] |  |
| Figures | Figure 6 reference and file | pass | reference=True; existing_files=['reports/publication_assets/class_level_v001/FIG_6_ZERO_RECALL_BREAKDOWN_BY_ENCODER.svg'] |  |
| Citations and bibliography | Inline citations detected | pass | 1 citation-like tokens detected | Add citations throughout Introduction, Methods context, and Discussion before submission. |
| Citations and bibliography | Reference list exists | fail | missing | Add a References section in the target journal style. |
| Citations and bibliography | Minimum citation blocks needed | warn | ESPI/speckle interferometry; fringe analysis/phase retrieval; deep learning for ESPI/optical metrology; frozen encoder/representation learning; grouped OOD/distribution-shift evaluation; frequency-control/leakage controls | Prepare citation blocks before journal formatting. |
| Terminology | image-derived frozen ESPI encoder | pass | present |  |
| Terminology | grouped board/material | pass | present |  |
| Terminology | material-held-out stress test | pass | present |  |
| Terminology | deterministic spectral | pass | present |  |
| Terminology | LeFFT-inspired | pass | present |  |
| Terminology | LeFFT-inspired, not trained LeFFT | pass | trained LeFFT wording absent or negated |  |
| Abstract readiness | Word count | pass | 198 words |  |
| Abstract readiness | Metric density | warn | 7 metric tokens | Consider compressing the abstract by retaining only fixed-k stratified, grouped v6.2-A, and frequency-only summary values. |

## Recommended Minimum Citation Blocks

- ESPI / speckle interferometry.
- Fringe analysis / phase retrieval.
- Deep learning for ESPI / optical metrology.
- Frozen encoder / representation learning.
- Grouped OOD / distribution-shift evaluation.
- Frequency-control / leakage-control baselines.