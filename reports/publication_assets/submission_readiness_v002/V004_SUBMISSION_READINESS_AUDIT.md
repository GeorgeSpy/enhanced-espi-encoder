# V004 Submission Readiness Audit

## Inputs

- Manuscript: `manuscripts\FROZEN_REPRESENTATION_AUDITS_DRAFT_v004_related_work_refs.md`
- Tables directory: `reports\publication_assets\final_tables_v001`
- Hardening directory: `reports\publication_assets\methodological_hardening_v001`

## Readiness Summary

- Ready for journal formatting: `yes`
- Ready for reference cleanup: `yes`
- Ready for figure generation: `yes`
- Checks passed/warned/failed: `69` / `2` / `0`

## Required Fixes Before Formatting

_No required fixes._

## Optional Improvements

| section | check | status | evidence | recommendation |
| --- | --- | --- | --- | --- |
| Citations and bibliography | References needing DOI/page/venue cleanup | warn | [3], [6], [7], [8] | Add DOI fields during venue-specific reference cleanup where available. |
| Abstract readiness | Metric density | warn | 7 metric tokens | Consider compressing the abstract by retaining only fixed-k stratified, grouped v6.2-A, and frequency-only summary values. |

## Detailed Checks

| section | check | status | evidence | recommendation |
| --- | --- | --- | --- | --- |
| Manuscript structure | Title | pass | present |  |
| Manuscript structure | Abstract | pass | present |  |
| Manuscript structure | Introduction | pass | present |  |
| Manuscript structure | Related Work | pass | present |  |
| Manuscript structure | Methods | pass | present |  |
| Manuscript structure | Results | pass | present |  |
| Manuscript structure | Discussion | pass | present |  |
| Manuscript structure | Limitations/Future Work | pass | present |  |
| Manuscript structure | Conclusion | pass | present |  |
| Manuscript structure | Data/Code/Reproducibility Availability | pass | present |  |
| Manuscript structure | References | pass | present |  |
| Manuscript structure | Tables/Figures list | pass | present |  |
| Claim boundary | v6.2-A as strongest image-derived frozen ESPI encoder baseline | pass | phrase present |  |
| Claim boundary | frequency_hz as dominant metadata-only predictor | pass | frequency dominance stated |  |
| Claim boundary | positive acoustic-response prediction claim | pass | absent or explicitly negated/scope-limited |  |
| Claim boundary | positive validated Physics-Aligned Encoder claim | pass | absent or explicitly negated/scope-limited |  |
| Claim boundary | positive LeFFT superiority claim | pass | absent or explicitly negated/scope-limited |  |
| Claim boundary | positive trained LeFFT model claim | pass | absent or explicitly negated/scope-limited |  |
| Claim boundary | positive retrained CNN LOBO/LOMO generalization claim | pass | absent or explicitly negated/scope-limited |  |
| Claim boundary | spectral descriptors supplementary/control-only | pass | supplementary/control wording present |  |
| Claim boundary | Material LOMO descriptive/stress-test caution | pass | material-held-out stress-test wording present |  |
| Metrics | v6.1 fixed k=10 stratified Macro-F1 | pass | 93.65% present |  |
| Metrics | v6.2-A fixed k=10 stratified Macro-F1 | pass | 92.78% present |  |
| Metrics | v6.1 diagnostic best-k Macro-F1 | pass | 95.07% present |  |
| Metrics | v6.2-A Board LOBO Macro-F1 | pass | 93.56% present |  |
| Metrics | v6.2-A Material LOMO Macro-F1 | pass | 93.49% present |  |
| Metrics | frequency-only stratified Macro-F1 | pass | 98.67% present |  |
| Metrics | frequency-only Board LOBO Macro-F1 | pass | 97.49% present |  |
| Metrics | frequency-only Material LOMO Macro-F1 | pass | 94.09% present |  |
| Metrics | frequency + v6.2-A stratified Macro-F1 | pass | 96.75% present |  |
| Metrics | frequency + v6.2-A Board LOBO Macro-F1 | pass | 94.79% present |  |
| Metrics | frequency + v6.2-A Material LOMO Macro-F1 | pass | 94.50% present |  |
| Metrics | v6.1 95.07% appears only as diagnostic/best-k | pass | The best-k diagnostic does not change the qualitative interpretation. v6.1 reaches 95.07% at `k=5`, whereas v6.2-A reaches 92.78% at `k=10`. Best-k values are reported as sensitivity analysis only; fixed `k=10` is the primary kNN protocol [Table 8]. |  |
| Metrics | spectral descriptor control remains supplementary | pass | supplementary spectral reference present |  |
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
| Citations and bibliography | Related Work exists | pass | present |  |
| Citations and bibliography | Reference list exists | pass | present |  |
| Citations and bibliography | Every in-text numbered citation has matching reference | pass | missing references=none |  |
| Citations and bibliography | Every reference item is cited at least once | pass | uncited references=none |  |
| Citations and bibliography | Unresolved citation placeholders | pass | none |  |
| Citations and bibliography | References needing DOI/page/venue cleanup | warn | [3], [6], [7], [8] | Add DOI fields during venue-specific reference cleanup where available. |
| Terminology | image-derived frozen ESPI encoder | pass | present |  |
| Terminology | grouped board/material | pass | present |  |
| Terminology | material-held-out stress test | pass | present |  |
| Terminology | deterministic spectral | pass | present |  |
| Terminology | LeFFT-inspired | pass | present |  |
| Terminology | frequency metadata | pass | present |  |
| Terminology | LeFFT-inspired, not trained LeFFT | pass | trained LeFFT wording absent or negated |  |
| Abstract readiness | Word count | pass | 198 words |  |
| Abstract readiness | Metric density | warn | 7 metric tokens | Consider compressing the abstract by retaining only fixed-k stratified, grouped v6.2-A, and frequency-only summary values. |