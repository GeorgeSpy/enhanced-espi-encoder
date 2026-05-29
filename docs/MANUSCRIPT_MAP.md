# Manuscript Map

This map links manuscript claims, tables, and figures to repository artifacts. Items marked `pending` should be completed before reviewer/public release.

## Working Title

```text
Toward a Physics-Aligned ESPI Encoder: Frozen Representation Evidence from
Full-Field Interferometric Measurements under Grouped OOD Evaluation
```

## Tables

| Manuscript item | Source script | Config | Input artifact | Output/report | Status |
|---|---|---|---|---|---|
| Table 1: v6.1 vs v6.2 classifier baseline | `scripts/v6_1/train_v6_1_head_only.py`, `scripts/v6_2/train_v6_2.py` | `configs/v6_2/config.antigravity.5class.yaml` | external manifest/checkpoints | `reports/v61_v62_comparison/COMPARE_SUMMARY.md` | done-summary |
| Table 2: frozen v6.2-A embedding audit | `scripts/encoder/audit_v62_embeddings.py` | not required after feature extraction | external `features_v62a_epoch25.npz` | `reports/encoder/EMBEDDING_AUDIT_SUMMARY.md` | done-summary |
| Table 3: grouped OOD frozen-embedding evaluation | `scripts/encoder/evaluate_encoder_grouped_generalization.py` | not required after feature extraction | external `features_v62a_epoch25.npz` | `reports/encoder/GROUPED_ENCODER_EVAL_SUMMARY.md` | done-summary |
| Table 4a: v6.2-A vs hierarchical frozen-embedding comparison | `scripts/encoder/compare_v62a_vs_hierarchical.py` | not required after audit reports | v6.2-A and hierarchical audit reports | `reports/encoder_comparison/COMPARE_V62A_vs_HIERARCHICAL.md` | done-external |
| Table 4b: Encoder baseline input audit | `scripts/encoder/audit_encoder_baseline_inputs.py` | not required | baseline manifests, checkpoints, feature dumps, and loaders | `reports/encoder_baselines/ENCODER_BASELINE_INPUT_AUDIT.md` | done-external |
| Table 4c: Feature schema normalization | `scripts/encoder/normalize_encoder_feature_schema.py` | not required | v6.2-A and hierarchical feature dumps plus manifest | `reports/encoder_baselines/FEATURE_SCHEMA_NORMALIZATION.md` | done-external |
| Table 4d: Random/ImageNet ResNet-18 baseline extraction | `scripts/encoder/extract_resnet18_baseline_embeddings.py` | not required | manifest and image root | `reports/encoder_baselines/RESNET18_BASELINE_EXTRACTION.md` | done-external |
| Table 4e: v6.1 embedding extraction | `scripts/encoder/extract_v61_embeddings.py` | not required | manifest, image root, and v6.1 checkpoint | `reports/encoder_baselines/V61_EMBEDDING_EXTRACTION.md` | done-external |
| Table 4: encoder baseline reference comparison | pending script | pending config | random/ImageNet/v6.1/v6.2/hierarchical embeddings | pending report | pending |
| Table 5: Unified encoder baseline evaluation | `scripts/encoder/evaluate_encoder_baselines.py` | not required after normalized feature dumps | normalized random/ImageNet/v6.1/v6.2-A/hierarchical feature dumps | `reports/encoder_baselines/eval_v001/ENCODER_BASELINE_EVALUATION.md` | done-external |
| Table 5 QA: Encoder evaluation consistency check | `scripts/encoder/check_encoder_eval_consistency.py` | not required after evaluation outputs | unified encoder evaluation outputs and normalized feature dumps | `reports/encoder_baselines/eval_v001/qa/ENCODER_EVAL_QA_REPORT.md` | done-external |
| Table 5b: Publication-grade encoder baseline evaluation | `scripts/encoder/evaluate_encoder_baselines.py` | not required after normalized feature dumps | normalized random/ImageNet/v6.1/v6.2-A/hierarchical feature dumps | `reports/encoder_baselines/eval_v002/ENCODER_BASELINE_EVALUATION.md` | done-external |
| Publication Table 1: Encoder baseline summary | `scripts/encoder/make_publication_tables_figures.py` | not required after eval_v002 | `reports/encoder_baselines/eval_v002/encoder_baseline_summary.csv` | `reports/publication_assets/encoder_baseline_v001/TABLE_1_ENCODER_BASELINE_SUMMARY.md` | done-external |
| Publication Table 2: Grouped robustness with CI | `scripts/encoder/make_publication_tables_figures.py` | not required after eval_v002 | `reports/encoder_baselines/eval_v002/encoder_baseline_summary.csv` | `reports/publication_assets/encoder_baseline_v001/TABLE_2_GROUPED_ROBUSTNESS_WITH_CI.md` | done-external |
| Publication Table 3: Controlled architecture comparison | `scripts/encoder/make_publication_tables_figures.py` | not required after eval_v002 | `reports/encoder_baselines/eval_v002/encoder_baseline_summary.csv` | `reports/publication_assets/encoder_baseline_v001/TABLE_3_CONTROLLED_ARCHITECTURE_COMPARISON.md` | done-external |
| Table 6: Class-level diagnostics | `scripts/encoder/export_class_level_diagnostics.py` | not required after eval_v002 | normalized feature dumps and `reports/encoder_baselines/eval_v002` | `reports/encoder_baselines/eval_v002/class_diagnostics/CLASS_LEVEL_DIAGNOSTICS_SUMMARY.md` | done-external |
| Table 6: Class support and imbalance | `scripts/encoder/make_class_level_publication_assets.py` | not required after class diagnostics | `reports/encoder_baselines/eval_v002/class_diagnostics/class_support_overall.csv` | `reports/publication_assets/class_level_v001/TABLE_4_CLASS_SUPPORT.md` | done-external |
| Table 7: Per-class grouped v6.2-A vs v6.1 comparison | `scripts/encoder/make_class_level_publication_assets.py` | not required after class diagnostics | `reports/encoder_baselines/eval_v002/class_diagnostics/class_level_key_numbers.json` | `reports/publication_assets/class_level_v001/TABLE_5_V62A_VS_V61_GROUPED_PER_CLASS_F1.md` | done-external |
| Table 8: Zero-recall breakdown | `scripts/encoder/make_class_level_publication_assets.py` | not required after class diagnostics | `reports/encoder_baselines/eval_v002/class_diagnostics/zero_recall_classes.csv` | `reports/publication_assets/class_level_v001/TABLE_6_ZERO_RECALL_BREAKDOWN.md` | done-external |
| Dataset composition and grouped-fold support audit | `scripts/encoder/make_dataset_composition_report.py` | not required after normalized feature dumps | manifest plus normalized v6.2-A feature metadata | `reports/publication_assets/methodological_hardening_v001/DATASET_COMPOSITION_REPORT.md` | done-external |
| Supplementary Table S1: board-level support | `scripts/encoder/make_dataset_composition_report.py` | not required after normalized feature dumps | normalized v6.2-A feature metadata | `reports/publication_assets/methodological_hardening_v001/dataset_composition_by_board.csv` | done-external |
| Supplementary Table S2: material-level support | `scripts/encoder/make_dataset_composition_report.py` | not required after normalized feature dumps | normalized v6.2-A feature metadata | `reports/publication_assets/methodological_hardening_v001/dataset_composition_by_material.csv` | done-external |
| Supplementary Table S3: class-by-board support | `scripts/encoder/make_dataset_composition_report.py` | not required after normalized feature dumps | normalized v6.2-A feature metadata | `reports/publication_assets/methodological_hardening_v001/class_by_board_support.csv` | done-external |
| Supplementary Table S4: class-by-material support | `scripts/encoder/make_dataset_composition_report.py` | not required after normalized feature dumps | normalized v6.2-A feature metadata | `reports/publication_assets/methodological_hardening_v001/class_by_material_support.csv` | done-external |
| Paired v6.1 vs v6.2-A protocol-level delta table | `scripts/encoder/make_paired_v61_v62a_grouped_difference_tables.py` | not required after eval_v002 | unified encoder evaluation summary | `reports/publication_assets/methodological_hardening_v001/paired_protocol_delta_v61_v62a.csv` | done-external |
| Paired v6.1 vs v6.2-A board LOBO fold-level table | `scripts/encoder/make_paired_v61_v62a_grouped_difference_tables.py` | not required after eval_v002 | LOBO summary plus board fold support | `reports/publication_assets/methodological_hardening_v001/paired_board_lobo_delta_v61_v62a.csv` | done-external |
| Paired v6.1 vs v6.2-A material LOMO fold-level table | `scripts/encoder/make_paired_v61_v62a_grouped_difference_tables.py` | not required after eval_v002 | LOMO summary plus material fold support | `reports/publication_assets/methodological_hardening_v001/paired_material_lomo_delta_v61_v62a.csv` | done-external |
| Frequency-only baseline table | `scripts/encoder/evaluate_frequency_only_baseline.py` | not required after normalized feature dump | normalized v6.2-A feature metadata and eval_v002 summaries | `reports/publication_assets/methodological_hardening_v001/frequency_only_baseline/frequency_only_summary.csv` | done-external |
| Frequency-only Board LOBO support/results | `scripts/encoder/evaluate_frequency_only_baseline.py` | not required after normalized feature dump | normalized v6.2-A feature metadata and board groups | `reports/publication_assets/methodological_hardening_v001/frequency_only_baseline/frequency_only_board_lobo.csv` | done-external |
| Frequency-only Material LOMO support/results | `scripts/encoder/evaluate_frequency_only_baseline.py` | not required after normalized feature dump | normalized v6.2-A feature metadata and material groups | `reports/publication_assets/methodological_hardening_v001/frequency_only_baseline/frequency_only_material_lomo.csv` | done-external |
| Frequency + embedding fusion information-budget comparison | `scripts/encoder/evaluate_frequency_embedding_fusion.py` | not required after normalized feature dumps | normalized frequency metadata, frozen embeddings, and frequency-only baseline outputs | `reports/publication_assets/methodological_hardening_v001/frequency_embedding_fusion/FREQUENCY_EMBEDDING_FUSION_REPORT.md` | done-external |
| Frequency + embedding fusion Board LOBO results | `scripts/encoder/evaluate_frequency_embedding_fusion.py` | not required after normalized feature dumps | normalized frequency metadata, frozen embeddings, and board groups | `reports/publication_assets/methodological_hardening_v001/frequency_embedding_fusion/frequency_embedding_fusion_board_lobo.csv` | done-external |
| Frequency + embedding fusion Material LOMO results | `scripts/encoder/evaluate_frequency_embedding_fusion.py` | not required after normalized feature dumps | normalized frequency metadata, frozen embeddings, and material groups | `reports/publication_assets/methodological_hardening_v001/frequency_embedding_fusion/frequency_embedding_fusion_material_lomo.csv` | done-external |
| kNN fixed-k protocol table | `scripts/encoder/make_knn_protocol_hardening_report.py` | not required after eval_v002 | existing kNN, Board LOBO, and Material LOMO summaries | `reports/publication_assets/methodological_hardening_v001/knn_protocol/knn_fixed_k_summary.csv` | done-external |
| kNN best-k diagnostic table | `scripts/encoder/make_knn_protocol_hardening_report.py` | not required after eval_v002 | existing stratified kNN summary | `reports/publication_assets/methodological_hardening_v001/knn_protocol/knn_best_k_diagnostic.csv` | done-external |
| Manuscript hardening patch plan | not applicable | not required | methodological hardening reports and key-number JSON files | `reports/publication_assets/methodological_hardening_v001/MANUSCRIPT_HARDENING_PATCH_PLAN.md` | done-external |
| Manuscript v002 consistency audit | `scripts/encoder/audit_manuscript_v002_consistency.py` | not required | hardened manuscript v002 and methodological hardening artifacts | `reports/publication_assets/methodological_hardening_v001/manuscript_v002_audit/MANUSCRIPT_V002_CONSISTENCY_AUDIT.md` | done-external |
| Table 6: acoustic-response prediction | pending script | pending config | acoustic-response manifest | pending report | future |

## Figures

| Manuscript item | Source script | Input artifact | Output/report | Status |
|---|---|---|---|---|
| Fig. 1: ESPI-to-encoder pipeline | manual diagram or pending plotting script | model lineage docs | `figures/pipeline.svg` | pending |
| Fig. 2: v6.1 to v6.2 lineage | manual diagram | `docs/MODEL_LINEAGE.md`, `docs/MODEL_ARCHITECTURE.md` | `figures/model_lineage.svg` | pending |
| Fig. 3: PCA by class | `scripts/encoder/make_encoder_technical_report.py` or audit output | external feature dump | external PCA PNG, summarized in encoder report | done-external |
| Fig. 4: PCA by board/material | `scripts/encoder/make_encoder_technical_report.py` or audit output | external feature dump | external PCA PNG, summarized in encoder report | done-external |
| Fig. 5: grouped evaluation bars | pending plotting script from CSV summaries | `reports/encoder/lobo_grouped_summary.csv`, `reports/encoder/lomo_summary.csv` | `figures/grouped_eval_summary.svg` | pending |
| Figure 4: Per-class grouped robustness | `scripts/encoder/export_class_level_diagnostics.py` | class-level diagnostics outputs | `reports/encoder_baselines/eval_v002/class_diagnostics/per_class_f1_v61_vs_v62a_grouped.svg` | done-external |
| Publication Fig. 1: Stratified vs grouped Macro-F1 | `scripts/encoder/make_publication_tables_figures.py` | `reports/encoder_baselines/eval_v002/encoder_baseline_summary.csv` | `reports/publication_assets/encoder_baseline_v001/FIG_1_STRATIFIED_VS_GROUPED_MACRO_F1.svg` | done-external |
| Publication Fig. 2: Grouped robustness gain | `scripts/encoder/make_publication_tables_figures.py` | `reports/encoder_baselines/eval_v002/encoder_baseline_summary.csv` | `reports/publication_assets/encoder_baseline_v001/FIG_2_GROUPED_ROBUSTNESS_GAIN.svg` | done-external |
| Publication Fig. 3: Encoder decision flow | `scripts/encoder/make_publication_tables_figures.py` | `reports/encoder_baselines/eval_v002/encoder_baseline_summary.csv` | `reports/publication_assets/encoder_baseline_v001/FIG_3_ENCODER_DECISION_FLOW.svg` | done-external |

| Figure 4: Class support imbalance | `scripts/encoder/make_class_level_publication_assets.py` | class-level diagnostics outputs | `reports/publication_assets/class_level_v001/FIG_4_CLASS_SUPPORT_IMBALANCE.svg` | done-external |
| Figure 5: v6.2-A vs v6.1 per-class grouped F1 | `scripts/encoder/make_class_level_publication_assets.py` | class-level diagnostics outputs | `reports/publication_assets/class_level_v001/FIG_5_V62A_VS_V61_PER_CLASS_GROUPED_F1.svg` | done-external |
| Figure 6: Zero-recall breakdown by encoder | `scripts/encoder/make_class_level_publication_assets.py` | class-level diagnostics outputs | `reports/publication_assets/class_level_v001/FIG_6_ZERO_RECALL_BREAKDOWN_BY_ENCODER.svg` | done-external |
## Architecture References

| Document | Scope | Status |
|---|---|---|
| `docs/MODEL_ARCHITECTURE.md` | Corrected v6.1, v6.2-A, and hierarchical v6.2 architecture definitions and publication claim boundaries | done |
| `docs/MODEL_LINEAGE.md` | Model lineage and development history | done-summary |

## Core Claims

| Claim | Evidence source | Manuscript section | Status |
|---|---|---|---|
| v6.2-A is a stronger reportable baseline than v6.1. | `reports/v61_v62_comparison/` | Results: classifier lineage | done-summary |
| Frozen v6.2-A embeddings preserve class structure. | `reports/encoder/EMBEDDING_AUDIT_SUMMARY.md` | Results: embedding geometry | done-summary |
| Frozen v6.2-A embeddings remain informative under board/material grouped OOD evaluation. | `reports/encoder/GROUPED_ENCODER_EVAL_SUMMARY.md` | Results: grouped OOD evaluation | done-summary |
| Hierarchical v6.2 is a candidate comparison branch. | `scripts/v6_2_hierarchical/train_v6_2_clean.py` | Methods: model variants | code-only |
| ESPI encoder improves acoustic-response prediction. | pending acoustic-response experiments | Future/second paper unless completed | pending |

## Technical Audits

| Audit item | Source script | Input artifact | Output/report | Status |
|---|---|---|---|---|
| Hierarchical v6.2 checkpoint load audit | `scripts/encoder/load_audit_v62_hierarchical.py` | external hierarchical checkpoint | `reports/hierarchical_encoder/LOAD_AUDIT_V62_HIERARCHICAL.md` | done-external |
| Hierarchical v6.2 wrapper smoke test | `scripts/encoder/test_v62_hierarchical_wrapper.py` | external hierarchical checkpoint and manifest | `reports/hierarchical_encoder/WRAPPER_SMOKE_TEST.md` | done-external |
| Hierarchical v6.2 small-sample embedding extraction | `scripts/encoder/extract_v62_hierarchical_embeddings.py` | external hierarchical checkpoint and manifest | `reports/hierarchical_encoder/EXTRACTION_SMALL_REPORT.md` | done-external |
| Hierarchical v6.2 full embedding extraction | `scripts/encoder/extract_v62_hierarchical_embeddings.py` | external hierarchical checkpoint and manifest | `reports/hierarchical_encoder/EXTRACTION_FULL_REPORT.md` | done-external |
| Hierarchical v6.2 embedding audit | `scripts/encoder/audit_v62_hierarchical_embeddings.py` | external hierarchical feature dump | `reports/hierarchical_encoder/audit_v001/HIERARCHICAL_ENCODER_AUDIT_SUMMARY.md` | done-external |

## Final Manuscript Tables

| Item | Source script | Input artifact | Output/report | Status |
|---|---|---|---|---|
| Final manuscript table pack | `scripts/encoder/make_final_manuscript_tables.py` | locked eval_v002/publication/hardening artifacts | `reports/publication_assets/final_tables_v001/TABLE_INDEX.md` | done-external |

## Spectral Descriptor Ablation

| Artifact | Source script | Input artifact | Output/report | Status |
|---|---|---|---|---|
| Spectral / LeFFT-inspired descriptor extraction | `scripts/encoder/extract_spectral_descriptors.py` | manifest + image root | `reports/publication_assets/methodological_hardening_v001/lefft_descriptor_ablation/SPECTRAL_DESCRIPTOR_EXTRACTION_REPORT.md` | done-external |

## Spectral Descriptor Ablation Evaluation

| Artifact | Source script | Input artifact | Output/report | Status |
|---|---|---|---|---|
| Spectral descriptor ablation evaluation | `scripts/encoder/evaluate_spectral_descriptor_ablation.py` | normalized spectral/v6.2-A feature dumps | `reports/publication_assets/methodological_hardening_v001/lefft_descriptor_ablation/eval_v001/SPECTRAL_DESCRIPTOR_ABLATION_REPORT.md` | done-external |

## Release Gate

Before submission, every `pending` manuscript item should have:

```text
script -> config -> input artifact reference -> output report -> checksum or DOI
```
