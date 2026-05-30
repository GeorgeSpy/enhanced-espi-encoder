# Experiment Registry

| ID | Experiment / artifact | Primary script or source | Status |
|---|---|---|---|
| EXP-001 | v6.2-A frozen embedding extraction | `scripts/encoder/extract_v62_embeddings.py` | done-locked |
| EXP-002 | v6.2-A embedding audit | `scripts/encoder/audit_v62_embeddings.py` | done-locked |
| EXP-003 | Grouped v6.2-A evaluation | `scripts/encoder/evaluate_encoder_grouped_generalization.py` | done-locked |
| EXP-004 | Hierarchical v6.2 load audit | `scripts/encoder/load_audit_v62_hierarchical.py` | done-locked |
| EXP-005 | Hierarchical wrapper smoke test | `scripts/encoder/test_v62_hierarchical_wrapper.py` | done-locked |
| EXP-006 | Hierarchical embedding extraction | `scripts/encoder/extract_v62_hierarchical_embeddings.py` | done-locked |
| EXP-007 | Hierarchical embedding audit | `scripts/encoder/audit_v62_hierarchical_embeddings.py` | done-locked |
| EXP-008 | v6.2-A vs hierarchical comparison | `scripts/encoder/compare_v62a_vs_hierarchical.py` | done-locked |
| EXP-009 | Encoder baseline input audit | `scripts/encoder/audit_encoder_baseline_inputs.py` | done-locked |
| EXP-010 | Feature schema normalization | `scripts/encoder/normalize_encoder_feature_schema.py` | done-locked |
| EXP-011 | Random/ImageNet ResNet extraction | `scripts/encoder/extract_resnet18_baseline_embeddings.py` | done-locked |
| EXP-012 | v6.1 embedding extraction | `scripts/encoder/extract_v61_embeddings.py` | done-locked |
| EXP-013 | Unified encoder baseline eval_v002 | `scripts/encoder/evaluate_encoder_baselines.py` | done-locked |
| EXP-014 | Dataset composition and fold support | `scripts/encoder/make_dataset_composition_report.py` | done-locked |
| EXP-015 | Paired v6.1 vs v6.2-A deltas | `scripts/encoder/make_paired_v61_v62a_grouped_difference_tables.py` | done-locked |
| EXP-016 | Frequency-only baseline | `scripts/encoder/evaluate_frequency_only_baseline.py` | done-locked |
| EXP-017 | Frequency + embedding fusion | `scripts/encoder/evaluate_frequency_embedding_fusion.py` | done-locked |
| EXP-018 | Deterministic spectral descriptor extraction | `scripts/encoder/extract_spectral_descriptors.py` | done-locked |
| EXP-019 | Spectral descriptor ablation | `scripts/encoder/evaluate_spectral_descriptor_ablation.py` | done-locked |
| EXP-020 | PCA dimension matching | `scripts/encoder/evaluate_v62a_dimension_matched_pca.py` | done-locked |
| EXP-021 | Frequency-residual correction | `scripts/encoder/evaluate_frequency_residual_correction.py` | done-locked |
| EXP-022 | Targeted `1_2` / `2_1` overlap analysis | `scripts/encoder/evaluate_frequency_overlap_pair.py` | done-locked |
| EXP-023 | Balanced subset evaluation | `scripts/encoder/evaluate_balanced_subsets.py` | done-locked |
| EXP-024 | Quantitative embedding geometry | `scripts/encoder/evaluate_embedding_geometry.py` | done-locked |
| EXP-025 | Final manuscript tables | `scripts/encoder/make_final_manuscript_tables.py` | done-locked |
| EXP-026 | Submission readiness audit | `scripts/encoder/audit_v003_submission_readiness.py` | done-locked |

No experiment in this registry trains a new deep model for the final manuscript claims.
