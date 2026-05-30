# Experiment Registry

This registry separates locked OLEN manuscript evidence from active or planned Enhanced ESPI Encoder development. Status values are:

- `done-locked`
- `active`
- `planned`
- `exploratory`
- `negative-control`
- `archived`

## Locked Experiments

EXP-001 to EXP-023 define the locked OLEN frozen representation evidence associated with `v0.9-olen-pre-submission-evidence`.

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
| EXP-014 | Class-level diagnostics and publication assets | `scripts/encoder/export_class_level_diagnostics.py` | done-locked |
| EXP-015 | Dataset composition and fold support | `scripts/encoder/make_dataset_composition_report.py` | done-locked |
| EXP-016 | Paired v6.1 vs v6.2-A deltas | `scripts/encoder/make_paired_v61_v62a_grouped_difference_tables.py` | done-locked |
| EXP-017 | Frequency-only baseline | `scripts/encoder/evaluate_frequency_only_baseline.py` | done-locked |
| EXP-018 | Frequency + embedding fusion | `scripts/encoder/evaluate_frequency_embedding_fusion.py` | done-locked |
| EXP-019 | Deterministic spectral descriptor extraction | `scripts/encoder/extract_spectral_descriptors.py` | negative-control |
| EXP-020 | Spectral descriptor ablation | `scripts/encoder/evaluate_spectral_descriptor_ablation.py` | negative-control |
| EXP-021 | PCA dimension matching | `scripts/encoder/evaluate_v62a_dimension_matched_pca.py` | done-locked |
| EXP-022 | Frequency-residual and targeted `1_2` / `2_1` overlap analysis | `scripts/encoder/evaluate_frequency_residual_correction.py`; `scripts/encoder/evaluate_frequency_overlap_pair.py` | done-locked |
| EXP-023 | Balanced subset robustness and quantitative embedding geometry | `scripts/encoder/evaluate_balanced_subsets.py`; `scripts/encoder/evaluate_embedding_geometry.py` | done-locked |

Supporting manuscript/table-generation scripts and submission-readiness audits are reproducibility infrastructure for the locked evidence package, not new scientific experiments.

## Active / Planned Experiments

| ID | Experiment / artifact | Purpose | Status |
|---|---|---|---|
| EXP-024 | Learned LeFFT ablation | Train and evaluate matched learned spectral / LeFFT modules under the locked grouped protocols. | planned |
| EXP-025 | Frequency-controlled image-only classification | Define protocols where scalar frequency is controlled or residualized before image-derived representation evaluation. | planned |
| EXP-026 | Acoustic-response target definition | Define acoustic-response targets, metadata schema, and validation splits. | planned |
| EXP-027 | Acoustic-response metadata baselines | Evaluate metadata-only, frequency-only, and geometry-only controls for acoustic-response targets. | planned |
| EXP-028 | ESPI embedding + acoustic-response fusion | Test whether ESPI embeddings add predictive value beyond scalar frequency and geometry for acoustic-response targets. | planned |
| EXP-029 | Contrastive representation fine-tuning | Test representation-aligned contrastive or metric-learning objectives under grouped validation. | planned |
| EXP-030 | Cross-setup / external validation | Evaluate robustness across additional instruments, boards, materials, or acquisition campaigns if data become available. | planned |

No active or planned experiment supports a manuscript-level claim until it has matched baselines, grouped validation where applicable, information-budget controls, sanitized reports, reproducible scripts, claim-boundary updates, and release/tag association.
