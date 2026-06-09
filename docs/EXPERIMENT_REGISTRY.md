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
| EXP-024A | H2 standalone CE-only LeFFT v001/v002 grouped screening | Archive standalone clean LeFFT grouped-transfer diagnostics and threshold decisions. | archived-negative |
| EXP-024B | H3 auxiliary LeFFT Phase-1 representation diagnostics | Archive auxiliary LeFFT mechanics/safety/representation audits. | archived-negative / development-only |
| EXP-024C | H3.2 LeFFT fusion | Define no-harm fusion only if explicitly drafted; do not run without transfer-positive evidence and no-harm protocol. | blocked |
| EXP-024D | Teacher-guided LeFFT distillation | Conditional future rescue path using v6.2-A as teacher/anchor under explicit go/no-go design. | planned-conditional |
| EXP-025 | Frequency-controlled image-only classification | Define protocols where scalar frequency is controlled or residualized before image-derived representation evaluation. | planned-conditional |
| EXP-026 | Acoustic-response target definition | Define acoustic-response targets, metadata schema, and validation splits. | planned-priority |
| EXP-027 | Acoustic-response metadata baselines | Evaluate metadata-only, frequency-only, and geometry-only controls for acoustic-response targets. | planned |
| EXP-028 | ESPI embedding + acoustic-response fusion | Test whether ESPI embeddings add predictive value beyond scalar frequency and geometry for acoustic-response targets. | planned |
| EXP-029 | Representation-aligned rescue objectives | Test contrastive, distillation, or other representation-aligned objectives only under new explicit go/no-go designs. | planned-conditional |
| EXP-030 | Cross-setup / external validation | Evaluate robustness across additional instruments, boards, materials, or acquisition campaigns if data become available. | planned |

No active or planned experiment supports a manuscript-level claim until it has matched baselines, grouped validation where applicable, information-budget controls, sanitized reports, reproducible scripts, claim-boundary updates, and release/tag association.

<!-- BEGIN H2_H3_DEVELOPMENT_REGISTRY -->
## H2/H3 Development-only LeFFT Registry

All entries below are development-only and do not alter locked OLEN v6.2-A evidence semantics.

| ID | Experiment / artifact | Purpose | Status | Final decision | Artifact path | Claim status |
|---|---|---|---|---|---|---|
| EXP-H2-00 | H2.0 design | Clean LeFFT/physics encoder protocol. | archived | Design complete; no training. | `experiments/clean_lefft_encoder_v001/reports/h2_0_clean_lefft_physics_encoder_design/` | development-only |
| EXP-H2-01 | H2.1 module skeleton | LeFFT module and synthetic smoke tests. | archived | Mechanics passed. | `experiments/clean_lefft_encoder_v001/reports/h2_1_lefft_module_skeleton*/` | development-only |
| EXP-H2-02 | H2.2 physics loss tests | Synthetic loss/gradient tests. | archived | Losses finite; gradients verified. | `experiments/clean_lefft_encoder_v001/reports/h2_2_physics_loss_unit_tests/` | development-only |
| EXP-H2-03 | H2.3 pair audit | Pair mining metadata support. | archived | CE-only/SupCon pair support ready. | `experiments/clean_lefft_encoder_v001/reports/h2_3_pair_statistics_audit/` | development-only |
| EXP-H2-04B | H2.4B reduced go/no-go | CE-only LeFFT LOBO threshold test. | archived | FAIL; LOBO Macro-F1 0.248644. | `experiments/clean_lefft_encoder_v001/reports/h2_4b_reduced_ce_only_lefft_go_no_go/` | development-only |
| EXP-H2-01B | H2.1B redesign | v002 architecture and W4 sampler plan. | archived | Smoke tests passed. | `experiments/clean_lefft_encoder_v001/reports/h2_1b_lefft_architecture_redesign/` | development-only |
| EXP-H2-04C | H2.4C factorial repair | S0/S1/S2/S3 grouped repair test. | archived | All failed/fail-bound locked; best S2 LOBO 0.294294. | `experiments/clean_lefft_encoder_v001/reports/h2_4c_final_factorial_postmortem/` | development-only |
| EXP-H3-00 | H3.0 design scaffold | Anchor-assisted LeFFT design. | archived | Design complete. | `experiments/clean_lefft_encoder_v001/reports/h3_0_anchor_lefft_design_scaffold/` | development-only |
| EXP-H3-01A | H3.1A synthetic mechanics | Synthetic branch/loss tests. | archived | Tests passed. | `experiments/clean_lefft_encoder_v001/reports/h3_1a_lefft_aux_physics_pretraining*/` | development-only |
| EXP-H3-01B | H3.1B real-data plan | Dataloader/safety/readiness plan. | archived | Plan complete. | `experiments/clean_lefft_encoder_v001/reports/h3_1b_realdata_phase1_pretraining_plan*/` | development-only |
| EXP-H3-01C | H3.1C guarded pilot | 3000-image Phase-1 pilot. | archived | Mechanics pilot complete. | `experiments/clean_lefft_encoder_v001/reports/h3_1c_guarded_realdata_phase1_pilot/` | development-only |
| EXP-H3-01D | H3.1D postmortem | H3.1C verification. | archived | Mechanics verified. | `experiments/clean_lefft_encoder_v001/reports/h3_1d_phase1_pilot_postmortem*/` | development-only |
| EXP-H3-01E | H3.1E scaled Phase-1 | 6000-image scaled baseline. | archived | Mechanics PASS; weak/near-static representation. | `experiments/clean_lefft_encoder_v001/reports/h3_1e_scaled_phase1_pretraining_retry02/` | development-only |
| EXP-H3-01F | H3.1F postmortem | H3.1E verification. | archived | H3.2 blocked. | `experiments/clean_lefft_encoder_v001/reports/h3_1f_scaled_phase1_postmortem_retry01/` | development-only |
| EXP-H3-01G | H3.1G sensitivity pilot | G0/G1/G2/G3 objective sensitivity. | archived | G1 selected with small partial signal. | `experiments/clean_lefft_encoder_v001/reports/h3_1g_phase1_sensitivity_pilot/` | development-only |
| EXP-H3-01H | H3.1H scaled G1 | 6000-image G1 scaling. | archived | Borderline representation signal. | `experiments/clean_lefft_encoder_v001/reports/h3_1h_scaled_g1_phase1_pretraining/` | development-only |
| EXP-H3-01I | H3.1I postmortem | H3.1H verification. | archived | Objective redesign recommended; H3.2 blocked. | `experiments/clean_lefft_encoder_v001/reports/h3_1i_scaled_g1_postmortem/` | development-only |
| EXP-H3-01J | H3.1J objective redesign | J0/J1/J2/J3 objective redesign pilot. | archived | J1 best non-diagnostic pilot. | `experiments/clean_lefft_encoder_v001/reports/h3_1j_objective_redesign_postmortem/` | development-only |
| EXP-H3-01K | H3.1K scaled J1 | 6000-image scaled J1 multiview. | archived | Mechanics PASS; negative cross-board transfer. | `experiments/clean_lefft_encoder_v001/reports/h3_1k_scaled_j1_postmortem_retry01/` | development-only |
| EXP-H3-CONS | H3 consolidation | Track-level consolidation. | archived | Mechanics/safety pass; transfer insufficient_or_negative. | `experiments/clean_lefft_encoder_v001/reports/h3_track_consolidation_v001/` | development-only |
<!-- END H2_H3_DEVELOPMENT_REGISTRY -->
