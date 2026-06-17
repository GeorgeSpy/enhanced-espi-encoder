# Experiment Registry

This registry separates historical locked OLEN evidence, archived H2/H3 development diagnostics, completed forensic audits, and the active v32 `M_ESPI` / WP2 response-target roadmap.

Status values:

- `done-locked`
- `completed-forensic`
- `archived`
- `archived-negative`
- `development-only`
- `blocked`
- `superseded`
- `merged`
- `planned-conditional`
- `planned-priority`
- `planned`

## Locked Historical OLEN Experiments

EXP-001 to EXP-023 define the locked OLEN frozen representation evidence associated with `v0.9-olen-pre-submission-evidence`. These entries are retained for reproducibility and historical WP1 forensic review.

| ID | Experiment / artifact | Primary script or source | Status | Claim status |
|---|---|---|---|---|
| EXP-001 | v6.2-A frozen embedding extraction | `scripts/encoder/extract_v62_embeddings.py` | done-locked | historical OLEN evidence |
| EXP-002 | v6.2-A embedding audit | `scripts/encoder/audit_v62_embeddings.py` | done-locked | historical OLEN evidence |
| EXP-003 | Grouped v6.2-A evaluation | `scripts/encoder/evaluate_encoder_grouped_generalization.py` | done-locked | historical OLEN evidence |
| EXP-004 | Hierarchical v6.2 load audit | `scripts/encoder/load_audit_v62_hierarchical.py` | done-locked | historical OLEN evidence |
| EXP-005 | Hierarchical wrapper smoke test | `scripts/encoder/test_v62_hierarchical_wrapper.py` | done-locked | historical OLEN evidence |
| EXP-006 | Hierarchical embedding extraction | `scripts/encoder/extract_v62_hierarchical_embeddings.py` | done-locked | historical OLEN evidence |
| EXP-007 | Hierarchical embedding audit | `scripts/encoder/audit_v62_hierarchical_embeddings.py` | done-locked | historical OLEN evidence |
| EXP-008 | v6.2-A vs hierarchical comparison | `scripts/encoder/compare_v62a_vs_hierarchical.py` | done-locked | historical OLEN evidence |
| EXP-009 | Encoder baseline input audit | `scripts/encoder/audit_encoder_baseline_inputs.py` | done-locked | historical OLEN evidence |
| EXP-010 | Feature schema normalization | `scripts/encoder/normalize_encoder_feature_schema.py` | done-locked | historical OLEN evidence |
| EXP-011 | Random/ImageNet ResNet extraction | `scripts/encoder/extract_resnet18_baseline_embeddings.py` | done-locked | historical OLEN evidence |
| EXP-012 | v6.1 embedding extraction | `scripts/encoder/extract_v61_embeddings.py` | done-locked | historical OLEN evidence |
| EXP-013 | Unified encoder baseline eval_v002 | `scripts/encoder/evaluate_encoder_baselines.py` | done-locked | historical OLEN evidence |
| EXP-014 | Class-level diagnostics and publication assets | `scripts/encoder/export_class_level_diagnostics.py` | done-locked | historical OLEN evidence |
| EXP-015 | Dataset composition and fold support | `scripts/encoder/make_dataset_composition_report.py` | done-locked | historical OLEN evidence |
| EXP-016 | Paired v6.1 vs v6.2-A deltas | `scripts/encoder/make_paired_v61_v62a_grouped_difference_tables.py` | done-locked | historical OLEN evidence |
| EXP-017 | Frequency-only baseline | `scripts/encoder/evaluate_frequency_only_baseline.py` | done-locked | historical OLEN evidence |
| EXP-018 | Frequency + embedding fusion | `scripts/encoder/evaluate_frequency_embedding_fusion.py` | done-locked | historical OLEN evidence |
| EXP-019 | Deterministic spectral descriptor extraction | `scripts/encoder/extract_spectral_descriptors.py` | done-locked | negative-control |
| EXP-020 | Spectral descriptor ablation | `scripts/encoder/evaluate_spectral_descriptor_ablation.py` | done-locked | negative-control |
| EXP-021 | PCA dimension matching | `scripts/encoder/evaluate_v62a_dimension_matched_pca.py` | done-locked | historical OLEN evidence |
| EXP-022 | Frequency-residual and targeted `1_2` / `2_1` overlap analysis | `scripts/encoder/evaluate_frequency_residual_correction.py`; `scripts/encoder/evaluate_frequency_overlap_pair.py` | done-locked | historical/local diagnostic |
| EXP-023 | Balanced subset robustness and quantitative embedding geometry | `scripts/encoder/evaluate_balanced_subsets.py`; `scripts/encoder/evaluate_embedding_geometry.py` | done-locked | historical OLEN evidence |

## Archived H2/H3 LeFFT Development Registry

All entries in this section are development-only and do not alter historical OLEN evidence semantics or current v32 claim boundaries.

| ID | Experiment / artifact | Purpose | Status | Final decision | Artifact path | Claim status |
|---|---|---|---|---|---|---|
| EXP-024A | H2 standalone CE-only LeFFT v001/v002 grouped screening | Archive standalone clean LeFFT grouped-transfer diagnostics and threshold decisions. | archived-negative | Standalone CE-only grouped transfer failed. | `experiments/clean_lefft_encoder_v001/reports/h2_4c_final_factorial_postmortem/` | development-only |
| EXP-024B | H3 auxiliary LeFFT Phase-1 representation diagnostics | Archive auxiliary LeFFT mechanics/safety/representation audits. | archived-negative | Mechanics/safety pass; representation transfer insufficient or negative. | `experiments/clean_lefft_encoder_v001/reports/h3_track_consolidation_v001/` | development-only |
| EXP-024C | H3.2 LeFFT fusion | Define no-harm fusion only if separately drafted; do not run without transfer-positive evidence and no-harm protocol. | blocked | H3.2 run remains blocked. | `experiments/clean_lefft_encoder_v001/reports/h3_track_consolidation_v001/` | development-only |
| EXP-024D | Teacher-guided LeFFT distillation | Conditional rescue path using v6.2-A as teacher/anchor under explicit go/no-go design. | planned-conditional | Not active priority after forensic audits. | not yet created | future-gated |
| EXP-H2-00 | H2.0 design | Clean LeFFT/physics encoder protocol. | archived | Design complete; no training. | `experiments/clean_lefft_encoder_v001/reports/h2_0_clean_lefft_physics_encoder_design/` | development-only |
| EXP-H2-01 | H2.1 module skeleton | LeFFT module and synthetic smoke tests. | archived | Mechanics passed. | `experiments/clean_lefft_encoder_v001/reports/h2_1_lefft_module_skeleton*/` | development-only |
| EXP-H2-02 | H2.2 physics loss tests | Synthetic loss/gradient tests. | archived | Losses finite; gradients verified. | `experiments/clean_lefft_encoder_v001/reports/h2_2_physics_loss_unit_tests/` | development-only |
| EXP-H2-03 | H2.3 pair audit | Pair mining metadata support. | archived | CE-only/SupCon pair support ready. | `experiments/clean_lefft_encoder_v001/reports/h2_3_pair_statistics_audit/` | development-only |
| EXP-H2-04B | H2.4B reduced go/no-go | CE-only LeFFT LOBO threshold test. | archived-negative | FAIL; LOBO Macro-F1 `0.248644`. | `experiments/clean_lefft_encoder_v001/reports/h2_4b_reduced_ce_only_lefft_go_no_go/` | development-only |
| EXP-H2-01B | H2.1B redesign | v002 architecture and W4 sampler plan. | archived | Smoke tests passed. | `experiments/clean_lefft_encoder_v001/reports/h2_1b_lefft_architecture_redesign/` | development-only |
| EXP-H2-04C | H2.4C factorial repair | S0/S1/S2/S3 grouped repair test. | archived-negative | All conditions failed/fail-bound; best S2 LOBO `0.294294`. | `experiments/clean_lefft_encoder_v001/reports/h2_4c_final_factorial_postmortem/` | development-only |
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
| EXP-H3-01K | H3.1K scaled J1 | 6000-image scaled J1 multiview. | archived-negative | Mechanics PASS; negative cross-board transfer. | `experiments/clean_lefft_encoder_v001/reports/h3_1k_scaled_j1_postmortem_retry01/` | development-only |
| EXP-H3-CONS | H3 consolidation | Track-level consolidation. | archived | Mechanics/safety pass; transfer insufficient_or_negative. | `experiments/clean_lefft_encoder_v001/reports/h3_track_consolidation_v001/` | development-only |

## Superseded or Merged Pre-v32 Plans

| ID | Experiment / artifact | Previous purpose | Current classification | Current decision |
|---|---|---|---|---|
| EXP-025 | Frequency-controlled image-only classification | Control scalar frequency before image representation evaluation. | superseded | Superseded by forensic audit and clean ROI / response-target gates. |
| EXP-026 | Acoustic-response target definition | Define acoustic-response targets and validation splits. | merged | Merged into EXP-040 WP2 target feasibility matrix. |
| EXP-027 | Acoustic-response metadata baselines | Evaluate metadata-only/frequency-only controls for response targets. | merged | Merged into EXP-041 B0-B6 response-target baselines. |
| EXP-028 | ESPI embedding + acoustic-response fusion | Test ESPI embeddings beyond scalar frequency and geometry. | planned-conditional | Allowed only after EXP-040/041 pass; B6 residual framing required. |
| EXP-029 | Representation-aligned rescue objectives | Contrastive/distillation or representation-aligned objectives. | superseded | Superseded by `M_ESPI` measurement-consistent representation direction. |
| EXP-030 | Cross-setup / external validation | Robustness across additional instruments/campaigns. | planned | Still required for any promoted external/generalization claim. |

## Post-forensic v32 Experiments

| ID | Experiment / artifact | Purpose | Status | Artifact path | Claim status |
|---|---|---|---|---|---|
| EXP-031 | v62 frequency-label forensic audit | Quantify label provenance, frequency dominance, board/material mismatch, and OLEN reframing implications. | completed-forensic | `experiments/clean_lefft_encoder_v001/reports/v62_frequency_label_forensic_audit_v001_retry02/` | WP1 forensic evidence |
| EXP-032 | Cross-model forensic audit | Test whether any historical model adds robust information beyond frequency under common diagnostics. | completed-forensic | `experiments/clean_lefft_encoder_v001/reports/cross_model_forensic_audit_v001_retry02/` | WP1 forensic evidence |
| EXP-033 | OLEN reframing / line-freeze package | Freeze current OLEN line as deprecated strong-representation path and bridge WP1 to WP2. | completed-forensic | `experiments/clean_lefft_encoder_v001/reports/olen_line_freeze_v001/`; `experiments/clean_lefft_encoder_v001/reports/wp1_to_wp2_transition_v001/` | decision evidence |
| EXP-034 | Visual frequency-overlay and ROI-mask audit | Audit actual-pixel overlays, ROI mask status, patch leakage, domain fingerprints, and split contamination. | completed-forensic | `experiments/visual_domain_leakage_audit_v002_retry01/` | leakage-risk evidence |
| EXP-035 | Clean ROI dataset manifest construction | Define deterministic clean ROI/no-overlay transform, manifest, and leakage-control supplement. | completed-forensic | `experiments/cleaned_roi_rerun_v001/` | leakage-control evidence |
| EXP-036 | `z_ESPI` semantic separation: hist/ROI/MC | Define representation semantics and promotion gates. | active documentation | `docs/Z_ESPI_SEMANTICS.md` | claim-boundary infrastructure |
| EXP-037 | `M_ESPI` operator specification | Specify differentiable probabilistic ESPI observation operator. | active documentation | `docs/MESPI_OPERATOR_SPEC.md` | future-gated |
| EXP-038 | `M_ESPI` synthetic re-render smoke tests | Validate synthetic encode-field-re-render behavior and differentiability. | planned-priority | not yet created | future-gated |
| EXP-039 | `M_ESPI` validation against SLDV/FEM/reference fields | Compare operator outputs against independent displacement or numerical fields. | planned-priority | not yet created | future-gated |
| EXP-040 | WP2 target feasibility matrix | Screen response targets for frequency/metadata baseline headroom and grouped-OOD feasibility. | planned-priority | `experiments/clean_lefft_encoder_v001/reports/wp2_target_screening_template_v001/wp2_target_feasibility_matrix.csv` | future-gated |
| EXP-041 | B0-B6 response-target baselines | Evaluate frequency, metadata, numerical, descriptor, embedding, fusion, and residual baselines. | planned-priority | not yet created | future-gated |
| EXP-042 | B6 residual response prediction pilot | Test whether ESPI information predicts residual response beyond B0-B5 under grouped-OOD splits. | planned-conditional | not yet created | future-gated |

No active or planned experiment supports a manuscript-level claim until it has matched baselines, grouped validation where applicable, information-budget controls, sanitized reports, reproducible scripts, claim-boundary updates, and release/tag association.
