# Manuscript Evidence Map

> Note: this file maps the historical locked OLEN package. It is not the active v32 PhD proposal map. For the active v32 direction, see `docs/PROPOSAL_V32_MAP.md`, `docs/WP1_FORENSIC_EVIDENCE_MAP.md`, and `docs/MESPI_OPERATOR_EVIDENCE_MAP.md`.

This map tracks the final OLEN frozen ESPI representation-audit manuscript evidence package. Acoustic-response prediction is future work and is not a pending table in the current manuscript.

## Main Tables

| Item | Description | Source artifact | Status |
|---|---|---|---|
| Table 1 | Encoder baseline summary, fixed k=10 | `reports/encoder_baselines/eval_v002/` | done-locked |
| Table 2 | Grouped robustness with confidence intervals | `reports/encoder_baselines/eval_v002/` | done-locked |
| Table 3 | Frequency information-budget comparison | `reports/publication_assets/methodological_hardening_v001/frequency_only_baseline/`, `reports/publication_assets/methodological_hardening_v001/frequency_embedding_fusion/` | done-locked |
| Table 4 | Dimension-matched PCA control | `reports/dimension_matched_v62a/` | done-locked |
| Table 5 | Targeted `1_2` / `2_1` frequency-overlap analysis | `reports/frequency_overlap_pair_v62a/`, `reports/frequency_residual_v62a/`, `reports/frequency_residual_v62a_rf/` | done-locked |
| Table 6 | Balanced subset robustness | `reports/balanced_subset_v62a/` | done-locked |
| Table 7 | Quantitative embedding geometry | `reports/embedding_geometry_v62a/` | done-locked |

## Supplementary Tables

| Item | Description | Source artifact | Status |
|---|---|---|---|
| S1 | Class support and imbalance | `reports/publication_assets/final_tables_v001/` | done-locked |
| S2 | v6.2-A vs v6.1 per-class grouped F1 | `reports/publication_assets/final_tables_v001/` | done-locked |
| S3 | Board-level support | `reports/publication_assets/final_tables_v001/` | done-locked |
| S4 | Material-level support | `reports/publication_assets/final_tables_v001/` | done-locked |
| S5 | Paired board/material deltas | `reports/publication_assets/final_tables_v001/` | done-locked |
| S6 | Per-class intra/inter cosine distance ratio | `reports/embedding_geometry_v62a/` | done-locked |
| S7 | Downstream head sensitivity | `reports/encoder_baselines/eval_v002/` | done-locked |
| S8 | Deterministic spectral descriptor control | `reports/publication_assets/final_tables_v001/`, `reports/publication_assets/methodological_hardening_v001/lefft_descriptor_ablation/` | done-locked |
| S9 | Fixed-k vs best-k diagnostic | `reports/publication_assets/final_tables_v001/`, `reports/publication_assets/methodological_hardening_v001/knn_protocol/` | done-locked |
| S10 | Zero-recall breakdown | `reports/publication_assets/final_tables_v001/` | done-locked |
| S11 | Controlled architecture comparison | `reports/encoder_comparison/`, `reports/hierarchical_encoder/` | done-locked |

## Figures

| Item | Description | Source artifact | Status |
|---|---|---|---|
| Figure 1 | Representative ESPI patterns | `manuscripts/figures/` | done-locked |
| Figure 2 | Central result comparison | `manuscripts/figures/central_result_comparison.*` | done-locked |
| Figure 3 | PCA/UMAP embedding diagnostics | `manuscripts/figures/` | done-locked |
| Figure 4 | Board LOBO confusion matrices | `manuscripts/figures/` | done-locked |

## Final Manuscript Package

| Artifact | Path | Status |
|---|---|---|
| OLEN LaTeX source | `manuscripts/OLEN_ESPI_frozen_representation_submission.tex` | done-locked |
| Cover letter draft | `manuscripts/cover_letter_draft.txt` | done-locked |
| Highlights | `manuscripts/highlights.txt` | done-locked |
| Figures | `manuscripts/figures/` | done-locked |

## Future Work Outside Current Manuscript

- Acoustic-response prediction.
- Frequency-controlled or frequency-residual labels beyond the current modal-label protocol.
- Learned spectral / LeFFT-style model ablations.
- Physics-aligned encoder validation.
- External-laboratory generalization.
