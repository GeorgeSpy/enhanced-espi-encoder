# Evidence Ledger

| Manuscript claim | Evidence artifacts | Status |
|---|---|---|
| v6.2-A is the strongest evaluated image-derived frozen ESPI representation under grouped board/material evaluation. | `reports/encoder_baselines/eval_v002/`, `reports/publication_assets/final_tables_v001/`, `reports/publication_assets/methodological_hardening_v001/` | supported |
| v6.1 remains competitive and slightly stronger under fixed-k stratified kNN. | `reports/encoder_baselines/eval_v002/`, `TABLE_8_FIXED_K_VS_BEST_K_DIAGNOSTIC.md` | supported |
| `frequency_hz` is the dominant metadata-only predictor for the current five-class task. | `reports/publication_assets/methodological_hardening_v001/frequency_only_baseline/` | supported |
| Frequency + v6.2-A improves over v6.2-A embedding-only but does not clearly exceed frequency-only except slightly in Material LOMO. | `reports/publication_assets/methodological_hardening_v001/frequency_embedding_fusion/` | supported |
| v6.2-A adds complementary morphology information in localized `1_2` / `2_1` frequency ambiguity. | `reports/frequency_overlap_pair_v62a/`, `reports/frequency_residual_v62a/`, `reports/frequency_residual_v62a_rf/` | targeted diagnostic |
| v6.2-A grouped advantage is not a dimensionality artifact. | `reports/dimension_matched_v62a/` | supported |
| v6.2-A grouped advantage persists under class/material/frequency-balanced controls. | `reports/balanced_subset_v62a/` | supported |
| v6.2-A has stronger frozen-embedding geometry than v6.1. | `reports/embedding_geometry_v62a/` | supported |
| Hierarchical v6.2 phase2 is technically valid but not superior. | `reports/hierarchical_encoder/`, `reports/encoder_comparison/` | supported |
| Deterministic spectral descriptors are supplementary controls and do not support LeFFT superiority. | `reports/publication_assets/methodological_hardening_v001/lefft_descriptor_ablation/`, `SUPPLEMENTARY_TABLE_S7_SPECTRAL_DESCRIPTOR_CONTROL.md` | supported |
| Acoustic-response prediction is established. | none | not claimed |
| Validated Physics-Aligned ESPI Encoder is established. | none | not claimed |

All evidence rows are bounded by `docs/CLAIM_BOUNDARIES.md`.
