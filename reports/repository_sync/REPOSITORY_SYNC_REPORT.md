# Repository Sync Report

## Scope

This sync updates the repository narrative from a private development repository to a pre-submission evidence package for the OLEN frozen ESPI representation-audit manuscript.

## Files Created

- `docs/EXPERIMENT_REGISTRY.md`
- `docs/EVIDENCE_LEDGER.md`
- `docs/ARTIFACT_POLICY.md`
- `docs/templates/ARTIFACT_MANIFEST_TEMPLATE.csv`
- `scripts/utils/sanitize_report_paths.py`
- `scripts/publication/format_olen_submission.py`
- `scripts/publication/make_uniform_appendix_tables.py`
- `scripts/publication/enhance_representative_patterns.py`
- `manuscripts/OLEN_ESPI_frozen_representation_submission.tex`
- `manuscripts/cover_letter_draft.txt`
- `manuscripts/highlights.txt`
- `manuscripts/ALL_TABLES_S1_S7_1_8.md`
- `manuscripts/figures/Figure_1_Representative_Patterns.pdf`
- `manuscripts/figures/Figure_1_Representative_Patterns.png`
- `manuscripts/figures/central_result_comparison.pdf`
- `manuscripts/figures/central_result_comparison.png`
- `manuscripts/figures/Figure_PCA_UMAP_Embeddings.pdf`
- `manuscripts/figures/Figure_PCA_UMAP_Embeddings.png`
- `manuscripts/figures/Figure_Confusion_Matrices_Board_LOBO.pdf`
- `manuscripts/figures/Figure_Confusion_Matrices_Board_LOBO.png`

## Files Updated

- `README.md`
- `REPRODUCE.md`
- `CLAIM_BOUNDARIES.md`
- `docs/CLAIM_BOUNDARIES.md`
- `docs/MANUSCRIPT_MAP.md`
- `docs/REPRODUCE.md`
- `docs/DATA_AVAILABILITY.md`
- `docs/MODEL_ARCHITECTURE.md`

## Claim-Boundary Changes

- Replaced the older future-oriented Physics-Aligned Encoder framing with the final manuscript framing: grouped and frequency-controlled frozen ESPI representation auditing.
- Marked frequency-only metadata dominance as supported.
- Marked v6.2-A grouped board/material image-derived representation strength as supported.
- Added PCA dimension matching, targeted overlap, balanced subset, quantitative geometry, and spectral-control evidence to supported scope.
- Kept acoustic-response prediction, validated Physics-Aligned Encoder, LeFFT superiority, neural operators, full retrained CNN LOBO/LOMO, and external-laboratory generalization outside current claims.

## Evidence Artifacts Reflected

- `reports/encoder_baselines/eval_v002/`
- `reports/dimension_matched_v62a/`
- `reports/frequency_residual_v62a/`
- `reports/frequency_residual_v62a_rf/`
- `reports/frequency_overlap_pair_v62a/`
- `reports/balanced_subset_v62a/`
- `reports/embedding_geometry_v62a/`
- `reports/publication_assets/final_tables_v001/`
- `reports/publication_assets/submission_readiness_v002/`

## Excluded Artifacts

- raw ESPI images,
- private manifests,
- full feature dumps,
- checkpoints,
- `.npz` / `.npy` arrays,
- local logs and intermediate outputs.

## Checks Performed

- Verified final manuscript package files exist.
- Verified LaTeX `\includegraphics` references resolve against `manuscripts/figures/`.
- Sanitized local absolute paths in small Markdown/TXT report files using `scripts/utils/sanitize_report_paths.py --write` (43 replacements across 15 files).
- Verified no push was performed.

## Remaining Tasks

- Review `git status` and decide staged file set.
- Optionally run a full LaTeX compile if a TeX distribution is available.
- Optionally run `scripts/utils/sanitize_report_paths.py reports --write` after reviewing dry-run output.
- Commit in the recommended sequence: documentation, scripts, reports, manuscript/figures, sync audit.
- Push only after explicit approval.

## Repository Sync Verdict

The repository now reflects the paper evidence package at the documentation and submission-package level, pending staging/commit review.
