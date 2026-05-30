# Manuscript Patch Application Report

## Source Manuscript Path

- `manuscript/MANUSCRIPT_SUBMISSION_DRAFT.md`

## Output Manuscript Path

- `manuscripts/FROZEN_REPRESENTATION_AUDITS_DRAFT_v002.md`

## Source Artifacts Used

- `reports/publication_assets/methodological_hardening_v001/MANUSCRIPT_HARDENING_PATCH_PLAN.md`
- `reports/publication_assets/methodological_hardening_v001/manuscript_hardening_patch_key_numbers.json`
- `reports/publication_assets/methodological_hardening_v001/DATASET_COMPOSITION_REPORT.md`
- `reports/publication_assets/methodological_hardening_v001/PAIRED_V61_V62A_GROUPED_DIFFERENCES.md`
- `reports/publication_assets/methodological_hardening_v001/frequency_only_baseline/FREQUENCY_ONLY_BASELINE_REPORT.md`
- `reports/publication_assets/methodological_hardening_v001/frequency_embedding_fusion/FREQUENCY_EMBEDDING_FUSION_REPORT.md`
- `reports/publication_assets/methodological_hardening_v001/knn_protocol/KNN_PROTOCOL_HARDENING_REPORT.md`

## Sections Modified

- Title retained exactly as requested.
- Abstract revised to:
  - state v6.2-A as the strongest image-derived frozen ESPI encoder baseline;
  - add the frequency-only metadata control;
  - replace best-k primary wording with fixed `k=10` primary kNN wording;
  - preserve acoustic-response as outside current scope.
- Introduction revised to:
  - add frequency metadata as a necessary control;
  - separate image-derived representation evidence from metadata-only predictability;
  - add a hardened claim-boundary paragraph.
- Methods revised to:
  - add dataset composition and grouped-fold support details;
  - define fixed `k=10` as the primary kNN protocol;
  - define best-k over `k = 1, 3, 5, 10, 20` as diagnostic only;
  - clarify grouped metrics as means over held-out-group Macro-F1 values;
  - describe Material LOMO as a descriptive material-held-out stress test;
  - add a frequency-only and frequency-fusion controls subsection.
- Results revised to:
  - use fixed `k=10` as the primary kNN result;
  - move best-k to diagnostic/sensitivity wording;
  - add paired v6.1 vs v6.2-A fold-level deltas;
  - add frequency-only and frequency + v6.2-A information-budget results;
  - preserve class-level and hierarchical interpretation boundaries.
- Discussion revised to:
  - state that the current label protocol is strongly frequency-structured;
  - state that frequency metadata is stronger than image-derived embeddings for this label task;
  - retain v6.2-A as the strongest image-derived frozen ESPI encoder baseline.
- Limitations revised to:
  - add frequency metadata dominance;
  - add Material LOMO two-material caution;
  - preserve the hierarchical metadata caveat;
  - preserve no acoustic-response, no LeFFT superiority, and no validated Physics-Aligned Encoder boundaries.
- Tables/Figures plan revised to:
  - add Table 7;
  - add Table 8;
  - add Supplementary Table S5;
  - add Supplementary Table S6.

## Key Metric Replacements and Additions

- Replaced primary stratified kNN wording:
  - v6.1 fixed `k=10` stratified Macro-F1 = 93.65%.
  - v6.2-A fixed `k=10` stratified Macro-F1 = 92.78%.
- Reframed best-k as diagnostic:
  - v6.1 best-k diagnostic = `k=5`, 95.07%.
  - v6.2-A best-k diagnostic = `k=10`, 92.78%.
- Preserved grouped fixed `k=10` values:
  - v6.2-A Board LOBO Macro-F1 = 93.56%.
  - v6.2-A Material LOMO Macro-F1 = 93.49%.
  - v6.1 Board LOBO Macro-F1 = 87.10%.
  - v6.1 Material LOMO Macro-F1 = 86.33%.
- Added paired fold deltas:
  - Board LOBO delta v6.2-A minus v6.1 = +6.47 pp.
  - Material LOMO delta v6.2-A minus v6.1 = +7.16 pp.
  - v6.2-A wins 5/6 boards and 2/2 materials.
  - C02 = -0.02 pp for v6.2-A, essentially tied.
  - W01 = +0.27 pp for v6.2-A, small positive margin.
- Added frequency-only controls:
  - Stratified Macro-F1 = 98.67%.
  - Board LOBO Macro-F1 = 97.49%.
  - Material LOMO Macro-F1 = 94.09%.
- Added frequency + v6.2-A fusion:
  - Stratified Macro-F1 = 96.75%.
  - Board LOBO Macro-F1 = 94.79%.
  - Material LOMO Macro-F1 = 94.50%.
- Added fusion deltas:
  - Fusion minus frequency-only: -1.92 pp stratified, -2.70 pp Board LOBO, +0.41 pp Material LOMO.
  - Fusion minus v6.2-A embedding-only: +3.27 pp stratified, +0.50 pp Board LOBO, +0.45 pp Material LOMO.
  - Material LOMO per-class fusion gains over frequency-only: `1_2` +6.43 pp and `2_1` +6.73 pp.

## Claim-Boundary Changes

- Removed/avoided the unqualified implication that v6.2-A is the strongest overall predictor.
- Added the refined claim:
  - v6.2-A is the strongest image-derived frozen ESPI encoder baseline among the evaluated encoders.
  - `frequency_hz` is a dominant metadata-only predictor for the current five-class modal-label protocol.
- Explicitly retained boundaries:
  - no claim that image embeddings are necessary for simple modal-label classification;
  - no acoustic-response predictive claim;
  - no LeFFT superiority claim;
  - no fully validated physics-aligned encoder claim;
  - no full retrained CNN LOBO/LOMO generalization claim.

## Unresolved Placeholders

- No unresolved manuscript placeholders were introduced by this patch.
- Table 7, Table 8, Supplementary Table S5, and Supplementary Table S6 are planned manuscript/supplementary items and should be materialized during final table formatting.

## Missing Source Artifacts

- None. All requested hardening source artifacts were present at patch time.

## Validation Summary

The hardened manuscript contains the required values:

- 93.65%, 92.78%, 95.07%, 93.56%, 93.49%, 87.10%, 86.33%.
- 98.67%, 97.49%, 94.09%, 96.75%, 94.79%, 94.50%.
- +6.47, +7.16, -0.02, +0.27, +6.43, +6.73.
- 12,944, 10,115, 78.14%.

No raw data, feature dumps, evaluation outputs, or scripts were modified.
