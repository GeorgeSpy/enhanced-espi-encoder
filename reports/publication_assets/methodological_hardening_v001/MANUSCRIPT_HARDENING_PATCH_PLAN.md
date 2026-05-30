# Manuscript Hardening Patch Plan

## Purpose

This patch plan specifies manuscript edits required after the methodological hardening controls. It is a planning artifact only: no manuscript text has been edited, no metrics have been changed, and no new experiments are proposed here.

## Core Claim Revision

The manuscript should no longer imply that v6.2-A is the strongest predictor overall when metadata is allowed. The revised claim should be:

> v6.2-A is the strongest image-derived frozen ESPI encoder baseline under the evaluated grouped board/material protocols, while `frequency_hz` is a stronger metadata-only predictor for the present five-class modal-label protocol.

This preserves the frozen representation audit while explicitly acknowledging that the current labels are strongly frequency-structured.

## 1. Abstract Patch

Required changes:

- Replace any unqualified claim that v6.2-A is the strongest overall predictor with "v6.2-A is the strongest image-derived frozen ESPI encoder baseline."
- Add one sentence: "A frequency-only metadata control showed that `frequency_hz` is a strong predictor of the present five-class modal labels, indicating that the label protocol is strongly frequency-structured."
- Update stratified kNN wording:
  - Primary fixed-k result: v6.1 fixed `k=10` stratified Macro-F1 = 93.65%.
  - v6.2-A fixed `k=10` stratified Macro-F1 = 92.78%.
  - v6.1 best-k Macro-F1 = 95.07% should be described as a diagnostic sensitivity result, not the primary kNN protocol.
- Keep Board LOBO and Material LOMO fixed `k=10` values for v6.2-A:
  - Board LOBO Macro-F1 = 93.56%.
  - Material LOMO Macro-F1 = 93.49%.

Suggested wording:

> Under a fixed `k=10` kNN protocol, v6.1 remained slightly stronger in stratified evaluation (93.65% Macro-F1 versus 92.78% for v6.2-A), while v6.2-A provided the strongest image-derived grouped robustness (93.56% Board LOBO and 93.49% Material LOMO Macro-F1). A frequency-only control reached 98.67%, 97.49%, and 94.09% Macro-F1 under stratified, Board LOBO, and Material LOMO protocols, respectively, showing that the present label protocol is strongly frequency-structured.

## 2. Introduction Patch

Required changes:

- Keep the frozen representation audit framing.
- Add a paragraph explaining why frequency metadata is a necessary control for modal-label classification:
  - Modal labels are physically related to resonance/frequency structure.
  - A strong image encoder result must therefore be interpreted against a metadata-only frequency baseline.
  - Frequency controls distinguish image-derived representation evidence from metadata-driven label separability.
- Keep acoustic-response prediction outside current scope.

Suggested paragraph:

> Because the five modal labels are associated with frequency-dependent vibration structure, `frequency_hz` is a necessary metadata control. Without this control, high label-classification performance could be misinterpreted as evidence that the image representation alone captures transferable ESPI structure. The present study therefore separates image-derived frozen encoder evidence from metadata-only predictability and treats acoustic-response prediction as future validation rather than a current claim.

## 3. Methods 2.1 Dataset Patch

Insert dataset composition details:

- Total samples: 12,944.
- Boards: 6 (`C01`, `C02`, `C03`, `W01`, `W02`, `W03`).
- Materials: 2 (`carbon`, `wood`).
- All boards contain all five classes.
- All materials contain all five classes.
- No LOBO or LOMO fold has missing held-out or reference classes.
- Supplementary Tables S1-S4 should be referenced as dataset composition/support tables generated from the dataset composition audit:
  - Supplementary Table S1: board-level support.
  - Supplementary Table S2: material-level support.
  - Supplementary Table S3: class-by-board support.
  - Supplementary Table S4: class-by-material support.

Suggested wording:

> The frozen-encoder evidence set contained 12,944 samples from six boards and two materials. All boards and both materials contained all five modal classes, and no board- or material-held-out fold lacked held-out or reference classes. Board-level, material-level, class-by-board, and class-by-material support are reported in Supplementary Tables S1-S4.

## 4. Methods 2.5 Evaluation Protocol Patch

Required changes:

- Define fixed `k=10` as the primary kNN protocol.
- State that `k=10` was the locked development/default grouped kNN setting.
- State that best-k over `k = 1, 3, 5, 10, 20` is diagnostic only.
- State that grouped metrics are means over held-out-group Macro-F1 values.
- State that Material LOMO is interpreted descriptively as a material-held-out stress test because only two material groups exist.

Suggested wording:

> The primary kNN protocol used cosine kNN with fixed `k=10`, selected as the locked development/default grouped kNN setting before manuscript hardening. Additional stratified kNN values for `k = 1, 3, 5, 10, 20` are reported only as diagnostic sensitivity analysis. Grouped Board LOBO and Material LOMO scores are computed as means over held-out-group Macro-F1 values. Material LOMO is interpreted as a descriptive material-held-out stress test because only two material groups are available.

## 5. Methods Frequency Controls Patch

Add subsection title:

### Frequency-only and frequency-fusion controls

Required content:

- Frequency-only logistic regression:
  - input: scalar `frequency_hz`;
  - balanced class weights;
  - `max_iter = 5000`;
  - training/reference-fold standardization only.
- Frequency-only random forest:
  - input: scalar `frequency_hz`;
  - class weighting;
  - used as a non-linear metadata-only control.
- Frequency + embedding fusion:
  - concatenate scalar `frequency_hz` with frozen embedding;
  - standardization fit only on training/reference folds;
  - evaluate with balanced logistic regression and random forest.
- No raw images, deep-model inference, fine-tuning, LeFFT, or acoustic-response prediction were used in these controls.

Suggested wording:

> To estimate the information budget of frequency metadata, we evaluated frequency-only logistic-regression and random-forest controls using `frequency_hz` as the sole input. We also evaluated frequency-plus-embedding fusion by concatenating `frequency_hz` with frozen encoder embeddings. All standardization and preprocessing were fit only on the training/reference fold and applied to the held-out fold.

## 6. Results 3.1 / 3.2 Patch

Required changes:

- Replace primary stratified best-k wording with fixed-k wording:
  - v6.1 fixed `k=10` stratified Macro-F1 = 93.65%.
  - v6.2-A fixed `k=10` stratified Macro-F1 = 92.78%.
- Move v6.1 best-k value to diagnostic wording:
  - v6.1 best-k diagnostic: `k=5`, 95.07%.
  - v6.2-A best-k diagnostic: `k=10`, 92.78%.
- State that best-k is diagnostic sensitivity analysis, not the primary protocol.

Suggested wording:

> Under the primary fixed `k=10` kNN protocol, v6.1 achieved 93.65% stratified Macro-F1, while v6.2-A achieved 92.78%. A diagnostic best-k sensitivity analysis over `k = 1, 3, 5, 10, 20` gave v6.1 its maximum stratified Macro-F1 at `k=5` (95.07%), whereas v6.2-A was already maximized at `k=10` (92.78%).

## 7. Results Grouped Robustness Patch

Add paired fold-level delta summary:

- Board LOBO mean delta: v6.2-A over v6.1 = +6.47 pp.
- Material LOMO mean delta: v6.2-A over v6.1 = +7.16 pp.
- v6.2-A wins 5/6 board folds and 2/2 material folds.
- Mention near-tie folds:
  - C02: -0.02 pp for v6.2-A versus v6.1, essentially tied.
  - W01: +0.27 pp for v6.2-A, small margin.
- Keep Material LOMO as descriptive/stress-test because only two material groups exist.

Suggested wording:

> Paired fold-level comparison showed that v6.2-A improved over v6.1 by +6.47 percentage points in mean Board LOBO Macro-F1 and by +7.16 percentage points in mean Material LOMO Macro-F1. v6.2-A won 5/6 board folds and 2/2 material folds. The C02 board was essentially tied (-0.02 pp for v6.2-A), and W01 showed only a small v6.2-A margin (+0.27 pp), indicating that the grouped advantage is broad but not uniform across all boards.

## 8. Results Frequency Controls Patch

Add frequency-control subsection:

### Frequency-only and frequency-fusion controls

Required numbers:

- Frequency-only:
  - Stratified Macro-F1 = 98.67%.
  - Board LOBO Macro-F1 = 97.49%.
  - Material LOMO Macro-F1 = 94.09%.
- Frequency + v6.2-A:
  - Stratified Macro-F1 = 96.75%.
  - Board LOBO Macro-F1 = 94.79%.
  - Material LOMO Macro-F1 = 94.50%.
- v6.2-A embedding-only in matched fusion heads:
  - Stratified Macro-F1 = 93.48%.
  - Board LOBO Macro-F1 = 94.28%.
  - Material LOMO Macro-F1 = 94.05%.
- Deltas:
  - Fusion minus frequency-only:
    - Stratified = -1.92 pp.
    - Board LOBO = -2.70 pp.
    - Material LOMO = +0.41 pp.
  - Fusion minus v6.2-A embedding-only:
    - Stratified = +3.27 pp.
    - Board LOBO = +0.50 pp.
    - Material LOMO = +0.45 pp.
- Per-class Material LOMO note:
  - For `1_2`, frequency + v6.2-A exceeds frequency-only by +6.43 pp.
  - For `2_1`, frequency + v6.2-A exceeds frequency-only by +6.73 pp.

Suggested wording:

> Frequency-only controls showed that the current five-class modal protocol is strongly frequency-structured. Frequency-only models reached 98.67%, 97.49%, and 94.09% Macro-F1 under stratified, Board LOBO, and Material LOMO protocols, respectively. Frequency + v6.2-A reached 96.75%, 94.79%, and 94.50%, improving over v6.2-A embedding-only in all three protocols but exceeding frequency-only only in Material LOMO (+0.41 pp). At the class level under Material LOMO, frequency + v6.2-A improved over frequency-only for the minority/difficult classes `1_2` (+6.43 pp) and `2_1` (+6.73 pp).

## 9. Discussion Patch

Required changes:

- State directly that the current label protocol is strongly frequency-structured.
- State that v6.2-A is not the strongest overall predictor when frequency metadata is allowed.
- State that v6.2-A remains the strongest image-derived frozen ESPI encoder baseline.
- Explain why this does not invalidate the representation audit:
  - The audit asks whether image-derived frozen embeddings are reusable and grouped-robust relative to other image-derived baselines.
  - The frequency control answers a different question: how much label information is carried by metadata.
  - Both are necessary for a defensible manuscript.
- Add future-work direction:
  - frequency-controlled labels;
  - frequency-residual tasks;
  - acoustic-response targets;
  - tasks where image-derived morphology must explain variance beyond frequency.

Suggested wording:

> The frequency controls refine the interpretation of the encoder study. The current modal-label protocol is strongly frequency-structured, and frequency metadata is a stronger overall predictor than any image-derived frozen representation. This does not invalidate the representation audit; instead, it clarifies its scope. The frozen-encoder experiments compare image-derived representations under matched grouped protocols, where v6.2-A remains the strongest image-derived baseline. Demonstrating image-derived information beyond modal frequency structure will require frequency-controlled labels, frequency-residual prediction, or acoustic-response targets in future work.

## 10. Limitations Patch

Add limitations:

- Frequency metadata dominates the current five-class modal-label protocol.
- v6.2-A should not be described as the strongest overall predictor when metadata is allowed.
- Material LOMO uses only two material groups and should be interpreted as a descriptive material-held-out stress test rather than a broad inferential estimate.
- Hierarchical v6.2 phase2 remains internal-only for final publication tables unless regenerated with correct source metadata.
- The current work does not test acoustic-response prediction, frequency-residual prediction, or validated Physics-Aligned Encoder behavior.

## 11. Tables/Figures Plan Patch

Add:

- Table 7: Frequency-only and frequency + embedding information-budget comparison.
- Table 8: Fixed-k versus best-k kNN diagnostic.
- Supplementary Table S5: paired v6.1 vs v6.2-A board/material fold deltas.
- Supplementary Table S6: frequency-fusion per-class changes.

Suggested mapping:

- Table 7 source: `frequency_only_baseline/FREQUENCY_ONLY_BASELINE_REPORT.md` and `frequency_embedding_fusion/FREQUENCY_EMBEDDING_FUSION_REPORT.md`.
- Table 8 source: `knn_protocol/KNN_PROTOCOL_HARDENING_REPORT.md`.
- Supplementary Table S5 source: paired board/material delta CSVs.
- Supplementary Table S6 source: `frequency_embedding_fusion_per_class.csv`.

## 12. Claim Boundary Paragraph

Polished insertion candidate:

> The present study establishes frozen, image-derived ESPI encoder evidence rather than a complete physical surrogate model. The current five-class modal-label protocol is strongly frequency-structured, and frequency metadata is a stronger overall predictor than the image-derived embeddings for this label task. Therefore, v6.2-A is interpreted as the strongest image-derived frozen ESPI encoder baseline under grouped board/material evaluation, not as the strongest possible classifier when metadata is available. The study does not claim acoustic-response predictive value, a validated Physics-Aligned Encoder, LeFFT superiority, metamaterials readiness, or full retrained CNN LOBO/LOMO generalization.

## Do Not Change

- Do not change locked metrics.
- Do not claim frequency + embedding fusion as a broad improvement over frequency-only.
- Do not claim image embeddings are necessary for simple five-class label classification.
- Do not introduce acoustic-response, LeFFT, or metamaterials claims.
- Do not move Material LOMO from descriptive stress-test to broad inferential generalization.

## Artifacts Supporting This Patch Plan

- Dataset composition audit: `DATASET_COMPOSITION_REPORT.md`.
- Paired v6.1 vs v6.2-A deltas: `PAIRED_V61_V62A_GROUPED_DIFFERENCES.md`.
- Frequency-only baseline: `frequency_only_baseline/FREQUENCY_ONLY_BASELINE_REPORT.md`.
- Frequency + embedding fusion: `frequency_embedding_fusion/FREQUENCY_EMBEDDING_FUSION_REPORT.md`.
- kNN protocol hardening: `knn_protocol/KNN_PROTOCOL_HARDENING_REPORT.md`.
