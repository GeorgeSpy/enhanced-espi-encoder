# Visual Leakage Audit Protocol

## 1. Purpose

Image-derived ESPI representation claims require verification that the model is not using non-physical visual information. Before any image-derived ESPI claim is promoted, the dataset and preprocessing pipeline must be audited for printed excitation frequency, class text, acquisition labels, axes, borders, colorbars, timestamps, instrument annotations, and other overlays inside the image pixels.

The purpose of this protocol is to separate physical ESPI fringe evidence from visual shortcuts and acquisition-domain artifacts. It is an integrity and claim-boundary protocol, not a model-development protocol.

## 2. Risk Model

The audit must account for four leakage channels:

- Explicit metadata leakage: `frequency_hz`, class labels, board labels, material labels, specimen IDs, or acquisition fields enter the model or evaluation path directly.
- Label-provenance leakage: modal labels are partly or fully derived from frequency windows, frequency maps, or fallback label rules.
- Visual leakage: frequency, source pressure, class, board, acquisition ID, axes, borders, timestamps, colorbars, or instrument text are printed inside the actual image pixels used for training, embedding extraction, or evaluation.
- Domain leakage: board, specimen, session, background, camera, illumination, mounting, or acquisition-root artifacts allow a model to infer domain identity instead of physical morphology.

These channels can coexist. Removing visual overlays does not automatically remove frequency-label circularity or acquisition-domain structure.

## 3. Clean ROI Requirements

Publishable image-derived claims require a clean ROI/no-overlay image path with:

- only the physical ESPI/fringe region retained;
- no printed frequency;
- no acquisition text;
- no axes, class labels, mode labels, board labels, or colorbars;
- no fixed borders carrying metadata or acquisition fingerprints;
- verified ROI-mask or crop application before embedding extraction and evaluation;
- reproducible clean-image manifest with original path, clean path or virtual transform ID, mask coordinates, dimensions, and QC status.

If a clean ROI image cannot be materialized, the virtual transform must be deterministic, versioned, and reproducible.

## 4. Audit A: Visual Sampling

Export representative samples before model or claim promotion. Sampling must cover:

- board or specimen;
- material;
- class;
- low, mid, and high frequency ranges;
- acquisition session or acquisition root if available;
- raw image version;
- masked-overlay version;
- ROI-only version.

Contact sheets should include full images, corner crops, border strips, known overlay regions, and central ROI crops. Visual sampling must be sufficient to detect both obvious overlays and fixed-position low-contrast metadata.

## 5. Audit B: Top-left Patch Baseline

Train a simple patch-only diagnostic using only the top-left patch or any known overlay region. Acceptable models include logistic regression, linear SVM, random forest, or simple handcrafted descriptors.

Targets should include:

- modal label;
- frequency bin;
- board ID;
- material;
- acquisition session or acquisition root if available.

Decision rule:

- If the patch-only model predicts modal labels or frequency bins above chance, visual leakage is present.
- If the patch-only model predicts board, material, or session above chance, domain fingerprinting is present.
- A positive patch baseline does not identify the full model's mechanism by itself, but it invalidates unqualified image-derived morphology claims until cleaned ROI controls are run.

## 6. Audit C: Overlay-masked Comparison

Compare at least three image variants:

- original full image;
- full image with overlay region masked;
- ROI-only image.

Report:

- model or descriptor performance deltas;
- embedding nearest-neighbor geometry changes;
- board/material/session prediction changes;
- frequency-bin prediction changes;
- class and cross-board same-class geometry changes.

Interpretation:

- A large drop after masking suggests reliance on non-physical overlay or border information.
- Stable performance after masking means only that the tested diagnostic performance survives that cleaning step.
- Stable performance does not prove global frequency-independent morphology recognition.

## 7. Audit D: OCR / Manual Overlay Check

OCR can be used as an optional advisory tool to flag printed frequencies, text labels, acquisition strings, or axes. OCR is low-trust for ESPI images and should not be the sole decision mechanism.

Manual inspection and fixed-region masking are sufficient when overlay coordinates are known. The audit should record:

- detected overlay type;
- image region;
- coordinate range;
- affected source roots;
- whether overlays appear in training/evaluation images or only in manuscript-rendered figures.

The distinction between illustrative manuscript overlays and actual model-input overlays is mandatory.

## 8. Audit E: Exact-frequency Clean ROI Morphology Diagnostic

Exact-frequency or near-exact-frequency morphology tests must be repeated only on clean ROI/no-overlay images.

For each adjacent class pair or candidate morphology contrast, report:

- exact frequency or narrow frequency bin;
- sample count per class;
- board/specimen support;
- material support;
- grouped split feasibility;
- shuffled-label control;
- permutation or bootstrap uncertainty where feasible.

Positive findings are local diagnostics unless sufficient grouped cells support broader generalization. Underpowered pairs must be labeled `inconclusive_underpowered`, not negative.

## 9. Required Outputs

Expected artifacts:

- `reports/visual_leakage_audit_v001/VISUAL_LEAKAGE_AUDIT.md`
- `reports/visual_leakage_audit_v001/top_left_patch_baseline.csv`
- `reports/visual_leakage_audit_v001/masked_vs_roi_comparison.csv`
- `reports/visual_leakage_audit_v001/clean_roi_sample_manifest.csv`
- `reports/visual_leakage_audit_v001/summary.json`

Optional supporting artifacts:

- contact sheets for raw, masked, and ROI variants;
- mask coordinate specification;
- OCR advisory log;
- patch-control confusion matrices;
- near-duplicate or acquisition-contamination audit.

## 10. Claim Interpretation

If visual leakage is present, historical embeddings remain diagnostic only.

If clean ROI performance remains strong, embeddings become eligible for response-target tests, not automatically eligible for global morphology claims.

Clean ROI success does not by itself prove global morphology recognition. Any morphology claim still requires:

- grouped evaluation;
- frequency-controlled diagnostics;
- label-provenance audit;
- metadata/path leakage audit;
- domain-fingerprint diagnostics;
- negative controls;
- sufficient exact-frequency or response-target support.

The final validation target remains B6 residual response validation: ESPI information must add value beyond frequency-coordinate, metadata, and numerical/FEM baselines under grouped-OOD evaluation.
