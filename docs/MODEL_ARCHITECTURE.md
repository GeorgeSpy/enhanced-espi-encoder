# Model Architecture Notes

This document defines the current architecture terminology used in the manuscript, reports, and encoder-evidence package. It separates the conservative classifier baselines from the hierarchical / physics-aware development branch and sets explicit claim boundaries.

## 1. v6.1 - Reference Frozen ResNet-18 Baseline

v6.1 is the conservative reference classifier baseline.

- Conservative 5-class ESPI mode classifier.
- Uses 1-channel ESPI image input.
- Uses a frozen ResNet18-style backbone.
- Adds a lightweight classifier head for the 5-class task.
- Does not include an explicit Fourier, phase, or physics branch.
- Role: reference baseline for later v6.2-A and hierarchical comparisons.

## 2. v6.2-A - Official Reportable Baseline and Frozen Encoder Candidate

v6.2-A is the official reportable 5-class baseline and the main frozen encoder-evidence source.

- Based on the v6.1 architecture.
- Uses a ResNet18-style frozen backbone.
- Keeps the architecture conservative rather than introducing a physics branch.
- Adds distribution-aware / domain-mixed training logic.
- Uses clean, averaged, and pseudo-noisy batch composition.
- Uses targeted noisy-like augmentation.
- Includes class-imbalance handling.
- Includes zero-recall safeguards for fragile minority classes.
- Role: official reportable 5-class baseline and main frozen ESPI encoder candidate.

v6.2-A should not be described as LeFFT-based or as the hierarchical physics-aware architecture. Its frozen-embedding evidence comes from the external `MCDropoutClassifier.global_pool` extractor, so MC Dropout should only be mentioned when tied directly to that actual extractor and checkpoint evidence.

## 3. v6.2 Hierarchical / Physics-Aware Branch

The hierarchical v6.2 branch is a controlled physics-aware architectural candidate, separate from the official v6.2-A reportable baseline.

- Includes denoising components.
- Includes a learnable Fourier prior / LeFTP path.
- Exposes wrapped phase, unwrapped phase, and amplitude outputs.
- Uses multiscale physics feature processing.
- Uses a compact convolutional classifier backbone.
- Includes a gatekeeper head.
- Includes an expert embedding.
- Includes an ArcFace head.
- Role: controlled physics-aware architectural candidate for comparison and future encoder development.

The current phase2 expert checkpoint is technically valid: it loads cleanly, avoids placeholder physics modules, exposes `outputs["embeddings"]`, and passes smoke and extraction checks. However, its frozen embedding audit underperforms v6.2-A in representation-space evaluation. No validated Physics-Aligned Encoder claim is currently supported by this branch.

## Why Classification Performance and Frozen Embedding Quality Differ

Classifier performance and frozen embedding quality are related but not equivalent.

- A classifier head can produce strong class predictions even when the underlying embedding geometry is not robust under kNN, prototype, or grouped retrieval-style evaluation.
- ArcFace or margin-based classification can improve supervised decision boundaries without necessarily yielding embeddings that are stable under nearest-neighbor or prototype classifiers.
- The phase2 expert checkpoint is not necessarily a final fine-tuned encoder checkpoint. It is a trained hierarchical classifier/expert checkpoint that can be audited as a frozen representation source, not a finished encoder.
- Architecture-level physics awareness does not guarantee representation-level physics alignment. Denoising, phase, Fourier, or physics-feature modules can be present while the final embedding space remains sensitive to board, material, or distribution signatures.
- Frozen embedding audits test whether the saved representation space is reusable without the original classifier head. They should therefore be reported separately from classifier accuracy.

## Publication Claim Boundary

Current evidence supports cautious representation-level claims, not final encoder claims.

Can claim:

- v6.2-A is the official reportable 5-class baseline.
- v6.2-A is a frozen ESPI encoder candidate.
- v6.2-A has grouped post-hoc representation evidence from saved frozen embeddings.
- The hierarchical v6.2 branch is a technically loadable and auditable physics-aware candidate branch.
- The hierarchical phase2 checkpoint exposes a valid expert embedding point for controlled frozen-representation audit.

Cannot claim yet:

- A validated Physics-Aligned Encoder.
- Acoustic-response predictive value from the ESPI embeddings.
- LeFFT / LeFTP superiority over the v6.2-A baseline.
- Full retrained-CNN LOBO / LOMO generalization.
- That hierarchical physics-aware architecture alone proves physics-aligned representation learning.
