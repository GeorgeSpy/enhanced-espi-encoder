# Model and Feature Sources

## v6.1 Reference Baseline

- Role: reference frozen ResNet18-style ESPI baseline.
- Input: 1-channel ESPI patterns.
- Feature point: pre-classifier / average-pool representation.
- Embedding dimension: 512.
- Scope: strong stratified representation baseline; weaker grouped board/material robustness than v6.2-A.

## v6.2-A Main Image-Derived Frozen Encoder Candidate

- Role: official reportable image-derived frozen ESPI encoder baseline.
- Lineage: conservative ResNet18-style architecture derived from v6.1.
- Training design: distribution-aware/domain-mixed sampling logic, clean/averaged/pseudo-noisy batch composition, targeted noisy-like augmentation, class-imbalance handling, and zero-recall safeguards.
- Feature point: frozen pre-head embedding.
- Embedding dimension: 1280.
- Scope: strongest evaluated image-derived frozen ESPI representation under grouped board/material evaluation.
- Boundary: v6.2-A is not LeFFT-based and is not the hierarchical physics-aware branch.

## Hierarchical v6.2 Phase2

- Role: controlled physics-aware architecture comparison.
- Components: denoising branch, learnable Fourier prior / LeFTP-style components, phase/amplitude outputs, gatekeeper head, expert embedding, expert classifier, and ArcFace head.
- Feature point used in audits: `outputs["embeddings"]`, corresponding to `z_expert_prelogit / z_arcface_input`.
- Embedding dimension: 512.
- Scope: technically valid checkpoint and wrapper, but not superior as a frozen representation source.
- Boundary: not a validated Physics-Aligned Encoder.

## Generic ResNet Controls

- Random ResNet-18: no pretrained weights; frozen 512-dimensional average-pool representation.
- ImageNet ResNet-18: ImageNet-pretrained visual baseline; frozen 512-dimensional average-pool representation.
- Scope: generic visual controls. They can perform non-trivially under stratified evaluation but collapse under grouped board/material evaluation.

## Frequency Metadata Control

- Input: scalar `frequency_hz`.
- Role: metadata-only information-budget control.
- Finding: dominant predictor for the current five-class modal-label protocol.
- Boundary: frequency metadata is not an image-derived ESPI representation.

## Deterministic Spectral Descriptor Control

- Input: deterministic FFT-derived spectral descriptors from ESPI images.
- Families: radial energy, angular energy, low/mid/high ratios, centroid, bandwidth, entropy, peak concentration, anisotropy, and patch spectral summaries.
- Role: supplementary LeFFT-inspired control.
- Boundary: not a trained LeFFT model and not evidence of LeFFT superiority.
