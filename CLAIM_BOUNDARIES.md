# Claim Boundaries and Scope

This document explicitly defines what the empirical results in this repository **do** and **do not** prove, ensuring a reviewer-safe interpretation of the findings.

## Supported Claims
1. **Grouped Specimen Invariance:** Under fixed board-grouped evaluation, the distribution-aware **v6.2-A** encoder generates frozen image-derived embeddings that are significantly more robust across unseen boards than generic controls (ImageNet) and the previous ESPI baseline (v6.1).
2. **Frequency Metadata Dominance:** The current 5-class ESPI modal protocol is strongly frequency-structured. Scalar frequency (`frequency_hz`) provides a higher overall classification accuracy than any single image-derived frozen embedding for the full five-class task.
3. **Resolving Frequency Ambiguity:** In targeted overlapping frequency intervals where scalar frequency is ambiguous (e.g., between the 1_2 and 2_1 modes at 550-556 Hz), the image-derived spatial morphology captured by v6.2-A effectively resolves the ambiguity and corrects the classification errors of the frequency-only baseline.
4. **Improved Representation Geometry:** The v6.2-A representation provides demonstrably better embedding geometry (higher Silhouette score, lower Davies-Bouldin index, tighter intra/inter-class distances) compared to v6.1, independent of the downstream classifier head.

## Unsupported Claims (Out of Scope)
1. **End-to-End Deep Learning Superiority:** This repository evaluates **frozen features** with lightweight downstream heads (kNN, Linear Probes). It does not evaluate end-to-end retrained CNN architectures under Leave-One-Board-Out validation.
2. **LeFFT / Learned Spectral Superiority:** The hierarchical physics-aware checkpoint (v6.2 phase2) was evaluated internally but did not outperform the ResNet-based v6.2-A under grouped representation metrics. The code includes a deterministic FFT-based spectral descriptor control, but it does **not** evaluate a fully trained LeFFT model.
3. **Acoustic-Response Prediction:** The study focuses on modal pattern classification. It does not test or claim whether these ESPI embeddings can predict continuous acoustic, phononic, or metamaterial response variables.
