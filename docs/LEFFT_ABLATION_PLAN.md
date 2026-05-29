# LeFFT Ablation Plan

This document defines a controlled LeFFT ablation study. The purpose is to test whether explicit Fourier / phase descriptors add measurable grouped-generalization value beyond the current v6.2-A frozen embedding evidence.

## 1. Why LeFFT Is an Ablation, Not a Main Claim

LeFFT should be treated as an ablation because the current strongest evidence comes from v6.2-A frozen embeddings, not from a LeFFT-enhanced architecture.

- v6.2-A already has post-hoc representation evidence from frozen embeddings.
- The hierarchical v6.2 branch is technically valid but underperforms v6.2-A in frozen embedding evaluation.
- The presence of Fourier, phase, or physics-inspired modules does not prove representation-level physics alignment.
- LeFFT superiority requires a matched comparison against v6.2-A under the same feature dump, split, metric, and grouped-evaluation protocol.
- Until such an ablation is completed, LeFFT is a candidate descriptor family or future architectural direction, not a central manuscript claim.

## 2. Compared Representations

### v6.2-A Frozen Encoder

The v6.2-A frozen encoder is the current official reportable encoder candidate.

- Source: v6.2-A reportable baseline checkpoint.
- Representation: pre-head frozen embedding from `MCDropoutClassifier.global_pool`.
- Role: main frozen ESPI encoder evidence.
- Expected use: kNN, prototype, linear probe, and grouped OOD representation evaluation.

### Hierarchical v6.2 Phase2 Expert

The hierarchical v6.2 phase2 expert is a physics-aware architectural candidate.

- Source: phase2 expert checkpoint from the hierarchical branch.
- Representation: `outputs["embeddings"]`, interpreted as `z_expert_prelogit / z_arcface_input`.
- Includes denoising, phase-related outputs, LeFTP / Fourier-prior components, gatekeeper, expert classifier, and ArcFace head.
- Role: controlled physics-aware candidate branch.
- Current status: technically valid and auditable, but weaker than v6.2-A in frozen embedding evaluation.

### LeFFT Descriptor-Only Baseline

The LeFFT descriptor-only baseline uses explicit hand-engineered Fourier / phase descriptors without CNN embeddings.

- Source: saved ESPI images or existing phase/amplitude representations, depending on the available preprocessing path.
- Representation: compact descriptor vector built from Fourier, phase, amplitude, fringe, and local coherence statistics.
- Role: tests whether explicit physics-inspired descriptors are independently predictive.
- Claim boundary: if weak, it can serve as a controlled negative baseline; if strong, it motivates deeper LeFFT integration.

### v6.2-A Embedding + LeFFT Descriptor Fusion

The fusion baseline concatenates frozen v6.2-A embeddings with LeFFT descriptors.

- Source: v6.2-A frozen embedding plus matched LeFFT descriptor vector for the same sample.
- Representation: `[z_v62A ; d_LeFFT]`, optionally standardized per training fold.
- Role: tests whether explicit LeFFT descriptors add information not already captured by the v6.2-A embedding.
- Critical comparison: fusion must be compared against v6.2-A embeddings alone under identical grouped splits.

## 3. Proposed LeFFT Descriptors

The descriptor set should be compact, deterministic, and reproducible.

- Wrapped phase statistics: circular mean, circular variance, phase entropy, wrapped phase histogram summaries.
- Amplitude statistics: mean, standard deviation, robust percentiles, skewness, kurtosis, amplitude entropy.
- Phase-gradient statistics: gradient magnitude mean / variance, directional gradient histograms, high-gradient fraction.
- Spectral energy concentration: low-, mid-, and high-frequency energy ratios; radial spectral concentration; spectral entropy.
- Dominant spectral radius / orientation: dominant radial frequency, dominant orientation, orientation concentration, anisotropy ratio.
- Fringe density: estimated fringe count, zero-crossing / phase-wrap density, local fringe spacing summaries.
- Local contrast / coherence descriptors: patch-level contrast, local phase coherence, amplitude-phase consistency, texture coherence statistics.

Descriptors should be computed without using validation or test labels. Any normalization must be fitted only on the training/reference fold and then applied to the held-out fold.

## 4. Evaluation Protocol

The LeFFT ablation must use the same frozen-representation audit protocol as the v6.2-A encoder evidence.

### Methods

- kNN using cosine or standardized Euclidean distance.
- Linear probe using balanced logistic regression.
- Nearest class prototype classifier.

### Splits

- Stratified train-to-validation evaluation, for continuity with the baseline audit.
- Board-grouped LOBO-style evaluation.
- Material-grouped LOMO-style evaluation.

### Representation Conditions

Each method and split should be run for:

- v6.2-A embedding only.
- LeFFT descriptor only.
- v6.2-A embedding + LeFFT descriptor fusion.
- Hierarchical v6.2 phase2 expert embedding, reported separately as a physics-aware branch reference.

### Metrics

- Accuracy.
- Macro Recall.
- Macro-F1.
- Per-class recall and F1 where possible.
- Mean grouped Macro-F1.
- Worst-group Macro-F1.
- Best-group Macro-F1.

Macro-F1 under board/material grouped evaluation is the primary decision metric.

## 5. Decision Criteria

The ablation should make a binary engineering decision.

Proceed toward a v6.2-B model only if:

- LeFFT descriptor fusion improves v6.2-A grouped Macro-F1 under matched board-grouped or material-grouped evaluation.
- The improvement is consistent across more than one grouped protocol or is concentrated in a scientifically meaningful failure mode.
- The fusion gain is not caused by leakage, metadata shortcuts, or test-fold normalization.

Keep LeFFT as a controlled negative result or future work if:

- Descriptor-only performance is weak.
- Fusion does not improve v6.2-A grouped Macro-F1.
- Improvements appear only in stratified validation and disappear under LOBO-style or LOMO-style evaluation.
- Descriptor effects are dominated by board/material signatures rather than mode-relevant structure.

If LeFFT improves grouped Macro-F1, the next architectural candidate can be called v6.2-B. If it does not, v6.2-A remains the main frozen encoder candidate and LeFFT remains an ablation result.

## 6. Claim Boundaries

The LeFFT ablation must preserve conservative publication wording.

Do not claim:

- LeFFT superiority without a matched ablation against v6.2-A.
- A validated Physics-Aligned Encoder.
- Acoustic-response predictive value.
- That Fourier / phase descriptors imply physics alignment by construction.
- That descriptor fusion is useful if gains occur only under stratified splits.

Can claim after successful ablation:

- LeFFT descriptors add measurable grouped-representation value to v6.2-A under the tested protocol.
- LeFFT fusion motivates a v6.2-B architecture.
- Explicit Fourier / phase descriptors are useful as controlled representation features, subject to grouped validation.

Can claim after failed ablation:

- LeFFT descriptors were tested under a matched protocol and did not improve grouped frozen-embedding performance.
- LeFFT remains future work or a controlled negative baseline.
- The main evidence remains the v6.2-A frozen encoder candidate.

## 7. Required Outputs

The ablation should produce the following artifacts:

- `LEFFT_DESCRIPTOR_AUDIT.md`: descriptor extraction summary, descriptor definitions, quality checks, descriptor-only audit results, and caveats.
- `LEFFT_FUSION_COMPARISON.md`: matched comparison of v6.2-A embedding only, LeFFT descriptor only, and v6.2-A + LeFFT fusion.
- `lefft_ablation_key_numbers.json`: machine-readable metrics, feature dimensions, split definitions, grouped Macro-F1 summaries, and decision flags.

Recommended supporting CSV outputs:

- `lefft_descriptor_quality_report.csv`
- `lefft_descriptor_only_report.csv`
- `lefft_fusion_knn_report.csv`
- `lefft_fusion_linear_probe_report.csv`
- `lefft_fusion_prototype_report.csv`
- `lefft_fusion_lobo_board_summary.csv`
- `lefft_fusion_lomo_material_summary.csv`

These outputs should be generated only after a dedicated implementation step. This plan does not implement descriptor extraction or evaluation scripts.
