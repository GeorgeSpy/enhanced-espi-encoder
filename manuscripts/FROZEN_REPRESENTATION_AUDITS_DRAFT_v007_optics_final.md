# Grouped and Frequency-Controlled Audits of Frozen ESPI Representations for Full-Field Vibration Pattern Analysis

## Abstract

Electronic speckle pattern interferometry (ESPI) provides full-field optical measurements of vibration patterns, but classification accuracy alone does not demonstrate that a model has learned reusable ESPI representations. This study presents a representation audit of frozen image-derived embeddings for five-class ESPI modal-pattern recognition under grouped board and material evaluation protocols. We compare random, ImageNet-pretrained, and two ESPI-trained residual encoders (v6.1 and distribution-aware v6.2-A). Frequency-only metadata controls outperform image-derived embeddings globally, demonstrating that the current label protocol is strongly frequency-dominated. However, under matched grouped evaluation, v6.2-A emerges as the strongest image-derived frozen ESPI encoder baseline, achieving 93.56% Board LOBO and 93.49% Material LOMO Macro-F1 at fixed $k=10$. A targeted frequency-overlap and geometry analysis shows that v6.2-A embeddings provide critical complementary morphology information---resolving frequency-ambiguous modes---and exhibit tighter intra-class representation geometry. This grouped advantage persists after controlling for class, material, and frequency-bin distribution. The results support v6.2-A as the strongest evaluated image-derived frozen ESPI representation, while establishing scalar frequency as the dominant metadata-only predictor. Acoustic-response prediction, learned spectral modules, and full physics-aligned encoder validation remain outside the present scope.

## 1. Introduction

Electronic speckle pattern interferometry (ESPI) is a full-field optical metrology technique for measuring vibration and deformation [1,2]. In vibration studies, ESPI fringe and speckle patterns encode spatially distributed modal information that is not available from a single point sensor. These full-field interferometric images are therefore attractive for automated modal-pattern recognition, but they also contain signatures associated with board geometry, material, acquisition condition, and excitation frequency.

In ESPI vibration measurements, fringe density, nodal-line organization, and spatial modulation patterns are linked to the underlying deformation field and modal response of the specimen. A reusable ESPI representation should therefore preserve full-field modal morphology rather than only reproduce label correlations under a random split.

Most supervised ESPI classification studies evaluate whether a trained model assigns each image to the correct modal class on a validation split. This is a necessary baseline, but it addresses only the final classification decision. It does not determine whether the internal feature space is reusable as an ESPI representation. A classifier head may separate classes under a stratified split while the corresponding embedding space remains strongly tied to board-, material-, frequency-, or acquisition-specific correlations.

This manuscript evaluates ESPI models as frozen encoder candidates. Encoder weights remain fixed, and the saved representations are tested with lightweight downstream heads: fixed-k kNN, prototype classification, and balanced logistic-regression linear probes. This approach separates representation geometry from the original classifier head and provides an audit of whether the embedding space remains informative under grouped evaluation.

Grouped evaluation is important for optical metrology applications because a reusable ESPI representation should remain useful when measurements come from boards or materials not present in the reference set. We therefore include board-held-out and material-held-out protocols. In the board-grouped evaluation, the reference set excludes the held-out board. In the material-grouped evaluation, the reference set excludes the held-out material. These tests are frozen-embedding grouped evaluations, not retrained CNN LOBO/LOMO experiments.

The five modal labels are also frequency-structured. Consequently, `frequency_hz` must be treated as an information-budget control. Without a frequency-only baseline, high modal-label accuracy could be mistaken for evidence that image morphology alone explains the label structure. This study therefore separates metadata-only predictability from image-derived frozen representation evidence. Acoustic-response prediction is treated as future validation rather than a current result.

The evaluation covers generic, ESPI-specific, physics-aware, and metadata-controlled sources. Random ResNet-18 and ImageNet-pretrained ResNet-18 embeddings serve as generic visual controls. v6.1 is the reference ESPI-specific baseline. v6.2-A is the primary ESPI-trained image-derived baseline evaluated in this study. Hierarchical v6.2 phase2 is included as a controlled physics-aware architectural comparison. Frequency-only and frequency-plus-embedding controls quantify how much modal-label information is carried by excitation frequency.

The main finding is a controlled trade-off. Under fixed `k=10`, v6.1 is slightly stronger than v6.2-A in stratified kNN evaluation (93.65% versus 92.78% Macro-F1). Under grouped fixed `k=10` evaluation, v6.2-A is strongest among image-derived frozen encoders, reaching 93.56% Board LOBO Macro-F1 and 93.49% Material LOMO Macro-F1. Frequency-only controls are stronger overall for the current label protocol, reaching 98.67%, 97.49%, and 94.09% Macro-F1 under stratified, Board LOBO, and Material LOMO protocols. The appropriate claim is therefore precise: v6.2-A is the strongest image-derived frozen ESPI encoder baseline among the evaluated encoders, while `frequency_hz` is the dominant metadata-only predictor for the current five-class modal-label protocol.

The present study establishes frozen image-derived ESPI representation evidence under fixed grouped evaluation protocols. The present scope is restricted to frozen image-derived ESPI representation auditing; acoustic-response prediction, learned spectral modules, and full physics-aligned encoder validation define subsequent validation stages.

### Contributions

- **Frozen ESPI encoder audit protocol.** We define a representation-level audit for ESPI embeddings using fixed `k=10` kNN, diagnostic best-k sensitivity analysis, prototype classification, linear probes, board-grouped testing, material-grouped testing, class-level metrics, and zero-recall diagnostics.
- **Controlled baseline and metadata comparison.** We evaluate Random ResNet-18, ImageNet ResNet-18, v6.1, v6.2-A, and hierarchical v6.2 phase2 embeddings, and add frequency-only and frequency-plus-embedding controls within a common normalized schema.
- **Grouped image-derived robustness result.** We show that v6.2-A is the strongest grouped-robust image-derived frozen ESPI encoder baseline, while v6.1 remains slightly stronger under fixed `k=10` stratified kNN and frequency metadata remains stronger for the current label task.
- **Dimension-matched PCA control.** We evaluate the v6.2-A representation under fold-local PCA projection down to 512, 256, and 128 dimensions, demonstrating that the specimen-level grouped advantage persists even at reduced embedding capacity.
- **Per-class and fold-level stability analysis.** We show that v6.2-A improves grouped F1 over v6.1 across all five modal classes and wins 5/6 board folds and 2/2 material folds, supported by Wilcoxon signed-rank and bootstrap significance testing, and a supplementary deterministic spectral descriptor control.

## 2. Related Work

### 2.1 Full-field ESPI Vibration Metrology

Electronic speckle pattern interferometry is a full-field optical metrology technique for measuring displacement and vibration-induced surface deformation through coherent speckle interference [1,2]. In vibration studies, ESPI fringe patterns encode spatially distributed modal information rather than isolated point measurements, which makes them well suited for mode-shape inspection, nondestructive testing, and specimen-level comparison. This full-field structure also creates a representation-learning question: a classifier may separate labels under a conventional split without necessarily producing image-derived embeddings that remain stable across boards, materials, or acquisition conditions. The present work therefore uses ESPI classification as the experimental context, but evaluates frozen representation quality under grouped board/material protocols.

### 2.2 Fringe, Phase, and Spectral Analysis

Classical interferometric analysis relies heavily on fringe, phase, and spectral structure. Fourier-transform fringe analysis provides a standard route from fringe patterns to phase-related quantities [3], while two-dimensional phase unwrapping methods address the recovery of continuous phase fields from wrapped measurements [4]. Speckle optics also motivates frequency-domain descriptions of spatial interference patterns [5]. These ideas motivate the deterministic spectral descriptors used here as supplementary controls. The descriptors summarize FFT energy distributions and anisotropy, but they are not a trained LeFFT model and are not used to claim LeFFT superiority. Recent deep learning approaches have explored neural networks for ESPI fringe denoising [9], phase retrieval [10], and continuous phase unwrapping [12], confirming the ongoing shift from analytical to learned interferometric analysis [11, 13]. Our study complements this shift by focusing on representation and frozen-feature evaluation [14].

### 2.3 CNN Representations and Frozen Encoder Evaluation

Residual convolutional networks provide a standard backbone family for visual representation learning [6], and ImageNet pretraining is widely used as a generic visual baseline [7]. In optical metrology, however, strong classification accuracy does not by itself establish reusable representation quality. Frozen-embedding audits using kNN, prototype classifiers, and linear probes provide a lightweight way to test whether the embedding space itself carries class-relevant structure, rather than relying only on the final classifier head. This distinction is central to the present comparison between generic ResNet controls, ESPI-specific v6.1 and v6.2-A embeddings, and the hierarchical v6.2 phase2 branch.

### 2.4 Grouped Evaluation and Distribution-shift Controls

Grouped evaluation is important when samples share specimen, acquisition, or material structure. Stratified splits can overestimate robustness if training and validation sets contain closely related samples from the same physical boards or materials. Domain-generalization benchmarks similarly emphasize that model selection and evaluation should respect group structure when the intended use involves distribution shift [8]. The board-held-out and material-held-out protocols used here are therefore treated as grouped evaluation tests for frozen ESPI representations. Material LOMO is interpreted descriptively as a material-held-out stress test because only two material groups are available.

### 2.5 Frequency Metadata and Information-budget Controls

Modal labels in this dataset are strongly frequency-structured, so frequency metadata is a necessary information-budget control. A model that predicts labels from `frequency_hz` alone does not demonstrate image-derived ESPI representation quality, but it does define how much label information is available outside the image. The present study therefore reports frequency-only and frequency-fusion controls alongside image-derived frozen embeddings. This framing keeps the main claim specific: v6.2-A is evaluated as an image-derived frozen ESPI encoder baseline, while frequency metadata is treated as a dominant metadata-only predictor for the current label protocol.

## 3. Methods

### 3.1 Dataset and Task Definition

The dataset contains ESPI full-field vibration patterns organized under a five-class modal protocol. Each sample belongs to one of five modal labels: `1_1H`, `1_1T`, `1_2`, `2_1`, or `higher`.

The frozen-encoder evidence set contains 12,944 samples from six boards (`C01`, `C02`, `C03`, `W01`, `W02`, `W03`) and two materials (`carbon`, `wood`). All boards and both materials contain all five modal classes. No board- or material-held-out fold lacks held-out or reference classes, so grouped Macro-F1 values are not artifacts of missing-class folds.

The class distribution is strongly imbalanced. The `higher` class contains 10,115 samples, corresponding to 78.14% of the dataset. The remaining classes contain 653 samples for `1_1H`, 669 samples for `1_1T`, 738 samples for `1_2`, and 769 samples for `2_1`.

Supplementary Tables S1-S4 document the grouped support structure. Supplementary Table S1 reports support by board, Supplementary Table S2 reports support by material, Supplementary Table S3 reports class-by-board support, and Supplementary Table S4 reports class-by-material support.

Macro-F1 is used as a primary summary metric because the dataset is class-imbalanced. Accuracy can be dominated by the majority `higher` class, whereas Macro-F1 gives equal weight to each modal class. This is necessary for assessing minority modal classes under grouped evaluation.

### 3.2 Encoder Sources

Five frozen representation sources are evaluated.

The Random ResNet-18 baseline is an untrained generic visual control. It tests whether architectural texture bias alone can produce apparent stratified separability.

The ImageNet ResNet-18 baseline is a generic pretrained control. ESPI images are passed through an ImageNet-pretrained ResNet-18 feature extractor, and embeddings are taken from the pre-classifier average-pooling representation. This tests whether natural-image features transfer to ESPI modal representation without ESPI-specific training.

v6.1 is the reference ESPI-specific baseline. It uses a one-channel ESPI input, a frozen ResNet18-style backbone, and a lightweight classifier head. It does not include an explicit Fourier, phase, or physics branch.

v6.2-A is the official reportable ESPI baseline and the main image-derived frozen encoder candidate. It is based on the v6.1 architecture and remains a conservative ResNet18-style model rather than a LeFFT-based or hierarchical physics-aware architecture. Its training configuration includes distribution-aware/domain-mixed sampling logic, clean/averaged/pseudo-noisy batch composition, targeted noisy-like augmentation, class-imbalance handling, and zero-recall safeguards. In this evaluation, v6.2-A is represented by its recorded 1280-dimensional pre-head embedding.

The hierarchical v6.2 phase2 branch is included as a controlled physics-aware architectural comparison. It contains denoising, a learnable Fourier-prior / LeFTP path, wrapped/unwrapped phase and amplitude outputs, multiscale physics features, a gatekeeper head, an expert embedding, and an ArcFace head. The phase2 checkpoint is technically valid and auditable as a frozen embedding source, but it is not superior to v6.2-A in the current representation evaluation.

### 3.3 Frozen Embedding Extraction

All comparisons use frozen embeddings. Encoder weights are not updated, and no deep model is retrained or fine-tuned. Lightweight evaluation heads are fitted only on the relevant training/reference fold.

| Encoder | Embedding point | Dimension |
| --- | --- | ---: |
| Random ResNet-18 | `avgpool_pre_fc` | 512 |
| ImageNet ResNet-18 | `avgpool_pre_fc` | 512 |
| v6.1 | `avgpool_pre_classifier` | 512 |
| v6.2-A | `MCDropoutClassifier.global_pool` | 1280 |
| hierarchical v6.2 phase2 | `z_expert_prelogit` | 512 |

For v6.1, the extraction report documents a clean checkpoint load audit with 124 matched keys, 0 missing keys, 0 unexpected keys, and 0 shape mismatches. The full extraction produced 12,944 embeddings of shape `[12944, 512]`, with 0 NaN values, 0 Inf values, 0 duplicate paths, nonzero variance, and metadata lengths matching the embedding array.

For the Random and ImageNet ResNet-18 controls, ESPI images were loaded as grayscale and resized to 256 x 256. The random baseline used a one-channel convolutional input adaptation and mean/std normalization of 0.5/0.5. The ImageNet baseline repeated grayscale ESPI images to RGB and applied ImageNet mean/std normalization. Both baselines produced 12,944 embeddings of dimension 512 with 0 NaN values, 0 Inf values, 0 duplicate paths, and nonzero variance.

### 3.4 Normalized Feature Schema

All feature dumps were normalized to a common NPZ schema before unified evaluation. The schema stores embeddings, labels, label names, paths, board identifiers, material identifiers, frequency metadata, split labels, split-group identifiers, distribution groups, checkpoint metadata, embedding point, model name, source feature identifier, and schema version.

The normalized schema ensures that all evaluated encoders use aligned labels, sample paths, boards, materials, split labels, split-group identifiers, and frequency metadata. Checkpoint paths and SHA256 values are stored when applicable.

The v6.2-A normalized feature dump contains 12,944 samples and 1280-dimensional embeddings. Its embedding point was set from the available embedding-layer metadata, and its checkpoint SHA256 was computed from the documented checkpoint path.

The hierarchical v6.2 phase2 normalized feature dump contains 12,944 samples and 512-dimensional embeddings. The hierarchical v6.2 phase2 row is retained as a controlled architecture comparison and is not used to support the primary v6.2-A conclusion.

### 3.5 Evaluation Protocol

The primary kNN protocol uses cosine kNN with fixed `k=10`. This value was locked as the default grouped kNN setting before manuscript hardening and is used for the main grouped Board LOBO and Material LOMO claims. Additional stratified kNN values for `k = 1, 3, 5, 10, 20` are reported only as diagnostic sensitivity analysis. Best-k results are not used as the primary manuscript protocol.

All feature scaling, standardization, or normalization used by lightweight heads is fitted only on the training/reference fold and then applied to the held-out fold. Grouped held-out samples are not used to fit preprocessing. This rule applies to stratified train/validation evaluation, board-grouped evaluation, material-grouped evaluation, frequency-only controls, frequency-fusion controls, and deterministic spectral descriptor controls.

The prototype classifier computes one class prototype per class from the reference fold and assigns each test embedding to the nearest prototype. This tests whether class structure is organized around stable class centers in the frozen embedding space.

The linear probe uses balanced logistic regression. The publication-grade evaluation used `linear_max_iter = 5000`, and the captured linear convergence warnings were 0.

The board-grouped protocol holds out each board as the test group and uses all other boards as the reference group. The material-grouped protocol holds out each material group and uses the other material group as the reference group. These are frozen-embedding grouped evaluations: encoder weights remain fixed, and only the reference/test grouping changes.

Grouped Board LOBO and Material LOMO scores are reported as arithmetic means of held-out-group Macro-F1 values. For each held-out board or material, Macro-F1 is computed on that held-out group for each method. The reported grouped score is the mean of those held-out-group Macro-F1 values.

The evaluation computed 95% bootstrap confidence intervals for the main summary Macro-F1 metrics using 200 iterations. For stratified and lightweight-probe metrics, bootstrap resampling was performed over prediction pairs at the sample level. For grouped summary metrics, bootstrap resampling was performed over held-out-group Macro-F1 values, with the grouped mean recomputed on each bootstrap draw.

### 3.6 Frequency-only, Fusion, and Spectral Descriptor Controls

Frequency-only controls use `frequency_hz` as the sole input feature. We evaluated balanced logistic regression and random forest under the same stratified, board-grouped, and material-grouped protocols. The logistic-regression control used balanced class weights, training/reference-fold standardization, and `max_iter = 5000`.

Frequency-plus-embedding fusion concatenates scalar `frequency_hz` with frozen encoder embeddings. Fusion conditions are evaluated with balanced logistic regression and random forest. All preprocessing is fitted only on the training/reference fold.

We also evaluated deterministic spectral / LeFFT-inspired descriptors as a lightweight supplementary control. These descriptors are computed from grayscale ESPI images using deterministic FFT-based statistics; they are not a trained LeFFT model. The control includes spectral-only descriptors, frequency + spectral descriptors, v6.2-A + spectral descriptors, and frequency + v6.2-A + spectral descriptors. These controls are included only to test whether simple spatial-frequency descriptors provide complementary information under the same lightweight-head protocols.

No frequency-control or spectral-control experiment updates encoder weights, trains a deep model, implements acoustic-response prediction, or supports a LeFFT superiority claim.

### 3.7 Class-Level Diagnostics and QA

Class-level diagnostics determine whether grouped robustness is distributed across modal classes or driven mainly by the majority class. The diagnostics include per-class precision, recall, F1, support, macro averages, weighted averages, confusion matrices, zero-recall rows, per-board class diagnostics, and per-material class diagnostics.

The class-level analysis compares v6.2-A against v6.1 under grouped evaluation. It reports grouped F1 for each modal class and the v6.2-A minus v6.1 difference in percentage points.

QA checks verified metadata consistency across all five image-derived encoders, label-permutation sanity, metric consistency across generated CSV/JSON/Markdown outputs, feature consistency, 0 NaN values, 0 Inf values, duplicate-path checks, and convergence-warning handling.

## 4. Results

### 4.1 Encoder Baseline Overview

The unified evaluation compares five image-derived frozen representation sources under the same fixed `k=10` protocol and grouped evaluation design [Table 1]. The generic Random and ImageNet ResNet-18 controls achieve non-trivial stratified scores but fail under grouped board/material testing. v6.1 is the strongest fixed-k stratified image-derived baseline, while v6.2-A is the strongest grouped image-derived baseline. Hierarchical v6.2 phase2 is technically valid but not competitive in frozen embedding evaluation.

### 4.2 Stratified Representation Performance

Under the primary fixed `k=10` protocol, v6.1 reaches 93.65% stratified kNN Macro-F1, and v6.2-A reaches 92.78%. Random ResNet-18, ImageNet ResNet-18, and hierarchical v6.2 phase2 reach 81.14%, 84.34%, and 37.80%, respectively.

The best-k diagnostic does not change the qualitative interpretation. v6.1 reaches 95.07% at `k=5`, whereas v6.2-A reaches 92.78% at `k=10`. Best-k values are reported as sensitivity analysis only; fixed `k=10` is the primary kNN protocol [Table 8].

### 4.3 Board-Grouped Robustness

The board-grouped protocol separates stratified embedding geometry from grouped robustness. At fixed `k=10`, v6.2-A achieves 93.56% Board LOBO Macro-F1, with a 95% confidence interval of [92.36%, 94.67%]. v6.1 reaches 87.10%, with a 95% confidence interval of [82.73%, 91.23%]. Random ResNet-18 and ImageNet ResNet-18 reach 26.17% and 39.30% under their best grouped evaluation methods, while hierarchical v6.2 phase2 reaches 24.99%.

Paired fold-level comparison shows that v6.2-A improves over v6.1 by +6.47 percentage points in mean Board LOBO Macro-F1 and wins 5/6 board folds. The C02 board is essentially tied (-0.02 pp for v6.2-A relative to v6.1), and W01 shows a small v6.2-A margin (+0.27 pp). The largest board gains occur for C01, W02, and W03 [Supplementary Table S3].

These results show that v6.2-A preserves class-relevant image-derived structure more effectively when the reference set excludes a board group. A directional paired Wilcoxon signed-rank test over the six Board LOBO folds supported the v6.2-A F1 improvement over v6.1 ($p = 0.03125$), supported by a paired t-test ($p = 0.01547$), a Cohen's $d$ effect size of $1.2149$, and a paired bootstrap 95% confidence interval of $[+2.09\text{ pp}, +10.15\text{ pp}]$ for the mean pairwise F1 delta. The collapse of Random and ImageNet controls under board grouping indicates that stratified texture similarity alone is insufficient for board-level ESPI representation robustness [Table 2] [Figure 3].

### 4.4 Material-Grouped Robustness

The material-grouped protocol gives the same image-derived encoder conclusion, with the required caution that only two material groups are available. At fixed `k=10`, v6.2-A reaches 93.49% Material LOMO Macro-F1, with a 95% confidence interval of [93.01%, 93.97%]. v6.1 reaches 86.33%, with a 95% confidence interval of [85.32%, 87.34%]. Random ResNet-18, ImageNet ResNet-18, and hierarchical v6.2 phase2 reach 24.59%, 22.52%, and 24.47%, respectively.

Paired fold-level comparison shows that v6.2-A improves over v6.1 by +7.16 percentage points in mean Material LOMO Macro-F1 and wins 2/2 material folds (see Supplementary Table S3 for fold-level details). Because the material grouping contains only `carbon` and `wood`, this result is interpreted as a descriptive material-held-out stress test rather than as a broad inferential estimate of material generalization.

The material-grouped result supports v6.2-A as the strongest grouped-robust image-derived frozen ESPI encoder baseline in this comparison. The result is a frozen-embedding grouped evaluation, not retrained CNN LOMO generalization [Table 2] [Figure 3].

### 4.5 Dimension-Matched PCA Control

To address the concern that the performance advantage of v6.2-A (1280 dimensions) over v6.1 (512 dimensions) is merely an artifact of a larger embedding capacity, we evaluated v6.2-A under dimensionality reduction. A fold-local Principal Component Analysis (PCA) was fitted on each training fold and applied to the corresponding held-out fold to project the 1280-dimensional embeddings down to 512, 256, and 128 dimensions. Under the fixed $k=10$ Board LOBO protocol, the dimension-reduced v6.2-A representations achieve 93.58% (512 dims), 93.59% (256 dims), and 93.65% (128 dims) Macro-F1. Under the Material LOMO protocol, the representations achieve 93.51% (512 dims), 93.49% (256 dims), and 93.51% (128 dims) Macro-F1. These results demonstrate that the specimen-level grouped robustness advantage of v6.2-A is fully preserved even at 128 dimensions, confirming that the representation advantage is morphological rather than a dimensional confound.

### 4.6 Frequency-only and Frequency-fusion Controls

Frequency-only controls show that the current modal-label protocol is strongly frequency-structured. Using `frequency_hz` as the only input feature, the best frequency-only models reach 98.67% stratified Macro-F1, 97.49% Board LOBO Macro-F1, and 94.09% Material LOMO Macro-F1. These values exceed the image-derived frozen embeddings for the current label task.

Frequency + v6.2-A fusion reaches 96.75% stratified Macro-F1, 94.79% Board LOBO Macro-F1, and 94.50% Material LOMO Macro-F1 [Table 7]. Under matched lightweight-head conditions, v6.2-A embedding-only reaches 93.48%, 94.28%, and 94.05% under the same protocols. Frequency + v6.2-A therefore improves over v6.2-A embedding-only by +3.27 pp stratified, +0.50 pp Board LOBO, and +0.45 pp Material LOMO.

Frequency + v6.2-A does not clearly exceed frequency-only overall. It is lower than frequency-only by -1.92 pp in stratified evaluation and -2.70 pp in Board LOBO, and slightly higher in Material LOMO by +0.41 pp. Under Material LOMO, frequency + v6.2-A improves over frequency-only for the minority/difficult classes `1_2` (+6.43 pp) and `2_1` (+6.73 pp). Thus, v6.2-A embeddings can add complementary image-derived information in selected grouped/class conditions, while frequency metadata remains the dominant predictor for the current modal-label task [Table 7].

### 4.7 Targeted Frequency-Overlap Analysis

To test whether the image-derived representation contributes information beyond scalar excitation frequency in a localized ambiguous regime, we isolated the two structurally similar minority modes `1_2` and `2_1`. Across the full binary subset ($N=1{,}507$), frequency + v6.2-A fusion substantially improved over the frequency-only baseline. Under Material LOMO, balanced logistic regression increased Macro-F1 from 85.67% to 96.81% (+11.14 pp), while a random forest head increased Macro-F1 from 85.67% to 94.76% (+9.09 pp). Under Board LOBO, balanced logistic regression increased Macro-F1 from 94.56% to 98.01% (+3.45 pp) and random forest increased Macro-F1 from 92.68% to 96.02% (+3.34 pp).

We further restricted the evaluation to the narrow frequency-overlap interval from 550.00 to 556.00 Hz ($N=43$), where scalar frequency becomes weakly discriminative. In this subset, the frequency-only logistic-regression baseline reached 52.85% Macro-F1, whereas the v6.2-A embedding-only and frequency + v6.2-A fusion conditions reached 97.65% Macro-F1. Because the strict overlap subset contains 43 samples, this result is interpreted as a targeted diagnostic of the localized ambiguity rather than as a standalone population-level estimate. This targeted result supports the interpretation that the v6.2-A embedding captures image-derived spatial morphology that resolves the localized frequency ambiguity between `1_2` and `2_1`.

### 4.8 Class Support and Imbalance

The dataset is strongly imbalanced. The `higher` class contains 10,115 of 12,944 samples, corresponding to 78.14% of the dataset. The remaining classes contain 653 samples for `1_1H`, 669 for `1_1T`, 738 for `1_2`, and 769 for `2_1`.

This imbalance makes class-level diagnostics essential. Macro-F1 reduces majority-class dominance, but per-class analysis is required to determine whether grouped robustness is distributed across modal classes [Table 4].

### 4.9 Per-Class Grouped Stability

v6.2-A improves grouped F1 over v6.1 across all five modal classes [Table 5]. The largest gains occur in the difficult minority modal classes. For `2_1`, grouped F1 increases from 77.88% to 91.32%, a gain of +13.43 percentage points. For `1_2`, grouped F1 increases from 78.35% to 90.57%, a gain of +12.21 percentage points.

The other classes also improve: `1_1H` increases by +3.53 percentage points, `1_1T` by +1.49 percentage points, and `higher` by +0.63 percentage points. The grouped advantage of v6.2-A is therefore visible both in aggregate Macro-F1 and in class-level stability [Table 5].

### 4.10 Zero-Recall Analysis

Zero-recall failures are concentrated in the generic and hierarchical baselines. The generated summary reports 20 zero-recall rows for ImageNet ResNet-18, 18 for Random ResNet-18, and 9 for hierarchical v6.2 phase2 [Table 6]. No zero-recall rows are attributed to v6.1 or v6.2-A in the generated summary.

This pattern shows that grouped failures in the weaker baselines include complete class-level failures in specific grouped conditions. v6.1 and v6.2-A avoid zero-recall rows in the generated diagnostic summary, with v6.2-A additionally providing higher grouped per-class F1 across all classes [Table 6].

### 4.11 Controlled Hierarchical Comparison

Hierarchical v6.2 phase2 provides a controlled physics-aware architecture comparison [Table 3]. The checkpoint is technically valid and produces frozen embeddings, but it is not superior in this representation evaluation. It reaches 37.80% stratified kNN Macro-F1, 24.99% Board LOBO Macro-F1, and 24.47% Material LOMO Macro-F1.

This result should be interpreted at the checkpoint and representation-evaluation level. Architecture-level physics-aware components do not automatically imply robust post-hoc embedding geometry. The resulting decision flow selects v6.2-A as the primary reportable image-derived frozen ESPI encoder baseline, retains v6.1 as the strongest stratified image-derived reference, and keeps hierarchical v6.2 phase2 as a controlled physics-aware alternative rather than a final validated encoder [Table 3].

## 5. Discussion

### 5.1 Grouped Robustness Rather than Overall Predictive Dominance

The central image-derived representation result is the distinction between stratified embedding geometry and grouped board/material robustness. Under fixed `k=10`, v6.1 is slightly stronger in stratified kNN Macro-F1 (93.65%) than v6.2-A (92.78%). Under grouped fixed `k=10` evaluation, v6.2-A is strongest among image-derived encoders, reaching 93.56% Board LOBO Macro-F1 and 93.49% Material LOMO Macro-F1.

The frequency controls define the information budget of the present label protocol. Frequency metadata is a stronger overall predictor than any image-derived frozen representation for this five-class task. This refines the representation claim rather than weakening it: the encoder comparison concerns image-derived representations under matched grouped protocols, where v6.2-A remains the strongest image-derived baseline. Demonstrating image-derived information beyond modal frequency structure will require frequency-controlled labels, frequency-residual prediction, or acoustic-response targets.

Furthermore, frequency + v6.2-A embeddings improve over frequency-only for the structurally confusable minority classes 1_2 and 2_1 under Material LOMO (+6.43 and +6.73 pp, respectively), suggesting that frozen image-derived features encode morphological information that is complementary to scalar excitation frequency, particularly for modes with overlapping frequency signatures.

### 5.2 Why Grouped Evaluation Matters for ESPI Representations

ESPI images preserve full-field interferometric structure, but this structure can include signatures associated with board geometry, material, acquisition condition, and frequency sampling. Stratified validation may overestimate representation robustness if related samples share these signatures across train and validation splits.

Grouped board/material evaluation provides a stricter representation-level test. The board-grouped protocol excludes the held-out board from the reference set; the material-grouped protocol excludes the held-out material. These tests evaluate whether a fixed embedding space remains useful under grouped distribution-shift conditions. The grouped performance of v6.2-A therefore supports its use as a frozen image-derived ESPI encoder baseline for transferable representation studies.

### 5.3 Per-Class Grouped Stability

The class-level diagnostics show that the grouped advantage of v6.2-A is not only an aggregate effect. v6.2-A improves grouped F1 over v6.1 across all five modal classes, with the largest gains in `2_1` (+13.43 percentage points) and `1_2` (+12.21 percentage points).

This is important because the dataset is dominated by the `higher` class, which contributes 10,115 samples, or 78.14% of the dataset. If the grouped gain were driven only by the majority class, it would be less informative as encoder evidence. Instead, the largest gains occur in minority modal classes, indicating improved grouped class stability.

### 5.4 Interpretation of Generic and Spectral Controls

Random and ImageNet ResNet-18 baselines show non-trivial stratified Macro-F1 but collapse under grouped evaluation. This indicates that generic texture features can support stratified separability when train and validation samples share acquisition or grouping structure, but do not provide stable grouped ESPI representations.

Deterministic FFT-based descriptors were evaluated as a supplementary spectral control. Their weak grouped performance indicates that simple handcrafted spatial-frequency summaries do not explain the grouped robustness achieved by v6.2-A. Detailed performance metrics for these supplementary controls are available in Supplementary Table S4.

### 5.5 Interpretation of Hierarchical v6.2 Phase2

Hierarchical v6.2 phase2 is technically valid but not superior in this frozen embedding evaluation. It provides a controlled physics-aware architectural comparison, but its frozen embeddings do not outperform v6.2-A under stratified, board-grouped, or material-grouped evaluation.

This result shows that physics-aware architecture alone does not guarantee robust frozen embedding geometry. A representation-aligned training objective, contrastive fine-tuning, or a different extraction point may be required before the hierarchical branch can serve as a stronger encoder candidate. The result should not be interpreted as a LeFFT failure; learned spectral modules and LeFFT-style architectures remain future work requiring matched ablation.

## 6. Limitations and Future Work

This study evaluates frozen embeddings rather than CNNs retrained under full LOBO/LOMO protocols. The grouped board/material tests change the reference and held-out groups while keeping encoder weights fixed. The results support frozen grouped representation evidence, not retrained CNN grouped generalization.

Frequency metadata dominates the current five-class modal-label protocol. Frequency-only controls exceed the image-derived embeddings for the present label task, so v6.2-A is not the strongest overall predictor when metadata is allowed. The image-derived frozen encoder claim is specifically about representation quality among frozen ESPI image encoders under matched grouped protocols.

Material LOMO uses only two material groups (`carbon` and `wood`) and is interpreted as a descriptive material-held-out stress test.

The study does not evaluate acoustic-response prediction. It does not test whether ESPI embeddings improve prediction of acoustic, phononic, or metamaterial response variables. Such prediction requires separate targets, baselines, and information-budget controls.

The study also does not establish a complete physics-aligned encoder. v6.2-A is identified as the primary reportable image-derived frozen ESPI encoder baseline for grouped representation studies, but further validation would be required before claiming a fully physics-aligned representation.

The hierarchical v6.2 phase2 feature dump is suitable for internal comparison, but it remains internal-only for final tables until regenerated with correct board and split-group metadata stored directly in the source feature dump. This caveat affects final publication readiness for the hierarchical row, not the primary v6.2-A conclusion.

Deterministic spectral / LeFFT-inspired descriptors were evaluated as a supplementary spectral control. Spectral-only descriptors were not competitive, and adding spectral descriptors to frequency + v6.2-A yielded only marginal grouped gains. No LeFFT superiority claim is supported. Learned spectral modules or LeFFT-style architectures remain future work.

Representation-aligned fine-tuning is another natural next step. Supervised contrastive learning, triplet objectives, or domain-aware batch construction could encourage same-mode samples from different boards or materials to occupy nearby regions of the embedding space while preserving class separation.

Future work should explicitly test image-derived information beyond frequency structure. Suitable protocols include frequency-controlled labels, frequency-residual prediction, acoustic-response targets, or tasks where image-derived morphology must explain variance beyond modal frequency metadata. Only after such validation should downstream use in acoustic or phononic systems be claimed.

## 7. Conclusion

This study audits frozen ESPI representation spaces under fixed stratified and grouped protocols. v6.1 remains slightly stronger under fixed `k=10` stratified kNN, indicating strong in-distribution embedding geometry. v6.2-A provides the strongest grouped board/material robustness among the evaluated image-derived frozen encoders.

Class-level diagnostics show that v6.2-A improves grouped F1 across all five modal classes, with the largest gains in the difficult minority classes `2_1` and `1_2`. Frequency-only and frequency-fusion controls show that the current modal-label protocol is strongly frequency-structured; frequency metadata is a dominant predictor, while frequency + v6.2-A provides limited complementary gains over embedding-only and a small Material LOMO gain over frequency-only.

The work establishes an encoder-evidence stage between ESPI classification and future physically grounded representation learning. It supports frozen grouped image-derived representation evidence while leaving acoustic-response prediction, frequency-controlled tasks, learned spectral ablations, and full encoder validation as future work.

## Data, Code, and Reproducibility Availability

The analysis is based on frozen feature dumps, normalized metadata schemas, lightweight evaluation scripts, metadata controls, deterministic spectral descriptor controls, and methodological hardening audits. No deep model retraining, acoustic-response prediction, trained LeFFT model, or LeFFT superiority claim is part of the reported results. The reproducibility package should include the normalized frozen feature schema, encoder evaluation scripts, class-level diagnostic scripts, frequency-control scripts, spectral descriptor control scripts, publication tables and figures, claim checklist, and audit reports. Data and code release details should be finalized according to institutional permissions and the target journal's repository policy.

### Reproducibility Notes

The hierarchical v6.2 phase2 normalized feature dump contains embeddings suitable for internal controlled comparison, but its source feature dump did not directly store reliable board and split-group fields (these were recovered during normalization). This dump should be regenerated with proper metadata before any external release that treats the hierarchical row as fully publication-ready. This caveat does not affect the primary v6.2-A conclusions.

## Tables and Figures

### Tables

#### Table 1: Encoder baseline summary under primary fixed $k=10$ kNN evaluation.
| Encoder | Embedding Dim | Stratified F1 | Board LOBO F1 | Material LOMO F1 |
| :--- | :--- | :---: | :---: | :---: |
| Random ResNet-18 | 512 | 81.14% | 26.17% | 24.59% |
| ImageNet ResNet-18 | 512 | 84.34% | 39.30% | 22.52% |
| v6.1 Reference Baseline | 512 | **93.65%** | 87.10% | 86.33% |
| v6.2-A Frozen Encoder | 1280 | 92.78% | **93.56%** | **93.49%** |
| Hierarchical v6.2 | 512 | 37.80% | 24.99% | 24.47% |

#### Table 2: Grouped robustness summary with 95% bootstrap confidence intervals.
| Encoder | Board LOBO F1 | Board LOBO 95% CI | Material LOMO F1 | Material LOMO 95% CI |
| :--- | :---: | :---: | :---: | :---: |
| Random ResNet-18 | 26.17% | N/A | 24.59% | N/A |
| ImageNet ResNet-18 | 39.30% | [33.16%, 46.36%] | 22.52% | N/A |
| v6.1 Baseline | 87.10% | [82.73%, 91.23%] | 86.33% | [85.32%, 87.34%] |
| v6.2-A Encoder | **93.56%** | [92.36%, 94.67%] | **93.49%** | [93.01%, 93.97%] |
| Hierarchical v6.2 | 24.99% | N/A | 24.47% | [23.23%, 25.71%] |

#### Table 3: Controlled architecture comparison.
| Encoder | Architecture role | Stratified F1 | Board LOBO F1 | Material LOMO F1 | Publication Decision |
| :--- | :--- | :---: | :---: | :---: | :--- |
| v6.1 reference | ESPI baseline | **93.65%** | 87.10% | 86.33% | Retain as reference. |
| v6.2-A baseline | Reportable visual | 92.78% | **93.56%** | **93.49%** | Primary encoder baseline. |
| Hierarchical v6.2 | Physics comparison | 37.80% | 24.99% | 24.47% | Controlled physics alternative. |

#### Table 4: Class support and imbalance.
| Modal Class | Stratified Train | Stratified Val | Total |
| :--- | :---: | :---: | :---: |
| `1_1H` | 522 | 131 | 653 |
| `1_1T` | 535 | 134 | 669 |
| `1_2` | 590 | 148 | 738 |
| `2_1` | 615 | 154 | 769 |
| `higher` | 8092 | 2023 | 10115 |
| **Total** | **10354** | **2590** | **12944** |

#### Table 5: v6.2-A vs v6.1 per-class grouped stability comparison.
| Class | v6.1 F1 | v6.2-A F1 | Difference (pp) | Status |
| :--- | :---: | :---: | :---: | :--- |
| `1_1H` | 88.54% | 92.07% | +3.53 pp | Improved |
| `1_1T` | 89.92% | 91.41% | +1.49 pp | Improved |
| `1_2` | 78.35% | 90.57% | +12.21 pp | Large improvement |
| `2_1` | 77.88% | 91.32% | +13.43 pp | Large improvement |
| `higher` | 98.43% | 99.06% | +0.63 pp | Improved / stable |

#### Table 6: Zero-recall breakdown by encoder.
| Encoder | Zero-Recall Rows | Split Conditions | Diagnosis |
| :--- | :---: | :--- | :--- |
| Random ResNet-18 | 18 | Board/Material Grouped | General visual features collapse. |
| ImageNet ResNet-18 | 20 | Board/Material Grouped | Pretrained weights fail specimen shift. |
| v6.1 reference | 0 | All protocols | Reliable ESPI baseline class recovery. |
| v6.2-A reportable | 0 | All protocols | Robust specimen-invariant recovery. |
| Hierarchical v6.2 | 9 | Board/Material Grouped | Unstable grouped class recovery. |

#### Table 7: Frequency-only and frequency+embedding information-budget comparison.
| Condition | Stratified Macro-F1 | Board LOBO Macro-F1 | Material LOMO Macro-F1 |
| :--- | :---: | :---: | :---: |
| Frequency-only | **98.67%** | **97.49%** | 94.09% |
| v6.2-A Embedding-only | 93.48% | 94.28% | 94.05% |
| Frequency + v6.2-A Fusion | 96.75% | 94.79% | **94.50%** |

### Supplementary Tables

#### Supplementary Table S1: Board-level support.
| Board | `1_1H` | `1_1T` | `1_2` | `2_1` | `higher` | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| C01 | 109 | 108 | 111 | 114 | 1485 | 1927 |
| C02 | 108 | 110 | 110 | 113 | 1494 | 1935 |
| C03 | 110 | 111 | 112 | 112 | 1478 | 1925 |
| W01 | 108 | 112 | 136 | 142 | 1888 | 2386 |
| W02 | 110 | 114 | 134 | 144 | 1892 | 2394 |
| W03 | 108 | 114 | 135 | 144 | 1878 | 2377 |
| **Total** | **653** | **669** | **738** | **769** | **10115** | **12944** |

#### Supplementary Table S2: Material-level support.
| Material | `1_1H` | `1_1T` | `1_2` | `2_1` | `higher` | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| carbon | 327 | 329 | 333 | 339 | 4457 | 5785 |
| wood | 326 | 340 | 405 | 430 | 5658 | 7159 |
| **Total** | **653** | **669** | **738** | **769** | **10115** | **12944** |

#### Supplementary Table S3: Paired board/material deltas (v6.2-A vs v6.1 Macro-F1).
| Held-out Fold | v6.1 F1 | v6.2-A F1 | Delta (pp) | Result |
| :--- | :---: | :---: | :---: | :--- |
| Board C01 | 81.38% | 92.54% | +11.16 pp | v6.2-A win |
| Board C02 | 93.62% | 93.60% | -0.02 pp | Tie / minor delta |
| Board C03 | 88.42% | 94.12% | +5.70 pp | v6.2-A win |
| Board W01 | 91.24% | 91.51% | +0.27 pp | v6.2-A win |
| Board W02 | 83.42% | 94.61% | +11.19 pp | v6.2-A win |
| Board W03 | 84.52% | 95.01% | +10.49 pp | v6.2-A win |
| **Mean Board LOBO** | **87.10%** | **93.56%** | **+6.47 pp** | **v6.2-A win (5/6 folds)** |
| Material Carbon $\rightarrow$ Wood | 85.32% | 93.01% | +7.69 pp | v6.2-A win |
| Material Wood $\rightarrow$ Carbon | 87.34% | 93.97% | +6.63 pp | v6.2-A win |
| **Mean Material LOMO** | **86.33%** | **93.49%** | **+7.16 pp** | **v6.2-A win (2/2 folds)** |

#### Supplementary Table S4: Deterministic spectral descriptor control.
| Condition | Stratified Macro-F1 | Board LOBO Macro-F1 | Material LOMO Macro-F1 |
| :--- | :---: | :---: | :---: |
| Spectral-only descriptors | 52.92% | 22.77% | 19.72% |
| v6.2-A embedding-only | 93.48% | 94.28% | 94.05% |
| v6.2-A + spectral descriptors | -- | 94.35% | 94.00% |

#### Supplementary Table S5: Per-class intra/inter cosine distance ratio.
| Class | v6.1 | v6.2-A | $\Delta$ |
| :--- | :---: | :---: | :---: |
| `1_1H` | 0.2293 | 0.1534 | -0.0760 |
| `1_1T` | 0.2762 | 0.1554 | -0.1209 |
| `1_2` | 0.4326 | 0.1951 | -0.2376 |
| `2_1` | 0.3909 | 0.1557 | -0.2352 |
| `higher` | 0.2858 | 0.1893 | -0.0965 |

#### Supplementary Table S6: Downstream head sensitivity analysis for frozen v6.2-A representation.
| Protocol | Cosine kNN ($k=10$) | Linear Probe (Logistic Regression) | Nearest Class Prototype |
| :--- | :---: | :---: | :---: |
| Board LOBO Mean | 93.56% | 91.40% | 70.97% |
| Material LOMO Mean | 93.49% | 91.17% | 71.35% |

## References

[1] R. Jones and C. Wykes, *Holographic and Speckle Interferometry*, Cambridge University Press, 1989.

[2] P. K. Rastogi, Ed., *Digital Speckle Pattern Interferometry and Related Techniques*, Wiley, 2001.

[3] M. Takeda, H. Ina, and S. Kobayashi, "Fourier-transform method of fringe-pattern analysis for computer-based topography and interferometry," *Journal of the Optical Society of America*, vol. 72, no. 1, pp. 156-160, 1982. https://doi.org/10.1364/JOSA.72.000156

[4] D. C. Ghiglia and M. D. Pritt, *Two-Dimensional Phase Unwrapping: Theory, Algorithms, and Software*, Wiley, 1998.

[5] J. W. Goodman, *Speckle Phenomena in Optics: Theory and Applications*, Roberts and Company, 2007.

[6] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770-778, 2016. https://doi.org/10.1109/CVPR.2016.90

[7] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A Large-Scale Hierarchical Image Database," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 248-255, 2009. https://doi.org/10.1109/CVPR.2009.5206848

[8] I. Gulrajani and D. Lopez-Paz, "In Search of Lost Domain Generalization," in *International Conference on Learning Representations (ICLR)*, 2021.

[9] K. Yan, Y. Yu, C. Huang, L. Sui, K. Qian, and A. Anand, "Fringe pattern denoising based on deep learning," *Optics Communications*, vol. 437, pp. 148-152, 2019. https://doi.org/10.1016/j.optcom.2018.12.058

[10] S. Feng, Q. Chen, G. Gu, T. Tao, L. Zhang, Y. Hu, W. Yin, and C. Zuo, "Fringe pattern analysis using deep learning," *Advanced Photonics*, vol. 1, 025001, 2019. https://doi.org/10.1117/1.AP.1.2.025001

[11] C. Zuo, J. Qian, S. Feng, W. Yin, Y. Li, P. Fan, J. Han, K. Qian, and Q. Chen, "Deep learning in optical metrology: a review," *Light: Science & Applications*, vol. 11, 39, 2022. https://doi.org/10.1038/s41377-022-00714-x

[12] K. Wang, Y. Li, Q. Kemao, J. Di, and J. Zhao, "One-step robust deep learning phase unwrapping," *Optics Express*, vol. 27, pp. 15100-15115, 2019. https://doi.org/10.1364/OE.27.015100

[13] M. Sun, Z. Zhang, and X. Gao, "Deep learning-based fringe pattern analysis for optical metrology," *Measurement Science and Technology*, vol. 32, 112001, 2021. https://doi.org/10.1088/1361-6501/ac0b4e

[14] Y. Bengio, A. Courville, and P. Vincent, "Representation learning: A review and new perspectives," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 35, pp. 1798-1828, 2013. https://doi.org/10.1109/TPAMI.2013.50
