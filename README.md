# Enhanced ESPI Encoder

This repository presents a reproducible evaluation of image-derived ESPI representations for modal-label classification. It develops the v6.2-A frozen encoder as an image-derived baseline and defines the next stage of the project: measurement-consistent representation learning.

## What this repository shows

The central empirical result is that **v6.2-A is the strongest image-derived frozen ESPI encoder baseline evaluated in this repository**. Under grouped board and material evaluation, it outperforms both a randomly initialised ResNet-18 and an ImageNet-pretrained ResNet-18 on the five-class modal-label task. The comparison therefore supports the value of task-specific ESPI image representations relative to generic image encoders.

The study also shows that **excitation frequency is the dominant predictor** in the current five-class protocol. A frequency-only baseline reaches approximately 0.9327 Macro-F1 under LOBO evaluation. This result clarifies the structure of the task: the labels largely encode frequency-bound modal intervals, so performance on this task cannot by itself be interpreted as evidence of frequency-independent recognition of mode morphology.

Within this setting, v6.2-A provides a useful image-derived baseline and shows complementary behaviour in targeted frequency-ambiguous comparisons. These comparisons are local and limited by the available data, so they motivate further controlled evaluation rather than support a global claim of morphology recognition.

The LeFFT experiments provide a systematic negative control. Standalone and auxiliary Fourier-prior variants were mechanically stable, but did not produce robust transfer gains under grouped evaluation. They therefore establish the empirical limits of the tested Fourier-prior designs and motivate the move toward a measurement-consistent formulation.

The active direction is to model ESPI observation directly through a probabilistic differentiable operator, \(M_{\mathrm{ESPI}}\), based on the time-averaged \(J_0^2\) Bessel-kernel model. The project will use this operator to test whether ESPI-derived representations add information for independent response targets beyond frequency, metadata, numerical and FEM baselines.

## Repository structure

- `src/enhanced_espi_encoder/`: frozen encoder wrapper and model-loading utilities.
- `scripts/encoder/`: feature extraction, audits, hardening controls, and report generation.
- `configs/`: experiment and evaluation configurations.
- `docs/`: claim boundaries, roadmap, operator specification, development status, and reproducibility documentation.
- `reports/`: small CSV, JSON, and Markdown evidence summaries.
- `manuscripts/`: historical OLEN manuscript source and figures.
- `experiments/` and `examples/`: experiment and usage material.

## Private artifacts

Raw ESPI image folders, model checkpoints, full feature dumps, private manifests, and large binary artifacts are intentionally excluded from Git. For review or DOI-backed release, `docs/templates/ARTIFACT_MANIFEST_TEMPLATE.csv` records external artifact identifiers, checksums, and release plans.
