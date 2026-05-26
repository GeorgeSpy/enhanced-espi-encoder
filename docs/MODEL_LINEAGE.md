# Model Lineage

This document records what is currently included in the repository and how the ESPI model line evolves toward the encoder-oriented publication package.

## v6.1 Baseline

Purpose:

- establish a compact baseline for ESPI mode classification
- use a frozen ResNet-18 backbone with a trainable 5-class classifier head
- export the baseline to ONNX for deployment/reference use

Included code:

- `scripts/v6_1/train_v6_1_head_only.py`
- `scripts/v6_1/export_v6_1_to_onnx.py`

Included evidence:

- `reports/v61_v62_comparison/COMPARE_SUMMARY.md`
- `reports/v61_v62_comparison/compare_summary.json`

Not included:

- v6.1 checkpoint files
- ONNX artifacts
- raw data
- full validation dumps

## v6.2 Baseline

Purpose:

- improve the v6.1 line with stronger representation learning and reproducibility controls
- provide the technical base for the later Physics-Aligned Encoder

Included code:

- `scripts/v6_2/espi_v6_2_fixed.py`
- `scripts/v6_2/train_v6_2.py`

Included configs:

- `configs/v6_2/config.antigravity.5class.yaml`
- `configs/v6_2/config.antigravity.5class.quickcheck.yaml`

Included evidence:

- `reports/v61_v62_comparison/COMPARE_SUMMARY.md`

## v6.2 Hierarchical

Purpose:

- explore a two-stage training strategy with gatekeeper and expert components
- separate signal/noise routing from mode expert classification
- provide an alternative v6.2 development branch for later encoder work

Included code:

- `scripts/v6_2_hierarchical/train_v6_2_clean.py`

Key structure:

- Phase 1: gatekeeper training
- Phase 2: expert training
- Phase 3: fine-tuning

Not included:

- hierarchical checkpoints
- full training logs
- generated manifests

## Encoder Transition

Purpose:

- reuse the v6.2 representation path as a Physics-Aligned Encoder
- extract intermediate embeddings from ESPI measurements
- audit embedding quality and grouped generalization

Included code:

- `scripts/encoder/extract_v62_embeddings.py`
- `scripts/encoder/audit_v62_embeddings.py`
- `scripts/encoder/evaluate_encoder_grouped_generalization.py`
- `scripts/encoder/make_encoder_technical_report.py`

Included evidence:

- `reports/encoder/EMBEDDING_AUDIT_SUMMARY.md`
- `reports/encoder/GROUPED_ENCODER_EVAL_SUMMARY.md`
- `reports/encoder/FROZEN_ENCODER_TECHNICAL_REPORT.md`
- `reports/encoder/*key_numbers.json`

## Publication Boundary

This repository currently documents the transition from ESPI classification to ESPI encoder development.

It does not yet prove the final acoustic-response claim. That requires:

- an acoustic-response manifest
- acoustic-only baselines
- geometry/material/frequency baselines
- ESPI-only prediction models
- multimodal fusion models
- ablation tables
- uncertainty and calibration evaluation
