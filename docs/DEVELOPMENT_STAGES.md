# Development Stages

## Stage A: v6.1 Baseline

Goal:

- establish a compact ESPI mode-classification baseline

Artifacts:

- head-only v6.1 training script
- ONNX export script
- v6.1 vs v6.2 comparison summary

## Stage B: v6.2 Baseline

Goal:

- improve the baseline and make it suitable as an embedding source

Artifacts:

- v6.2 model/training scripts
- 5-class config snapshots
- v6.1 vs v6.2 comparison summary

## Stage C: v6.2 Hierarchical Branch

Goal:

- test gatekeeper/expert decomposition
- separate signal/noise handling from mode expert classification

Artifacts:

- `scripts/v6_2_hierarchical/train_v6_2_clean.py`

## Stage D: Frozen ESPI Encoder Candidate

Goal:

- evaluate the v6.2-A representation path as a frozen ESPI encoder candidate
- extract embeddings
- audit embedding geometry and grouped OOD behavior

Artifacts:

- encoder extraction script
- embedding audit script
- grouped evaluation script
- technical report generator
- summary reports

## Stage E: Acoustic-Response Prediction

Goal:

- test whether ESPI-derived embeddings add value for acoustic-response prediction

Still required:

- acoustic-response manifest
- target extraction utilities
- metadata/frequency/geometry baseline reference models
- ESPI-only model
- multimodal fusion model
- ablation and uncertainty evaluation
