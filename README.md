# Enhanced ESPI Encoder

Private development repository for the encoder-oriented evolution of the Enhanced ESPI v6.2 line.

The repository is intentionally minimalist. It maintains only the code, documentation, and empirical evidence required to construct a publication-grade optics/acoustics package around ESPI representation learning and a future Physics-Aligned ESPI Encoder.

## Scope

- preserve the minimal v6.2-A encoder code path
- document the architectural transition from v6.1 to v6.2-A
- audit learned ESPI representations under grouped OOD splits
- prepare the downstream acoustic-response prediction layer
- keep raw data, model checkpoints, and feature dumps outside version control

## Layout

```text
configs/v6_2/              v6.2 config snapshots
docs/                      methods, reproducibility protocols, data audit notes
examples/                  lightweight example manifests
reports/v61_v62_comparison/ summary comparison metrics
reports/encoder/           encoder audit and grouped OOD evaluation summaries
scripts/encoder/           representation extraction and evaluation scripts
scripts/v6_1/              v6.1 baseline reference scripts
scripts/v6_2/              minimal v6.2 model/training scripts
scripts/v6_2_hierarchical/ v6.2 gatekeeper/expert training branch
src/enhanced_espi_encoder/ modular and reusable package code
```

## Publication Direction

The intended optics-focused manuscript is not framed as a "better classifier" paper. The current working claim is:

```text
The v6.2-A ESPI classifier line is evaluated as a frozen ESPI encoder candidate,
extracting learned full-field interferometric representations that preserve modal
structure under grouped OOD evaluation.
```

The full acoustic-response claim requires future validation through metadata, frequency, geometry, ESPI-only, and fused baseline reference models.

## Current Status

This repository is under private development and is not yet reviewer-ready.

The current publication boundary and manuscript traceability are documented in:

- `docs/CLAIM_BOUNDARIES.md`
- `docs/MANUSCRIPT_MAP.md`
- `docs/REPRODUCE.md`

Documentation and technical reports should be maintained exclusively in English to support peer review.

Mandatory pre-submission protocol:

- enforce strict environment/dependency locking
- integrate artifact hashes or DOI-backed dataset references
- execute baseline comparisons against random initialization, ImageNet priors, v6.1, v6.2-A, and hierarchical v6.2 representations
- generate publication-quality visualizations under `figures/`
- tag a stable, immutable submission release

## Model Lineage

The explicit model-development trajectory is documented in:

- `docs/MODEL_LINEAGE.md`
- `docs/DATA_AND_CODE_DESCRIPTION.md`
- `docs/DEVELOPMENT_STAGES.md`
