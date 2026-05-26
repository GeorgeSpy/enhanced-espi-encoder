# Enhanced ESPI Encoder

Private development repository for the encoder-oriented evolution of the Enhanced ESPI v6.2 line.

The repository is intentionally small. It keeps only the code and evidence needed to develop a publication-grade optics/acoustics package around a Physics-Aligned ESPI Encoder.

## Scope

- preserve the minimal v6.2 encoder code path
- document the v6.1 to v6.2 improvement
- audit ESPI embeddings under grouped/domain splits
- prepare the future acoustic-response prediction layer
- keep raw data, checkpoints, and feature dumps outside Git

## Layout

```text
configs/v6_2/              v6.2 config snapshots
docs/                      methods, reproducibility, data notes
examples/                  lightweight example manifests
reports/v61_v62_comparison/ summary comparison only
reports/encoder/           encoder audit and grouped-generalization summaries
scripts/encoder/           embedding extraction and audit scripts
scripts/v6_1/              v6.1 baseline scripts
scripts/v6_2/              minimal v6.2 model/training scripts
scripts/v6_2_hierarchical/ v6.2 gatekeeper/expert training branch
src/enhanced_espi_encoder/ future reusable package code
```

## Publication Direction

The intended optics-paper claim is not "a better classifier". The working claim is:

```text
The v6.2 ESPI classifier line can be repurposed as a Physics-Aligned Encoder
for response-relevant full-field interferometric representations under domain shift.
```

The full acoustic-response claim will require future metadata/frequency/geometry baselines and ablation against ESPI-only and fused models.

## Current Status

This is a private development repo. It is not yet public/reviewer-ready.

The current publication boundary and manuscript traceability are documented in:

- `docs/CLAIM_BOUNDARIES.md`
- `docs/MANUSCRIPT_MAP.md`
- `docs/REPRODUCE.md`

Before submission:

- add exact environment/dependency lock
- add artifact hashes or DOI-backed artifact references
- add baseline encoder comparisons against random, ImageNet, v6.1, v6.2-A, and hierarchical v6.2 embeddings
- add publication-quality figures under `figures/`
- tag a stable submission release

## Model Lineage

The included model-development line is documented in:

- `docs/MODEL_LINEAGE.md`
- `docs/DATA_AND_CODE_DESCRIPTION.md`
- `docs/DEVELOPMENT_STAGES.md`
