# Enhanced ESPI Encoder Evidence Package

This repository is maintained as a pre-submission evidence package for a frozen ESPI representation-audit manuscript targeting optical metrology / applied optics venues.

The repository does not track raw ESPI images, model checkpoints, full feature dumps, or private binary artifacts. It tracks source code, reproducibility documentation, manuscript evidence, final tables, small reports, figures, and claim-boundary documentation.

## Current Evidence Snapshot

- Dataset: 12,944 ESPI samples.
- Boards: C01, C02, C03, W01, W02, W03.
- Materials: carbon and wood.
- Task: five-class modal-label protocol.
- Coverage: all boards and both materials contain all five modal classes.

| Condition | Stratified Macro-F1 | Board LOBO Macro-F1 | Material LOMO Macro-F1 |
|---|---:|---:|---:|
| v6.1, fixed k=10 | 93.65% | 87.10% | 86.33% |
| v6.2-A, fixed k=10 | 92.78% | 93.56% | 93.49% |
| frequency-only | 98.67% | 97.49% | 94.09% |
| frequency + v6.2-A | 96.75% | 94.79% | 94.50% |

## Methodological Hardening

- v6.2-A grouped advantage persists under fold-local PCA dimension matching to 512, 256, and 128 components.
- v6.2-A provides complementary morphology information in the targeted `1_2` / `2_1` frequency-overlap analysis.
- v6.2-A grouped advantage persists under class-balanced, class-material-balanced, and class-material-frequency-matched subset controls.
- Quantitative embedding-geometry diagnostics support stronger class structure and cross-group consistency for v6.2-A.
- Deterministic spectral descriptors are included only as supplementary controls and do not support a LeFFT superiority claim.

## Claim Boundary

The central supported claim is that v6.2-A is the strongest evaluated image-derived frozen ESPI representation under grouped board/material evaluation. The current five-class modal-label task remains strongly frequency-structured, and `frequency_hz` is the dominant metadata-only predictor.

This repository does not claim acoustic-response prediction, a validated Physics-Aligned ESPI Encoder, LeFFT superiority, neural-operator implementation, full retrained CNN LOBO/LOMO generalization, or external-laboratory generalization.

## Repository Guide

- `scripts/encoder/`: feature extraction, normalized-schema utilities, audits, hardening controls, and table generation.
- `docs/CLAIM_BOUNDARIES.md`: supported and unsupported claims.
- `docs/MANUSCRIPT_MAP.md`: final manuscript table/figure/evidence mapping.
- `docs/REPRODUCE.md`: layered reproduction guide using private-artifact placeholders.
- `docs/DATA_AVAILABILITY.md`: data and review-artifact release policy.
- `docs/EXPERIMENT_REGISTRY.md`: numbered experiment registry.
- `docs/EVIDENCE_LEDGER.md`: manuscript-claim to evidence-artifact map.
- `reports/`: small CSV/JSON/Markdown evidence summaries.
- `manuscripts/`: final OLEN submission source and figures.

## Private Artifacts

Large or private artifacts are intentionally excluded from Git:

- raw ESPI image folders,
- model checkpoints,
- full normalized feature dumps,
- `.npz` / `.npy` arrays,
- local logs,
- private manifests.

For review or DOI-backed release, use `docs/templates/ARTIFACT_MANIFEST_TEMPLATE.csv` to record external artifact identifiers, checksums, and release plans.
