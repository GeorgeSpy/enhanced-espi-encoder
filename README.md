# Enhanced ESPI Encoder

This repository is the ongoing research and development home for the Enhanced ESPI Encoder project. It tracks code, documentation, sanitized reports, manuscript-support artifacts, and reproducibility records for image-derived ESPI representation research.

The repository contains one locked paper-evidence milestone:

- `v0.9-olen-pre-submission-evidence`: frozen evidence snapshot for the OLEN frozen ESPI representation-audit manuscript.

The `main` branch remains active for future encoder, learned spectral / LeFFT, acoustic-response, physics-aware representation, and external-validation work. The OLEN paper is one locked milestone within a broader research-development trajectory.

The repository does not track raw ESPI images, model checkpoints, full feature dumps, or private binary artifacts.

## How to Read This Repository

### 1. Frozen paper evidence layer

Use the release tag `v0.9-olen-pre-submission-evidence` when reproducing or auditing the locked OLEN manuscript evidence package. This layer includes:

- reproducible reports,
- final tables,
- final figures,
- evaluation and audit scripts,
- claim-boundary documentation,
- stable release tag: `v0.9-olen-pre-submission-evidence`.

The locked paper-evidence layer supports grouped frozen ESPI representation auditing, frequency-controlled interpretation, dimensionality controls, balanced-subset controls, targeted frequency-overlap diagnostics, and embedding-geometry diagnostics.

### 2. Ongoing development layer

Use `main` for current development. This layer may contain evolving documentation and future work for:

- learned spectral modules,
- LeFFT ablations,
- acoustic-response prediction,
- physics-aware representations,
- representation-aligned training,
- external validation.

Exploratory development does not automatically create manuscript-supported claims. Manuscript-level claims require matched baselines, grouped validation where applicable, information-budget controls, sanitized reports, reproducible scripts, claim-boundary updates, and release/tag association.

## Repository Status

| Branch / Tag | Purpose | Stability |
|---|---|---|
| `main` | Active development of Enhanced ESPI Encoder | evolving |
| `v0.9-olen-pre-submission-evidence` | Frozen OLEN manuscript evidence package | locked |
| future `v1.0-olen-submitted` | Exact submitted manuscript snapshot | planned |
| future `dev/lefft-ablation` | Learned spectral / LeFFT ablation work | planned |
| future `dev/acoustic-response` | Acoustic-response prediction validation | planned |

## Current Locked Evidence Snapshot

The following values describe the frozen OLEN evidence snapshot and must not be changed without a new tagged release.

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

## Locked Methodological Hardening

- v6.2-A grouped advantage persists under fold-local PCA dimension matching to 512, 256, and 128 components.
- v6.2-A provides complementary morphology information in the targeted `1_2` / `2_1` frequency-overlap analysis.
- v6.2-A grouped advantage persists under class-balanced, class-material-balanced, and class-material-frequency-matched subset controls.
- Quantitative embedding-geometry diagnostics support stronger class structure and cross-group consistency for v6.2-A.
- Deterministic spectral descriptors are included only as supplementary controls and do not support a LeFFT superiority claim.

## Claim Boundary

The central locked OLEN claim is that v6.2-A is the strongest evaluated image-derived frozen ESPI representation under grouped board/material evaluation. The current five-class modal-label task remains strongly frequency-structured, and `frequency_hz` is the dominant metadata-only predictor.

This repository does not claim acoustic-response prediction, a validated Physics-Aligned ESPI Encoder, LeFFT superiority, neural-operator implementation, full retrained CNN LOBO/LOMO generalization, or external-laboratory generalization.

Development branches may explore these directions, but exploratory results remain development evidence until they satisfy the claim-promotion policy in `docs/DEVELOPMENT_STATUS.md` and `docs/CLAIM_BOUNDARIES.md`.

## Repository Guide

- `scripts/encoder/`: feature extraction, normalized-schema utilities, audits, hardening controls, and table generation.
- `docs/ROADMAP.md`: staged development roadmap beyond the locked OLEN milestone.
- `docs/DEVELOPMENT_STATUS.md`: locked evidence, active development areas, and claim-promotion policy.
- `docs/CLAIM_BOUNDARIES.md`: supported, unsupported, and development-only claims.
- `docs/MANUSCRIPT_MAP.md`: final manuscript table/figure/evidence mapping for the locked OLEN package.
- `docs/REPRODUCE.md`: layered reproduction guide using private-artifact placeholders.
- `docs/DATA_AVAILABILITY.md`: data and review-artifact release policy.
- `docs/EXPERIMENT_REGISTRY.md`: locked and planned experiment registry.
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
