# Enhanced ESPI Encoder

This repository is the research and development home for the Enhanced ESPI Encoder project. It now contains three clearly separated layers:

1. a historical locked WP1 OLEN evidence snapshot,
2. archived H2/H3 LeFFT diagnostic development evidence,
3. the active v32 PhD roadmap centered on measurement-consistent ESPI representations and independent response targets.

The repository does not track raw ESPI images, model checkpoints, full feature dumps, private manifests, or large binary artifacts.

## Current Repository Status

| Layer | Purpose | Current status |
|---|---|---|
| Historical OLEN / WP1 evidence | Locked forensic evidence around v6.2-A frozen ESPI representations | retained as historical and diagnostic evidence |
| H2/H3 LeFFT development | Standalone and auxiliary LeFFT diagnostics | archived as development-only evidence |
| Active v32 roadmap | Measurement-consistent ESPI representation learning and response-target validation | active direction |

The release tag `v0.9-olen-pre-submission-evidence` is a historical locked WP1 evidence snapshot. It remains useful for forensic review and reproducibility, but it is no longer the active strong standalone representation-submission path.

The previous planned `v1.0-olen-submitted` milestone is deprecated as the active next milestone. Any future manuscript snapshot must be based on the post-forensic claim boundaries in `docs/CLAIM_BOUNDARIES.md`.

## Current Scientific Direction

The active PhD direction is no longer "show that the current five-class modal-label task proves frequency-independent morphology recognition." The active direction is:

- define a probabilistic differentiable ESPI observation operator, `M_ESPI`;
- distinguish historical, clean-ROI, and measurement-consistent representations:
  - `z_ESPI^hist`: historical v6.2-A frozen embedding, diagnostic only;
  - `z_ESPI^ROI`: clean ROI/no-overlay candidate embedding for response-target tests;
  - `z_ESPI^MC`: future measurement-consistent representation constrained by `M_ESPI`;
- build clean ROI/no-overlay image controls before promoting image-derived representation claims;
- screen independent WP2 response targets with frequency-coordinate, metadata, and numerical/FEM baselines;
- use B0-B6 information-budget controls, with B6 residual response prediction as the decisive validation gate.

`M_ESPI` is the planned core artifact: a probabilistic differentiable ESPI observation/measurement operator based on the time-averaged ESPI `J_0^2` Bessel-kernel observation model.

## Historical OLEN / WP1 Evidence

The locked OLEN evidence remains historically useful because it documents grouped representation audits, frequency controls, dimensionality controls, balanced-subset controls, visual/metadata risks, and forensic limitations.

Key post-forensic interpretation:

- v6.2-A remains the strongest available historical image-derived frozen baseline.
- The current five-class modal-label protocol is strongly frequency-structured.
- Frequency-only LOBO Macro-F1 is approximately `0.932677` in the primary-safe forensic audit.
- v6.2-A does not support a global frequency-independent morphology-recognition claim.
- Clean LOMO/material generalization is not supported because material and board provenance are not cleanly separable.
- Exact-frequency morphology diagnostics are local/underpowered and cannot support global morphology recognition.

Older locked numerical OLEN evidence is retained as historical context, not as current active submission framing.

## Archived H2/H3 LeFFT Diagnostics

H2 and H3 remain development-only diagnostic evidence.

- H2 standalone CE-only LeFFT variants failed the grouped LOBO gate.
- H2.5 remains blocked.
- H3 auxiliary LeFFT Phase-1 training was mechanically stable but transfer-insufficient or negative.
- H3.2 fusion remains blocked.
- H2/H3 do not rescue the current frequency-structured modal-label task.
- H2/H3 do not support claims of LeFFT superiority, validated physics-aware representation, or acoustic-response prediction.

See:

- `docs/H2_TRACK_STATUS.md`
- `docs/H3_TRACK_STATUS.md`
- `docs/LEFFT_DEVELOPMENT_SUMMARY.md`

## v32 Documentation Entry Points

- `docs/CLAIM_BOUNDARIES.md`: post-forensic claim boundaries and historical OLEN semantics.
- `docs/ROADMAP.md`: active v32 M_ESPI / response-target roadmap.
- `docs/DEVELOPMENT_STATUS.md`: current development status and claim-promotion policy.
- `docs/Z_ESPI_SEMANTICS.md`: `z_ESPI^hist`, `z_ESPI^ROI`, and `z_ESPI^MC` semantics.
- `docs/MESPI_OPERATOR_SPEC.md`: planned `M_ESPI` observation-operator specification.
- `docs/CLEAN_ROI_REQUIREMENTS.md`: clean ROI/no-overlay requirements.
- `docs/WP2_RESPONSE_TARGET_SCREENING.md`: WP2 response-target gates.
- `docs/B0_B6_INFORMATION_BUDGET.md`: information-budget baseline ladder.
- `docs/EXPERIMENT_REGISTRY.md`: locked, archived, and future experiment registry.

## Claim Boundary

This repository does not currently claim:

- global frequency-independent ESPI morphology recognition,
- independent topology labels for the current five-class task,
- strong clean material generalization from LOMO,
- validated physics-aware ESPI representation,
- LeFFT superiority,
- H3.2 fusion readiness,
- acoustic-response prediction,
- neural/operator-model validation.

Future image-derived representation claims require clean ROI/no-overlay images, visual leakage controls, grouped validation, frequency/metadata/numerical baselines, residual B6 testing, sanitized reports, reproducible scripts, and explicit claim-boundary updates.

## Repository Guide

- `scripts/encoder/`: feature extraction, normalized-schema utilities, audits, hardening controls, and table generation.
- `docs/`: claim boundaries, roadmap, status, evidence maps, operator specs, and reproducibility documentation.
- `reports/`: small CSV/JSON/Markdown evidence summaries.
- `manuscripts/`: historical OLEN manuscript source and figures.

## Private Artifacts

Large or private artifacts are intentionally excluded from Git:

- raw ESPI image folders,
- model checkpoints,
- full normalized feature dumps,
- `.npz` / `.npy` arrays,
- local logs,
- private manifests.

For review or DOI-backed release, use `docs/templates/ARTIFACT_MANIFEST_TEMPLATE.csv` to record external artifact identifiers, checksums, and release plans.
