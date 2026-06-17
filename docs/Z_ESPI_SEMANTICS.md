# z_ESPI Semantics

This file defines the allowed meanings and promotion gates for ESPI representation symbols used in the v32 roadmap.

## `z_ESPI^hist`

Definition: historical v6.2-A frozen embedding produced by the locked OLEN/WP1 pipeline.

Allowed uses:

- historical baseline,
- forensic audit reference,
- cross-model comparison,
- diagnostic geometry/domain analysis.

Forbidden claims:

- global frequency-independent morphology recognition,
- clean image-derived morphology signal,
- validated physics-aware representation,
- response-target prediction,
- superiority over frequency-only baselines.

Promotion status: not promotable to current active representation claim without clean ROI/no-overlay re-extraction and new response-target validation.

## `z_ESPI^ROI`

Definition: candidate embedding extracted from clean ROI/no-overlay ESPI images.

Allowed uses:

- leakage-controlled image representation testing,
- exact-frequency local diagnostics,
- response-target screening if paired targets exist,
- B0-B6 baseline comparisons.

Forbidden claims before validation:

- global morphology recognition,
- physics-aware representation,
- acoustic/response prediction,
- grouped-OOD generalization.

Required validation for promotion:

1. deterministic clean ROI/no-overlay images,
2. visual patch and border leakage controls,
3. shuffled-label controls,
4. grouped-OOD validation,
5. frequency/metadata/numerical baselines,
6. response-target residual testing where applicable.

## `z_ESPI^MC`

Definition: future measurement-consistent representation constrained by `M_ESPI`.

Allowed uses after construction:

- encode-field-re-render self-supervision,
- measurement-consistent representation learning,
- response-target residual prediction,
- physical consistency diagnostics.

Forbidden claims before validation:

- validated physics-aware encoder,
- general acoustic-response prediction,
- external-laboratory generalization,
- morphology superiority over baselines.

Required validation for promotion:

1. `M_ESPI` operator specification,
2. `J_0^2` kernel tests,
3. probabilistic likelihood tests,
4. differentiability/gradient tests,
5. synthetic re-render smoke tests,
6. validation against SLDV/FEM/reference fields,
7. B0-B6 residual response-target evaluation under grouped-OOD splits.
