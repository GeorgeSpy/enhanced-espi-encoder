# Development Status

## Historical Locked Evidence

The OLEN manuscript evidence package is frozen at:

`v0.9-olen-pre-submission-evidence`

This release is retained as a historical WP1 forensic evidence snapshot. It supports reproducibility and methodological review of the v6.2-A frozen representation line, but it is no longer the active strong standalone representation-submission path.

The post-forensic interpretation is:

- v6.2-A remains the strongest available historical image-derived frozen baseline.
- The current five-class modal-label task is strongly frequency-structured.
- Frequency-only LOBO Macro-F1 is approximately `0.932677`.
- Global frequency-independent morphology recognition is not supported.
- Strong clean material generalization from LOMO is not supported.
- Future image-derived claims require clean ROI/no-overlay evidence and leakage controls.

## v32 Active Direction

The active development direction is measurement-consistent representation validation via `M_ESPI`, not generic physics-aware encoder naming.

Current focus:

- `M_ESPI`: probabilistic differentiable ESPI observation/measurement operator based on the time-averaged ESPI `J_0^2` Bessel kernel.
- `z_ESPI^hist`: historical v6.2-A frozen embedding, diagnostic only.
- `z_ESPI^ROI`: clean ROI/no-overlay candidate embedding for response-target tests.
- `z_ESPI^MC`: future measurement-consistent representation constrained by `M_ESPI`.
- WP2 independent response-target residual validation under B0-B6 information-budget controls.
- B6 residual response prediction under grouped-OOD evaluation as the decisive validation gate.

## Archived H2/H3 LeFFT Development Tracks

### H2 Clean LeFFT Standalone Track

Status: completed and archived as a negative diagnostic development track.

Outcome:

- Design, module skeleton, synthetic smoke tests, physics-loss tests, and pair audits completed.
- Grouped transfer failed under H2.4B and H2.4C.
- H2.5 remains blocked.
- No further standalone CE-only H2 scaling is recommended.

### H3 LeFFT Auxiliary Track

Status: completed and archived as a diagnostic development track.

Outcome:

- Real-data Phase-1 auxiliary training mechanics passed.
- Safety guardrails passed.
- Representation transfer evidence is insufficient or negative.
- No further simple H3.1 scaling is recommended.
- H3.2 no-harm fusion design-only may be drafted, but H3.2 run remains blocked.

## Current Development Priorities

1. Finalize clean ROI/no-overlay requirements and leakage-control evidence.
2. Specify and validate `M_ESPI` at the operator level.
3. Screen independent WP2 response targets before large acquisition or model work.
4. Run B0-B6 information-budget baselines for any candidate response target.
5. Attempt B6 residual prediction only when frequency-coordinate, metadata, and numerical/FEM baselines leave measurable residual headroom.

## Claim Policy

Development branches may contain exploratory code and negative results. Manuscript-level claims require:

1. matched baselines,
2. grouped validation,
3. information-budget controls,
4. visual leakage and clean ROI controls for image-derived claims,
5. sanitized reports,
6. reproducible scripts,
7. claim-boundary update.

A result should also be associated with a release tag before it is treated as a stable manuscript evidence snapshot.

## Branching Convention

Recommended branch name for the current roadmap:

- `dev/v32-mespi-response-roadmap`

Archived or conditional branches:

- `dev/lefft-ablation`: archived/conditional only.
- `dev/acoustic-response`: superseded by WP2 response-target residual validation.
- `dev/physics-aware-validation`: superseded by `M_ESPI` measurement-consistent validation.
- `dev/external-validation`: future grouped-OOD validation if data become available.

The `main` branch remains evolving development history. The tag `v0.9-olen-pre-submission-evidence` remains the locked historical OLEN evidence snapshot.
