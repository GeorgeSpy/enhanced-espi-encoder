# Development Status

## Locked Evidence

The OLEN manuscript evidence package is frozen at:

`v0.9-olen-pre-submission-evidence`

The locked release supports:

- grouped frozen ESPI representation auditing,
- frequency-controlled interpretation,
- dimension-matched PCA controls,
- balanced subset controls,
- targeted frequency-overlap diagnostics.

The locked release is the correct reference point for reproducing or reviewing the OLEN frozen representation-audit manuscript.

## Active Development

The following areas remain active on `main` or planned development branches:

- learned spectral / LeFFT ablations,
- acoustic-response prediction,
- physics-aware encoder validation,
- representation-aligned training,
- external validation.

Active development may include exploratory code, prototype reports, negative results, and partial audits. These artifacts do not automatically become manuscript-supported evidence.

## Claim Policy

Development branches may contain exploratory code and negative results. Manuscript-level claims require:

1. matched baselines,
2. grouped validation,
3. information-budget controls,
4. sanitized reports,
5. reproducible scripts,
6. claim-boundary update.

A result should also be associated with a release tag before it is treated as a stable manuscript evidence snapshot.

## Branching Convention

Recommended branch names for future work:

- `dev/lefft-ablation`
- `dev/acoustic-response`
- `dev/physics-aware-validation`
- `dev/external-validation`

The `main` branch remains evolving development history. The existing tag `v0.9-olen-pre-submission-evidence` remains the locked OLEN evidence snapshot.

<!-- BEGIN H2_H3_ARCHIVED_DEVELOPMENT_STATUS -->
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
- No further simple scaling is recommended.
- H3.2 no-harm fusion design-only is optional; H3.2 run remains blocked.
- Acoustic-response target acquisition remains a future research direction.
<!-- END H2_H3_ARCHIVED_DEVELOPMENT_STATUS -->
