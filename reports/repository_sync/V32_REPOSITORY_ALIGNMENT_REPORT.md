# v32 Repository Alignment Report

## Scope

This documentation update aligns the repository with the current post-forensic v32 PhD direction. It does not train models, run H2/H3, modify raw data, delete locked OLEN evidence, or edit manuscript LaTeX.

## Files Updated

- `README.md`
- `docs/CLAIM_BOUNDARIES.md`
- `docs/DEVELOPMENT_STATUS.md`
- `docs/ROADMAP.md`
- `docs/EXPERIMENT_REGISTRY.md`
- `docs/MANUSCRIPT_MAP.md`

## Files Created

- `docs/PROPOSAL_V32_MAP.md`
- `docs/WP1_FORENSIC_EVIDENCE_MAP.md`
- `docs/MESPI_OPERATOR_EVIDENCE_MAP.md`
- `docs/Z_ESPI_SEMANTICS.md`
- `docs/MESPI_OPERATOR_SPEC.md`
- `docs/CLEAN_ROI_REQUIREMENTS.md`
- `docs/WP2_RESPONSE_TARGET_SCREENING.md`
- `docs/B0_B6_INFORMATION_BUDGET.md`
- `reports/repository_sync/V32_REPOSITORY_ALIGNMENT_REPORT.md`
- `reports/repository_sync/v32_repository_alignment_summary.json`

## Claim Changes

- Deprecated the OLEN/v6.2-A line as the active strong standalone representation-submission path.
- Preserved `v0.9-olen-pre-submission-evidence` as a historical locked WP1 evidence snapshot.
- Clarified that v6.2-A remains the strongest available historical image-derived frozen baseline.
- Marked global frequency-independent ESPI morphology recognition as unsupported.
- Marked independent topology labels for the current five-class task as unsupported.
- Marked strong clean material generalization from LOMO as unsupported.
- Marked LeFFT/H3 rescue of the current modal-label task as unsupported.
- Marked `z_ESPI^hist` as diagnostic only.
- Defined `z_ESPI^ROI` as a clean ROI/no-overlay candidate representation requiring verification.
- Defined `z_ESPI^MC` as the future measurement-consistent representation constrained by `M_ESPI`.
- Defined B6 residual response prediction as the decisive future validation gate.

## Roadmap Changes

The roadmap was restructured into:

1. Stage 0: historical OLEN/v6.2-A forensic baseline.
2. Stage 1: visual leakage and clean ROI/no-overlay audit.
3. Stage 2: `M_ESPI` operator specification and synthetic validation.
4. Stage 3: measurement-consistent representation learning (`z_ESPI^MC`).
5. Stage 4: WP2 response-target screening and B0-B6 baselines.
6. Stage 5: B6 residual response prediction under grouped-OOD evaluation.
7. Stage 6: optional neural/operator models only after accepted targets pass gates.

Archived H2/H3 LeFFT work remains documented but is no longer framed as the main next step.

## Experiment Registry Changes

- Added EXP-031 through EXP-042 for forensic audits, clean ROI controls, `z_ESPI` semantics, `M_ESPI`, WP2 target screening, B0-B6 baselines, and B6 residual prediction.
- Preserved EXP-001 through EXP-023 as historical locked OLEN evidence.
- Preserved H2/H3 archived entries as development-only evidence.
- Reclassified EXP-025 through EXP-030 as superseded, merged, planned-conditional, or planned according to the v32 direction.

## Remaining TODOs

- Finalize `M_ESPI` implementation plan and unit-test scope.
- Create synthetic `J_0^2` kernel validation tests.
- Build WP2 target feasibility matrix with candidate measured response targets.
- Define grouped-OOD splits for WP2 targets before model work.
- Run B0-B6 baselines once paired response data exist.
- Decide whether a compact WP1 leakage/frequency-confound audit note is worth preparing.

## Risks

- Historical OLEN files may still be misread as active strong-representation submission evidence unless readers start from the updated README and claim boundaries.
- Clean ROI/no-overlay evidence remains a requirement for future image-derived representation claims.
- WP2 targets may still be frequency-dominated; target screening must precede acquisition-scale investment.
- `M_ESPI` may be non-identifiable without independent displacement/reference-field validation.

## Recommended Branch

Suggested branch name for this documentation alignment and follow-up work:

`dev/v32-mespi-response-roadmap`
