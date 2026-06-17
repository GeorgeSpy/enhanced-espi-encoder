# Proposal v32 Map

This map connects the active v32 PhD proposal direction to repository evidence, current status, and future evidence requirements.

## Proposal Structure

| v32 section | Repository evidence | Current status | Future evidence required |
|---|---|---|---|
| WP1 forensic baseline | `docs/WP1_FORENSIC_EVIDENCE_MAP.md`; v62 forensic audit; cross-model audit; visual leakage audit | completed as methodological evidence | final write-up and claim-boundary-safe synthesis |
| Representation semantics | `docs/Z_ESPI_SEMANTICS.md` | defined | promotion evidence for `z_ESPI^ROI` and `z_ESPI^MC` |
| Clean ROI / no-overlay controls | `docs/CLEAN_ROI_REQUIREMENTS.md`; `experiments/cleaned_roi_rerun_v001/` | initial leakage-control supplement complete | full clean ROI dataset if future image-derived claims are pursued |
| `M_ESPI` observation operator | `docs/MESPI_OPERATOR_SPEC.md` | specification drafted | synthetic kernel tests, likelihood tests, differentiability checks |
| Measurement-consistent representation | `docs/MESPI_OPERATOR_EVIDENCE_MAP.md` | planned | encode-field-re-render smoke tests and reference-field validation |
| WP2 response targets | `docs/WP2_RESPONSE_TARGET_SCREENING.md` | planned priority | paired response data and feasibility matrix |
| B0-B6 information budget | `docs/B0_B6_INFORMATION_BUDGET.md` | defined | response-target baselines and B6 residual tests |
| Grouped-OOD validation | `docs/ROADMAP.md` | required future gate | grouped splits by board/material/specimen/acquisition campaign |

## Active Thesis Logic

1. Historical modal-label classification is not sufficient for a strong morphology claim because it is frequency-structured and partly frequency-derived.
2. Historical embeddings remain useful as forensic baselines, not final representation proof.
3. New response targets must be screened before any strong representation claim.
4. `M_ESPI` should constrain future representations to measurement-consistent structure.
5. B6 residual prediction is the decisive validation gate: ESPI information must explain response residuals beyond frequency, metadata, and numerical/FEM descriptors.

## Current Decision

The active v32 path is a measurement-consistent response-learning program, not a continuation of the old OLEN strong-representation submission line.
