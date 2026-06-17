# Development Roadmap

This roadmap reflects the post-forensic v32 direction. The repository now separates historical OLEN/v6.2-A evidence, archived H2/H3 LeFFT diagnostics, and the active measurement-consistent ESPI representation program.

## Stage 0 - Historical OLEN/v6.2-A Forensic Baseline

Status: retained as historical locked WP1 evidence.

Scope:

- v6.1 vs v6.2-A frozen encoder comparison.
- Grouped Board LOBO / Material LOMO evaluation.
- Frequency-only and frequency-fusion controls.
- Dimension-matched PCA audit.
- Balanced subset robustness analysis.
- Quantitative embedding geometry diagnostics.
- v62 frequency-label forensic audit.
- Cross-model forensic audit.

Post-forensic decision:

- v6.2-A remains the strongest available historical image-derived frozen baseline.
- The current modal-label task is frequency-structured and partly frequency-derived.
- Frequency-only LOBO Macro-F1 is approximately `0.932677`.
- Global frequency-independent morphology recognition is unsupported.
- Clean LOMO/material generalization is unsupported.

## Stage 1 - Visual Leakage and Clean ROI/No-overlay Audit

Status: active hardening requirement.

Purpose:

- Determine whether actual model images contain visible `Freq:` text, source-pressure text, borders, axes, or acquisition overlays.
- Verify ROI masks/crops before embedding extraction.
- Build clean ROI/no-overlay image requirements.
- Run patch, border, background, shuffled-label, and cleaned ROI diagnostics.

Decision rule:

- No future image-derived representation claim is promoted without clean ROI/no-overlay controls and leakage-sensitive baselines.

## Stage 2 - `M_ESPI` Operator Specification and Synthetic Validation

Status: active design target.

Purpose:

- Specify `M_ESPI`, a probabilistic differentiable ESPI observation operator.
- Use the time-averaged ESPI `J_0^2` Bessel-kernel observation model.
- Include background `A(x,y)`, visibility `B(x,y)`, sensitivity coefficient `c`, speckle/noise likelihoods, and differentiability requirements.
- Validate with synthetic fields and controlled re-render smoke tests before using real response targets.

## Stage 3 - Measurement-consistent Representation Learning (`z_ESPI^MC`)

Status: future gated work.

Purpose:

- Move beyond historical `z_ESPI^hist` and clean candidate `z_ESPI^ROI`.
- Train or constrain `z_ESPI^MC` through measurement consistency under `M_ESPI`.
- Use encode-field-re-render self-supervision only after operator validation.

Required gates:

- synthetic operator tests pass,
- gradients are stable,
- clean ROI/no-overlay data are available,
- no hidden frequency/metadata leakage enters the representation objective.

## Stage 4 - WP2 Response-target Screening and B0-B6 Baselines

Status: next substantive research gate.

Candidate targets:

- measured `TL(f)`,
- FRF or response curves,
- damping / Q-factor / bandwidth,
- broadband/transient response descriptors,
- radiated-field descriptors,
- residual response after frequency-coordinate regression.

Required baselines:

- B0 frequency-coordinate / mean-response baseline,
- B1 metadata baseline,
- B2 FEM/numerical descriptors,
- B3 hand-crafted ESPI descriptors,
- B4 `z_ESPI` embedding,
- B5 fusion,
- B6 residual prediction.

Decision rule:

- A target is viable only if frequency-coordinate and metadata baselines leave measurable residual headroom under grouped-OOD evaluation.

## Stage 5 - B6 Residual Response Prediction Under Grouped-OOD Evaluation

Status: future decisive validation gate.

Purpose:

- Test whether ESPI information explains response residuals beyond frequency, metadata, and numerical/FEM descriptors.
- Use grouped-OOD splits by board, material, specimen, acquisition campaign, or condition where applicable.
- Report residual improvement, confidence intervals, fold-wise results, and failure modes.

Promotion rule:

- A strong ESPI representation claim requires B6 residual improvement under grouped-OOD evaluation. Raw response prediction alone is not sufficient.

## Stage 6 - Optional Neural / Operator Models After Target Gates

Status: future optional.

Scope:

- Neural operators, differentiable simulators, or response-field models may be explored only after response targets pass screening gates.
- These models must not be claimed as validated unless they beat B0-B6 baselines under grouped-OOD evaluation.

## Archived LeFFT Development Tracks

H2/H3 are archived, not active roadmap drivers.

- H2 standalone CE-only LeFFT track is archived.
- H2.5 remains blocked.
- H3.1 auxiliary LeFFT track is archived.
- H3.2 run remains blocked.
- H3.2 no-harm fusion design-only is optional, but execution is not ready.
- No further simple H2/H3 scaling is recommended.
- Future LeFFT work would require new objectives, clean ROI controls, response-target validation, or new transfer-positive evidence.
