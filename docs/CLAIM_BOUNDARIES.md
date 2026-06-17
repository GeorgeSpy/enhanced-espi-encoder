# Claim Boundaries

This document defines supported, historical/diagnostic, unsupported, and future-gated claims for the Enhanced ESPI Encoder repository after the v62 forensic audits, cross-model audits, visual leakage audit, and H2/H3 consolidation.

The repository contains a historical locked OLEN evidence release and active v32 PhD development. The locked manuscript snapshot remains tagged as `v0.9-olen-pre-submission-evidence`, but it is not the current active strong-representation submission framing.

## Post-forensic v32 Claim Boundaries

### Supported

| Claim | Status | Scope |
|---|---|---|
| v6.2-A is the strongest available historical image-derived frozen ESPI baseline in this repository. | supported | historical WP1 diagnostic baseline only |
| The current five-class modal-label protocol is strongly frequency-structured. | supported | frequency-only LOBO Macro-F1 approximately `0.932677` in the primary-safe forensic audit |
| H2/H3 LeFFT development results are archived diagnostic evidence and do not change the OLEN decision. | supported | H2/H3 consolidation reports |
| Clean ROI/no-overlay controls are required before promoting future image-derived morphology claims. | supported | visual/domain leakage audit and cleaned ROI rerun |
| The active PhD direction is measurement-consistent representation validation via `M_ESPI` and independent response targets. | supported as roadmap | future evidence still required |

### Historical / Diagnostic Only

| Claim or artifact | Classification | Allowed use |
|---|---|---|
| `z_ESPI^hist` | historical/diagnostic only | v6.2-A frozen embedding for forensic baselines and historical comparison |
| Targeted exact-frequency morphology diagnostics | historical/local/underpowered only | local diagnostic, not global morphology recognition |
| Locked OLEN v6.2-A grouped tables | historical evidence semantics | audit trail and WP1 methodological motivation |
| H2 standalone LeFFT results | development-only diagnostic | negative grouped-transfer development evidence |
| H3 auxiliary LeFFT results | development-only diagnostic | mechanics/safety evidence; insufficient or negative transfer evidence |

### Unsupported

| Claim | Status |
|---|---|
| Global frequency-independent ESPI morphology recognition from the current five-class labels | unsupported |
| Independent topology labels for the current five-class task | unsupported |
| Strong material generalization from LOMO | unsupported because material/board provenance is not cleanly separable |
| LeFFT/H3 rescue of the current modal-label task | unsupported |
| H3.2 fusion ready to run | unsupported |
| Validated physics-aware ESPI representation | unsupported |
| Acoustic-response prediction | unsupported |
| Clean image-derived morphology signal from historical overlaid images | unsupported |

### Future / Gated

| Future claim | Required gate |
|---|---|
| `z_ESPI^ROI` supports image-derived information | clean ROI/no-overlay dataset, visual leakage controls, grouped validation, and frequency/metadata baselines |
| `z_ESPI^MC` supports measurement-consistent ESPI representation learning | `M_ESPI` operator specification, synthetic validation, differentiability tests, and physical/reference-field validation |
| `M_ESPI` is a useful observation operator | Bessel `J_0^2` kernel tests, probabilistic likelihood tests, gradient checks, and comparison against SLDV/FEM/reference fields |
| ESPI adds value for independent response targets | B0-B6 response-target baselines with B6 residual improvement under grouped-OOD evaluation |
| Acoustic or response prediction | paired response measurements, target feasibility screening, frequency-coordinate baseline failure or residual headroom, and grouped-OOD validation |

## Historical Locked OLEN Evidence Semantics, Not Current Active Submission Framing

The table below preserves the older locked OLEN evidence semantics for historical reproducibility. It must not be read as the current v32 active manuscript framing.

| Claim | Historical status | Current post-forensic interpretation |
|---|---|---|
| v6.2-A is the strongest evaluated image-derived frozen ESPI representation under grouped board/material evaluation. | supported in locked snapshot | retained as historical baseline language, not global morphology proof |
| `frequency_hz` is the dominant metadata-only predictor for the present five-class modal-label task. | supported | strengthened by forensic audit; central limitation |
| v6.2-A adds complementary morphology information in localized frequency-ambiguous cases. | targeted diagnostic | local/underpowered only; not a global claim |
| v6.2-A grouped advantage persists under fold-local PCA dimension matching. | historical diagnostic | useful audit record; not sufficient for beyond-frequency morphology |
| v6.2-A grouped advantage persists under balanced class/material/frequency subset controls. | historical diagnostic | constrained by label provenance, frequency dominance, and visual/domain leakage risks |
| Quantitative embedding geometry supports stronger class structure for v6.2-A than v6.1. | historical diagnostic | domain/frequency organization must be reported; not sufficient for global morphology |
| Hierarchical v6.2 phase2 is technically valid but not superior as a frozen representation source. | supported | remains historical model-family audit |
| Deterministic FFT/spectral descriptors do not support a LeFFT superiority claim. | supplementary control | remains negative-control evidence |

## Development Claims vs Manuscript Claims

A result is not manuscript-supported unless it has:

1. a reproducible script,
2. a sanitized report,
3. matched baseline comparison,
4. grouped validation where applicable,
5. frequency/metadata/numerical information-budget controls where applicable,
6. leakage and ROI controls where image pixels are used,
7. claim-boundary entry,
8. release/tag association.

Development branches may contain exploratory experiments, negative results, prototype reports, or partial validations. These must not be promoted to manuscript-level claims until the promotion requirements above are satisfied.

## Development-only H2 Clean LeFFT Track

### Supported

- Standalone CE-only clean LeFFT variants failed the grouped LOBO threshold.
- H2.4C best condition S2_v002_W0 reached LOBO Macro-F1 `0.294294` versus the `0.608` threshold.
- H2.5 remains blocked.
- H2 results motivate redesign, distillation, anchored/fusion design, or stronger domain-invariant objectives if future work explicitly reopens LeFFT.

### Unsupported

- LeFFT generally failed.
- Physics-informed ESPI encoding failed.
- Acoustic-response prediction failed.
- Physics losses failed on real data.
- SupCon failed.
- v6.2-A fine-tuning failed.
- H2 supports any OLEN core claim.

## Development-only H3 LeFFT Auxiliary Track

### Supported

- Stable LeFFT auxiliary Phase-1 training is feasible on real ESPI data.
- H3 guardrails and audit infrastructure worked.
- H3.1K showed board-local consistency increase under scaled J1.
- H3.1K degraded cross-board same-class geometry.
- H3.2 run remains blocked.

### Unsupported

- LeFFT improves v6.2-A.
- H3.1 objectives improve grouped transfer.
- Board invariance was achieved.
- Physics-informed representation was validated.
- Acoustic-response prediction was tested.
- H3.2 fusion is ready.
