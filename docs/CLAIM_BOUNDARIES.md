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

### Scope Delimitation of Evaluated Evidence

| Evaluated Aspect | Measured Finding and Scope |
|---|---|
| Modal-label task structure | The five-class task operationalizes frequency-bound modal intervals rather than frequency-independent topological mode shapes. |
| Topology labeling origin | Nominal labels were generated from frequency intervals and specimen metadata heuristics rather than independent visual annotations. |
| Material holdout evaluation | Held-out material evaluation reflects joint material and board provenance due to the available specimen set size. |
| LeFFT / auxiliary branch transfer | LeFFT architectures (H2/H3) serve as diagnostic baselines characterizing Fourier prior mechanics; transfer improvements under grouped LOBO were not observed. |
| Physics-guided representation | Current models provide image-derived baselines; rigorous physics consistency requires the differentiable Bessel observation operator $M_{\text{ESPI}}$. |
| Response target scope | Current evaluations focus on optical fringe pattern categorization, motivating independent response targets (WP2) for future representation tests. |

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

### Evaluated Findings and Scope

- Standalone CE-only clean LeFFT variants establish a grouped LOBO baseline (best condition S2_v002_W0 reached LOBO Macro-F1 `0.294294` versus the `0.608` target threshold).
- Findings demonstrate that standalone Fourier prior training without anchored spatial features characterizes a lower-bound representation, motivating anchored/hybrid fusion or distillation for any future LeFFT exploration.
- H2 results provide systematic baseline data on the mechanics of unanchored Fourier priors under strict board holdouts.

## Development-only H3 LeFFT Auxiliary Track

### Evaluated Findings and Scope

- Phase-1 auxiliary training confirms that optimizing a joint Fourier branch with scaled physics loss is mechanically stable on real ESPI data.
- Evaluation demonstrates that board-local consistency increased under scaled J1, while cross-board same-class geometry was characterized under grouped evaluation.
- Results define the empirical behavior of auxiliary Fourier losses under grouped evaluation, providing the foundational motivation for the differentiable $M_{\text{ESPI}}$ measurement operator.
