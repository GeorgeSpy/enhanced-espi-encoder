# Development Roadmap

This roadmap describes the broader Enhanced ESPI Encoder development trajectory. It separates the completed OLEN frozen representation audit from future learned spectral, acoustic-response, physics-aware, and external-validation work.

## Stage 0 — Completed: OLEN Frozen Representation Audit

Status: completed and locked at `v0.9-olen-pre-submission-evidence`.

Completed evidence blocks:

- v6.1 vs v6.2-A frozen encoder comparison.
- Grouped Board LOBO / Material LOMO evaluation.
- Frequency-only and frequency-fusion controls.
- Dimension-matched PCA audit.
- Targeted `1_2` / `2_1` frequency-overlap analysis.
- Balanced subset robustness analysis.
- Quantitative embedding geometry diagnostics.

Stage 0 supports the locked OLEN claim that v6.2-A is the strongest evaluated image-derived frozen ESPI representation under grouped board/material evaluation, while `frequency_hz` is the dominant metadata-only predictor for the current five-class modal-label task.

## Stage 1 — Learned Spectral / LeFFT Diagnostics

Status: archived development evidence.

Outcome:

- H2 standalone CE-only LeFFT v001/v002 did not pass the grouped Board LOBO threshold.
- H3 auxiliary LeFFT trained safely but did not produce transfer-positive representation evidence.
- No LeFFT superiority claim is supported.
- No H3.2 fusion run is justified at this stage.
- Further simple scaling is not recommended.

## Stage 1B — Conditional Representation-Aligned Rescue

Status: optional / planned-conditional.

Possible future work:

- Teacher-guided distillation from frozen v6.2-A.
- Stronger backbone or representation head redesign.
- Representation-aligned training under explicit grouped go/no-go criteria.
- H3.2 no-harm fusion design-only, not execution, unless transfer-positive evidence and no-harm protocol exist.

## Stage 2 — Acoustic-Response Prediction

Status: next substantive research direction.

Goal: test whether ESPI representations predict acoustic-response targets beyond metadata-only controls.

Planned work:

- Define acoustic-response targets.
- Build metadata-only, frequency-only, geometry-only, ESPI-only, and fusion baselines.
- Test whether ESPI embeddings add predictive value beyond scalar frequency and geometry.
- Preserve explicit separation between modal-label classification and acoustic-response prediction.

## Stage 3 — Physics-Aware Encoder Validation

Goal: validate physics-aware representation learning rather than only architecture naming.

Planned work:

- Representation-aligned training objectives.
- Contrastive / metric-learning objectives.
- Phase/fringe morphology diagnostics.
- Material/specimen/domain robustness.
- Matched comparisons against frozen encoder, frequency-only, and spectral controls.

## Stage 4 — External Validation

Goal: test robustness beyond the current laboratory evidence set.

Planned work:

- Additional instruments, boards, materials, or acquisition campaigns.
- External-laboratory or cross-setup evaluation if data become available.
- Cross-setup metadata audits and data-sharing policy updates.
- Release-tagged evidence snapshots for any promoted external-validation claims.

<!-- BEGIN H2_H3_ARCHIVED_LEFFT_ROADMAP -->
## Archived LeFFT Development Tracks

- H2 standalone CE-only LeFFT track is archived.
- H2.5 remains blocked.
- H3.1 auxiliary LeFFT track is archived.
- H3.2 run remains blocked.
- H3.2 no-harm fusion design-only is optional.
- No further simple H2/H3 scaling is recommended.
- Future LeFFT work requires stronger objectives, anchored/fusion design, downstream acoustic validation, or new transfer-positive evidence.
- Next high-value direction: acoustic-response target acquisition and downstream validation.
- OLEN submission remains focused on H0/H1/v6.2-A locked evidence.
<!-- END H2_H3_ARCHIVED_LEFFT_ROADMAP -->
