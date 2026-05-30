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

## Stage 1 — Learned Spectral / LeFFT Ablation

Goal: test learned spectral modules under matched representation-audit protocols.

Planned work:

- Implement trained LeFFT / spectral module under matched protocols.
- Compare frequency-only, v6.2-A, LeFFT, frequency+LeFFT, and v6.2-A+LeFFT conditions.
- Test grouped Board LOBO and Material LOMO.
- Avoid LeFFT superiority claims until matched ablations support them.

## Stage 2 — Acoustic-Response Prediction

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
