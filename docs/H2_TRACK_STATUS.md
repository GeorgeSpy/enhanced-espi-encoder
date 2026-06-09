# H2 Clean LeFFT Standalone Development Track Status

## Purpose

H2 tested clean LeFFT / learned spectral modules as standalone CE-only grouped-transfer classifiers before the later H3 auxiliary-branch work.

H2 is development-only diagnostic evidence. It does not alter the locked OLEN v6.2-A claim boundary and does not support a general claim that LeFFT or physics-informed ESPI encoding failed.

## Relation to v6.2-A and H3

- v6.2-A remains the frozen OLEN anchor/baseline.
- H2 tested standalone clean LeFFT variants, not v6.2-A fusion.
- H2 negative grouped-transfer results motivated H3 auxiliary/anchored LeFFT exploration.
- H2.5 physics-loss extension remains blocked because standalone CE-only robustness did not pass the predefined gate.

## Timeline

| Stage | Outcome |
|---|---|
| H2.0 | Clean LeFFT physics encoder design; Option A learnable 2D bandpass filter bank; independent sigmoid masks; K=4/8/16 sensitivity planned; LOBO Macro-F1 threshold 0.608; design only. |
| H2.1 | LeFFT module skeleton with synthetic smoke tests; K=4/8/16; no real training. |
| H2.2 | Physics/self-supervised loss unit tests; finite losses and gradients on synthetic tensors; no real training. |
| H2.3 | Pair statistics audit; 39,562 metadata rows loaded, 36,434 primary-safe rows, 12,265 high-confidence rows; H2.4 CE-only and SupCon pair mining ready. |
| H2.4B | Reduced CE-only LeFFT go/no-go; best K=8; LOBO Macro-F1 0.248644; worst fold 0.143138; threshold 0.608; FAIL. |
| H2.1B | Architecture redesign; v002 modules, GroupNorm residual backbone, SE/GELU, radial/oriented/hybrid masks, W4 sampler spec; smoke tests passed. |
| H2.4C | 2x2 factorial repair completed; all conditions failed or fail-bound locked; best condition S2_v002_W0 LOBO Macro-F1 0.294294. |

## H2.4C Factorial Table

| Condition | Intervention | Status | LOBO Macro-F1 | Interpretation |
|---|---|---:|---:|---|
| S0_v001_W0_reference | v001 radial LeFFT + W0 | FAIL | 0.248644 | Verified H2.4B reference failure. |
| S1_v001_W4 | v001 radial LeFFT + W4 board-balanced sampler | FAIL_BOUND_LOCKED | not complete; max possible below threshold | Sampler-only effect partially improved matched folds but was insufficient. |
| S2_v002_W0 | v002 hybrid/GroupNorm residual architecture + W0 | FAIL_BOUND_LOCKED | 0.294294 | Best H2.4C condition but far below 0.608 threshold. |
| S3_v002_W4 | v002 architecture + W4 | FAIL_BOUND_LOCKED | 0.196368 | Combined repair did not rescue grouped transfer. |

## Final H2 Decision

- H2 completed as a negative diagnostic development track.
- H2 mechanics/design modules were implemented and smoke-tested.
- Standalone CE-only LeFFT variants failed the grouped LOBO go/no-go threshold.
- H2.5 remains blocked.

## Supported Claims

- Standalone CE-only clean LeFFT variants failed the grouped LOBO threshold.
- Board-balanced sampling alone was insufficient.
- v002 architecture alone was insufficient.
- v002 + W4 was insufficient.
- H2.5 remains blocked.
- Domain-invariant, distillation, anchored-fusion, or stronger representation objectives would be needed before any physics-loss extension.

## Unsupported Claims

- LeFFT generally ineffective.
- Physics-informed ESPI encoding ineffective.
- Acoustic-response prediction failed.
- Physics losses failed on real data.
- SupCon failed.
- v6.2-A fine-tuning failed.
- H2 invalidates H3.
- H2 changes the OLEN paper claim.

## Local Reports

Detailed reports are under `experiments/clean_lefft_encoder_v001/reports/`. Raw images, checkpoints, full feature dumps, and private manifests are not included in this documentation index.
