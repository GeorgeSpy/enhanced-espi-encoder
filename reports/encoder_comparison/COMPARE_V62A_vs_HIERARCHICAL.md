# v6.2-A vs Hierarchical v6.2 Frozen Embedding Comparison

## Scope

This report compares the existing frozen v6.2-A encoder evidence against the hierarchical v6.2 phase2 expert embedding audit. It is a report-only comparison over existing artifacts. It does not implement retraining, fine-tuning, acoustic-response prediction, z_physics_fusion extraction, or LeFFT scripts.

## Comparison table

| Encoder | Architecture role | Checkpoint | Embedding point | Dim | kNN Macro-F1 | Linear Macro-F1 | Prototype Macro-F1 | Board LOBO mean Macro-F1 | Material LOMO mean Macro-F1 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| v6.2-A frozen encoder | Official reportable 5-class baseline and primary frozen ESPI encoder candidate | `artifacts\checkpoints\checkpoint_epoch25_2026...` | `MCDropoutClassifier.global_pool` | 1280 | 93.44% | 92.85% | 86.42% | 93.56% | 93.49% |
| hierarchical v6.2 phase2 expert | Controlled physics-aware architectural candidate branch | `ebb0a1ed05fc480b9e7f575800db1df8e7dffa027b75c...` | `z_expert_prelogit / z_arcface_input from outputs['embeddings']` | 512 | 40.17% | 29.28% | 26.55% | 26.41% | 24.47% |

## Decision

- v6.2-A remains the primary reportable frozen ESPI encoder baseline.
- Hierarchical v6.2 phase2 expert embeddings are technically valid but not superior to v6.2-A under the current frozen embedding audits.
- Hierarchical v6.2 remains a development branch unless retrained or fine-tuned with representation-aligned objectives.
- No validated Physics-Aligned Encoder claim is supported from this hierarchical checkpoint.

## Technical interpretation

- Classification performance is not equivalent to frozen embedding quality. A classifier can make useful predictions while its pre-head representation is weak under kNN, prototype, or grouped retrieval-style evaluation.
- ArcFace and classifier heads can improve supervised decision boundaries without yielding robust nearest-neighbor or prototype geometry in the frozen embedding space.
- The phase2 expert checkpoint is not necessarily a final fine-tuned encoder checkpoint. It is a technically valid hierarchical expert checkpoint that can be audited, not a validated final encoder.
- Physics-aware architecture alone does not guarantee representation robustness. Denoising, phase, Fourier, and multiscale physics modules must still demonstrate grouped representation value under matched evaluation.

## Metadata caveat

The hierarchical audit recovered `board` and `split_group` from `distribution_group` / `path` because the hierarchical NPZ stored these fields as `unknown`. This is acceptable for the current technical audit, but it must be fixed in a future extraction version before final publication tables.

## Claim boundary

- Supported: v6.2-A is the stronger current frozen ESPI encoder candidate under the available representation audits.
- Supported: hierarchical v6.2 phase2 expert embeddings are loadable, extractable, and auditable.
- Not supported: validated Physics-Aligned Encoder.
- Not supported: hierarchical superiority over v6.2-A.
- Not supported: acoustic-response predictive value.
- Not supported: LeFFT / LeFTP superiority.

## Source artifacts

- v6.2-A report: `reports\encoder\FROZEN_ENCODER_TECHNICAL_REPORT.md`
- v6.2-A grouped report: `reports\encoder\GROUPED_ENCODER_EVAL_SUMMARY.md`
- hierarchical key numbers: `reports\hierarchical_encoder\audit_v001\hierarchical_audit_key_numbers.json`
- hierarchical report: `reports\hierarchical_encoder\audit_v001\HIERARCHICAL_ENCODER_AUDIT_SUMMARY.md`

## Warnings

- Requested report not found: reports\encoder\ENCODER_TECHNICAL_REPORT_GR.md
- Using fallback report: reports\encoder\FROZEN_ENCODER_TECHNICAL_REPORT.md
- Hierarchical metadata caveat: board/split_group were recovered from distribution_group/path because the NPZ stored them as unknown.
