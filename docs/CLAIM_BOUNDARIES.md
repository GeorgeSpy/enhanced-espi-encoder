# Claim Boundaries

This document defines what the current repository can and cannot support in a manuscript or reviewer package.

## Current Evidence Level

The repository is an internal, curated development package for the first ESPI encoder paper. It contains code, selected reports, and traceability notes. It does not contain raw ESPI image data, acoustic-response data, model checkpoints, or full feature dumps.

## Claims Supported Now

| Claim | Evidence in repository | Status |
|---|---|---|
| v6.1 is the historical baseline for the 5-class ESPI mode-classification line. | `scripts/v6_1/`, `reports/v61_v62_comparison/` | supported as lineage evidence |
| v6.2-A is the reportable 5-class ESPI classifier baseline. | `scripts/v6_2/`, `configs/v6_2/`, comparison report | supported as baseline evidence |
| The frozen v6.2-A representation space has strong post-hoc class geometry. | `reports/encoder/EMBEDDING_AUDIT_SUMMARY.md`, key-number JSON | supported as frozen-embedding evidence |
| Grouped frozen-embedding evaluation supports v6.2-A as an ESPI encoder candidate. | `reports/encoder/GROUPED_ENCODER_EVAL_SUMMARY.md`, grouped CSV summaries | supported as candidate evidence |
| The hierarchical v6.2 branch is a valid development branch to compare against flat v6.2. | `scripts/v6_2_hierarchical/train_v6_2_clean.py`, `docs/MODEL_LINEAGE.md` | code-level support only |

## Claims Not Yet Supported

| Claim | Missing requirement | Status |
|---|---|---|
| Fully validated Physics-Aligned Encoder. | external baseline reference models, ablations, frozen/unfrozen comparison, statistical repeats | not yet |
| Acoustic-response predictive value. | acoustic-response manifest, acoustic-only baselines, ESPI-only/fusion models | not yet |
| ESPI embeddings improve over geometry/frequency/material metadata baseline reference models. | metadata-only and acoustic-only baseline experiments | not yet |
| True CNN LOBO/LOMO training generalization. | retrained CNN experiments under leave-one-board/material-out splits | not yet |
| Hierarchical v6.2 is superior to flat v6.2 as an encoder. | matched hierarchical embedding extraction and grouped evaluation | not yet |
| Neural-operator / PNO implementation. | operator architecture, training code, acoustic-response targets | not included |

## Safe Manuscript Framing

Use this framing for the current paper draft:

```text
The v6.2-A ESPI classifier line is evaluated as a frozen ESPI encoder candidate.
Post-hoc embedding audits and grouped OOD evaluations show that learned
full-field interferometric representations preserve modal structure under
board/material grouping, motivating the next acoustic-response prediction stage.
```

Avoid this framing until additional experiments are complete:

```text
The model is a validated Physics-Aligned Encoder that predicts acoustic response.
```

## Reviewer Risk Controls

- State that current evidence is frozen-embedding evidence, not full acoustic prediction.
- Separate classifier performance from encoder usefulness.
- Report grouped evaluations separately from true retrained CNN generalization.
- Mark hierarchical v6.2 as a development branch until matched encoder metrics exist.
- Keep raw-data and checkpoint availability in `docs/DATA_AVAILABILITY.md`.
