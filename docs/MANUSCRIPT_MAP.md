# Manuscript Map

This map links manuscript claims, tables, and figures to repository artifacts. Items marked `pending` should be completed before reviewer/public release.

## Working Title

```text
Physics-Aligned ESPI Encoder Evidence from Full-Field Interferometric Measurements
under Domain and Material Shift
```

## Tables

| Manuscript item | Source script | Config | Input artifact | Output/report | Status |
|---|---|---|---|---|---|
| Table 1: v6.1 vs v6.2 classifier baseline | `scripts/v6_1/train_v6_1_head_only.py`, `scripts/v6_2/train_v6_2.py` | `configs/v6_2/config.antigravity.5class.yaml` | external manifest/checkpoints | `reports/v61_v62_comparison/COMPARE_SUMMARY.md` | done-summary |
| Table 2: frozen v6.2-A embedding audit | `scripts/encoder/audit_v62_embeddings.py` | not required after feature extraction | external `features_v62a_epoch25.npz` | `reports/encoder/EMBEDDING_AUDIT_SUMMARY.md` | done-summary |
| Table 3: grouped frozen-embedding evaluation | `scripts/encoder/evaluate_encoder_grouped_generalization.py` | not required after feature extraction | external `features_v62a_epoch25.npz` | `reports/encoder/GROUPED_ENCODER_EVAL_SUMMARY.md` | done-summary |
| Table 4: encoder baseline comparison | pending script | pending config | random/ImageNet/v6.1/v6.2/hierarchical embeddings | pending report | pending |
| Table 5: acoustic-response prediction | pending script | pending config | acoustic-response manifest | pending report | future |

## Figures

| Manuscript item | Source script | Input artifact | Output/report | Status |
|---|---|---|---|---|
| Fig. 1: ESPI-to-encoder pipeline | manual diagram or pending plotting script | model lineage docs | `figures/pipeline.svg` | pending |
| Fig. 2: v6.1 to v6.2 lineage | manual diagram | `docs/MODEL_LINEAGE.md` | `figures/model_lineage.svg` | pending |
| Fig. 3: PCA by class | `scripts/encoder/make_encoder_technical_report.py` or audit output | external feature dump | external PCA PNG, summarized in encoder report | done-external |
| Fig. 4: PCA by board/material | `scripts/encoder/make_encoder_technical_report.py` or audit output | external feature dump | external PCA PNG, summarized in encoder report | done-external |
| Fig. 5: grouped evaluation bars | pending plotting script from CSV summaries | `reports/encoder/lobo_grouped_summary.csv`, `reports/encoder/lomo_summary.csv` | `figures/grouped_eval_summary.svg` | pending |

## Core Claims

| Claim | Evidence source | Manuscript section | Status |
|---|---|---|---|
| v6.2-A is a stronger reportable baseline than v6.1. | `reports/v61_v62_comparison/` | Results: classifier lineage | done-summary |
| Frozen v6.2-A embeddings preserve class structure. | `reports/encoder/EMBEDDING_AUDIT_SUMMARY.md` | Results: embedding geometry | done-summary |
| Frozen v6.2-A embeddings remain informative under board/material grouping. | `reports/encoder/GROUPED_ENCODER_EVAL_SUMMARY.md` | Results: grouped evaluation | done-summary |
| Hierarchical v6.2 is a candidate comparison branch. | `scripts/v6_2_hierarchical/train_v6_2_clean.py` | Methods: model variants | code-only |
| ESPI encoder improves acoustic-response prediction. | pending acoustic-response experiments | Future/second paper unless completed | pending |

## Release Gate

Before submission, every `pending` manuscript item should have:

```text
script -> config -> input artifact reference -> output report -> checksum or DOI
```
