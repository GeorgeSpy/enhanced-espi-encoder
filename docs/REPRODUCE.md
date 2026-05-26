# Reproduce Internal Reports

This document records command-level reproduction for the current internal evidence package. Paths are intentionally repository-relative or argument-driven. Raw images, checkpoints, manifests, and feature dumps are not stored in Git.

## Expected External Artifacts

Place or mount the private artifacts outside Git, then pass their paths explicitly.

| Artifact | Suggested local path | Tracked in Git |
|---|---|---|
| 5-class ESPI manifest | `artifacts/manifests/manifest_v1_5class.npz` | no |
| v6.2-A epoch-25 checkpoint | `artifacts/checkpoints/checkpoint_epoch25_20260211_035150.pt` | no |
| v6.2 helper package | `external/v6_2_fixed_package/` or `ESPI_V62_PACKAGE_ROOT` | no |
| extracted feature dump | `outputs/features_v62a_epoch25.npz` | no |

## 1. Extract v6.2-A Frozen Embeddings

```powershell
python scripts/encoder/extract_v62_embeddings.py `
  --package-root external/v6_2_fixed_package `
  --config configs/v6_2/config.antigravity.5class.yaml `
  --manifest artifacts/manifests/manifest_v1_5class.npz `
  --checkpoint artifacts/checkpoints/checkpoint_epoch25_20260211_035150.pt `
  --out outputs/features_v62a_epoch25.npz `
  --device cuda
```

CPU smoke test:

```powershell
python scripts/encoder/extract_v62_embeddings.py `
  --package-root external/v6_2_fixed_package `
  --config configs/v6_2/config.antigravity.5class.quickcheck.yaml `
  --manifest artifacts/manifests/manifest_v1_5class.npz `
  --checkpoint artifacts/checkpoints/checkpoint_epoch25_20260211_035150.pt `
  --out outputs/features_v62a_epoch25.smoke.npz `
  --device cpu `
  --max-samples 64
```

Expected internal reference:

| Field | Expected |
|---|---|
| Samples | 12944 |
| Embedding dimension | 1280 |
| Checkpoint load missing/unexpected | 0/0 |
| Embedding layer | `MCDropoutClassifier.global_pool.pre_fc` |

## 2. Run Embedding Audit

```powershell
python scripts/encoder/audit_v62_embeddings.py `
  --features outputs/features_v62a_epoch25.npz `
  --out-dir outputs/embedding_audit_v001
```

Expected internal reference:

| Metric | Reference value |
|---|---:|
| kNN Macro-F1 | about 93.44% |
| Linear probe Macro-F1 | about 92.85% |
| Prototype Macro-F1 | about 86.42% |

## 3. Run Grouped Encoder Evaluation

```powershell
python scripts/encoder/evaluate_encoder_grouped_generalization.py `
  --features outputs/features_v62a_epoch25.npz `
  --out-dir outputs/embedding_audit_v001/grouped_generalization_v001
```

Expected outputs:

```text
outputs/embedding_audit_v001/grouped_generalization_v001/
outputs/embedding_audit_v001/grouped_generalization_v001/GROUPED_ENCODER_EVAL_SUMMARY.md
outputs/embedding_audit_v001/grouped_generalization_v001/grouped_encoder_key_numbers.json
```

## 4. Generate Technical Report

```powershell
python scripts/encoder/make_encoder_technical_report.py `
  --feature-summary outputs/features_v62a_epoch25.FEATURE_DUMP_SUMMARY.md `
  --features outputs/features_v62a_epoch25.npz `
  --metadata outputs/features_v62a_epoch25.metadata.csv `
  --audit-dir outputs/embedding_audit_v001 `
  --out outputs/FROZEN_ENCODER_TECHNICAL_REPORT.md `
  --key-numbers outputs/encoder_report_key_numbers.json
```

If this command changes because the report generator evolves, update this file and `docs/MANUSCRIPT_MAP.md` together.

## 5. Baseline Encoder Comparison

This is not complete yet. Before submission, add a common script that evaluates the same grouped metrics for:

```text
random ResNet-18 embeddings
ImageNet-pretrained ResNet-18 embeddings
v6.1 embeddings
v6.2-A embeddings
optional hierarchical v6.2 embeddings
```

Target table:

| Encoder | Embedding point | kNN Macro-F1 | Linear Macro-F1 | LOBO Macro-F1 | LOMO Macro-F1 |
|---|---|---:|---:|---:|---:|
| random ResNet-18 | pre-head | pending | pending | pending | pending |
| ImageNet ResNet-18 | pre-head | pending | pending | pending | pending |
| v6.1 | pre-head | pending | pending | pending | pending |
| v6.2-A | global pool pre-FC | done-summary | done-summary | done-summary | done-summary |
| v6.2 hierarchical | matched embedding | pending | pending | pending | pending |

## 6. Artifact Hashes

Before public/reviewer release, record checksums for private artifacts in a non-Git or DOI-backed manifest:

```powershell
Get-FileHash artifacts/manifests/manifest_v1_5class.npz -Algorithm SHA256
Get-FileHash artifacts/checkpoints/checkpoint_epoch25_20260211_035150.pt -Algorithm SHA256
Get-FileHash outputs/features_v62a_epoch25.npz -Algorithm SHA256
```
