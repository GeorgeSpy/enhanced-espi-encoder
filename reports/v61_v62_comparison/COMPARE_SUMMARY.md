# Apples-to-Apples: v6.1 vs v6.2 (same VAL split)

- Manifest: `artifacts/manifests/manifest_v1_5class.npz`
- v6.2 config: `configs/v6_2/config.antigravity.5class.yaml`
- VAL samples: **2589**

## Global Metrics

| Metric | v6.1 | v6.2 | Delta (v6.2-v6.1) |
|---|---:|---:|---:|
| Accuracy | 94.36% | 96.18% | +1.82 pp |
| Macro Recall | 91.05% | 92.70% | +1.65 pp |
| Macro F1 | 87.65% | 90.97% | +3.33 pp |

## Per-class Recall

| Class | v6.1 Recall | v6.2 Recall | Delta (pp) |
|---:|---:|---:|---:|
| 0 | 91.60% | 94.66% | +3.05 |
| 1 | 93.28% | 93.28% | +0.00 |
| 2 | 87.07% | 88.44% | +1.36 |
| 3 | 87.66% | 89.61% | +1.95 |
| 4 | 95.65% | 97.53% | +1.88 |

## Load Audit

- v6.1 missing/unexpected: 0/0
- v6.2 missing/unexpected: 0/0

## Files
- `compare_summary.json`
- `confusion_v61.csv`
- `confusion_v62.csv`
