# Artifact Manifest

## Included

- selected scripts for v6.2 and encoder evaluation
- selected config snapshots
- summary-level reports only
- one example acoustic manifest placeholder

## Excluded

| Pattern | Reason |
|---|---|
| `*.pt`, `*.pth` | checkpoints are large and should be externally versioned |
| `*.onnx` | generated model artifact |
| `*.npz`, `*.npy` | large features/manifests/arrays |
| `*.zip` | package archives |
| raw image folders | dataset ownership and size |
| full confusion matrix folders | too much report noise for the development repo |

## Archive

The previous full import was moved outside this curated repository and is intentionally not part of the reviewer package.

```text
../enhanced-espi-encoder_archive_20260525_full_import
```
