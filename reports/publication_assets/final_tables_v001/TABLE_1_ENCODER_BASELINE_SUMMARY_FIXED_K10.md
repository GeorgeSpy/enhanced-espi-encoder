| Encoder | Role | Embedding dimension | Primary stratified kNN Macro-F1 at fixed k=10 | Board LOBO Macro-F1 at fixed k=10 | Material LOMO Macro-F1 at fixed k=10 | Main interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| Random ResNet-18 | Random generic visual-control encoder | 512 | 81.14% | 26.17% | 24.59% | Non-trivial stratified texture baseline but weak grouped robustness. |
| ImageNet ResNet-18 | ImageNet generic visual-control encoder | 512 | 84.34% | 39.30% | 22.52% | Generic pretrained features remain insufficient under grouped evaluation. |
| v6.1 | Reference ESPI-specific frozen ResNet18-style baseline | 512 | 93.65% | 87.10% | 86.33% | Strong stratified ESPI-specific representation, weaker grouped robustness than v6.2-A. |
| v6.2-A | Primary image-derived frozen ESPI encoder baseline | 1280 | 92.78% | 93.56% | 93.49% | Strongest image-derived frozen ESPI encoder under grouped board/material evaluation. |
| hierarchical v6.2 phase2 | Controlled physics-aware architecture comparison | 512 | 37.80% | 24.99% | 24.47% | Technically valid but not superior; internal comparison only pending metadata regeneration. |
