# Table 1 - Encoder Baseline Summary

| Encoder | Role | Embedding dimension | Stratified kNN Macro-F1 | Board LOBO Macro-F1 | Material LOMO Macro-F1 | Main interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| Random ResNet-18 | Untrained texture/control encoder | 512 | 84.40% | 31.99% | 25.44% | High stratified texture similarity but weak grouped robustness. |
| ImageNet ResNet-18 | Generic natural-image pretrained control encoder | 512 | 87.15% | 39.30% | 22.95% | Generic features improve stratified retrieval but collapse under grouped OOD. |
| v6.1 | Reference frozen ResNet-18 ESPI baseline | 512 | 95.07% | 87.10% | 86.33% | Best stratified kNN, but weaker board/material grouped robustness than v6.2-A. |
| v6.2-A | Primary reportable frozen ESPI encoder baseline | 1280 | 92.78% | 93.56% | 93.49% | Best grouped board/material robustness; primary reportable frozen encoder baseline. |
| hierarchical v6.2 phase2 | Controlled physics-aware development branch | 512 | 40.17% | 26.41% | 24.47% | Technically valid embeddings, but not representation-robust in this checkpoint. |
