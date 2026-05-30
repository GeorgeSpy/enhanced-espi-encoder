| Encoder | Architecture role | Embedding point | Embedding dimension | Stratified fixed k=10 Macro-F1 | Board LOBO Macro-F1 | Material LOMO Macro-F1 | Publication decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v6.1 | Reference frozen ResNet18-style ESPI baseline | avgpool_pre_classifier | 512 | 93.65% | 87.10% | 86.33% | Retain as strongest stratified ESPI-specific reference. |
| v6.2-A | Official reportable image-derived frozen encoder baseline | MCDropoutClassifier.global_pool | 1280 | 92.78% | 93.56% | 93.49% | Use as primary reportable image-derived frozen ESPI encoder baseline. |
| hierarchical v6.2 phase2 | Physics-aware hierarchical phase2 expert comparison branch | z_expert_prelogit / z_arcface_input | 512 | 37.80% | 24.99% | 24.47% | Keep internal/controlled; not final-table ready until source metadata are regenerated. |
