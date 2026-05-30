| Condition | Stratified Macro-F1 | Board LOBO Macro-F1 | Material LOMO Macro-F1 | Delta vs frequency-only | Delta vs embedding-only | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| frequency-only | 98.67% | 97.49% | 94.09% | reference | not applicable | Frequency_hz is the dominant metadata-only predictor for the current label protocol. |
| v6.2-A embedding-only | 93.48% | 94.28% | 94.05% | Stratified -5.19 pp, Board -3.20 pp, Material -0.03 pp | reference | Image-derived v6.2-A embeddings remain the strongest frozen ESPI encoder baseline. |
| frequency + v6.2-A | 96.75% | 94.79% | 94.50% | Stratified -1.92 pp, Board -2.70 pp, Material +0.41 pp | Stratified +3.27 pp, Board +0.50 pp, Material +0.45 pp | Fusion improves over embedding-only but does not clearly exceed frequency-only except in Material LOMO. |
