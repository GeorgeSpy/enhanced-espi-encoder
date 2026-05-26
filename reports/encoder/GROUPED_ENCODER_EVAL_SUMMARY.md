# Grouped Encoder Evaluation Summary

## Scope

This is a **frozen grouped embedding evaluation** for the **v6.2-A frozen ESPI encoder candidate**. It uses only the saved embeddings and metadata. It is **not a validated Physics-Aligned Encoder yet** and does not retrain any CNN.

## Source feature dump
- Feature dump: `outputs/features_v62a_epoch25.npz`
- Samples: `12944`
- Embedding dimension: `1280`
- Checkpoint: `artifacts/checkpoints/checkpoint_epoch25_20260211_035150.pt`
- Embedding layer: `MCDropoutClassifier.global_pool.pre_fc`
- Smoke test: `False`

## Methods used

- `knn_cosine_k10`
- `linear_probe_logistic_regression`
- `nearest_class_prototype_cosine`

All folds standardize features using only the reference/train fold and then apply the transform to the held-out test fold.

## Board-level LOBO-style summary

| Method | Groups | Mean Accuracy | Mean Macro-F1 | Min Macro-F1 | Max Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| knn_cosine_k10 | 6 | 96.95% | 93.56% | 90.82% | 95.69% |
| linear_probe_logistic_regression | 6 | 95.90% | 91.40% | 88.02% | 94.37% |
| nearest_class_prototype_cosine | 6 | 80.07% | 70.97% | 66.99% | 74.35% |

- Best board result: `C03` / `knn_cosine_k10`: Macro-F1 `95.69%`, Accuracy `97.62%`
- Worst board result: `W01` / `knn_cosine_k10`: Macro-F1 `90.82%`, Accuracy `95.08%`

## Material-level LOMO-style summary

| Method | Directions | Mean Accuracy | Mean Macro-F1 | Min Macro-F1 | Max Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| knn_cosine_k10 | 2 | 96.89% | 93.49% | 93.01% | 93.97% |
| linear_probe_logistic_regression | 2 | 95.72% | 91.17% | 89.19% | 93.15% |
| nearest_class_prototype_cosine | 2 | 79.65% | 71.35% | 69.00% | 73.70% |

- Best material direction: `wood_to_carbon` / `knn_cosine_k10`: Macro-F1 `93.97%`, Accuracy `97.23%`
- Worst material direction: `carbon_to_wood` / `knn_cosine_k10`: Macro-F1 `93.01%`, Accuracy `96.55%`

## Automatic interpretation

This evaluation tests whether the already extracted v6.2-A embedding space remains useful when the reference set excludes a board/material. It is stricter than a stratified retrieval/probe audit, but it is still not a new CNN training run.
- Board-grouped Macro-F1 remains relatively high, strengthening the frozen ESPI encoder candidate claim.
- Material-grouped Macro-F1 remains relatively high, supporting cross-material embedding usefulness.

## Caveat

This is a strict grouped evaluation over frozen embeddings, but not a new CNN training run. It tests whether the already extracted v6.2-A embedding space remains useful when the reference set excludes a board/material.
