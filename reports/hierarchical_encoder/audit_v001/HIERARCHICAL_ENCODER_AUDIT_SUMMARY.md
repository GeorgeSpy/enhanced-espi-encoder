# Hierarchical v6.2 Frozen Embedding Audit v001

## Scope
This report audits the already extracted hierarchical v6.2 frozen embeddings from `z_expert_prelogit / z_arcface_input`.
It is a frozen grouped embedding evaluation over saved features only. It does not train a new CNN, does not reload raw images, and does not validate a final Physics-Aligned Encoder.

## Source artifact
- Feature dump: `outputs\hierarchical_embeddings_v001\features_hier_z_expert_prelogit.npz`
- Model name: `EnhancedHybridPhysicsESPI_V6_2`
- Embedding point: `z_expert_prelogit / z_arcface_input from outputs['embeddings']`
- Checkpoint SHA256: `ebb0a1ed05fc480b9e7f575800db1df8e7dffa027b75cc6bd59debdd81e39e00`
- Samples: `12944`
- Embedding dimension: `512`

## Metadata recovery
- `board`: `derived_from_distribution_group_or_path`
- `material`: `saved_with_board_path_fallback`
- `domain`: `derived_from_distribution_group_or_path`
- `split_group`: `fallback_distribution_group`

## Quality checks
- NaN count: `0`
- Inf count: `0`
- Duplicate paths: `0`
- Nonzero variance: `True`
- Metadata lengths match embeddings: `True`
- Train/val path overlap count: `0`
- Train/val split-group overlap count: `13`

## Stratified train-to-validation evaluation
| method | accuracy | macro_recall | macro_f1 |
| --- | --- | --- | --- |
| knn_cosine_k1 | 72.42% | 39.99% | 40.17% |
| nearest_class_prototype | 55.54% | 36.27% | 26.55% |
| balanced_logistic_regression | 50.75% | 37.83% | 29.28% |


## Leave-split-group evaluation
| Method | Groups | Mean Macro-F1 | Worst group | Worst Macro-F1 | Best group | Best Macro-F1 |
|---|---:|---:|---|---:|---|---:|
| balanced_logistic_regression | 13 | 14.42% | W02_ESPI_90db-Averaged | 1.30% | C01_ESPI_90db | 34.20% |
| knn_cosine_k10 | 13 | 21.28% | W03_ESPI_90db-Averaged | 17.31% | W03_ESPI_90db | 33.30% |
| nearest_class_prototype | 13 | 12.25% | W02_ESPI_90db-Averaged | 1.32% | W01_ESPI_90db | 30.49% |


## Board-grouped LOBO-style evaluation
| Method | Groups | Mean Macro-F1 | Worst group | Worst Macro-F1 | Best group | Best Macro-F1 |
|---|---:|---:|---|---:|---|---:|
| balanced_logistic_regression | 6 | 26.41% | C03 | 18.20% | C01 | 32.42% |
| knn_cosine_k10 | 6 | 24.99% | C02 | 20.42% | W03 | 31.89% |
| nearest_class_prototype | 6 | 23.87% | C03 | 15.99% | C01 | 36.34% |


## Material-grouped LOMO-style evaluation
| Method | Groups | Mean Macro-F1 | Worst group | Worst Macro-F1 | Best group | Best Macro-F1 |
|---|---:|---:|---|---:|---|---:|
| balanced_logistic_regression | 2 | 23.47% | wood | 23.40% | carbon | 23.54% |
| knn_cosine_k10 | 2 | 24.47% | wood | 23.23% | carbon | 25.71% |
| nearest_class_prototype | 2 | 22.15% | wood | 22.06% | carbon | 22.25% |


## PCA diagnostics
- `reports\hierarchical_encoder\audit_v001\pca_by_class_z_expert_prelogit.png`
- `reports\hierarchical_encoder\audit_v001\pca_by_board_z_expert_prelogit.png`
- `reports\hierarchical_encoder\audit_v001\pca_by_material_z_expert_prelogit.png`

## Interpretation
The hierarchical v6.2 feature dump exposes a valid frozen embedding point through `outputs["embeddings"]`, used here as `z_expert_prelogit / z_arcface_input`.
The stratified and grouped metrics quantify whether this saved embedding space remains useful when the reference set excludes split groups, boards, or materials.
This is not a comparison with v6.2-A and it is not a claim of a validated Physics-Aligned Encoder.

## Caveats
- Grouped evaluations operate on frozen embeddings from an already trained hierarchical checkpoint.
- They test representation usefulness under excluded-reference settings, not a new CNN training run with held-out boards or materials.
- Metadata fields `board` and `split_group` in the feature dump may require recovery from `distribution_group` and `path`; the metadata sources are reported above.

## Warnings
- Board metadata was recovered using derived_from_distribution_group_or_path.
- Material metadata was recovered using saved_with_board_path_fallback.
- Domain metadata was recovered using derived_from_distribution_group_or_path.
- Split-group metadata uses fallback_distribution_group.
