# Embedding Geometry & Nearest-Neighbor Consistency Analysis

This report presents a direct quantitative analysis of the frozen embedding geometry for **v6.1** and **v6.2-A**.

- v6.1 features: `outputs\encoder_features_normalized_v001\features_v61.normalized.npz`
- v6.2-A features: `outputs\encoder_features_normalized_v001\features_v62a_epoch25.normalized.npz`

## Global Geometry & Generalization Metrics

The global metrics assess the cohesion, separation, and generalization (cross-specimen and cross-material consistency) of the representation spaces:

| Metric                                               | v6.1 Reference | v6.2-A Encoder | Delta   |
| ---------------------------------------------------- | -------------- | -------------- | ------- |
| Silhouette Score (Cosine, higher is better)          | 0.5789         | 0.7248         | 0.1459  |
| Davies-Bouldin Index (L2-norm, lower is better)      | 1.0540         | 0.6453         | -0.4087 |
| Intra/Inter Distance Ratio (Cosine, lower is better) | 0.3230         | 0.1698         | -0.1532 |
| Same-Class 1-NN Consistency Rate (higher is better)  | 0.9789         | 0.9730         | -0.0059 |
| Cross-Board Same-Class NN Rate (higher is better)    | 0.9537         | 0.9638         | 0.0100  |
| Cross-Material Same-Class NN Rate (higher is better) | 0.9421         | 0.9652         | 0.0232  |

## Per-Class Intra/Inter Cosine Distance Ratio

Lower Intra/Inter ratios indicate tighter class clusters relative to the background distribution. A negative Delta indicates a better-contained class cluster in v6.2-A:

| Class ID | Class Name | Support | v6.1 Intra-Dist | v6.2-A Intra-Dist | v6.1 Ratio | v6.2-A Ratio | Ratio Delta |
| -------- | ---------- | ------- | --------------- | ----------------- | ---------- | ------------ | ----------- |
| 0        | 0          | 653     | 0.1006          | 0.1336            | 0.2293     | 0.1534       | -0.0760     |
| 1        | 1          | 669     | 0.1608          | 0.1328            | 0.2762     | 0.1554       | -0.1209     |
| 2        | 2          | 738     | 0.2184          | 0.1399            | 0.4326     | 0.1951       | -0.2376     |
| 3        | 3          | 769     | 0.1936          | 0.1142            | 0.3909     | 0.1557       | -0.2352     |
| 4        | 4          | 10115   | 0.1423          | 0.1498            | 0.2858     | 0.1893       | -0.0965     |

## Output Files

- `reports\embedding_geometry_v62a\embedding_geometry_global.csv`
- `reports\embedding_geometry_v62a\embedding_geometry_per_class.csv`
- `reports\embedding_geometry_v62a\embedding_geometry_data.json`
