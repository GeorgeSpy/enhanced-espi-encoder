# v6.2-A Encoder Evidence Audit

## 1. Source checkpoint
- Checkpoint: `artifacts/checkpoints/checkpoint_epoch25_20260211_035150.pt`
- Baseline reference: `96.18% Accuracy / 90.97% Macro-F1`
- Embedding layer: `MCDropoutClassifier.global_pool.pre_fc`
- Embedding dimension: `1280`

## 2. Dataset / manifest
- Features: `features_v62a_epoch25.npz`
- Manifest: `artifacts/manifests/manifest_v1_5class.npz`
- Samples: `12944`
- Split counts: `{'train': 10355, 'val': 2589}`
- Split group definition: `board`, falling back to source group when board cannot be parsed

## 3. Embedding extraction audit
- NaN count: `0`
- Inf count: `0`
- Duplicate paths: `0`
- Class supports: `class_supports_by_split.csv`

## 4. Frozen embedding evaluation
- `knn_report.csv`: train-to-val kNN on frozen embeddings
- `linear_probe_report.csv`: logistic-regression linear probe when scikit-learn is available
- `prototype_report.csv`: nearest class prototype on train-to-val
- `leave_split_group_out_report.csv`: leave-`split_group`-out retrieval/prototype audit
- `leave_split_group_out_aggregate.csv`: grouped mean/min/max summary by method
- `distance_report.csv`: intra-class prototype compactness and nearest prototype separation

Important caveat: the frozen v6.2-A checkpoint was trained on the original stratified split. The leave-`split_group`-out report removes groups from the retrieval/probe reference set only; it is post-hoc representation-geometry evidence, not a true LOBO-trained encoder result.

## 5. Grouped structure diagnostics
- PCA plots are diagnostic only, not proof of generalization.
- Use board/material/source-group coloring to check whether embeddings encode grouped dataset signatures.

## 6. Interpretation Gate
- Strong encoder evidence requires LOBO retrieval/prototype macro-F1 to remain non-collapsed.
- If PCA/UMAP clusters primarily by board/material instead of label, the representation is group-biased.

## 7. Next step
- If grouped retrieval collapses, run supervised contrastive or group-aware fine-tuning before claiming progress toward a Physics-Aligned Encoder.

## Plots
- `embedding_audit_v001\pca_by_label_name.png`
- `embedding_audit_v001\pca_by_board.png`
- `embedding_audit_v001\pca_by_material.png`
- `embedding_audit_v001\pca_by_domain.png`

