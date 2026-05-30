# Reproduce the Evidence Package

This guide documents the layered reproduction path for the frozen ESPI representation-audit evidence package. Full reproduction requires private artifacts that are intentionally excluded from Git.

## External Artifact Requirements

Use local paths or a DOI-backed review artifact bundle for:

```powershell
$MANIFEST      = "artifacts/manifests/manifest_v1_5class.npz"
$IMAGE_ROOT    = "path/to/private/espi/images"
$CKPT_V61      = "path/to/private/v6_1_checkpoint.pt"
$CKPT_V62A     = "path/to/private/v6_2A_checkpoint.pt"
$CKPT_HIER     = "path/to/private/hierarchical_v6_2_phase2_checkpoint.pt"
$FEATURES_V62A = "outputs/encoder_features_normalized_v001/features_v62a_epoch25.normalized.npz"
$FEATURES_V61  = "outputs/encoder_features_normalized_v001/features_v61.normalized.npz"
```

Raw ESPI images, checkpoints, full feature dumps, and private manifests are not tracked in Git.

## Core Reproduction Layers

Run commands from the repository root.

### 1. Feature Schema Normalization

```powershell
python scripts\encoder\normalize_encoder_feature_schema.py `
  --v62a-features $FEATURES_V62A `
  --hier-features "outputs/hierarchical_embeddings_v001/features_hier_z_expert_prelogit.npz" `
  --manifest $MANIFEST `
  --out-dir outputs\encoder_features_normalized_v001
```

### 2. Encoder Feature Extraction

Optional if normalized feature dumps are already available.

```powershell
python scripts\encoder\extract_v61_embeddings.py --manifest $MANIFEST --image-root $IMAGE_ROOT --checkpoint $CKPT_V61 --out-dir outputs\encoder_features_normalized_v001 --device cpu --full --overwrite
python scripts\encoder\extract_resnet18_baseline_embeddings.py --manifest $MANIFEST --image-root $IMAGE_ROOT --out-dir outputs\encoder_features_normalized_v001 --device cpu --variant random --full --overwrite
python scripts\encoder\extract_resnet18_baseline_embeddings.py --manifest $MANIFEST --image-root $IMAGE_ROOT --out-dir outputs\encoder_features_normalized_v001 --device cpu --variant imagenet --full --overwrite
python scripts\encoder\extract_v62_hierarchical_embeddings.py --checkpoint $CKPT_HIER --manifest $MANIFEST --out-dir outputs\hierarchical_embeddings_v001 --device cpu --full --overwrite
```

### 3. Unified Encoder Baseline Evaluation

```powershell
python scripts\encoder\evaluate_encoder_baselines.py `
  --features-dir outputs\encoder_features_normalized_v001 `
  --out-dir reports\encoder_baselines\eval_v002 `
  --linear-max-iter 5000 `
  --enable-bootstrap `
  --bootstrap-iters 200
```

### 4. Frequency Controls

```powershell
python scripts\encoder\evaluate_frequency_only_baseline.py `
  --features outputs\encoder_features_normalized_v001\features_v62a_epoch25.normalized.npz `
  --eval-dir reports\encoder_baselines\eval_v002 `
  --composition-dir reports\publication_assets\methodological_hardening_v001 `
  --out-dir reports\publication_assets\methodological_hardening_v001\frequency_only_baseline

python scripts\encoder\evaluate_frequency_embedding_fusion.py `
  --features-dir outputs\encoder_features_normalized_v001 `
  --eval-dir reports\encoder_baselines\eval_v002 `
  --frequency-dir reports\publication_assets\methodological_hardening_v001\frequency_only_baseline `
  --out-dir reports\publication_assets\methodological_hardening_v001\frequency_embedding_fusion
```

### 5. Methodological Hardening

```powershell
python scripts\encoder\evaluate_v62a_dimension_matched_pca.py
python scripts\encoder\evaluate_frequency_residual_correction.py
python scripts\encoder\evaluate_frequency_overlap_pair.py
python scripts\encoder\evaluate_balanced_subsets.py
python scripts\encoder\evaluate_embedding_geometry.py
```

These scripts reproduce the PCA dimension-matching, frequency-residual correction, targeted `1_2` / `2_1` overlap, balanced subset, and quantitative geometry reports.

### 6. Spectral Descriptor Control

```powershell
python scripts\encoder\extract_spectral_descriptors.py --manifest $MANIFEST --image-root $IMAGE_ROOT --out-dir outputs\encoder_features_normalized_v001 --full --overwrite
python scripts\encoder\evaluate_spectral_descriptor_ablation.py --features-dir outputs\encoder_features_normalized_v001 --eval-dir reports\encoder_baselines\eval_v002 --frequency-dir reports\publication_assets\methodological_hardening_v001\frequency_only_baseline --fusion-dir reports\publication_assets\methodological_hardening_v001\frequency_embedding_fusion --out-dir reports\publication_assets\methodological_hardening_v001\lefft_descriptor_ablation\eval_v001
```

### 7. Final Tables and Submission Audit

```powershell
python scripts\encoder\make_final_manuscript_tables.py `
  --eval-dir reports\encoder_baselines\eval_v002 `
  --publication-assets reports\publication_assets\encoder_baseline_v001 `
  --hardening-dir reports\publication_assets\methodological_hardening_v001 `
  --out-dir reports\publication_assets\final_tables_v001

python scripts\encoder\audit_v003_submission_readiness.py `
  --manuscript manuscripts\OLEN_ESPI_frozen_representation_submission.tex `
  --tables-dir reports\publication_assets\final_tables_v001 `
  --hardening-dir reports\publication_assets\methodological_hardening_v001 `
  --out-dir reports\publication_assets\submission_readiness_v002
```

## Notes

- All lightweight-head preprocessing and feature scaling must be fit only on the training/reference fold.
- Material LOMO is interpreted as a material-held-out stress test because only two material groups exist.
- The deterministic spectral descriptor control is not a trained LeFFT model.
- This package does not reproduce raw-image feature dumps without private data access.
