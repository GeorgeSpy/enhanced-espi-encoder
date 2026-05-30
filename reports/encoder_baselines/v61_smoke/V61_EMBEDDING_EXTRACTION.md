# v6.1 Frozen Embedding Extraction

## Scope

This report documents deterministic eval-mode extraction of v6.1 frozen embeddings. No training, fine-tuning, unified evaluation, LeFFT, or acoustic-response prediction was performed.

## Inputs

- Manifest: `<LOCAL_WORKSPACE_ROOT>\ESPI_v62_v62A_Encoder_package_20260430_224510\core_v62_files\manifest\manifest_v1_5class.npz`
- Image root: `<PRIVATE_IMAGE_ROOT>`
- Checkpoint: `<PRIVATE_CHECKPOINT_ROOT>\espi_modes_v6_1_head.pt`
- Checkpoint SHA256: `28d7b06cc7da091f4924be5c2c6390f0b61c593fe5a6ac2efd1d3c7609a8a1c7`
- Device: `cpu`
- Batch size: `32`
- Full extraction: `False`
- Max samples when not full: `200`

## Model

- Model: `v6.1 ResNet18_HeadOnly`
- Architecture source: reconstructed from `scripts/v6_1/train_v6_1_head_only.py`
- Backbone: ResNet-18 style, 1-channel ESPI input
- Classifier head: Linear(512, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 5)
- Embedding point: `avgpool_pre_classifier`
- Embedding dimension: `512`
- Preprocessing: `v6.1 documented preprocessing: grayscale 1-channel, resize 256x256, per-image z-score normalization`

## Load audit

- Status: `clean`
- Checkpoint keys: `124`
- Model keys: `124`
- Matched keys: `124`
- Missing keys: `0`
- Unexpected keys: `0`
- Shape mismatches: `0`

## Quality checks

- Sample count: `200`
- Embedding shape: `[200, 512]`
- NaN count: `0`
- Inf count: `0`
- Duplicate paths: `0`
- Nonzero variance: `True`
- Metadata lengths match embeddings: `True`

## Output

- NPZ: `outputs\encoder_features_normalized_v001\v61_smoke\features_v61.normalized.npz`
- Internal comparison acceptable: `True`
- Final publication tables acceptable: `True`
- Schema version: `encoder_feature_schema_v001`
