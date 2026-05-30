# ResNet-18 Baseline Embedding Extraction

## Scope

This report documents deterministic eval-mode embedding extraction for random and ImageNet-pretrained ResNet-18 baselines. No training, fine-tuning, evaluation, LeFFT, or acoustic-response prediction was performed.

## Inputs

- Manifest: `<LOCAL_WORKSPACE_ROOT>\ESPI_v62_v62A_Encoder_package_20260430_224510\core_v62_files\manifest\manifest_v1_5class.npz`
- Image root: `<PRIVATE_IMAGE_ROOT>`
- Output directory: `outputs\encoder_features_normalized_v001\resnet18_small_test_cpu`
- Device: `cpu`
- Batch size: `32`
- Full extraction: `False`
- Max samples when not full: `200`

## Preprocessing note

All ESPI images are loaded as grayscale and resized to `256 x 256`.
The random baseline uses a one-channel ResNet-18 `conv1` adaptation with mean/std `0.5/0.5`.
The ImageNet baseline repeats grayscale to RGB and applies ImageNet mean/std normalization.

## Quality summary

| Variant | Preprocessing strategy | Samples | Dim | NaN | Inf | Duplicate paths | Nonzero variance | Internal comparison | Final publication tables | Output NPZ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| imagenet | grayscale ESPI image repeated to RGB, resized to 256x256, ImageNet mean/std normalization | 200 | 512 | 0 | 0 | 0 | True | True | True | `outputs\encoder_features_normalized_v001\resnet18_small_test_cpu\features_resnet18_imagenet.normalized.npz` |

## Output schema

Each NPZ uses schema `encoder_feature_schema_v001` and contains:

`embedding, label, label_name, path, board, material, frequency_hz, split, split_group, distribution_group, checkpoint_path, checkpoint_sha256, embedding_point, model_name, source_npz, schema_version`
