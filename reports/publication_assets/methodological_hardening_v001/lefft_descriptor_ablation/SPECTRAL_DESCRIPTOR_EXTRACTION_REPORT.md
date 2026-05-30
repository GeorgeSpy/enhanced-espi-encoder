# Spectral Descriptor Extraction Report

## Scope

This report documents deterministic spatial-frequency descriptor extraction for ESPI images. The descriptors are spectral / LeFFT-inspired descriptors, not a trained LeFFT model. No deep model training, encoder fine-tuning, evaluation, LeFFT superiority claim, or acoustic-response prediction was performed.

## Inputs

- Manifest: `<LOCAL_WORKSPACE_ROOT>\ESPI_v62_v62A_Encoder_package_20260430_224510\core_v62_files\manifest\manifest_v1_5class.npz`
- Image root: `<PRIVATE_IMAGE_ROOT>`
- ROI mask: `none`
- Device argument: `cpu` (descriptors are computed deterministically on CPU with NumPy)
- Full extraction: `True`
- Max samples when not full: `200`

## Output

- Output NPZ: `outputs\encoder_features_normalized_v001\features_spectral_descriptors.normalized.npz`
- Model name: `spectral_descriptors_lefft_inspired`
- Embedding point: `deterministic_fft_spectral_descriptors`
- Schema version: `encoder_feature_schema_v001`

## Preprocessing strategy

Each ESPI image is loaded as grayscale, resized to `256 x 256`, robustly normalized per image using percentile clipping, optionally masked with the supplied ROI mask, Hann-windowed, transformed with a 2D FFT, converted to log-magnitude spectrum, and processed after suppressing the DC component.

## Descriptor families

- 8 normalized radial FFT energy bands
- 12 normalized angular FFT energy bins
- low/mid/high spatial-frequency energy ratios
- spectral centroid radius, bandwidth, and entropy
- top-1/top-5/top-10 peak concentration
- orientation anisotropy and horizontal/vertical/diagonal energy ratios
- 4x4 local patch low/mid/high energy mean and standard deviation

## Quality checks

| Check | Value |
|---|---:|
| sample_count | 12944 |
| descriptor_dimension | 39 |
| embedding_shape | [12944, 39] |
| nan_count | 0 |
| inf_count | 0 |
| duplicate_paths | 0 |
| nonzero_variance | True |
| metadata_lengths_match | True |
| descriptor_names_count | 39 |
| descriptor_names_match_dimension | True |

## Descriptor names

`radial_energy_bin_00, radial_energy_bin_01, radial_energy_bin_02, radial_energy_bin_03, radial_energy_bin_04, radial_energy_bin_05, radial_energy_bin_06, radial_energy_bin_07, angular_energy_bin_00, angular_energy_bin_01, angular_energy_bin_02, angular_energy_bin_03, angular_energy_bin_04, angular_energy_bin_05, angular_energy_bin_06, angular_energy_bin_07, angular_energy_bin_08, angular_energy_bin_09, angular_energy_bin_10, angular_energy_bin_11, low_frequency_energy_ratio, mid_frequency_energy_ratio, high_frequency_energy_ratio, spectral_centroid_radius, spectral_bandwidth, spectral_entropy, peak_concentration_top1, peak_concentration_top5, peak_concentration_top10, orientation_anisotropy_max_over_mean, horizontal_energy_ratio, vertical_energy_ratio, diagonal_energy_ratio, patch_low_energy_mean, patch_mid_energy_mean, patch_high_energy_mean, patch_low_energy_std, patch_mid_energy_std, patch_high_energy_std`

## Claim boundary

These outputs are deterministic spectral descriptors and may be used as a lightweight spectral / LeFFT-inspired baseline. They are not a trained LeFFT model and do not support any claim of LeFFT superiority.
