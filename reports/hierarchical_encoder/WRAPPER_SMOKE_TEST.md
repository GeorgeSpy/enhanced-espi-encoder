# v6.2 Hierarchical Encoder Wrapper Smoke Test

## Scope

This smoke test validates the frozen hierarchical v6.2 encoder wrapper on a tiny manifest subset. It does not train and does not run full embedding extraction.

## Inputs

- Model file: `<LOCAL_PROJECT_ROOT>\scripts\v6_2\espi_v6_2_fixed.py`
- Checkpoint: `<PRIVATE_LOG_ROOT>\train_v6.2_opt\ckpt_phase2_expert.pt`
- Manifest: `<LOCAL_WORKSPACE_ROOT>\ESPI_v62_v62A_Encoder_package_20260430_224510\core_v62_files\manifest\manifest_v1_5class.npz`
- Device: `cpu`
- Samples tested: `5`
- Image size: `64`
- Created UTC: `2026-05-26T07:37:59.561030+00:00`

## Result

- Overall status: **pass**
- Required embeddings available: `True`
- No NaN: `True`
- No Inf: `True`
- Deterministic repeated forward: `True`
- Repeated forward max absolute difference: `{'z_expert_prelogit': 0.0, 'z_arcface_input': 0.0, 'gatekeeper_logits': 0.0, 'denoised': 0.0, 'wrapped_phase': 0.0, 'unwrapped_phase': 0.0, 'amplitude': 0.0}`

## Output Tensor Shapes

| Output | Available | Shape | NaN | Inf |
|---|---:|---|---:|---:|
| `z_expert_prelogit` | True | `[5, 512]` | 0 | 0 |
| `z_arcface_input` | True | `[5, 512]` | 0 | 0 |
| `gatekeeper_logits` | True | `[5, 2]` | 0 | 0 |
| `denoised` | True | `[5, 1, 64, 64]` | 0 | 0 |
| `wrapped_phase` | True | `[5, 1, 64, 64]` | 0 | 0 |
| `unwrapped_phase` | True | `[5, 1, 64, 64]` | 0 | 0 |
| `amplitude` | True | `[5, 1, 64, 64]` | 0 | 0 |

## Sample Paths

- `<PRIVATE_IMAGE_ROOT>\W01_ESPI_90db-PseudoNoisy_MATCH_v25\0150Hz_90.0db.png`
- `<PRIVATE_IMAGE_ROOT>\W01_ESPI_90db-PseudoNoisy_MATCH_v25\0320Hz_90.0db.png`
- `<PRIVATE_IMAGE_ROOT>\W01_ESPI_90db-PseudoNoisy_MATCH_v25\0500Hz_90.0db.png`
- `<PRIVATE_IMAGE_ROOT>\W01_ESPI_90db-PseudoNoisy_MATCH_v25\0550Hz_90.0db.png`
- `<PRIVATE_IMAGE_ROOT>\W01_ESPI_90db-PseudoNoisy_MATCH_v25\0700Hz_90.0db.png`

## Notes

- `z_expert_prelogit` and `z_arcface_input` intentionally reference `outputs["embeddings"]`.
- `z_physics_fusion` is intentionally not exposed in this wrapper.
- Placeholder physics modules are refused by the wrapper import path.