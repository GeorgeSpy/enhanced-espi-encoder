# v6.2 Hierarchical Checkpoint Load Audit

## Scope

This is a safe, read-only checkpoint load audit for the hierarchical / physics-aware v6.2 branch. It does not train a model, does not load raw images, and does not run full dataset inference.

## Inputs

- Model file: `<LOCAL_PROJECT_ROOT>\scripts\v6_2\espi_v6_2_fixed.py`
- Checkpoint path: `<PRIVATE_LOG_ROOT>\train_v6.2_opt\ckpt_phase2_expert.pt`
- Checkpoint SHA256: `ebb0a1ed05fc480b9e7f575800db1df8e7dffa027b75cc6bd59debdd81e39e00`
- Device: `cpu`
- Created UTC: `2026-05-26T07:25:08.735233+00:00`

## Model Import

- Model class: `v62_hierarchical_model_audit.EnhancedHybridPhysicsESPI_V6_2`
- Base physics modules imported: `True`
- Placeholder physics modules refused: `False`
- Total parameters: `2653944`

## Load Status

- Load mismatch status: **clean**
- Explanation: Checkpoint keys match the hierarchical model state_dict cleanly.
- Checkpoint state source: `checkpoint root`
- State-dict variant: `raw`
- Matched keys: `237`
- Missing keys: `0`
- Unexpected keys: `0`
- Shape mismatches: `0`

## Dummy Forward Output Keys

- Dummy forward attempted: `True`
- Dummy forward OK: `True`
- Dummy shape: `[1, 1, 64, 64]`
- Output keys: `['amplitude', 'denoised', 'embeddings', 'expert_logits', 'gate_logits', 'unwrapped_phase', 'wrapped_phase']`
- Dummy forward error: `none`

## Output Availability

| Output | Available |
|---|---:|
| `embeddings` | True |
| `denoised` | True |
| `wrapped_phase` | True |
| `unwrapped_phase` | True |
| `amplitude` | True |
| `physics_feats` | False |
| `spectral_filter` | False |

`outputs["embeddings"]` is the first practical embedding point for `z_expert_prelogit` / `z_arcface_input`. `physics_feats` and `spectral_filter` are expected to be unavailable in the current forward output unless the model is later extended or instrumented with hooks.

## Missing Keys

- none

## Unexpected Keys

- none

## Shape Mismatches

- none

## Warnings

- none

## Errors

- none

