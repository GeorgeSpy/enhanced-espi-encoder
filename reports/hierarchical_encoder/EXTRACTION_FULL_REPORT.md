# Hierarchical v6.2 Full Embedding Extraction

## Scope

This report summarizes a frozen hierarchical v6.2 embedding extraction run. It does not train. Full extraction is only enabled when `--full` is explicitly used.

## Inputs

- Checkpoint: `<PRIVATE_LOG_ROOT>\train_v6.2_opt\ckpt_phase2_expert.pt`
- Checkpoint SHA256: `ebb0a1ed05fc480b9e7f575800db1df8e7dffa027b75cc6bd59debdd81e39e00`
- Manifest: `<LOCAL_WORKSPACE_ROOT>\ESPI_v62_v62A_Encoder_package_20260430_224510\core_v62_files\manifest\manifest_v1_5class.npz`
- Output NPZ: `<LOCAL_PROJECT_ROOT>\outputs\hierarchical_embeddings_v001\features_hier_z_expert_prelogit.npz`
- Device: `cpu`
- Full extraction: `True`
- Max samples argument: `200`
- Samples extracted: `12944`
- Embedding point: `z_expert_prelogit / z_arcface_input from outputs['embeddings']`
- Model name: `EnhancedHybridPhysicsESPI_V6_2`
- Created UTC: `2026-05-26T08:22:56.468184+00:00`

## Quality Checks

- Overall status: **pass**
- Embedding shape: `[12944, 512]`
- No NaN: `True` (`0` NaN values)
- No Inf: `True` (`0` Inf values)
- Nonzero variance: `True`
- Consistent embedding shape: `True`
- Duplicate paths: `0`
- Metadata length equals embedding length: `True`

## Notes

- `z_expert_prelogit` and `z_arcface_input` are extracted from `outputs["embeddings"]`.
- `z_physics_fusion` is intentionally not exposed or extracted in this run.
- This is a small-sample extraction by default; full extraction requires `--full`.