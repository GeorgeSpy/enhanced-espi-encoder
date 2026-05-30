# Encoder Feature Schema Normalization

## Scope

This report documents normalization of existing encoder feature dumps into a common schema for internal baseline comparison. The process does not modify original NPZ files, does not run model inference, does not train, and does not evaluate embeddings.

## Normalized schema

`embedding, label, label_name, path, board, material, frequency_hz, split, split_group, distribution_group, checkpoint_path, checkpoint_sha256, embedding_point, model_name, source_npz, schema_version`

## Summary

| encoder | source_npz | output_npz | n_samples | embedding_dim | recovered_fields | unrecoverable_fields | acceptable_for_internal_comparison | acceptable_for_final_publication_tables |
|---|---|---|---:|---:|---|---|---:|---:|
| v6.2-A | <LOCAL_WORKSPACE_ROOT>\v62_encoder_evidence\features_v62a_epoch25.npz | outputs\encoder_features_normalized_v001\features_v62a_epoch25.normalized.npz | 12944 | 1280 | ['checkpoint_sha256', 'embedding_point'] | [] | True | True |
| hierarchical_v6.2_phase2_expert | outputs\hierarchical_embeddings_v001\features_hier_z_expert_prelogit.npz | outputs\encoder_features_normalized_v001\features_hier_z_expert_prelogit.normalized.npz | 12944 | 512 | ['board', 'material', 'split_group'] | [] | True | False |

## Recovered fields

- v6.2-A `embedding_point`: MCDropoutClassifier.global_pool.pre_fc (embedding_layer)
- v6.2-A `checkpoint_sha256`: Computed SHA256 from <PRIVATE_ESPI_ROOT>\FIXED_PACKAGE\baseline_v62_5class\checkpoints\checkpoint_epoch25_20260211_035150.pt. (checkpoint_path)
- hierarchical_v6.2_phase2_expert `board`: Recovered 12944 values from manifest path match. (manifest)
- hierarchical_v6.2_phase2_expert `material`: Recovered 143 values from manifest path match. (manifest)
- hierarchical_v6.2_phase2_expert `split_group`: Recovered 12944 values from manifest path match. (manifest)

## Unrecoverable fields

- None

## Final-publication rule

- v6.2-A normalized dump may be acceptable for final tables if checkpoint SHA is computed or the checkpoint path is documented.
- Hierarchical normalized dump is acceptable for internal comparison, but not final publication tables until extraction is regenerated with correct `board` and `split_group` directly stored in the source NPZ.

## Output files

- `outputs\encoder_features_normalized_v001\features_v62a_epoch25.normalized.npz`
- `outputs\encoder_features_normalized_v001\features_hier_z_expert_prelogit.normalized.npz`
