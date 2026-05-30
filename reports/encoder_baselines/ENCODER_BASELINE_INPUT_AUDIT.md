# Encoder Baseline Input Audit

## Scope

This report audits whether the inputs required for a unified encoder baseline comparison are available. It does not train models, extract embeddings, run evaluations, implement LeFFT scripts, or implement acoustic-response prediction.

## Target encoders

1. Random ResNet-18 embeddings.
2. ImageNet-pretrained ResNet-18 embeddings.
3. v6.1 embeddings, if checkpoint/model is available.
4. Existing v6.2-A frozen embeddings.
5. Hierarchical v6.2 phase2 expert embeddings.

## Availability matrix

| Encoder | Required artifact | Found? | Path | Status | Blocking issue |
|---|---|---:|---|---|---|
| all baselines | manifest file with expected metadata | yes | <LOCAL_WORKSPACE_ROOT>\ESPI_v62_v62A_Encoder_package_20260430_224510\core_v62_files\manifest\manifest_v1_5class.npz | incomplete | Missing fields: material, split_group |
| random ResNet-18 embeddings | image paths / root | yes | <PRIVATE_IMAGE_ROOT> | available |  |
| ImageNet-pretrained ResNet-18 embeddings | image paths / root | yes | <PRIVATE_IMAGE_ROOT> | available |  |
| random ResNet-18 embeddings | baseline extraction script / loader | yes | scripts\encoder\extract_v62_embeddings.py | available |  |
| ImageNet-pretrained ResNet-18 embeddings | baseline extraction script / loader | yes | scripts\encoder\extract_v62_embeddings.py | available |  |
| v6.1 embeddings | v6.1 model/checkpoint | yes | <PRIVATE_CHECKPOINT_ROOT>\espi_modes_v6_1_head.pt | available |  |
| v6.1 embeddings | v6.1 model/training script | yes | scripts\v6_1\train_v6_1_head_only.py | available |  |
| v6.2-A frozen embeddings | existing v6.2-A feature dump | yes | <LOCAL_WORKSPACE_ROOT>\v62_encoder_evidence\features_v62a_epoch25.npz | usable-with-caveat | Metadata issue requires regeneration before final publication tables. |
| hierarchical v6.2 phase2 expert embeddings | existing hierarchical feature dump | yes | outputs\hierarchical_embeddings_v001\features_hier_z_expert_prelogit.npz | usable-with-caveat | Hierarchical NPZ stored board/split_group as unknown; acceptable for internal comparison after recovery from distribution_group/path, but must be regenerated before final publication tables. |
| hierarchical v6.2 phase2 expert embeddings | hierarchical phase2 checkpoint | yes | <PRIVATE_LOG_ROOT>\train_v6.2_opt\ckpt_phase2_expert.pt | available |  |
| required scripts/loaders | random_imagenet_extraction | yes | scripts\encoder\extract_v62_embeddings.py | available |  |
| required scripts/loaders | v62a_audit | yes | scripts\encoder\audit_v62_embeddings.py | available |  |
| required scripts/loaders | hierarchical_wrapper | yes | src\enhanced_espi_encoder\v62_hierarchical_encoder.py | available |  |
| required scripts/loaders | hierarchical_extraction | yes | scripts\encoder\extract_v62_hierarchical_embeddings.py | available |  |
| required scripts/loaders | grouped_evaluation | yes | scripts\encoder\evaluate_encoder_grouped_generalization.py | available |  |
| required scripts/loaders | v61_training_script | yes | scripts\v6_1\train_v6_1_head_only.py | available |  |

## Feature dump checks

### v6.2-A

- Feature dump exists: `True`
- Path: `<LOCAL_WORKSPACE_ROOT>\v62_encoder_evidence\features_v62a_epoch25.npz`
- Samples: `12944`
- Embedding dimension: `1280`
- Missing expected fields: `none`
- Metadata issues: `checkpoint SHA256 is missing; checkpoint_path is available instead, embedding_point is missing; embedding_layer is available instead`

### Hierarchical v6.2 phase2 expert

- Feature dump exists: `True`
- Path: `outputs\hierarchical_embeddings_v001\features_hier_z_expert_prelogit.npz`
- Samples: `12944`
- Embedding dimension: `512`
- Missing expected fields: `none`
- Metadata issues: `board is stored as unknown for nearly all samples, split_group is stored as unknown for nearly all samples`

## Known metadata caveat

The hierarchical NPZ stored `board` and `split_group` as `unknown`. Previous audits recovered those fields from `distribution_group` and `path`. This is acceptable for internal comparison, but the hierarchical feature dump must be regenerated with correct `board` and `split_group` metadata before final publication tables.

## Blocking issues

- None

## Non-blocking caveats

- all baselines: manifest file with expected metadata (Missing fields: material, split_group)
- v6.2-A frozen embeddings: existing v6.2-A feature dump (Metadata issue requires regeneration before final publication tables.)
- hierarchical v6.2 phase2 expert embeddings: existing hierarchical feature dump (Hierarchical NPZ stored board/split_group as unknown; acceptable for internal comparison after recovery from distribution_group/path, but must be regenerated before final publication tables.)

## Decision

- Existing v6.2-A and hierarchical reports can be used for report-level comparison.
- No blocking missing artifact was found for internal baseline-comparison preparation.
- Unified baseline extraction should still wait for an explicit implementation step.
- Publication-grade tables require metadata cleanup/regeneration for the caveats listed above.
- v6.1 should be included only if a compatible model/checkpoint is available.
- Random and ImageNet ResNet-18 baselines require image-root and manifest availability before extraction.
