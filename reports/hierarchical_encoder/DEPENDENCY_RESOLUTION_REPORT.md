# Hierarchical v6.2 Dependency Resolution Report

## Scope

This report documents the safe dependency resolution required to import the legacy physics components used by `scripts/v6_2/espi_v6_2_fixed.py` for hierarchical v6.2 checkpoint load-auditing.

No training was run. No embedding extraction was run. Placeholder physics modules were not allowed.

## Original Failure

The hierarchical load audit initially failed with:

```text
BASE_IMPORTED=False
Model file fell back to placeholder physics modules; refusing to instantiate for a scientific load audit.
```

The immediate failed import was:

```text
from balanced_batch_sampler_v2 import BalancedBatchSampler
```

inside the legacy pipeline file:

```text
ESPI_v62_v62A_Encoder_package_20260430_224510/core_v62_files/enhanced_espi_pipeline.py
```

After resolving that import, a second training-only helper import was exposed:

```text
from checkpointing import save_checkpoint
```

## Required Symbols

The missing training/data-loader symbols were:

| Module | Required symbol | Runtime role |
|---|---|---|
| `balanced_batch_sampler_v2` | `BalancedBatchSampler` | Training/data-loader helper |
| `balanced_batch_sampler_v2` | `__version__` | Sampler logging/version metadata |
| `checkpointing` | `save_checkpoint` | Training checkpoint save helper |

These symbols are not used for hierarchical model instantiation or dummy forward inference.

## Resolution

No original `balanced_batch_sampler_v2.py` file was found in the local searched project trees:

```text
<LOCAL_WORKSPACE_ROOT>
<PRIVATE_ESPI_ROOT>
```

Therefore, minimal compatibility shims were added under:

```text
external/compat/balanced_batch_sampler_v2.py
external/compat/checkpointing.py
```

The shims are intentionally fail-closed:

- `BalancedBatchSampler(...)` raises `RuntimeError` if instantiated.
- `save_checkpoint(...)` raises `RuntimeError` if called.

This prevents accidental training or data-loader construction through compatibility code.

## Why This Is Safe for Load Audit / Inference

The hierarchical load audit only needs to:

1. import the real physics/model classes,
2. instantiate `EnhancedHybridPhysicsESPI_V6_2`,
3. load an external checkpoint,
4. optionally run one dummy forward pass.

It does not construct training data loaders and does not save checkpoints. Therefore the shimmed symbols are sufficient for import-time compatibility while still failing loudly if used for training.

## Final Audit Result

The hierarchical load audit was rerun with:

```text
PYTHONPATH=external/compat;<legacy core_v62_files path>
checkpoint=<PRIVATE_LOG_ROOT>\train_v6.2_opt\ckpt_phase2_expert.pt
```

Result:

| Field | Value |
|---|---:|
| `BASE_IMPORTED` | `True` |
| Placeholder physics modules used | `False` |
| Placeholder physics modules still refused by audit script | `True` |
| Load mismatch status | `clean` |
| Matched keys | `237` |
| Missing keys | `0` |
| Unexpected keys | `0` |
| Shape mismatches | `0` |
| Dummy forward OK | `True` |
| `outputs["embeddings"]` available | `True` |
| Physics outputs available | `denoised`, `wrapped_phase`, `unwrapped_phase`, `amplitude` |
| Still unavailable from forward dict | `physics_feats`, `spectral_filter` |

The clean load-audit report is:

```text
reports/hierarchical_encoder/LOAD_AUDIT_V62_HIERARCHICAL.md
reports/hierarchical_encoder/load_audit_v62_hierarchical.json
```

## Conclusion

The dependency blocker is resolved safely for load-audit and dummy-forward inference. The hierarchical checkpoint `ckpt_phase2_expert.pt` is compatible with `EnhancedHybridPhysicsESPI_V6_2` under the real physics component import path. The next step can be a dedicated hierarchical embedding extractor using `outputs["embeddings"]` as `z_expert_prelogit / z_arcface_input`, while keeping `z_physics_fusion` as a later extension.
