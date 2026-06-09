# H2 Track Consolidation Index

This index archives the H2 standalone clean LeFFT development track as development-only diagnostic evidence.

| Stage | Purpose | Final decision | Local report path |
|---|---|---|---|
| H2.0 | Design fair clean LeFFT / physics-informed encoder protocol. | Design complete; no training. | `experiments/clean_lefft_encoder_v001/reports/h2_0_clean_lefft_physics_encoder_design/` |
| H2.1 | Implement LeFFT module skeleton and synthetic smoke tests. | Module mechanics passed. | `experiments/clean_lefft_encoder_v001/reports/h2_1_lefft_module_skeleton*/` |
| H2.2 | Implement/unit-test physics/self-supervised losses. | Synthetic losses and gradients passed. | `experiments/clean_lefft_encoder_v001/reports/h2_2_physics_loss_unit_tests/` |
| H2.3 | Audit pair statistics for frequency-controlled SupCon/domain-aware training. | H2.4 CE-only ready; SupCon pair support ready. | `experiments/clean_lefft_encoder_v001/reports/h2_3_pair_statistics_audit/` |
| H2.4B | Reduced valid CE-only LeFFT LOBO go/no-go. | FAIL; K=8 LOBO Macro-F1 0.248644. | `experiments/clean_lefft_encoder_v001/reports/h2_4b_reduced_ce_only_lefft_go_no_go/` |
| H2.1B | Architecture redesign after H2.4B. | v002/GroupNorm/residual/hybrid-mask design created and smoke-tested. | `experiments/clean_lefft_encoder_v001/reports/h2_1b_lefft_architecture_redesign/` |
| H2.4C | Factorial repair: S0/S1/S2/S3. | All failed or fail-bound locked; best S2_v002_W0 LOBO Macro-F1 0.294294. | `experiments/clean_lefft_encoder_v001/reports/h2_4c_final_factorial_postmortem/` |

## Authoritative Final Decision

Standalone CE-only LeFFT grouped transfer failed. The best H2.4C condition was S2_v002_W0 with LOBO Macro-F1 0.294294 versus the predefined 0.608 threshold. H2.5 remains blocked.

Detailed reports may remain local if they are too large or contain private paths/metadata. This index does not include checkpoints, raw ESPI data, private manifests, or feature dumps.
