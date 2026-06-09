# H3 LeFFT Auxiliary Development Track Status

## Purpose

H3 tested a LeFFT auxiliary branch as a Phase-1 self-supervised / physics-aware pretraining component after H2 standalone CE-only LeFFT failed grouped LOBO thresholds.

## Relation to H2 and v6.2-A

- H2 showed standalone CE-only clean LeFFT variants did not pass grouped transfer gates.
- v6.2-A remains the frozen anchor/baseline.
- H3.1 evaluated only the standalone auxiliary branch mechanics and representation audits.
- H3.2 fusion remains blocked unless a no-harm design and transfer-positive evidence exist.

## Timeline Summary

| Stage | Outcome |
|---|---|
| H3.0 | Anchor-assisted LeFFT design scaffold; v6.2-A frozen anchor role defined; no checkpoint loaded. |
| H3.1A | Synthetic-only Phase-1 LeFFT auxiliary mechanics/loss tests passed. |
| H3.1B | Controlled real-data Phase-1 plan and dataloader readiness created. |
| H3.1C | Guarded 3000-image real-data mechanics pilot completed. |
| H3.1D | H3.1C verified as mechanics-valid. |
| H3.1E/F | 6000-image scaled Phase-1 mechanics PASS; representation weak/near-static. |
| H3.1G | G0/G1/G2/G3 sensitivity pilot; G1 selected with small partial signal. |
| H3.1H/I | Scaled G1 mechanics PASS; representation partial but borderline. |
| H3.1J | Objective redesign pilot; J1 multiview best non-diagnostic condition. |
| H3.1K | Scaled J1 mechanics PASS but negative for cross-board transfer. |

## Final Frozen H3 Decision

- H3 mechanics overall: PASS.
- H3 safety overall: PASS.
- H3 representation transfer overall: insufficient_or_negative.
- H3.1L ready: false.
- H3.2 design-only ready: true.
- H3.2 run ready: false.
- No further simple H3.1 scaling is recommended.

## H3.1K Key Result

H3.1K scaled J1 multiview training completed on 6000 real images for 20 epochs and 3760 optimizer steps. It used no v6.2-A checkpoint/embeddings, no fusion, no acoustic targets, no CE/SupCon/frequency auxiliary, and no class/board/material/frequency_hz labels in the loss.

Metrics:

- NMI board: 0.508168 -> 0.508168
- NMI class: 0.410973 -> 0.410973
- NMI frequency_bin: 0.756642 -> 0.756642
- same-board NN: 0.595117 -> 0.626367, delta +0.031250
- same-class NN: 0.486328 -> 0.494336, delta +0.008008
- cross-board same-class NN: 0.460498 -> 0.454022, delta -0.006476

Interpretation: scaled J1 strengthened board-local consistency and hurt cross-board same-class geometry. It is not transfer-positive.

## Supported Claims

- Stable LeFFT auxiliary Phase-1 training on real ESPI images is feasible.
- H3 guardrails and audit infrastructure are functional.
- Reconstruction/structure/multiview objectives can be trained without severe collapse in the completed scaled runs.
- Multiview J1 scaled training increases board-local consistency.
- J1 does not improve cross-board same-class geometry in the scaled run.
- H3.2 must remain blocked pending no-harm fusion design and stronger transfer-positive evidence.

## Unsupported Claims

- LeFFT improves frozen v6.2-A.
- H3.1 objectives improve grouped transfer.
- Board invariance achieved.
- Physics-informed representation validated.
- Acoustic-response prediction tested.
- H3.2 fusion ready to run.
- H3.1K should be scaled further.

## Local Reports

Consolidated H3 reports are indexed at `experiments/clean_lefft_encoder_v001/reports/h3_track_consolidation_v001`.
