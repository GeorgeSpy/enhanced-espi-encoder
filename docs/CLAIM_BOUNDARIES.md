# Claim Boundaries

This document defines the supported, unsupported, and development-only scientific claims for the Enhanced ESPI Encoder repository.

The repository contains both a locked OLEN manuscript evidence release and active development work. The locked manuscript snapshot is tagged as `v0.9-olen-pre-submission-evidence`. The `main` branch may evolve beyond that snapshot.

## Supported Claims in the Locked OLEN Evidence Release

| Claim | Status | Evidence scope |
|---|---|---|
| v6.2-A is the strongest evaluated image-derived frozen ESPI representation under grouped board/material evaluation. | supported | unified encoder baseline evaluation, paired board/material deltas, fixed k=10 protocol |
| `frequency_hz` is the dominant metadata-only predictor for the present five-class modal-label task. | supported | frequency-only and frequency-fusion controls |
| v6.2-A adds complementary morphology information in localized frequency-ambiguous `1_2` / `2_1` cases. | supported as targeted diagnostic | overlap-pair and frequency-residual correction analyses |
| v6.2-A grouped advantage persists under fold-local PCA dimension matching. | supported | PCA projections to 512, 256, and 128 dimensions |
| v6.2-A grouped advantage persists under balanced class/material/frequency subset controls. | supported | class-balanced, class-material-balanced, and class-material-frequency-matched subset evaluation |
| Quantitative embedding geometry supports stronger class structure for v6.2-A than v6.1. | supported | Silhouette, Davies-Bouldin, intra/inter distance ratio, same-class nearest-neighbor consistency |
| Hierarchical v6.2 phase2 is technically valid but not superior as a frozen representation source. | supported | hierarchical load audit, extraction audit, and frozen embedding comparison |
| Deterministic FFT/spectral descriptors do not support a LeFFT superiority claim. | supported as supplementary control | spectral descriptor extraction and ablation evaluation |

## Unsupported or Out-of-Scope Claims

| Claim | Status |
|---|---|
| Validated Physics-Aligned ESPI Encoder | not claimed |
| Acoustic-response prediction | future work |
| LeFFT superiority | not claimed |
| Neural operator, PNO, FNO, or DeepONet implementation | not included |
| Full retrained CNN LOBO/LOMO generalization | not claimed |
| External-laboratory generalization | not yet evaluated |
| Metamaterials-ready encoder | not claimed |

## Architecture-Specific Boundaries

- v6.1 is a reference frozen ResNet18-style baseline.
- v6.2-A is the main reportable image-derived frozen ESPI encoder candidate in the locked OLEN evidence release.
- v6.2-A is not LeFFT-based and is not the hierarchical physics-aware branch.
- Hierarchical v6.2 phase2 is a controlled physics-aware architecture comparison, not a validated Physics-Aligned Encoder.
- Deterministic spectral descriptors are LeFFT-inspired controls, not a trained LeFFT model.

## Frequency Information Budget

The present five-class modal-label task is strongly frequency-structured. Frequency-only metadata controls exceed image-derived embeddings for the current label protocol. This does not invalidate the image-derived representation audit; it defines the information budget and motivates frequency-controlled, frequency-residual, or acoustic-response targets for future studies.

## Development Claims vs Manuscript Claims

This repository contains both locked manuscript evidence and active development work.

A result is not considered manuscript-supported unless it has:

- a reproducible script,
- a sanitized report,
- matched baseline comparison,
- grouped validation where applicable,
- claim-boundary entry,
- release/tag association.

Development branches may contain exploratory experiments that do not yet support manuscript claims. Exploratory code, negative results, prototype reports, or partial validations must not be promoted to manuscript-level claims until they satisfy the claim-promotion requirements above.

## Manuscript-Safe Central Claim

The locked OLEN manuscript may state that v6.2-A is the strongest evaluated image-derived frozen ESPI representation under grouped board/material evaluation, while `frequency_hz` is the dominant metadata-only predictor for the current five-class modal-label protocol.

Future manuscripts may promote additional claims only after the evidence is added to this document and associated with a release tag.
