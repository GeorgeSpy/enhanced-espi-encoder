# v003 Language Edit Report

## Source and Output

- Source manuscript: `manuscripts/FROZEN_REPRESENTATION_AUDITS_DRAFT_v002_spectral_control.md`
- Output manuscript: `manuscripts/FROZEN_REPRESENTATION_AUDITS_DRAFT_v003_OLE_style.md`
- Target style: Optics and Lasers in Engineering / optical metrology / engineering-facing scientific machine learning.

## Sections Edited

- Abstract
- Introduction
- Contributions
- Methods
- Results
- Discussion
- Limitations and Future Work
- Conclusion
- Data, Code, and Reproducibility Availability
- Tables and Figures map

## Editing Actions

- Improved ESPI and full-field optical metrology framing.
- Reduced repetition of the central claim while keeping claim boundaries explicit.
- Reorganized language around the core trade-off: v6.1 is slightly stronger under fixed-k stratified kNN, while v6.2-A is stronger under grouped board/material image-derived evaluation.
- Clarified that frequency metadata defines the information budget for the current five-class modal-label protocol.
- Kept representation-learning terminology precise but less ML-centric.
- Kept deterministic spectral / LeFFT-inspired descriptors as supplementary/control-only evidence.
- Preserved all table and figure references, including Supplementary Table S7.

## Metrics Preserved

The following locked values were checked and are present in v003:

- v6.1 fixed `k=10` stratified Macro-F1: `93.65%`
- v6.2-A fixed `k=10` stratified Macro-F1: `92.78%`
- v6.2-A Board LOBO Macro-F1: `93.56%`
- v6.2-A Material LOMO Macro-F1: `93.49%`
- v6.1 best-k diagnostic: `95.07%`
- v6.1 Board LOBO Macro-F1: `87.10%`
- v6.1 Material LOMO Macro-F1: `86.33%`
- Board LOBO paired delta: `+6.47 pp`
- Material LOMO paired delta: `+7.16 pp`
- C02 near tie: `-0.02 pp`
- W01 small margin: `+0.27 pp`
- `2_1` grouped class gain: `+13.43 percentage points`
- `1_2` grouped class gain: `+12.21 percentage points`
- `higher` support: `10,115` samples / `78.14%`
- Total samples: `12,944`
- Frequency-only Macro-F1: `98.67%`, `97.49%`, `94.09%`
- Frequency + v6.2-A Macro-F1: `96.75%`, `94.79%`, `94.50%`
- Spectral-only Macro-F1: `52.92%`, `22.77%`, `19.72%`
- Frequency + v6.2-A + spectral deltas: `+0.19 pp`, `+0.14 pp`, `-0.33 pp`

## Claim Boundaries Preserved

- v6.2-A is described as the strongest image-derived frozen ESPI encoder baseline.
- `frequency_hz` is described as the dominant metadata-only predictor for the current five-class modal-label protocol.
- v6.2-A is not described as the strongest overall predictor when frequency metadata is allowed.
- No acoustic-response prediction is claimed.
- No validated Physics-Aligned Encoder is claimed.
- No LeFFT superiority is claimed.
- The spectral descriptor ablation is deterministic, supplementary/control-only, and not a trained LeFFT model.
- Material LOMO is described as a descriptive material-held-out stress test because only two material groups exist.
- Hierarchical v6.2 phase2 is described as technically valid but not superior and remains a controlled comparison.

## Validation Summary

- Locked metric scan: passed.
- Sections present: passed.
- Tables 1-8 and Supplementary Tables S1-S7 remain referenced.
- Figures 1-6 remain referenced.
- No new experiments were added.
- No scripts, feature dumps, tables, figures, or evaluation reports were modified by the language-edit step.

## Unresolved Issues

- None requiring scientific revision.
- Venue-specific formatting, reference style, figure caption style, and final supplementary packaging remain to be handled during journal formatting.

## Readiness Verdict

v003 is ready for journal-formatting preparation and final author review for an Optics and Lasers in Engineering style submission.
