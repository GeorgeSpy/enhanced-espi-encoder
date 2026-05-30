# Class-Level Diagnostics Summary

## Scope
- This is a class-level diagnostic export for the frozen encoder baseline study.
- No deep model training, fine-tuning, LeFFT, or acoustic-response prediction was run.
- Lightweight kNN, prototype, and logistic-regression heads were rerun on frozen embeddings only where per-sample predictions were not already stored.

## Dataset support
- Total samples: `12944`
- Smallest class: `1_1H` with `653` samples.
- Largest class: `higher` with `10115` samples.

## Hardest classes
- `2_1`: mean grouped best-method F1 `39.95%`.
- `1_2`: mean grouped best-method F1 `41.12%`.
- `1_1H`: mean grouped best-method F1 `49.46%`.
- `1_1T`: mean grouped best-method F1 `52.48%`.
- `higher`: mean grouped best-method F1 `87.60%`.

## Manuscript interpretation
- v6.2-A improves grouped class stability over v6.1 primarily because its board/material grouped aggregate class F1 remains higher and more balanced.
- Random/ImageNet baselines can show high stratified kNN scores, but their grouped per-class failures confirm that stratified texture similarity is not sufficient for board/material robustness.
- Hierarchical v6.2 phase2 failure pattern is `broad` under grouped evaluation, so it remains a controlled internal alternative rather than a superior encoder.
- These diagnostics support the frozen grouped embedding claim only; they do not establish acoustic-response value or a validated Physics-Aligned Encoder.
