# Spectral Descriptor Manuscript Patch Report

## Inputs

- Manuscript source: `manuscripts\FROZEN_REPRESENTATION_AUDITS_DRAFT_v002.md`
- Spectral ablation report: `reports/publication_assets/methodological_hardening_v001/lefft_descriptor_ablation/eval_v001/SPECTRAL_DESCRIPTOR_ABLATION_REPORT.md`
- Spectral ablation key numbers: `reports\publication_assets\methodological_hardening_v001\lefft_descriptor_ablation\eval_v001\spectral_descriptor_ablation_key_numbers.json`

## Outputs

- Updated manuscript: `manuscripts\FROZEN_REPRESENTATION_AUDITS_DRAFT_v002_spectral_control.md`
- Supplementary Table S7 Markdown: `reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S7_SPECTRAL_DESCRIPTOR_CONTROL.md`
- Supplementary Table S7 CSV: `reports\publication_assets\final_tables_v001\SUPPLEMENTARY_TABLE_S7_SPECTRAL_DESCRIPTOR_CONTROL.csv`
- Updated table index: `reports\publication_assets\final_tables_v001\TABLE_INDEX.md`

## Manuscript changes

- Added a short Limitations/Future Work paragraph stating that deterministic spectral / LeFFT-inspired descriptors were evaluated as a lightweight supplementary control.
- Stated that spectral-only descriptors were not competitive.
- Stated that adding spectral descriptors to frequency + v6.2-A produced only marginal grouped gains: `+0.19 pp` Board LOBO and `+0.14 pp` Material LOMO, while reducing stratified Macro-F1 by `-0.33 pp`.
- Added `Supplementary Table S7` to the supplementary table plan.

## Claim boundary

- Title unchanged.
- Abstract unchanged.
- Main Results claim unchanged.
- v6.2-A remains the strongest image-derived frozen ESPI encoder baseline.
- Frequency-only interpretation unchanged.
- No trained LeFFT model is claimed.
- No LeFFT superiority is claimed.
- No acoustic-response prediction is claimed.
- No experiments were run by this patch step.
- No scripts were modified by this patch step.
