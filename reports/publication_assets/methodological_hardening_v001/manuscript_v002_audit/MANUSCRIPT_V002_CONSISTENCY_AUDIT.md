# Manuscript v002 Consistency Audit

## Inputs

- Manuscript: `manuscripts\FROZEN_REPRESENTATION_AUDITS_DRAFT_v002.md`
- Hardening directory: `reports\publication_assets\methodological_hardening_v001`

## Source Artifact Availability

| artifact | found |
| --- | --- |
| patch_plan | True |
| patch_key_numbers | True |
| dataset_composition | True |
| paired_differences | True |
| frequency_only | True |
| frequency_fusion | True |
| knn_protocol | True |

## Section Status Summary

| section | status | n_pass | n_warn | n_fail |
| --- | --- | --- | --- | --- |
| Claim boundary | pass | 9 | 0 | 0 |
| Dataset composition | pass | 6 | 0 | 0 |
| Frequency controls | pass | 9 | 0 | 0 |
| Hierarchical caveat | pass | 4 | 0 | 0 |
| Material LOMO caution | pass | 2 | 0 | 0 |
| Paired v6.1 vs v6.2-A | pass | 6 | 0 | 0 |
| Source artifacts | pass | 7 | 0 | 0 |
| Tables and figures | pass | 15 | 0 | 0 |
| kNN protocol | pass | 6 | 0 | 0 |

## Detailed Checks

| section | check | status | evidence | recommendation |
| --- | --- | --- | --- | --- |
| Source artifacts | Source artifact `patch_plan` present | pass | found |  |
| Source artifacts | Source artifact `patch_key_numbers` present | pass | found |  |
| Source artifacts | Source artifact `dataset_composition` present | pass | found |  |
| Source artifacts | Source artifact `paired_differences` present | pass | found |  |
| Source artifacts | Source artifact `frequency_only` present | pass | found |  |
| Source artifacts | Source artifact `frequency_fusion` present | pass | found |  |
| Source artifacts | Source artifact `knn_protocol` present | pass | found |  |
| Claim boundary | v6.2-A described as strongest image-derived frozen ESPI encoder baseline | pass | phrase present | Use the exact phrase `strongest image-derived frozen ESPI encoder baseline` near the abstract/introduction claim. |
| Claim boundary | No positive strongest overall predictor | pass | absent or explicitly negated: v6.2-A is the strongest overall predictor |  |
| Claim boundary | No positive acoustic-response prediction claim | pass | absent or explicitly negated: acoustic-response predictive value |  |
| Claim boundary | No positive validated Physics-Aligned Encoder claim | pass | absent or explicitly negated: validated Physics-Aligned Encoder |  |
| Claim boundary | No positive LeFFT superiority claim | pass | absent or explicitly negated: LeFFT superiority |  |
| Claim boundary | No positive retrained CNN LOBO/LOMO generalization claim | pass | absent or explicitly negated: retrained CNN LOBO/LOMO generalization |  |
| Claim boundary | Frequency metadata is dominant/strong metadata control | pass | phrases found |  |
| Claim boundary | No simple modal-label necessity claim | pass | phrases found |  |
| Claim boundary | Acoustic response outside scope/future | pass | phrases found |  |
| kNN protocol | Primary fixed k=10 protocol stated | pass | phrases found |  |
| kNN protocol | v6.1 fixed k=10 stratified value present | pass | phrases found |  |
| kNN protocol | v6.2-A fixed k=10 stratified value present | pass | phrases found |  |
| kNN protocol | v6.1 best-k diagnostic value present | pass | phrases found |  |
| kNN protocol | Best-k diagnostic/sensitivity not primary | pass | phrases found |  |
| kNN protocol | Grouped kNN locked at k=10 | pass | phrases found |  |
| Dataset composition | Total sample count present | pass | phrases found |  |
| Dataset composition | Six boards present | pass | phrases found |  |
| Dataset composition | Two materials present | pass | phrases found |  |
| Dataset composition | All boards contain all five classes | pass | phrases found |  |
| Dataset composition | All materials contain all five classes | pass | phrases found |  |
| Dataset composition | No LOBO/LOMO missing held-out/reference classes | pass | phrases found |  |
| Material LOMO caution | Material LOMO descriptive/stress-test caveat | pass | caveat present | State that Material LOMO is descriptive/material-held-out stress test because only two materials exist. |
| Material LOMO caution | Material LOMO not over-interpreted | pass | no positive broad-population interpretation detected |  |
| Paired v6.1 vs v6.2-A | Board LOBO delta present | pass | phrases found |  |
| Paired v6.1 vs v6.2-A | Material LOMO delta present | pass | phrases found |  |
| Paired v6.1 vs v6.2-A | v6.2-A wins board folds | pass | phrases found |  |
| Paired v6.1 vs v6.2-A | v6.2-A wins material folds | pass | phrases found |  |
| Paired v6.1 vs v6.2-A | C02 near tie present | pass | phrases found |  |
| Paired v6.1 vs v6.2-A | W01 small margin present | pass | phrases found |  |
| Frequency controls | Frequency-only stratified present | pass | phrases found |  |
| Frequency controls | Frequency-only Board LOBO present | pass | phrases found |  |
| Frequency controls | Frequency-only Material LOMO present | pass | phrases found |  |
| Frequency controls | Frequency + v6.2-A stratified present | pass | phrases found |  |
| Frequency controls | Frequency + v6.2-A Board LOBO present | pass | phrases found |  |
| Frequency controls | Frequency + v6.2-A Material LOMO present | pass | phrases found |  |
| Frequency controls | frequency_hz dominant metadata predictor | pass | phrases found |  |
| Frequency controls | Fusion improves over embedding-only | pass | phrases found |  |
| Frequency controls | Fusion does not clearly exceed frequency-only except Material LOMO | pass | phrases found |  |
| Hierarchical caveat | Hierarchical technically valid | pass | phrases found |  |
| Hierarchical caveat | Hierarchical not superior | pass | phrases found |  |
| Hierarchical caveat | Hierarchical internal-only metadata caveat | pass | phrases found |  |
| Hierarchical caveat | Hierarchical not validated physics-aligned encoder | pass | phrases found |  |
| Tables and figures | Table 1 listed/referenced | pass | present |  |
| Tables and figures | Table 2 listed/referenced | pass | present |  |
| Tables and figures | Table 3 listed/referenced | pass | present |  |
| Tables and figures | Table 4 listed/referenced | pass | present |  |
| Tables and figures | Table 5 listed/referenced | pass | present |  |
| Tables and figures | Table 6 listed/referenced | pass | present |  |
| Tables and figures | Table 7 listed/referenced | pass | present |  |
| Tables and figures | Table 8 listed/referenced | pass | present |  |
| Tables and figures | Supplementary Table S1 listed/referenced | pass | present |  |
| Tables and figures | Supplementary Table S2 listed/referenced | pass | present |  |
| Tables and figures | Supplementary Table S3 listed/referenced | pass | present |  |
| Tables and figures | Supplementary Table S4 listed/referenced | pass | present |  |
| Tables and figures | Supplementary Table S5 listed/referenced | pass | present |  |
| Tables and figures | Supplementary Table S6 listed/referenced | pass | present |  |
| Tables and figures | Unresolved placeholders | pass | none |  |

## Required Fixes / Warnings

_No rows._

## Unresolved Placeholders

- None detected.

## Readiness Verdict

- Overall audit status: `pass`.
- Ready for scientific language editing: `True`.
- Ready for journal formatting: `True`.

If the status is `pass`, v002 correctly reflects the final methodological hardening decisions and can proceed to scientific language editing. Journal formatting should wait until planned Tables 7-8 and Supplementary Tables S5-S6 are materialized.