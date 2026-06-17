# WP1 Forensic Evidence Map

This file indexes the WP1 forensic evidence that reframed the historical OLEN/v6.2-A line.

| Evidence block | Artifact path | Status | Decision contribution |
|---|---|---|---|
| v62 frequency-label forensic audit | `experiments/clean_lefft_encoder_v001/reports/v62_frequency_label_forensic_audit_v001_retry02/` | completed | Frequency-only LOBO Macro-F1 approximately `0.932677`; no global beyond-frequency morphology support. |
| Cross-model forensic audit | `experiments/clean_lefft_encoder_v001/reports/cross_model_forensic_audit_v001_retry02/` | completed | No historical model beat frequency-only beyond threshold; v6.2-A remains historical baseline. |
| OLEN reframing package | `experiments/clean_lefft_encoder_v001/reports/olen_reframing_decision_v001/` | completed if present | Classifies current OLEN line as WP1/methods motivation rather than active strong submission. |
| OLEN line freeze package | `experiments/clean_lefft_encoder_v001/reports/olen_line_freeze_v001/` | completed | Freezes strong OLEN framing as deprecated for submission. |
| WP1-to-WP2 transition package | `experiments/clean_lefft_encoder_v001/reports/wp1_to_wp2_transition_v001/` | completed | Moves validation target toward independent response measurements. |
| Visual frequency-overlay and ROI-mask audit | `experiments/visual_domain_leakage_audit_v002_retry01/` | completed | Confirms actual-pixel overlays and high publication risk without cleaned ROI controls. |
| Clean ROI leakage-control rerun | `experiments/cleaned_roi_rerun_v001/` | completed | Removes/reduces patch leakage, but global morphology remains unsupported and exact-frequency diagnostics are underpowered. |
| H2/H3 consolidation | `experiments/clean_lefft_encoder_v001/reports/h3_track_consolidation_v001/`; H2 final reports | completed | LeFFT tracks remain archived development-only evidence. |

## WP1 Summary

WP1 is now best framed as a forensic/methodological evidence layer:

- historical v6.2-A baseline is useful,
- current modal labels are frequency-structured,
- visual/domain leakage controls are mandatory,
- no historical model supports global frequency-independent morphology recognition,
- the next publication-grade validation requires independent response targets.
