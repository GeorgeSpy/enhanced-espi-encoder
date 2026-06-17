# WP2 Response-target Screening

WP2 must start with target feasibility and baseline screening, not model training.

## Candidate Targets

- measured `TL(f)` or transmission-loss curves,
- FRF / response curves,
- damping,
- Q-factor,
- bandwidth,
- broadband or transient response descriptors,
- radiated-field descriptors,
- residual response after frequency-coordinate regression.

## Required Paired Measurements

Each candidate target requires:

- sample or condition identifier aligned to ESPI data,
- excitation coordinate or frequency grid,
- board/specimen/material metadata,
- acquisition session or campaign metadata where available,
- uncertainty or replicate information where possible.

## Split Requirements

Preferred grouped-OOD splits:

- board-held-out,
- material-held-out only when material provenance is clean,
- specimen-held-out,
- acquisition-campaign-held-out,
- condition-held-out.

Random splits are diagnostic only and cannot support grouped generalization claims.

## Baseline Gates

Before training an ESPI model, run:

- frequency-coordinate baseline,
- metadata baseline,
- FEM/numerical descriptor baseline,
- mean-response or smooth-curve baseline,
- hand-crafted ESPI descriptor baseline if images are used.

## B6 Residual Target

B6 asks whether ESPI information predicts residual response beyond:

1. frequency coordinate,
2. metadata,
3. numerical/FEM descriptors,
4. hand-crafted optical descriptors,
5. direct embedding/fusion baselines.

## Go / No-go Thresholds

A target is viable only if:

- frequency/metadata baselines do not already saturate the target,
- grouped-OOD splits have sufficient support,
- response residuals are stable enough to model,
- ESPI-derived predictors improve residual metrics with confidence intervals,
- leakage and repeated-measure contamination are controlled.

If frequency alone explains the target, the target is not suitable for a strong ESPI representation claim.
