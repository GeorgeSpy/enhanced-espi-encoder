# B0-B6 Information Budget

B0-B6 is an experimental hygiene ladder for response-target validation. It is not standalone methodological novelty.

## Baseline Ladder

| Level | Name | Inputs | Purpose |
|---|---|---|---|
| B0 | Frequency-coordinate / mean response baseline | frequency coordinate, mean target curve, or smooth response prior | establish unavoidable frequency structure |
| B1 | Metadata baseline | board, material, geometry, acquisition metadata where allowed | quantify non-image metadata signal |
| B2 | FEM / numerical descriptors | simulated modes, response descriptors, geometry descriptors | quantify numerical prior signal |
| B3 | Hand-crafted ESPI descriptors | Zernike/Fourier/wavelet/fringe descriptors from clean ROI images | quantify simple image-derived signal |
| B4 | `z_ESPI` embedding | `z_ESPI^ROI` or `z_ESPI^MC` | test learned representation signal |
| B5 | Fusion | frequency, metadata, numerical descriptors, and `z_ESPI` | test combined information budget |
| B6 | Residual prediction | residual target after B0-B5 components | decisive test for added ESPI value |

## Metrics

Use target-appropriate metrics:

- RMSE / MAE,
- R2 and grouped-OOD R2,
- calibration error for probabilistic targets,
- curve distance metrics,
- frequency-band summary errors,
- confidence intervals over grouped folds.

## Required Reporting

Every response-target experiment must report:

- sample counts and grouped split support,
- target missingness,
- B0-B6 metrics,
- fold-wise results,
- confidence intervals,
- residual plots,
- leakage controls,
- failure cases.

## Decision Rule

A representation claim requires B6 residual improvement under grouped-OOD validation. Higher raw prediction accuracy is insufficient if frequency, metadata, or numerical descriptors already explain the target.
