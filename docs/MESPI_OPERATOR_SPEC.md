# M_ESPI Operator Specification

`M_ESPI` is the planned probabilistic differentiable ESPI observation/measurement operator for the v32 PhD direction. It is not yet a validated model.

## Inputs

- `W(x,y,f)`: displacement or response field at spatial coordinate `(x,y)` and excitation coordinate `f`.
- `A(x,y)`: optical background / mean intensity field.
- `B(x,y)`: fringe visibility / contrast field.
- `c`: sensitivity coefficient mapping displacement amplitude to optical phase argument.
- optional nuisance parameters: illumination drift, speckle gain, sensor noise, saturation, and mask/ROI support.

## Time-averaged ESPI Kernel

The core observation model uses the time-averaged ESPI Bessel kernel:

```text
I(x,y,f) = A(x,y) + B(x,y) * J0(c * W(x,y,f))^2 + noise
```

where `J0` is the zeroth-order Bessel function of the first kind. The exact parameterization may include normalization, phase offsets, visibility attenuation, and masked ROI support, but any extension must preserve the measurement interpretation.

## Probabilistic Observation Model

Candidate likelihoods:

- Gaussian likelihood for normalized intensity residuals.
- Heteroscedastic Gaussian likelihood with spatially varying variance.
- Poisson-like or variance-stabilized likelihood for photon-counting approximations.
- Robust likelihoods for saturation, speckle outliers, and imperfect masks.

The likelihood must expose uncertainty rather than treating rendered ESPI images as deterministic labels.

## Differentiability Requirements

`M_ESPI` must support:

- differentiable rendering from `W(x,y,f)` to expected intensity,
- stable gradients through `J0(c * W)^2`,
- bounded or regularized nuisance parameters,
- ROI-aware masking,
- batched execution for synthetic validation and representation learning.

## Validation Gates

1. Unit tests for the `J_0^2` kernel against known values.
2. Synthetic field re-render tests with known `W`, `A`, `B`, and `c`.
3. Noise/likelihood calibration tests.
4. Gradient checks and finite-loss checks.
5. Comparison against SLDV/FEM/reference displacement fields where available.
6. Failure-mode report for non-identifiability between `A`, `B`, `c`, and `W`.

## Relation to Self-supervised Learning

`M_ESPI` enables encode-field-re-render self-supervision:

```text
image -> encoder -> latent z_ESPI^MC -> field/response decoder -> M_ESPI -> reconstructed observation
```

This is a future measurement-consistent path. It does not by itself validate morphology recognition or response prediction; validation still requires independent targets and B0-B6 residual testing.
