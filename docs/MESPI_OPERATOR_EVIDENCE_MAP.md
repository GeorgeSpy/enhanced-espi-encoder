# M_ESPI Operator Evidence Map

This file lists the evidence required before `M_ESPI` can support measurement-consistent ESPI representation claims.

| Future artifact | Required evidence | Status |
|---|---|---|
| Operator specification | Define input displacement/response field `W(x,y,f)`, optical background `A(x,y)`, visibility `B(x,y)`, sensitivity coefficient `c`, and time-averaged ESPI `J_0^2` kernel. | drafted in `docs/MESPI_OPERATOR_SPEC.md` |
| Bessel `J_0^2` kernel tests | Verify correct intensity behavior for controlled displacement amplitudes and frequencies. | planned |
| Probabilistic speckle likelihood tests | Compare Gaussian, Poisson-like, heteroscedastic, and robust likelihood options on synthetic data. | planned |
| Differentiability / gradient tests | Confirm stable gradients through field-to-image rendering and likelihood terms. | planned |
| Synthetic re-render smoke tests | Encode-field-re-render tests with known fields, known masks, and known noise. | planned |
| SLDV/FEM/reference-field validation | Compare rendered ESPI distributions against independent displacement fields or numerical simulations. | planned |
| Measurement-consistent representation smoke test | Train or constrain `z_ESPI^MC` only after operator validation and clean ROI controls. | future-gated |

## Claim Gate

`M_ESPI` becomes claim-supporting only after synthetic, differentiability, and reference-field validation. Until then it is a planned core artifact, not a validated physics-aware encoder.
