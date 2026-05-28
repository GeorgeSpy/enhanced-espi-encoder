# LeFFT / Spectral Branch Audit Decision

## Summary

This audit examined whether the current repository and hierarchical v6.2 phase2 checkpoint support a manuscript claim about LeFFT or learned spectral superiority.

## Findings

1. The hierarchical checkpoint contains only four Fourier-prior scalar parameters:
   - `leftp.bandwidth`
   - `leftp.filter_sharpness`
   - `leftp.fx_carrier`
   - `leftp.fy_carrier`

2. The checkpoint does not expose a rich learned LeFFT embedding branch suitable for direct publication-grade comparison against v6.2-A.

3. Existing deterministic FFT / spectral descriptor ablations were already available and show weak performance:
   - Spectral-only Stratified Macro-F1: 52.92%
   - Spectral-only Board LOBO Macro-F1: 22.77%
   - Spectral-only Material LOMO Macro-F1: 19.72%

4. Fusion of deterministic spectral descriptors with v6.2-A provides no substantive grouped improvement and is within noise-level variation.

## Manuscript Decision

LeFFT is not used as a main claim in the current manuscript.

The deterministic spectral descriptor experiment is retained as a supplementary negative spatial-frequency control. It supports the conclusion that simple hand-engineered spectral summaries do not explain the grouped robustness advantage of v6.2-A.

## Claim Boundary

The manuscript may state:

> Deterministic FFT-based descriptors were evaluated as a supplementary spectral control. Their weak grouped performance and negligible fusion gains indicate that simple handcrafted spatial-frequency summaries do not account for the grouped robustness achieved by v6.2-A.

The manuscript should not state:

- LeFFT improves ESPI grouped robustness.
- The hierarchical v6.2 phase2 branch validates a physics-aware spectral encoder.
- Learned spectral modules have been proven unnecessary.
- The negative deterministic descriptor result disproves future learned LeFFT architectures.

## Future Work

A learned LeFFT / spectral encoder remains a future ablation requiring matched training, fixed grouped protocols, frequency-only controls, and comparison against v6.2-A under the same frozen-representation audit framework.
