# Established Scope and Evidence Boundaries

The canonical scope and evidence document is `docs/CLAIM_BOUNDARIES.md`.

In short, the repository establishes:

- **Strongest Frozen Baseline**: v6.2-A provides the strongest evaluated image-derived frozen ESPI representation under grouped board/material evaluation.
- **Dominant Frequency Predictor**: `frequency_hz` serves as the dominant metadata-only predictor for the present five-class modal-label task (LOBO Macro-F1 $\approx 0.9327$).
- **Rigorous Methodological Controls**: Verified controls cover PCA dimension matching, targeted `1_2` / `2_1` overlap analysis, balanced subsets, embedding geometry diagnostics, and deterministic spectral descriptor baselines.
- **Diagnostic Architecture Evidence**: H2 and H3 LeFFT development evidence characterizes standalone and auxiliary Fourier branch behavior, establishing the foundation for future measurement-consistent representations ($M_{\text{ESPI}}$).
