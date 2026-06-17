# Clean ROI Requirements

Clean ROI/no-overlay controls are mandatory before future image-derived ESPI representation claims are promoted.

## Image Requirements

Clean ROI images must exclude:

- visible excitation frequency text,
- source-pressure text,
- class labels or mode labels,
- board/material labels,
- axes,
- borders,
- colorbars,
- timestamps,
- acquisition or instrument annotations,
- non-physical margins,
- filename/path-derived overlays.

## ROI Verification

Each cleaned image or virtual cleaning transform must record:

- original image path,
- clean image path or virtual transform ID,
- mask/crop coordinates,
- cleaning strategy,
- image dimensions before and after cleaning,
- board/material/class/frequency metadata for audit only,
- QC status.

## Required Leakage Controls

At minimum, run:

1. top-left patch baseline,
2. top-border patch baseline,
3. corner and border patch baselines,
4. background-only board/material prediction if background exists,
5. original-vs-masked comparison,
6. shuffled-label controls,
7. near-duplicate or acquisition-contamination checks where possible.

## Visual Audit Protocol

QC contact sheets should cover:

- every board,
- both materials,
- all available classes,
- low/mid/high frequency ranges,
- examples with known overlays,
- original vs masked vs ROI-only views.

## Claim Rule

If cleaned ROI performance survives, the allowed wording is limited to:

"This diagnostic performance was not materially reduced by the specified visual-overlay cleaning."

It must not be rewritten as global frequency-independent morphology recognition without frequency/metadata/numerical residual validation.
