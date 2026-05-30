# Data Availability and Artifact Policy

Raw ESPI images, model checkpoints, full feature dumps, and private manifests are not tracked in this Git repository. They may contain large binary artifacts, local acquisition structure, or private file-system metadata.

## What Is Tracked in Git

- Source scripts required to reproduce extraction, evaluation, controls, and table generation.
- Small Markdown, CSV, and JSON report summaries.
- Claim-boundary and reproducibility documentation.
- Final manuscript source, figures, and small submission-support files.
- Artifact manifest templates.

## What Is Excluded from Git

- Raw ESPI image directories.
- Checkpoints (`*.pt`, `*.pth`, `*.ckpt`).
- Full feature dumps (`*.npz`, `*.npy`).
- Local logs and intermediate outputs.
- Private manifest bundles.

## Review Artifact Bundle

For review or DOI-backed release, the following can be provided outside Git:

- sanitized manifest,
- normalized feature dumps,
- checkpoint and feature SHA256 checksums,
- final table CSVs,
- final figures,
- source scripts,
- environment or dependency notes,
- claim checklist and audit reports.

## Checksums

Every external binary artifact used in the manuscript should have a recorded SHA256 checksum in a non-Git or DOI-backed artifact manifest. Use `docs/templates/ARTIFACT_MANIFEST_TEMPLATE.csv` as the starting point.

## Release Plan

The public repository can support code-level and report-level reproducibility. Full numerical reproduction requires access to the private or DOI-backed artifact bundle described above.
