# Artifact Policy

This repository separates small, reviewable evidence artifacts from large or private binary artifacts.

## Tracked

- Python source scripts.
- Markdown documentation.
- Small CSV/JSON/Markdown reports.
- Final manuscript source.
- Final manuscript figures when size is reasonable.
- Table bundles and captions.
- Artifact manifest templates.

## Excluded

- Raw ESPI images.
- Full feature dumps.
- Checkpoints.
- Private manifests.
- Local logs and temporary outputs.
- Large zip bundles.

## External Artifact Manifest

Use `docs/templates/ARTIFACT_MANIFEST_TEMPLATE.csv` for any DOI-backed or private review bundle. Every binary artifact required for full numerical reproduction should include a SHA256 checksum.

## Path Hygiene

Reports committed to Git should avoid absolute user-specific paths. Use placeholders such as:

- `<LOCAL_PROJECT_ROOT>`
- `<PRIVATE_IMAGE_ROOT>`
- `<PRIVATE_CHECKPOINT_ROOT>`
- `<ARTIFACT_BUNDLE_ROOT>`

## Release Modes

- Private development: local private artifacts may be used.
- Reviewer package: provide sanitized artifacts and checksums externally.
- Public release: publish only artifacts cleared for redistribution.
