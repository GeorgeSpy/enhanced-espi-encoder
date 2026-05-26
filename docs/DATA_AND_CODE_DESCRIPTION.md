# Data and Code Description

## Data Scope

This repository does not include raw data. It documents the expected data organization and stores lightweight evidence reports.

## ESPI Classification Data

The historical v6.1/v6.2 line expects ESPI image-derived examples with:

- image path or feature source
- class label
- material/source-group metadata
- board/specimen identifier
- frequency information where available

The 5-class mode space used by the current v6.2 evidence is:

- `mode_1_1H`
- `mode_1_1T`
- `mode_1_2`
- `mode_2_1`
- `mode_higher`

## Encoder Data

The encoder workflow expects a trained or loadable v6.2-compatible model and an input manifest that can be used to extract embeddings.

The generated feature dumps are intentionally excluded from Git. Summary metadata and reports are included instead.

## Future Acoustic-Response Data

The future optics/acoustics paper direction requires an acoustic-response dataset manifest. A placeholder example is provided:

- `examples/sample_acoustic_manifest.csv`

Expected fields include:

- specimen id
- material
- geometry id
- boundary condition
- excitation condition
- frequency
- acoustic target reference
- optional ESPI reference
- split or grouped-OOD group

## Code Scope

The repository currently contains script-level code rather than a fully refactored package.

Core script groups:

- `scripts/v6_1/`: baseline classification and export
- `scripts/v6_2/`: v6.2 model/training path
- `scripts/v6_2_hierarchical/`: gatekeeper/expert training branch
- `scripts/encoder/`: embedding extraction and encoder evaluation

The reusable package namespace is reserved at:

- `src/enhanced_espi_encoder/`

## Development Status

The repository is under active development. The next engineering step is to refactor stable script logic into reusable package modules.
