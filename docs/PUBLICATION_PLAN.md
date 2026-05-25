# Development to Publication Plan

## Stage 1: Private Development

- keep the repo private
- maintain only core scripts and summary reports
- avoid raw data and model checkpoints
- stabilize the encoder claim

## Stage 2: Manuscript Support

- align repository results with the optics manuscript
- map every table and figure to scripts/configs/reports
- document limitations and failure cases
- define data/model availability

## Stage 3: Reviewer-Safe Package

- remove local paths
- add environment lock
- tag `v0.1-optics-submission`
- provide artifact DOI or private reviewer links if needed

## Stage 4: Public Release

- publish a clean version after acceptance or when journal policy allows
- keep large artifacts outside GitHub
- cite the release with DOI metadata
