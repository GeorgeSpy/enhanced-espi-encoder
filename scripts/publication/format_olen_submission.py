#!/usr/bin/env python
"""Check the local OLEN submission package structure.

This script does not format the manuscript automatically. It verifies that the
expected LaTeX source and figure assets are present in the repository package.
"""

from __future__ import annotations

from pathlib import Path


REQUIRED = [
    Path("manuscripts/OLEN_ESPI_frozen_representation_submission.tex"),
    Path("manuscripts/figures/Figure_1_Representative_Patterns.pdf"),
    Path("manuscripts/figures/central_result_comparison.pdf"),
    Path("manuscripts/figures/Figure_PCA_UMAP_Embeddings.pdf"),
    Path("manuscripts/figures/Figure_Confusion_Matrices_Board_LOBO.pdf"),
    Path("manuscripts/cover_letter_draft.txt"),
    Path("manuscripts/highlights.txt"),
]


def main() -> int:
    missing = [path for path in REQUIRED if not path.exists()]
    for path in REQUIRED:
        print(f"{'OK' if path.exists() else 'MISSING'} {path}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
