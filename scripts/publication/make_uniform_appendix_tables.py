#!/usr/bin/env python
"""List final table assets expected by the manuscript appendix.

The final table generation is handled by scripts/encoder/make_final_manuscript_tables.py.
This helper is a lightweight publication-package check.
"""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    table_dir = Path("reports/publication_assets/final_tables_v001")
    expected = sorted(table_dir.glob("*.md")) if table_dir.exists() else []
    if not expected:
        print("No final Markdown tables found.")
        return 1
    for path in expected:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
