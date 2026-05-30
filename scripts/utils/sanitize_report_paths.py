#!/usr/bin/env python
"""Sanitize local absolute paths in small report artifacts.

This utility is intentionally conservative: it rewrites text files only and
skips binary artifacts. Use it before committing report summaries that may have
been generated on a local Windows workstation.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_REPLACEMENTS: dict[str, str] = {}

TEXT_SUFFIXES = {".md", ".txt", ".csv", ".json", ".tex", ".yml", ".yaml"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Files or directories to sanitize.")
    parser.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Replacement pair. Can be passed multiple times.",
    )
    parser.add_argument("--write", action="store_true", help="Rewrite files in place. Default is dry run.")
    return parser.parse_args()


def iter_text_files(paths: list[Path]):
    for path in paths:
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            yield path
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and child.suffix.lower() in TEXT_SUFFIXES:
                    yield child


def parse_replacements(pairs: list[str]) -> dict[str, str]:
    replacements = dict(DEFAULT_REPLACEMENTS)
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid replacement pair: {pair!r}. Expected OLD=NEW.")
        old, new = pair.split("=", 1)
        replacements[old] = new
    return replacements


def sanitize_text(text: str, replacements: dict[str, str]) -> tuple[str, int]:
    count = 0
    for old, new in replacements.items():
        occurrences = text.count(old)
        if occurrences:
            text = text.replace(old, new)
            count += occurrences
    return text, count


def main() -> int:
    args = parse_args()
    replacements = parse_replacements(args.replace)
    total = 0
    changed_files = 0
    for path in iter_text_files(args.paths):
        text = path.read_text(encoding="utf-8", errors="ignore")
        sanitized, count = sanitize_text(text, replacements)
        if count:
            changed_files += 1
            total += count
            print(f"{path}: {count} replacements")
            if args.write:
                path.write_text(sanitized, encoding="utf-8")
    print(f"changed_files={changed_files} replacements={total} write={args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
