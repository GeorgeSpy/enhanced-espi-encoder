#!/usr/bin/env python
"""Audit manuscript submission readiness before journal formatting.

The script is read-only with respect to manuscripts, tables, figures, and
experimental artifacts. It does not run experiments or alter scientific outputs.
It remains backward-compatible with the v003 filename while emitting versioned
audit files based on the manuscript path, e.g. V004_* for a v004 manuscript.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


MAIN_TABLE_FILES = {
    1: "TABLE_1_ENCODER_BASELINE_SUMMARY_FIXED_K10.md",
    2: "TABLE_2_GROUPED_ROBUSTNESS_WITH_CI.md",
    3: "TABLE_3_CONTROLLED_ARCHITECTURE_COMPARISON.md",
    4: "TABLE_4_CLASS_SUPPORT_AND_IMBALANCE.md",
    5: "TABLE_5_V62A_VS_V61_PER_CLASS_GROUPED_F1.md",
    6: "TABLE_6_ZERO_RECALL_BREAKDOWN.md",
    7: "TABLE_7_FREQUENCY_INFORMATION_BUDGET.md",
    8: "TABLE_8_FIXED_K_VS_BEST_K_DIAGNOSTIC.md",
}

SUPP_TABLE_FILES = {
    1: "SUPPLEMENTARY_TABLE_S1_BOARD_LEVEL_SUPPORT.md",
    2: "SUPPLEMENTARY_TABLE_S2_MATERIAL_LEVEL_SUPPORT.md",
    3: "SUPPLEMENTARY_TABLE_S3_CLASS_BY_BOARD_SUPPORT.md",
    4: "SUPPLEMENTARY_TABLE_S4_CLASS_BY_MATERIAL_SUPPORT.md",
    5: "SUPPLEMENTARY_TABLE_S5_PAIRED_BOARD_AND_MATERIAL_DELTAS_V61_V62A.md",
    6: "SUPPLEMENTARY_TABLE_S6_FREQUENCY_FUSION_PER_CLASS_CHANGES.md",
    7: "SUPPLEMENTARY_TABLE_S7_SPECTRAL_DESCRIPTOR_CONTROL.md",
}

FIGURE_FILES = {
    1: ["reports/publication_assets/encoder_baseline_v001/FIG_1_STRATIFIED_VS_GROUPED_MACRO_F1.svg"],
    2: ["reports/publication_assets/encoder_baseline_v001/FIG_2_GROUPED_ROBUSTNESS_GAIN.svg"],
    3: ["reports/publication_assets/encoder_baseline_v001/FIG_3_ENCODER_DECISION_FLOW.svg"],
    4: ["reports/publication_assets/class_level_v001/FIG_4_CLASS_SUPPORT_IMBALANCE.svg"],
    5: ["reports/publication_assets/class_level_v001/FIG_5_V62A_VS_V61_PER_CLASS_GROUPED_F1.svg"],
    6: ["reports/publication_assets/class_level_v001/FIG_6_ZERO_RECALL_BREAKDOWN_BY_ENCODER.svg"],
}

LOCKED_METRICS = [
    ("v6.1 fixed k=10 stratified Macro-F1", "93.65%"),
    ("v6.2-A fixed k=10 stratified Macro-F1", "92.78%"),
    ("v6.1 diagnostic best-k Macro-F1", "95.07%"),
    ("v6.2-A Board LOBO Macro-F1", "93.56%"),
    ("v6.2-A Material LOMO Macro-F1", "93.49%"),
    ("frequency-only stratified Macro-F1", "98.67%"),
    ("frequency-only Board LOBO Macro-F1", "97.49%"),
    ("frequency-only Material LOMO Macro-F1", "94.09%"),
    ("frequency + v6.2-A stratified Macro-F1", "96.75%"),
    ("frequency + v6.2-A Board LOBO Macro-F1", "94.79%"),
    ("frequency + v6.2-A Material LOMO Macro-F1", "94.50%"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", required=True, type=Path)
    parser.add_argument("--tables-dir", required=True, type=Path)
    parser.add_argument("--hardening-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def manuscript_version(path: Path) -> str:
    match = re.search(r"v(\d{3})", path.name, re.IGNORECASE)
    return f"V{match.group(1)}" if match else "V003"


def status_row(section: str, check: str, status: str, evidence: str, recommendation: str = "") -> dict[str, str]:
    return {
        "section": section,
        "check": check,
        "status": status,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def heading_present(text: str, heading_regex: str) -> bool:
    return re.search(heading_regex, text, re.IGNORECASE | re.MULTILINE) is not None


def contains_any(text: str, phrases: list[str]) -> bool:
    lower = text.lower()
    return any(phrase.lower() in lower for phrase in phrases)


def line_contains_positive_claim(
    text: str,
    phrase: str,
    negators: tuple[str, ...] = (
        "no ",
        "not ",
        "does not ",
        "do not ",
        "outside",
        "future",
        "leaving ",
        "without ",
        "nor ",
        "should not ",
        "unless ",
    ),
) -> bool:
    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    for match in pattern.finditer(text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        if line_end == -1:
            line_end = len(text)
        line = text[line_start:line_end].lower()
        window = text[max(0, match.start() - 120) : match.end() + 120].lower()
        if any(negator in line for negator in negators) or any(negator in window for negator in negators):
            continue
        return True
    return False


def sentence_windows(text: str, token: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if token in line]
    if lines:
        return lines
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [part for part in parts if token in part]


def extract_numbered_citations(text: str) -> set[int]:
    numbers: set[int] = set()
    for content in re.findall(r"\[([0-9,\-\s]+)\]", text):
        for part in content.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                bounds = [value.strip() for value in part.split("-", 1)]
                if len(bounds) == 2 and bounds[0].isdigit() and bounds[1].isdigit():
                    start, end = int(bounds[0]), int(bounds[1])
                    if start <= end:
                        numbers.update(range(start, end + 1))
                continue
            if part.isdigit():
                numbers.add(int(part))
    return numbers


def audit_structure(text: str) -> list[dict[str, str]]:
    checks = [
        ("Title", r"^#\s+.+"),
        ("Abstract", r"^##\s+Abstract"),
        ("Introduction", r"^##\s+1\.\s+Introduction"),
        ("Related Work", r"^##\s+2\.\s+Related Work"),
        ("Methods", r"^##\s+\d+\.\s+Methods"),
        ("Results", r"^##\s+\d+\.\s+Results"),
        ("Discussion", r"^##\s+\d+\.\s+Discussion"),
        ("Limitations/Future Work", r"^##\s+\d+\.\s+Limitations and Future Work"),
        ("Conclusion", r"^##\s+\d+\.\s+Conclusion"),
        ("Data/Code/Reproducibility Availability", r"^##\s+Data, Code, and Reproducibility Availability"),
        ("References", r"^##\s+References"),
        ("Tables/Figures list", r"^##\s+Tables and Figures"),
    ]
    rows = []
    for label, regex in checks:
        present = heading_present(text, regex)
        rows.append(status_row("Manuscript structure", label, "pass" if present else "fail", "present" if present else "missing"))
    return rows


def audit_claims(text: str) -> list[dict[str, str]]:
    rows = [
        status_row(
            "Claim boundary",
            "v6.2-A as strongest image-derived frozen ESPI encoder baseline",
            "pass" if "strongest image-derived frozen ESPI encoder baseline" in text else "fail",
            "phrase present" if "strongest image-derived frozen ESPI encoder baseline" in text else "phrase missing",
        ),
        status_row(
            "Claim boundary",
            "frequency_hz as dominant metadata-only predictor",
            "pass" if contains_any(text, ["dominant metadata-only predictor", "dominant predictor"]) else "fail",
            "frequency dominance stated" if contains_any(text, ["dominant metadata-only predictor", "dominant predictor"]) else "frequency dominance missing",
        ),
    ]
    for label, phrase in [
        ("positive acoustic-response prediction claim", "acoustic-response prediction"),
        ("positive validated Physics-Aligned Encoder claim", "validated Physics-Aligned Encoder"),
        ("positive LeFFT superiority claim", "LeFFT superiority"),
        ("positive trained LeFFT model claim", "trained LeFFT model"),
        ("positive retrained CNN LOBO/LOMO generalization claim", "retrained CNN LOBO/LOMO"),
    ]:
        positive = line_contains_positive_claim(text, phrase)
        rows.append(
            status_row(
                "Claim boundary",
                label,
                "fail" if positive else "pass",
                "positive claim detected" if positive else "absent or explicitly negated/scope-limited",
            )
        )
    rows.append(
        status_row(
            "Claim boundary",
            "spectral descriptors supplementary/control-only",
            "pass" if contains_any(text, ["supplementary controls", "supplementary/control", "negative/control"]) else "fail",
            "supplementary/control wording present" if contains_any(text, ["supplementary controls", "supplementary/control", "negative/control"]) else "wording missing",
        )
    )
    rows.append(
        status_row(
            "Claim boundary",
            "Material LOMO descriptive/stress-test caution",
            "pass" if contains_any(text, ["material-held-out stress test", "descriptively as a material-held-out stress test"]) else "fail",
            "material-held-out stress-test wording present" if contains_any(text, ["material-held-out stress test", "descriptively as a material-held-out stress test"]) else "wording missing",
        )
    )
    return rows


def audit_metrics(text: str) -> list[dict[str, str]]:
    rows = []
    for label, value in LOCKED_METRICS:
        present = value in text
        rows.append(status_row("Metrics", label, "pass" if present else "fail", f"{value} {'present' if present else 'missing'}"))
    windows = sentence_windows(text, "95.07%")
    diagnostic_ok = bool(windows) and all(
        contains_any(window, ["diagnostic", "sensitivity", "best-k", "best k"]) for window in windows
    )
    rows.append(
        status_row(
            "Metrics",
            "v6.1 95.07% appears only as diagnostic/best-k",
            "pass" if diagnostic_ok else "fail",
            " | ".join(windows) if windows else "95.07% not found",
            "Move 95.07% to diagnostic/sensitivity wording only." if not diagnostic_ok else "",
        )
    )
    rows.append(
        status_row(
            "Metrics",
            "spectral descriptor control remains supplementary",
            "pass" if contains_any(text, ["Supplementary Table S7", "supplementary controls"]) else "fail",
            "supplementary spectral reference present" if contains_any(text, ["Supplementary Table S7", "supplementary controls"]) else "missing",
        )
    )
    return rows


def audit_tables(text: str, tables_dir: Path) -> list[dict[str, str]]:
    rows = []
    for idx, filename in MAIN_TABLE_FILES.items():
        ref_present = re.search(rf"\bTable {idx}\b|\[Table {idx}\]", text) is not None
        exists = (tables_dir / filename).exists()
        rows.append(
            status_row(
                "Tables",
                f"Table {idx} reference and file",
                "pass" if ref_present and exists else "fail",
                f"reference={ref_present}; file_exists={exists}; path={tables_dir / filename}",
            )
        )
    for idx, filename in SUPP_TABLE_FILES.items():
        ref_present = re.search(rf"Supplementary Table S{idx}\b|\[Supplementary Table S{idx}\]", text) is not None
        exists = (tables_dir / filename).exists()
        rows.append(
            status_row(
                "Tables",
                f"Supplementary Table S{idx} reference and file",
                "pass" if ref_present and exists else "fail",
                f"reference={ref_present}; file_exists={exists}; path={tables_dir / filename}",
            )
        )
    unresolved = re.findall(r"\[(?:Table|Figure)\s+X\]", text)
    rows.append(
        status_row(
            "Tables",
            "Unresolved [Table X]/[Figure X] placeholders",
            "fail" if unresolved else "pass",
            ", ".join(unresolved) if unresolved else "none",
        )
    )
    return rows


def audit_figures(text: str) -> list[dict[str, str]]:
    rows = []
    for idx, candidates in FIGURE_FILES.items():
        ref_present = re.search(rf"\bFigure {idx}\b|\[Figure {idx}\]", text) is not None
        existing = [path for path in candidates if Path(path).exists()]
        rows.append(
            status_row(
                "Figures",
                f"Figure {idx} reference and file",
                "pass" if ref_present and existing else "fail",
                f"reference={ref_present}; existing_files={existing if existing else 'none'}",
                "" if existing else "Generate or export this figure before venue formatting.",
            )
        )
    return rows


def split_references(text: str) -> tuple[str, str]:
    match = re.search(r"^##\s+References\s*$", text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return text, ""
    return text[: match.start()], text[match.end() :]


def audit_citations(text: str) -> list[dict[str, str]]:
    in_text, references = split_references(text)
    rows = []
    has_references = bool(references.strip())
    citation_numbers = {number for number in extract_numbered_citations(in_text) if number <= 100}
    reference_numbers = {int(num) for num in re.findall(r"^\[(\d+)\]", references, re.MULTILINE)}
    missing_refs = sorted(citation_numbers - reference_numbers)
    uncited_refs = sorted(reference_numbers - citation_numbers)
    unresolved = re.findall(r"\[(?:citation needed|TODO|REF|X)\]", text, re.IGNORECASE)
    doi_cleanup = []
    for line in references.splitlines():
        if re.match(r"^\[(3|6|7|8)\]", line) and "doi" not in line.lower():
            doi_cleanup.append(line.split("]", 1)[0] + "]")
    rows.extend(
        [
            status_row(
                "Citations and bibliography",
                "Related Work exists",
                "pass" if heading_present(text, r"^##\s+2\.\s+Related Work") else "fail",
                "present" if heading_present(text, r"^##\s+2\.\s+Related Work") else "missing",
            ),
            status_row(
                "Citations and bibliography",
                "Reference list exists",
                "pass" if has_references else "fail",
                "present" if has_references else "missing",
                "Add a References section in the target journal style." if not has_references else "",
            ),
            status_row(
                "Citations and bibliography",
                "Every in-text numbered citation has matching reference",
                "pass" if not missing_refs else "fail",
                f"missing references={missing_refs if missing_refs else 'none'}",
            ),
            status_row(
                "Citations and bibliography",
                "Every reference item is cited at least once",
                "pass" if not uncited_refs else "warn",
                f"uncited references={uncited_refs if uncited_refs else 'none'}",
            ),
            status_row(
                "Citations and bibliography",
                "Unresolved citation placeholders",
                "pass" if not unresolved else "fail",
                ", ".join(unresolved) if unresolved else "none",
            ),
            status_row(
                "Citations and bibliography",
                "References needing DOI/page/venue cleanup",
                "warn" if doi_cleanup else "pass",
                ", ".join(doi_cleanup) if doi_cleanup else "none",
                "Add DOI fields during venue-specific reference cleanup where available." if doi_cleanup else "",
            ),
        ]
    )
    return rows


def audit_terminology(text: str) -> list[dict[str, str]]:
    terms = [
        "image-derived frozen ESPI encoder",
        "grouped board/material",
        "material-held-out stress test",
        "deterministic spectral",
        "LeFFT-inspired",
        "frequency metadata",
    ]
    rows = []
    for term in terms:
        present = term.lower() in text.lower()
        rows.append(status_row("Terminology", term, "pass" if present else "warn", "present" if present else "not found"))
    positive_trained = line_contains_positive_claim(text, "trained LeFFT")
    rows.append(
        status_row(
            "Terminology",
            "LeFFT-inspired, not trained LeFFT",
            "fail" if positive_trained else "pass",
            "positive trained LeFFT wording detected" if positive_trained else "trained LeFFT wording absent or negated",
        )
    )
    return rows


def audit_abstract(text: str) -> list[dict[str, str]]:
    match = re.search(r"^##\s+Abstract\s+(.*?)(?=^##\s+)", text, re.DOTALL | re.MULTILINE)
    if not match:
        return [status_row("Abstract readiness", "Abstract exists", "fail", "missing")]
    abstract = match.group(1).strip()
    words = re.findall(r"\S+", abstract)
    metric_tokens = re.findall(r"\d+\.\d+%|\+\d+\.\d+ pp|-\d+\.\d+ pp", abstract)
    status = "warn" if len(metric_tokens) > 6 else "pass"
    return [
        status_row("Abstract readiness", "Word count", "pass" if len(words) <= 250 else "warn", f"{len(words)} words"),
        status_row(
            "Abstract readiness",
            "Metric density",
            status,
            f"{len(metric_tokens)} metric tokens",
            "Consider compressing the abstract by retaining only fixed-k stratified, grouped v6.2-A, and frequency-only summary values."
            if status == "warn"
            else "",
        ),
    ]


def summarize_status(rows: list[dict[str, str]]) -> dict[str, Any]:
    fails = [row for row in rows if row["status"] == "fail"]
    warns = [row for row in rows if row["status"] == "warn"]
    reference_failures = any(row["section"] == "Citations and bibliography" and row["status"] == "fail" for row in rows)
    figure_failures = any(row["section"] == "Figures" and row["status"] == "fail" for row in rows)
    return {
        "n_checks": len(rows),
        "n_fail": len(fails),
        "n_warn": len(warns),
        "n_pass": len(rows) - len(fails) - len(warns),
        "ready_for_journal_formatting": len(fails) == 0,
        "ready_for_reference_cleanup": not reference_failures,
        "ready_for_reference_completion": not reference_failures,
        "ready_for_figure_generation": not figure_failures,
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["section", "check", "status", "evidence", "recommendation"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def markdown_table(rows: list[dict[str, str]]) -> str:
    fields = ["section", "check", "status", "evidence", "recommendation"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(row.get(field, "")).replace("|", "\\|").replace("\n", "<br>") for field in fields)
            + " |"
        )
    return "\n".join(lines)


def render_report(args: argparse.Namespace, rows: list[dict[str, str]], summary: dict[str, Any], version: str) -> str:
    required = [row for row in rows if row["status"] == "fail"]
    optional = [row for row in rows if row["status"] == "warn"]
    return "\n".join(
        [
            f"# {version} Submission Readiness Audit",
            "",
            "## Inputs",
            "",
            f"- Manuscript: `{args.manuscript}`",
            f"- Tables directory: `{args.tables_dir}`",
            f"- Hardening directory: `{args.hardening_dir}`",
            "",
            "## Readiness Summary",
            "",
            f"- Ready for journal formatting: `{'yes' if summary['ready_for_journal_formatting'] else 'no'}`",
            f"- Ready for reference cleanup: `{'yes' if summary['ready_for_reference_cleanup'] else 'no'}`",
            f"- Ready for figure generation: `{'yes' if summary['ready_for_figure_generation'] else 'no'}`",
            f"- Checks passed/warned/failed: `{summary['n_pass']}` / `{summary['n_warn']}` / `{summary['n_fail']}`",
            "",
            "## Required Fixes Before Formatting",
            "",
            markdown_table(required) if required else "_No required fixes._",
            "",
            "## Optional Improvements",
            "",
            markdown_table(optional) if optional else "_No optional improvements._",
            "",
            "## Detailed Checks",
            "",
            markdown_table(rows),
        ]
    )


def main() -> int:
    args = parse_args()
    if not args.manuscript.exists():
        raise FileNotFoundError(args.manuscript)
    text = args.manuscript.read_text(encoding="utf-8")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    version = manuscript_version(args.manuscript)
    version_lower = version.lower()

    rows: list[dict[str, str]] = []
    rows.extend(audit_structure(text))
    rows.extend(audit_claims(text))
    rows.extend(audit_metrics(text))
    rows.extend(audit_tables(text, args.tables_dir))
    rows.extend(audit_figures(text))
    rows.extend(audit_citations(text))
    rows.extend(audit_terminology(text))
    rows.extend(audit_abstract(text))

    summary = summarize_status(rows)
    payload = {
        "manuscript": str(args.manuscript),
        "tables_dir": str(args.tables_dir),
        "hardening_dir": str(args.hardening_dir),
        "summary": summary,
        "checks": rows,
    }
    (args.out_dir / f"{version_lower}_submission_readiness_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_csv(args.out_dir / f"{version_lower}_required_fixes.csv", [row for row in rows if row["status"] == "fail"])
    (args.out_dir / f"{version}_SUBMISSION_READINESS_AUDIT.md").write_text(
        render_report(args, rows, summary, version), encoding="utf-8"
    )

    print(f"[done] wrote audit to {args.out_dir}")
    print(f"[done] ready_for_journal_formatting={summary['ready_for_journal_formatting']}")
    print(f"[done] failures={summary['n_fail']} warnings={summary['n_warn']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
