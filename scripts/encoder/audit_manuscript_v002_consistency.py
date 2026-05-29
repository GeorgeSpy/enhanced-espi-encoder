#!/usr/bin/env python
"""Audit manuscript v002 against methodological hardening decisions.

This script reads manuscript/report artifacts only. It does not modify the
manuscript, run experiments, implement LeFFT, or run acoustic-response
prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", required=True, type=Path)
    parser.add_argument("--hardening-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def lower(text: str) -> str:
    return text.lower()


def contains_all(text: str, phrases: list[str], case_sensitive: bool = False) -> bool:
    haystack = text if case_sensitive else lower(text)
    needles = phrases if case_sensitive else [lower(phrase) for phrase in phrases]
    return all(needle in haystack for needle in needles)


def contains_any(text: str, phrases: list[str], case_sensitive: bool = False) -> bool:
    haystack = text if case_sensitive else lower(text)
    needles = phrases if case_sensitive else [lower(phrase) for phrase in phrases]
    return any(needle in haystack for needle in needles)


def add_check(
    rows: list[dict[str, Any]],
    section: str,
    check: str,
    status: str,
    evidence: str,
    recommendation: str = "",
) -> None:
    rows.append(
        {
            "section": section,
            "check": check,
            "status": status,
            "evidence": evidence,
            "recommendation": recommendation,
        }
    )


def status_from_bool(ok: bool) -> str:
    return "pass" if ok else "fail"


def table_refs_present(text: str) -> dict[str, bool]:
    return {f"Table {idx}": f"Table {idx}" in text for idx in range(1, 9)}


def supp_refs_present(text: str) -> dict[str, bool]:
    return {f"Supplementary Table S{idx}": f"Supplementary Table S{idx}" in text for idx in range(1, 7)}


def find_unresolved_placeholders(text: str) -> list[str]:
    patterns = [
        r"\[(?:Table|Figure)\s+[A-ZxX?]+\]",
        r"\bTBD\b",
        r"\bTODO\b",
        r"\[INSERT[^\]]*\]",
        r"\[PLACEHOLDER[^\]]*\]",
    ]
    found: list[str] = []
    for pattern in patterns:
        found.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return sorted(set(found))


def source_artifact_status(hardening_dir: Path) -> dict[str, bool]:
    required = {
        "patch_plan": hardening_dir / "MANUSCRIPT_HARDENING_PATCH_PLAN.md",
        "patch_key_numbers": hardening_dir / "manuscript_hardening_patch_key_numbers.json",
        "dataset_composition": hardening_dir / "DATASET_COMPOSITION_REPORT.md",
        "paired_differences": hardening_dir / "PAIRED_V61_V62A_GROUPED_DIFFERENCES.md",
        "frequency_only": hardening_dir / "frequency_only_baseline" / "FREQUENCY_ONLY_BASELINE_REPORT.md",
        "frequency_fusion": hardening_dir / "frequency_embedding_fusion" / "FREQUENCY_EMBEDDING_FUSION_REPORT.md",
        "knn_protocol": hardening_dir / "knn_protocol" / "KNN_PROTOCOL_HARDENING_REPORT.md",
    }
    return {name: path.exists() for name, path in required.items()}


def audit_claim_boundaries(text: str, rows: list[dict[str, Any]]) -> None:
    section = "Claim boundary"
    text_lc = lower(text)

    def forbidden_positive_present(phrase: str) -> bool:
        phrase_lc = lower(phrase)
        if phrase_lc not in text_lc:
            return False
        candidate_sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text_lc)
            if phrase_lc in sentence
        ]
        negation_terms = [
            "does not",
            "do not",
            "not ",
            "no ",
            "nor ",
            "without",
            "outside the scope",
            "future",
            "before claiming",
            "before claim",
            "rather than",
        ]
        if candidate_sentences and all(any(term in sentence for term in negation_terms) for sentence in candidate_sentences):
            return False
        negation_patterns = [
            f"does not claim {phrase_lc}",
            f"does not establish {phrase_lc}",
            f"does not validate {phrase_lc}",
            f"no {phrase_lc}",
            f"not {phrase_lc}",
            f"nor do they establish {phrase_lc}",
            f"before claiming {phrase_lc}",
            f"before claim {phrase_lc}",
        ]
        return not any(pattern in text_lc for pattern in negation_patterns)

    good_claim = "strongest image-derived frozen ESPI encoder baseline" in text
    add_check(
        rows,
        section,
        "v6.2-A described as strongest image-derived frozen ESPI encoder baseline",
        status_from_bool(good_claim),
        "phrase present" if good_claim else "required phrase missing",
        "Use the exact phrase `strongest image-derived frozen ESPI encoder baseline` near the abstract/introduction claim.",
    )

    forbidden = [
        (
            "strongest overall predictor",
            "v6.2-A is the strongest overall predictor",
        ),
        (
            "acoustic-response prediction claim",
            "acoustic-response predictive value",
        ),
        (
            "validated Physics-Aligned Encoder claim",
            "validated Physics-Aligned Encoder",
        ),
        (
            "LeFFT superiority claim",
            "LeFFT superiority",
        ),
        (
            "retrained CNN LOBO/LOMO generalization claim",
            "retrained CNN LOBO/LOMO generalization",
        ),
    ]
    for label, phrase in forbidden:
        present = forbidden_positive_present(phrase)
        status = "fail" if present else "pass"
        evidence = (
            f"positive forbidden phrase found: {phrase}"
            if present
            else f"absent or explicitly negated: {phrase}"
        )
        add_check(
            rows,
            section,
            f"No positive {label}",
            status,
            evidence,
            f"Remove any positive claim that states `{phrase}`." if status == "fail" else "",
        )

    # Specific scope statements that should be present.
    for check, phrases in [
        ("Frequency metadata is dominant/strong metadata control", ["frequency metadata", "dominant"]),
        ("No simple modal-label necessity claim", ["do not claim that image embeddings are necessary"]),
        ("Acoustic response outside scope/future", ["acoustic-response", "outside the scope"]),
    ]:
        ok = contains_all(text, phrases)
        add_check(rows, section, check, status_from_bool(ok), "phrases found" if ok else f"missing: {phrases}")


def audit_knn(text: str, rows: list[dict[str, Any]]) -> None:
    section = "kNN protocol"
    checks = [
        ("Primary fixed k=10 protocol stated", ["primary fixed `k=10`", "kNN"]),
        ("v6.1 fixed k=10 stratified value present", ["v6.1", "93.65%"]),
        ("v6.2-A fixed k=10 stratified value present", ["v6.2-A", "92.78%"]),
        ("v6.1 best-k diagnostic value present", ["best-k", "k=5", "95.07%"]),
        ("Best-k diagnostic/sensitivity not primary", ["diagnostic", "sensitivity"]),
        ("Grouped kNN locked at k=10", ["grouped fixed `k=10`"]),
    ]
    for check, phrases in checks:
        ok = contains_all(text, phrases)
        add_check(rows, section, check, status_from_bool(ok), "phrases found" if ok else f"missing: {phrases}")


def audit_dataset(text: str, rows: list[dict[str, Any]]) -> None:
    section = "Dataset composition"
    checks = [
        ("Total sample count present", ["12,944"]),
        ("Six boards present", ["six boards", "C01", "C02", "C03", "W01", "W02", "W03"]),
        ("Two materials present", ["two materials", "carbon", "wood"]),
        ("All boards contain all five classes", ["All boards", "all five modal classes"]),
        ("All materials contain all five classes", ["both materials", "all five modal classes"]),
        ("No LOBO/LOMO missing held-out/reference classes", ["No board- or material-held-out fold lacks held-out or reference classes"]),
    ]
    for check, phrases in checks:
        ok = contains_all(text, phrases)
        add_check(rows, section, check, status_from_bool(ok), "phrases found" if ok else f"missing: {phrases}")


def audit_material_lomo(text: str, rows: list[dict[str, Any]]) -> None:
    section = "Material LOMO caution"
    ok = contains_all(text, ["Material LOMO", "descriptive", "material-held-out stress test", "two material"])
    add_check(
        rows,
        section,
        "Material LOMO descriptive/stress-test caveat",
        status_from_bool(ok),
        "caveat present" if ok else "missing descriptive two-material caveat",
        "State that Material LOMO is descriptive/material-held-out stress test because only two materials exist.",
    )
    overinterpret = "broad inferential estimate" in text and "rather than as a broad inferential estimate" not in text
    add_check(
        rows,
        section,
        "Material LOMO not over-interpreted",
        "fail" if overinterpret else "pass",
        "no positive broad-population interpretation detected" if not overinterpret else "possible broad inferential phrasing detected",
    )


def audit_paired(text: str, rows: list[dict[str, Any]]) -> None:
    section = "Paired v6.1 vs v6.2-A"
    checks = [
        ("Board LOBO delta present", ["+6.47"]),
        ("Material LOMO delta present", ["+7.16"]),
        ("v6.2-A wins board folds", ["5/6 board folds"]),
        ("v6.2-A wins material folds", ["2/2 material folds"]),
        ("C02 near tie present", ["C02", "-0.02"]),
        ("W01 small margin present", ["W01", "+0.27"]),
    ]
    for check, phrases in checks:
        ok = contains_all(text, phrases)
        add_check(rows, section, check, status_from_bool(ok), "phrases found" if ok else f"missing: {phrases}")


def audit_frequency(text: str, rows: list[dict[str, Any]]) -> None:
    section = "Frequency controls"
    checks = [
        ("Frequency-only stratified present", ["frequency-only", "98.67%"]),
        ("Frequency-only Board LOBO present", ["frequency-only", "97.49%"]),
        ("Frequency-only Material LOMO present", ["frequency-only", "94.09%"]),
        ("Frequency + v6.2-A stratified present", ["Frequency + v6.2-A", "96.75%"]),
        ("Frequency + v6.2-A Board LOBO present", ["Frequency + v6.2-A", "94.79%"]),
        ("Frequency + v6.2-A Material LOMO present", ["Frequency + v6.2-A", "94.50%"]),
        ("frequency_hz dominant metadata predictor", ["frequency", "dominant", "metadata"]),
        ("Fusion improves over embedding-only", ["improves over v6.2-A embedding-only in all three protocols"]),
        ("Fusion does not clearly exceed frequency-only except Material LOMO", ["does not clearly exceed frequency-only", "Material LOMO"]),
    ]
    for check, phrases in checks:
        ok = contains_all(text, phrases)
        add_check(rows, section, check, status_from_bool(ok), "phrases found" if ok else f"missing: {phrases}")


def audit_hierarchical(text: str, rows: list[dict[str, Any]]) -> None:
    section = "Hierarchical caveat"
    checks = [
        ("Hierarchical technically valid", ["hierarchical v6.2 phase2", "technically valid"]),
        ("Hierarchical not superior", ["not superior"]),
        ("Hierarchical internal-only metadata caveat", ["internal-only", "correct board and split-group metadata"]),
        ("Hierarchical not validated physics-aligned encoder", ["rather than a final validated encoder"]),
    ]
    for check, phrases in checks:
        ok = contains_all(text, phrases)
        add_check(rows, section, check, status_from_bool(ok), "phrases found" if ok else f"missing: {phrases}")


def audit_tables(text: str, rows: list[dict[str, Any]]) -> list[str]:
    section = "Tables and figures"
    for ref, present in table_refs_present(text).items():
        add_check(rows, section, f"{ref} listed/referenced", status_from_bool(present), "present" if present else "missing")
    for ref, present in supp_refs_present(text).items():
        add_check(rows, section, f"{ref} listed/referenced", status_from_bool(present), "present" if present else "missing")
    unresolved = find_unresolved_placeholders(text)
    add_check(
        rows,
        section,
        "Unresolved placeholders",
        "warn" if unresolved else "pass",
        ", ".join(unresolved) if unresolved else "none",
        "Resolve listed placeholders before journal formatting." if unresolved else "",
    )
    return unresolved


def overall_status(rows: list[dict[str, Any]]) -> str:
    statuses = [row["status"] for row in rows]
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def section_statuses(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for section in sorted({row["section"] for row in rows}):
        subset = [row for row in rows if row["section"] == section]
        status = overall_status(subset)
        output.append(
            {
                "section": section,
                "status": status,
                "n_pass": sum(row["status"] == "pass" for row in subset),
                "n_warn": sum(row["status"] == "warn" for row in subset),
                "n_fail": sum(row["status"] == "fail" for row in subset),
            }
        )
    return output


def build_required_fixes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fixes: list[dict[str, Any]] = []
    for row in rows:
        if row["status"] in {"fail", "warn"}:
            fixes.append(
                {
                    "section": row["section"],
                    "issue": row["check"],
                    "status": row["status"],
                    "evidence": row["evidence"],
                    "recommended_fix": row.get("recommendation", ""),
                }
            )
    return fixes


def render_report(
    manuscript: Path,
    hardening_dir: Path,
    source_status: dict[str, bool],
    section_rows: list[dict[str, Any]],
    check_rows: list[dict[str, Any]],
    fixes: list[dict[str, Any]],
    unresolved: list[str],
) -> str:
    verdict = overall_status(check_rows)
    ready_scientific = verdict in {"pass", "warn"} and not any(row["status"] == "fail" for row in check_rows)
    ready_formatting = verdict == "pass"
    source_rows = [{"artifact": key, "found": value} for key, value in source_status.items()]
    return "\n".join(
        [
            "# Manuscript v002 Consistency Audit",
            "",
            "## Inputs",
            "",
            f"- Manuscript: `{manuscript}`",
            f"- Hardening directory: `{hardening_dir}`",
            "",
            "## Source Artifact Availability",
            "",
            markdown_table(source_rows, ["artifact", "found"]),
            "",
            "## Section Status Summary",
            "",
            markdown_table(section_rows, ["section", "status", "n_pass", "n_warn", "n_fail"]),
            "",
            "## Detailed Checks",
            "",
            markdown_table(check_rows, ["section", "check", "status", "evidence", "recommendation"]),
            "",
            "## Required Fixes / Warnings",
            "",
            markdown_table(fixes, ["section", "issue", "status", "evidence", "recommended_fix"]),
            "",
            "## Unresolved Placeholders",
            "",
            "- " + ("\n- ".join(unresolved) if unresolved else "None detected."),
            "",
            "## Readiness Verdict",
            "",
            f"- Overall audit status: `{verdict}`.",
            f"- Ready for scientific language editing: `{ready_scientific}`.",
            f"- Ready for journal formatting: `{ready_formatting}`.",
            "",
            "If the status is `pass`, v002 correctly reflects the final methodological hardening decisions and can proceed to scientific language editing. Journal formatting should wait until planned Tables 7-8 and Supplementary Tables S5-S6 are materialized.",
        ]
    )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    text = load_text(args.manuscript)
    source_status = source_artifact_status(args.hardening_dir)

    check_rows: list[dict[str, Any]] = []
    for name, found in source_status.items():
        add_check(
            check_rows,
            "Source artifacts",
            f"Source artifact `{name}` present",
            status_from_bool(found),
            "found" if found else "missing",
            "Restore or regenerate the missing hardening artifact." if not found else "",
        )

    audit_claim_boundaries(text, check_rows)
    audit_knn(text, check_rows)
    audit_dataset(text, check_rows)
    audit_material_lomo(text, check_rows)
    audit_paired(text, check_rows)
    audit_frequency(text, check_rows)
    audit_hierarchical(text, check_rows)
    unresolved = audit_tables(text, check_rows)

    section_rows = section_statuses(check_rows)
    fixes = build_required_fixes(check_rows)
    payload = {
        "manuscript": str(args.manuscript),
        "hardening_dir": str(args.hardening_dir),
        "overall_status": overall_status(check_rows),
        "ready_for_scientific_language_editing": overall_status(check_rows) in {"pass", "warn"}
        and not any(row["status"] == "fail" for row in check_rows),
        "ready_for_journal_formatting": overall_status(check_rows) == "pass",
        "source_artifacts": source_status,
        "section_statuses": section_rows,
        "checks": check_rows,
        "required_fixes": fixes,
        "unresolved_placeholders": unresolved,
    }

    write_json(args.out_dir / "manuscript_v002_consistency_audit.json", payload)
    write_csv(args.out_dir / "manuscript_v002_required_fixes.csv", fixes)
    (args.out_dir / "MANUSCRIPT_V002_CONSISTENCY_AUDIT.md").write_text(
        render_report(args.manuscript, args.hardening_dir, source_status, section_rows, check_rows, fixes, unresolved),
        encoding="utf-8",
    )
    print(f"[done] wrote manuscript v002 consistency audit to {args.out_dir}")
    print(f"[done] overall_status={payload['overall_status']}")
    print(f"[done] required_fixes={len(fixes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
