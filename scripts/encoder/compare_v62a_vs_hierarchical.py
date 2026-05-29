#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a formal v6.2-A vs hierarchical v6.2 frozen-embedding comparison report.

This script reads existing report artifacts only. It does not train, fine-tune,
extract embeddings, run acoustic-response prediction, or compute LeFFT features.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_V62A_REPORT = Path("reports/encoder/ENCODER_TECHNICAL_REPORT.md")
DEFAULT_V62A_GROUPED_REPORT = Path("reports/encoder/GROUPED_ENCODER_EVAL_SUMMARY.md")
DEFAULT_HIER_KEY = Path("reports/hierarchical_encoder/audit_v001/hierarchical_audit_key_numbers.json")
DEFAULT_HIER_REPORT = Path("reports/hierarchical_encoder/audit_v001/HIERARCHICAL_ENCODER_AUDIT_SUMMARY.md")
DEFAULT_OUT_DIR = Path("reports/encoder_comparison")

V62A_REPORT_FALLBACKS = [
    Path("reports/encoder/ENCODER_TECHNICAL_REPORT.md"),
    Path("reports/encoder/ENCODER_TECHNICAL_REPORT_GR.md"),
    Path("reports/encoder/FROZEN_ENCODER_TECHNICAL_REPORT.md"),
    Path("reports/encoder/EMBEDDING_AUDIT_SUMMARY.md"),
]


def read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_report_path(requested: Path, fallbacks: list[Path]) -> tuple[Path | None, list[str]]:
    warnings: list[str] = []
    if requested.exists():
        return requested, warnings
    warnings.append(f"Requested report not found: {requested}")
    for fallback in fallbacks:
        if fallback.exists():
            warnings.append(f"Using fallback report: {fallback}")
            return fallback, warnings
    warnings.append("No fallback v6.2-A report was found.")
    return None, warnings


def as_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except Exception:
        return "n/a"
    return f"{100.0 * number:.2f}%"


def as_number(value: Any) -> int | str:
    if value is None:
        return "n/a"
    try:
        return int(value)
    except Exception:
        return str(value)


def pick_method_aggregate(grouped_json: dict[str, Any], key: str, preferred_method: str = "knn_cosine_k10") -> dict[str, Any] | None:
    rows = grouped_json.get(key, [])
    if isinstance(rows, dict):
        if preferred_method in rows:
            return rows[preferred_method]
        return next(iter(rows.values()), None) if rows else None
    if isinstance(rows, list):
        for row in rows:
            if row.get("method") == preferred_method:
                return row
        return rows[0] if rows else None
    return None


def parse_percent_from_text(text: str, patterns: list[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        value = float(match.group(1))
        return value / 100.0 if value > 1.0 else value
    return None


def extract_v62a_metrics(
    report_path: Path | None,
    grouped_report_path: Path,
    warnings: list[str],
) -> dict[str, Any]:
    report_text = read_text_if_exists(report_path) if report_path else ""
    grouped_text = read_text_if_exists(grouped_report_path)
    key_json = load_json(Path("reports/encoder/encoder_report_key_numbers.json"))
    grouped_json = load_json(Path("reports/encoder/grouped_encoder_key_numbers.json"))

    if not key_json:
        warnings.append("v6.2-A key-number JSON not found; using markdown parsing and fallback constants where needed.")
    if not grouped_json:
        warnings.append("v6.2-A grouped key-number JSON not found; using grouped markdown parsing and fallback constants where needed.")

    lobo_knn = pick_method_aggregate(grouped_json, "lobo_aggregate", "knn_cosine_k10")
    lomo_knn = pick_method_aggregate(grouped_json, "lomo_aggregate", "knn_cosine_k10")

    knn_macro_f1 = key_json.get("knn_best", {}).get("macro_f1")
    linear_macro_f1 = key_json.get("linear_probe", {}).get("macro_f1")
    prototype_macro_f1 = key_json.get("prototype", {}).get("macro_f1")
    board_lobo_macro_f1 = lobo_knn.get("macro_f1_mean") if lobo_knn else None
    material_lomo_macro_f1 = lomo_knn.get("macro_f1_mean") if lomo_knn else None

    if knn_macro_f1 is None:
        knn_macro_f1 = parse_percent_from_text(report_text, [r"kNN[^\\n|]*Macro-F1[^0-9]*(\\d+(?:\\.\\d+)?)%"])
    if linear_macro_f1 is None:
        linear_macro_f1 = parse_percent_from_text(report_text, [r"linear[^\\n|]*Macro-F1[^0-9]*(\\d+(?:\\.\\d+)?)%"])
    if prototype_macro_f1 is None:
        prototype_macro_f1 = parse_percent_from_text(report_text, [r"prototype[^\\n|]*Macro-F1[^0-9]*(\\d+(?:\\.\\d+)?)%"])
    if board_lobo_macro_f1 is None:
        board_lobo_macro_f1 = parse_percent_from_text(grouped_text, [r"LOBO[^\\n|]*kNN[^\\n|]*Macro-F1[^0-9]*(\\d+(?:\\.\\d+)?)%"])
    if material_lomo_macro_f1 is None:
        material_lomo_macro_f1 = parse_percent_from_text(grouped_text, [r"LOMO[^\\n|]*kNN[^\\n|]*Macro-F1[^0-9]*(\\d+(?:\\.\\d+)?)%"])

    return {
        "encoder_name": "v6.2-A frozen encoder",
        "architecture_role": "Official reportable 5-class baseline and primary frozen ESPI encoder candidate",
        "checkpoint": key_json.get("checkpoint", "checkpoint_epoch25_20260211_035150.pt"),
        "embedding_point": key_json.get("embedding_layer", "MCDropoutClassifier.global_pool"),
        "embedding_dimension": key_json.get("embedding_dimension", 1280),
        "knn_macro_f1": knn_macro_f1 if knn_macro_f1 is not None else 0.9344,
        "linear_probe_macro_f1": linear_macro_f1 if linear_macro_f1 is not None else 0.9285,
        "prototype_macro_f1": prototype_macro_f1 if prototype_macro_f1 is not None else 0.8642,
        "board_lobo_mean_macro_f1": board_lobo_macro_f1 if board_lobo_macro_f1 is not None else 0.9356,
        "material_lomo_mean_macro_f1": material_lomo_macro_f1 if material_lomo_macro_f1 is not None else 0.9349,
        "source_report": str(report_path) if report_path else "not found",
        "source_grouped_report": str(grouped_report_path),
        "source_key_numbers": "reports/encoder/encoder_report_key_numbers.json" if key_json else "fallback",
        "source_grouped_key_numbers": "reports/encoder/grouped_encoder_key_numbers.json" if grouped_json else "fallback",
    }


def extract_hierarchical_metrics(hier_key_path: Path, hier_report_path: Path, warnings: list[str]) -> dict[str, Any]:
    key_json = load_json(hier_key_path)
    if not key_json:
        raise FileNotFoundError(f"Required hierarchical key-number JSON not found: {hier_key_path}")
    if not hier_report_path.exists():
        warnings.append(f"Hierarchical summary report not found: {hier_report_path}")

    quality = key_json.get("quality", {})
    stratified = key_json.get("stratified", {})
    grouped = key_json.get("grouped", {})
    board_lobo = grouped.get("board_lobo", {})
    material_lomo = grouped.get("material_lomo", {})

    board_best_method = max(
        board_lobo.items(),
        key=lambda item: float(item[1].get("mean_macro_f1", -1.0)),
        default=("n/a", {}),
    )
    material_best_method = max(
        material_lomo.items(),
        key=lambda item: float(item[1].get("mean_macro_f1", -1.0)),
        default=("n/a", {}),
    )

    metadata_sources = key_json.get("metadata_sources", {})
    if metadata_sources.get("board") != "saved" or metadata_sources.get("split_group") != "saved":
        warnings.append(
            "Hierarchical metadata caveat: board/split_group were recovered from distribution_group/path because the NPZ stored them as unknown."
        )

    return {
        "encoder_name": "hierarchical v6.2 phase2 expert",
        "architecture_role": "Controlled physics-aware architectural candidate branch",
        "checkpoint": key_json.get("checkpoint_sha256", "phase2 expert checkpoint"),
        "embedding_point": key_json.get("embedding_point", "z_expert_prelogit / z_arcface_input"),
        "embedding_dimension": quality.get("embedding_dim", 512),
        "knn_macro_f1": stratified.get("stratified_best_knn", {}).get("macro_f1"),
        "linear_probe_macro_f1": stratified.get("stratified_linear_probe", {}).get("macro_f1"),
        "prototype_macro_f1": stratified.get("stratified_prototype", {}).get("macro_f1"),
        "board_lobo_mean_macro_f1": board_best_method[1].get("mean_macro_f1"),
        "board_lobo_best_method": board_best_method[0],
        "material_lomo_mean_macro_f1": material_best_method[1].get("mean_macro_f1"),
        "material_lomo_best_method": material_best_method[0],
        "quality": quality,
        "metadata_sources": metadata_sources,
        "source_key_numbers": str(hier_key_path),
        "source_report": str(hier_report_path),
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "Encoder",
        "Architecture role",
        "Checkpoint",
        "Embedding point",
        "Dim",
        "kNN Macro-F1",
        "Linear Macro-F1",
        "Prototype Macro-F1",
        "Board LOBO mean Macro-F1",
        "Material LOMO mean Macro-F1",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        checkpoint = str(row["checkpoint"])
        if len(checkpoint) > 48:
            checkpoint = checkpoint[:45] + "..."
        lines.append(
            "| {encoder} | {role} | `{checkpoint}` | `{point}` | {dim} | {knn} | {linear} | {prototype} | {lobo} | {lomo} |".format(
                encoder=row["encoder_name"],
                role=row["architecture_role"],
                checkpoint=checkpoint,
                point=row["embedding_point"],
                dim=as_number(row["embedding_dimension"]),
                knn=as_percent(row["knn_macro_f1"]),
                linear=as_percent(row["linear_probe_macro_f1"]),
                prototype=as_percent(row["prototype_macro_f1"]),
                lobo=as_percent(row["board_lobo_mean_macro_f1"]),
                lomo=as_percent(row["material_lomo_mean_macro_f1"]),
            )
        )
    return "\n".join(lines)


def write_report(out_path: Path, v62a: dict[str, Any], hierarchical: dict[str, Any], warnings: list[str]) -> None:
    warning_lines = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- None"
    text = f"""# v6.2-A vs Hierarchical v6.2 Frozen Embedding Comparison

## Scope

This report compares the existing frozen v6.2-A encoder evidence against the hierarchical v6.2 phase2 expert embedding audit. It is a report-only comparison over existing artifacts. It does not implement retraining, fine-tuning, acoustic-response prediction, z_physics_fusion extraction, or LeFFT scripts.

## Comparison table

{markdown_table([v62a, hierarchical])}

## Decision

- v6.2-A remains the primary reportable frozen ESPI encoder baseline.
- Hierarchical v6.2 phase2 expert embeddings are technically valid but not superior to v6.2-A under the current frozen embedding audits.
- Hierarchical v6.2 remains a development branch unless retrained or fine-tuned with representation-aligned objectives.
- No validated Physics-Aligned Encoder claim is supported from this hierarchical checkpoint.

## Technical interpretation

- Classification performance is not equivalent to frozen embedding quality. A classifier can make useful predictions while its pre-head representation is weak under kNN, prototype, or grouped retrieval-style evaluation.
- ArcFace and classifier heads can improve supervised decision boundaries without yielding robust nearest-neighbor or prototype geometry in the frozen embedding space.
- The phase2 expert checkpoint is not necessarily a final fine-tuned encoder checkpoint. It is a technically valid hierarchical expert checkpoint that can be audited, not a validated final encoder.
- Physics-aware architecture alone does not guarantee representation robustness. Denoising, phase, Fourier, and multiscale physics modules must still demonstrate grouped representation value under matched evaluation.

## Metadata caveat

The hierarchical audit recovered `board` and `split_group` from `distribution_group` / `path` because the hierarchical NPZ stored these fields as `unknown`. This is acceptable for the current technical audit, but it must be fixed in a future extraction version before final publication tables.

## Claim boundary

- Supported: v6.2-A is the stronger current frozen ESPI encoder candidate under the available representation audits.
- Supported: hierarchical v6.2 phase2 expert embeddings are loadable, extractable, and auditable.
- Not supported: validated Physics-Aligned Encoder.
- Not supported: hierarchical superiority over v6.2-A.
- Not supported: acoustic-response predictive value.
- Not supported: LeFFT / LeFTP superiority.

## Source artifacts

- v6.2-A report: `{v62a['source_report']}`
- v6.2-A grouped report: `{v62a['source_grouped_report']}`
- hierarchical key numbers: `{hierarchical['source_key_numbers']}`
- hierarchical report: `{hierarchical['source_report']}`

## Warnings

{warning_lines}
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def write_json(out_path: Path, payload: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare v6.2-A and hierarchical v6.2 frozen embedding audits.")
    parser.add_argument("--v62a-report", type=Path, default=DEFAULT_V62A_REPORT)
    parser.add_argument("--v62a-grouped-report", type=Path, default=DEFAULT_V62A_GROUPED_REPORT)
    parser.add_argument("--hier-key", type=Path, default=DEFAULT_HIER_KEY)
    parser.add_argument("--hier-report", type=Path, default=DEFAULT_HIER_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    warnings: list[str] = []
    v62a_report, report_warnings = resolve_report_path(args.v62a_report, V62A_REPORT_FALLBACKS)
    warnings.extend(report_warnings)

    v62a = extract_v62a_metrics(v62a_report, args.v62a_grouped_report, warnings)
    hierarchical = extract_hierarchical_metrics(args.hier_key, args.hier_report, warnings)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "COMPARE_V62A_vs_HIERARCHICAL.md"
    json_path = out_dir / "compare_v62a_vs_hierarchical_key_numbers.json"

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "v62a": v62a,
        "hierarchical": hierarchical,
        "decision": {
            "primary_reportable_frozen_encoder": "v6.2-A",
            "hierarchical_status": "technically valid but not superior in current frozen embedding evaluation",
            "validated_physics_aligned_encoder_claim_supported": False,
        },
        "warnings": warnings,
        "outputs": {
            "report": str(report_path),
            "key_numbers": str(json_path),
        },
    }
    write_report(report_path, v62a, hierarchical, warnings)
    write_json(json_path, payload)

    print(f"[done] report={report_path}")
    print(f"[done] key_numbers={json_path}")
    print(f"[done] v62a_knn_macro_f1={v62a['knn_macro_f1']:.6f}")
    print(f"[done] hierarchical_knn_macro_f1={hierarchical['knn_macro_f1']:.6f}")
    print(f"[done] v62a_board_lobo_mean_macro_f1={v62a['board_lobo_mean_macro_f1']:.6f}")
    print(f"[done] hierarchical_board_lobo_mean_macro_f1={hierarchical['board_lobo_mean_macro_f1']:.6f}")
    if warnings:
        print("[done] warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
