#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA and consistency checks for unified encoder baseline evaluation artifacts.

This script reads existing evaluation outputs and normalized feature dumps only.
It does not rerun evaluation, LeFFT, inference, training, or acoustic-response
prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_ENCODERS = [
    "random_resnet18",
    "imagenet_resnet18",
    "v6_1",
    "v6_2_a",
    "hierarchical_v6_2",
]

FEATURE_FILES = {
    "random_resnet18": "features_resnet18_random.normalized.npz",
    "imagenet_resnet18": "features_resnet18_imagenet.normalized.npz",
    "v6_1": "features_v61.normalized.npz",
    "v6_2_a": "features_v62a_epoch25.normalized.npz",
    "hierarchical_v6_2": "features_hier_z_expert_prelogit.normalized.npz",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def close_enough(a: Any, b: Any, tol: float = 1e-9) -> bool:
    af = as_float(a)
    bf = as_float(b)
    if math.isnan(af) and math.isnan(bf):
        return True
    return abs(af - bf) <= tol


def percent(value: Any) -> str:
    number = as_float(value)
    return "n/a" if math.isnan(number) else f"{100.0 * number:.2f}%"


def add_check(rows: list[dict[str, Any]], check: str, status: str, detail: str) -> None:
    rows.append({"check": check, "status": status, "detail": detail})


def load_eval_artifacts(eval_dir: Path) -> dict[str, Any]:
    json_path = eval_dir / "encoder_baseline_key_numbers.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing key-number JSON: {json_path}")
    return {
        "report_text": (eval_dir / "ENCODER_BASELINE_EVALUATION.md").read_text(encoding="utf-8"),
        "key_numbers": json.loads(json_path.read_text(encoding="utf-8")),
        "summary": read_csv(eval_dir / "encoder_baseline_summary.csv"),
        "quality": read_csv(eval_dir / "quality_summary.csv"),
        "knn": read_csv(eval_dir / "knn_summary.csv"),
        "linear": read_csv(eval_dir / "linear_probe_summary.csv"),
        "prototype": read_csv(eval_dir / "prototype_summary.csv"),
        "lobo": read_csv(eval_dir / "lobo_board_summary.csv"),
        "lomo": read_csv(eval_dir / "lomo_material_summary.csv"),
        "permutation": read_csv(eval_dir / "label_permutation_summary.csv"),
    }


def parse_md_main_table(report_text: str) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    in_table = False
    for line in report_text.splitlines():
        if line.startswith("| Encoder | Dim | Best kNN"):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            parts = [part.strip() for part in line.strip().strip("|").split("|")]
            if len(parts) < 7:
                continue
            encoder = parts[0]
            rows[encoder] = {
                "stratified_best_knn_macro_f1": float(parts[2].rstrip("%")) / 100.0,
                "linear_probe_macro_f1": float(parts[3].rstrip("%")) / 100.0,
                "prototype_macro_f1": float(parts[4].rstrip("%")) / 100.0,
                "best_lobo_mean_macro_f1": float(parts[5].rstrip("%")) / 100.0,
                "best_lomo_mean_macro_f1": float(parts[6].rstrip("%")) / 100.0,
            }
            continue
        if in_table and not line.startswith("|"):
            break
    return rows


def check_bootstrap(artifacts: dict[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any]:
    key_numbers = artifacts["key_numbers"]
    bootstrap_used = bool(key_numbers.get("bootstrap_used"))
    bootstrap_iters = int(key_numbers.get("bootstrap_iters", 0))
    ci_columns = []
    ci_values = []
    for name in ["knn", "linear", "prototype", "lobo", "lomo", "permutation"]:
        rows = artifacts[name]
        if rows and "macro_f1_ci95_low" in rows[0]:
            ci_columns.append(name)
            for row in rows:
                ci_values.append(row.get("macro_f1_ci95_low", "nan"))
                ci_values.append(row.get("macro_f1_ci95_high", "nan"))
    finite_ci = [
        value
        for value in ci_values
        if value not in {"", "nan", "NaN", "None", None}
        and not math.isnan(as_float(value))
    ]
    expected = "enabled" if bootstrap_used else "skipped"
    add_check(
        checks,
        "bootstrap_status",
        "pass",
        f"Bootstrap was {expected}; bootstrap_iters={bootstrap_iters}; CI columns in {ci_columns}; finite CI values={len(finite_ci)}.",
    )
    if bootstrap_used and not finite_ci:
        add_check(checks, "bootstrap_ci_presence", "fail", "Bootstrap is marked enabled but no finite CI values were found.")
    elif not bootstrap_used and finite_ci:
        add_check(checks, "bootstrap_ci_presence", "warn", "Bootstrap is marked skipped but finite CI values were found.")
    else:
        add_check(checks, "bootstrap_ci_presence", "pass", "Bootstrap CI presence matches bootstrap setting.")
    return {
        "bootstrap_used": bootstrap_used,
        "bootstrap_iters": bootstrap_iters,
        "ci_columns": ci_columns,
        "finite_ci_values": len(finite_ci),
    }


def check_convergence(eval_dir: Path, artifacts: dict[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any]:
    searched_files = []
    hits = []
    for pattern in ["*.md", "*.json", "*.csv", "*.log", "*.txt"]:
        for path in eval_dir.glob(pattern):
            searched_files.append(str(path))
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if "ConvergenceWarning" in text or "failed to converge" in text:
                hits.append(str(path))
    if hits:
        add_check(checks, "linear_probe_convergence_warnings", "warn", f"Convergence warning text found in: {hits}")
        recommendation = "Rerun linear probe with higher max_iter before citing final linear-probe values."
    else:
        add_check(
            checks,
            "linear_probe_convergence_warnings",
            "warn",
            "No convergence-warning text found in saved eval artifacts. Console warnings are not captured by the current report format.",
        )
        recommendation = "Rerun linear probe with higher max_iter if linear-probe metrics will be cited; otherwise kNN/grouped conclusions are unaffected."
    return {"searched_files": searched_files, "warning_files": hits, "recommendation": recommendation}


def best_rows(rows: list[dict[str, str]], encoder_id: str, method_prefix: str | None = None) -> dict[str, str] | None:
    candidates = [
        row
        for row in rows
        if row.get("encoder_id") == encoder_id
        and row.get("macro_f1", "") not in {"", "nan"}
        and (method_prefix is None or row.get("method", "").startswith(method_prefix))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: as_float(row["macro_f1"]))


def grouped_aggregate(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("macro_f1", "") in {"", "nan"}:
            continue
        buckets[(row["encoder_id"], row["method"])].append(row)
    output = {}
    for key, bucket in buckets.items():
        values = [as_float(row["macro_f1"]) for row in bucket]
        worst = min(bucket, key=lambda row: as_float(row["macro_f1"]))
        best = max(bucket, key=lambda row: as_float(row["macro_f1"]))
        output[key] = {
            "mean_macro_f1": float(np.mean(values)),
            "worst_group": worst.get("held_out_group", ""),
            "worst_group_macro_f1": as_float(worst["macro_f1"]),
            "best_group": best.get("held_out_group", ""),
            "best_group_macro_f1": as_float(best["macro_f1"]),
        }
    return output


def check_metric_consistency(artifacts: dict[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {row["encoder_id"]: row for row in artifacts["summary"]}
    key_summary = {row["encoder_id"]: row for row in artifacts["key_numbers"]["summary"]}
    md_summary = parse_md_main_table(artifacts["report_text"])
    lobo_agg = grouped_aggregate(artifacts["lobo"])
    lomo_agg = grouped_aggregate(artifacts["lomo"])
    issues = []

    for encoder_id, row in summary.items():
        key_row = key_summary.get(encoder_id)
        if not key_row:
            issues.append(f"{encoder_id}: missing from key-number JSON")
            continue
        for field in [
            "stratified_best_knn_macro_f1",
            "linear_probe_macro_f1",
            "prototype_macro_f1",
            "best_lobo_mean_macro_f1",
            "best_lomo_mean_macro_f1",
        ]:
            if not close_enough(row[field], key_row[field]):
                issues.append(f"{encoder_id}: summary CSV vs JSON mismatch for {field}")

        md_row = md_summary.get(row["encoder_name"])
        if not md_row:
            issues.append(f"{encoder_id}: missing from Markdown table")
        else:
            for field in md_row:
                if abs(as_float(row[field]) - md_row[field]) > 5e-5:
                    issues.append(f"{encoder_id}: Markdown rounded value mismatch for {field}")

        best_knn = best_rows(artifacts["knn"], encoder_id, "knn_cosine")
        if best_knn and (
            best_knn["method"] != row["stratified_best_knn_method"]
            or not close_enough(best_knn["macro_f1"], row["stratified_best_knn_macro_f1"])
        ):
            issues.append(f"{encoder_id}: best kNN mismatch")
        linear = best_rows(artifacts["linear"], encoder_id)
        if linear and not close_enough(linear["macro_f1"], row["linear_probe_macro_f1"]):
            issues.append(f"{encoder_id}: linear macro-F1 mismatch")
        proto = best_rows(artifacts["prototype"], encoder_id)
        if proto and not close_enough(proto["macro_f1"], row["prototype_macro_f1"]):
            issues.append(f"{encoder_id}: prototype macro-F1 mismatch")

        best_lobo = max(
            [value for (enc, _), value in lobo_agg.items() if enc == encoder_id],
            key=lambda value: value["mean_macro_f1"],
            default=None,
        )
        if best_lobo and (
            not close_enough(best_lobo["mean_macro_f1"], row["best_lobo_mean_macro_f1"])
            or best_lobo["worst_group"] != row["best_lobo_worst_group"]
            or not close_enough(best_lobo["worst_group_macro_f1"], row["best_lobo_worst_group_macro_f1"])
        ):
            issues.append(f"{encoder_id}: LOBO aggregate mismatch")
        best_lomo = max(
            [value for (enc, _), value in lomo_agg.items() if enc == encoder_id],
            key=lambda value: value["mean_macro_f1"],
            default=None,
        )
        if best_lomo and (
            not close_enough(best_lomo["mean_macro_f1"], row["best_lomo_mean_macro_f1"])
            or best_lomo["worst_group"] != row["best_lomo_worst_group"]
            or not close_enough(best_lomo["worst_group_macro_f1"], row["best_lomo_worst_group_macro_f1"])
        ):
            issues.append(f"{encoder_id}: LOMO aggregate mismatch")

    add_check(
        checks,
        "metric_consistency",
        "pass" if not issues else "fail",
        "All metrics are consistent across Markdown, JSON, summary CSV, and detailed CSVs." if not issues else "; ".join(issues),
    )
    return {"issues": issues, "markdown_rows_found": list(md_summary.keys())}


def check_feature_consistency(features_dir: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    arrays = {}
    metadata = {}
    for encoder_id, filename in FEATURE_FILES.items():
        path = features_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing feature dump: {path}")
        with np.load(str(path), allow_pickle=True) as data:
            arrays[encoder_id] = {
                "labels": data["label"].astype(np.int64),
                "paths": data["path"].astype(str),
                "split_group": data["split_group"].astype(str),
                "board": data["board"].astype(str),
                "material": data["material"].astype(str),
            }
            metadata[encoder_id] = {
                "sample_count": int(data["embedding"].shape[0]),
                "embedding_dim": int(data["embedding"].shape[1]),
                "model_name": str(data["model_name"].item()),
                "source_npz": str(data["source_npz"].item()),
            }
    reference_id = EXPECTED_ENCODERS[0]
    ref = arrays[reference_id]
    issues = []
    for encoder_id, values in arrays.items():
        if not np.array_equal(values["labels"], ref["labels"]):
            issues.append(f"{encoder_id}: labels differ from {reference_id}")
        if not np.array_equal(values["paths"], ref["paths"]):
            issues.append(f"{encoder_id}: paths differ from {reference_id}")
        if not np.array_equal(values["split_group"], ref["split_group"]):
            issues.append(f"{encoder_id}: split_group differs from {reference_id}")
        if not np.array_equal(values["board"], ref["board"]):
            issues.append(f"{encoder_id}: board differs from {reference_id}")
        if not np.array_equal(values["material"], ref["material"]):
            issues.append(f"{encoder_id}: material differs from {reference_id}")
    add_check(
        checks,
        "feature_metadata_consistency",
        "pass" if not issues else "fail",
        "All five feature dumps use identical labels, paths, board, material, and split_group arrays." if not issues else "; ".join(issues),
    )

    recovery_status = {}
    normalization_json = Path("reports/encoder_baselines/feature_schema_normalization.json")
    if normalization_json.exists():
        norm = json.loads(normalization_json.read_text(encoding="utf-8"))
        for summary in norm.get("summaries", []):
            recovery_status[summary["encoder"]] = {
                "recovered_fields": summary.get("recovered_fields", []),
                "acceptable_for_internal_comparison": summary.get("acceptable_for_internal_comparison"),
                "acceptable_for_final_publication_tables": summary.get("acceptable_for_final_publication_tables"),
            }
    recovery_status["hierarchical_note"] = "hierarchical normalized dump recovered board/split_group and remains internal-only for final publication tables"
    return {"metadata": metadata, "issues": issues, "recovery_status": recovery_status}


def check_label_permutation(artifacts: dict[str, Any], checks: list[dict[str, Any]], threshold: float = 0.25) -> dict[str, Any]:
    rows = artifacts["permutation"]
    issues = []
    by_encoder = defaultdict(list)
    for row in rows:
        by_encoder[row["encoder_id"]].append(row)
        macro_f1 = as_float(row.get("macro_f1"))
        if math.isnan(macro_f1) or macro_f1 > threshold:
            issues.append(f"{row.get('encoder_id')} {row.get('method')}: permutation Macro-F1={macro_f1}")
    for encoder_id in EXPECTED_ENCODERS:
        methods = {row["method"] for row in by_encoder.get(encoder_id, [])}
        if not any("knn" in method for method in methods):
            issues.append(f"{encoder_id}: missing kNN permutation row")
        if not any("logistic" in method for method in methods):
            issues.append(f"{encoder_id}: missing linear permutation row")
    add_check(
        checks,
        "label_permutation_sanity",
        "pass" if not issues else "warn",
        f"Permutation Macro-F1 threshold={threshold}; all permutation metrics are near chance." if not issues else "; ".join(issues),
    )
    return {"threshold": threshold, "issues": issues, "rows": rows}


def recommendations(bootstrap: dict[str, Any], convergence: dict[str, Any], feature: dict[str, Any]) -> list[str]:
    output = []
    output.append(convergence["recommendation"])
    if not bootstrap["bootstrap_used"]:
        output.append("Rerun bootstrap before final manuscript tables if confidence intervals will be reported.")
    else:
        output.append("Bootstrap is already enabled; inspect CI widths before final manuscript tables.")
    output.append("Regenerate the hierarchical NPZ before final publication tables because board/split_group were recovered rather than stored correctly in the source NPZ.")
    output.append("Do not rerun acoustic-response or LeFFT experiments as part of this QA step.")
    return output


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    checks = payload["checks"]
    permutation = payload["label_permutation"]
    recs = payload["recommendations"]
    check_lines = "\n".join(f"- `{row['check']}`: **{row['status']}** - {row['detail']}" for row in checks)
    summary_lines = [
        "| Encoder | Stratified kNN Macro-F1 | Board LOBO mean Macro-F1 | Board worst-group Macro-F1 | Material LOMO mean Macro-F1 | Material worst-group Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        summary_lines.append(
            f"| {row['encoder_name']} | {percent(row['stratified_best_knn_macro_f1'])} | {percent(row['best_lobo_mean_macro_f1'])} | {percent(row['best_lobo_worst_group_macro_f1'])} | {percent(row['best_lomo_mean_macro_f1'])} | {percent(row['best_lomo_worst_group_macro_f1'])} |"
        )
    permutation_lines = [
        "| Encoder | Method | Macro-F1 |",
        "|---|---|---:|",
    ]
    for row in permutation["rows"]:
        permutation_lines.append(f"| {row['encoder_name']} | {row['method']} | {percent(row['macro_f1'])} |")
    rec_lines = "\n".join(f"- {item}" for item in recs)
    text = f"""# Encoder Evaluation QA Report

## Scope

This QA report checks consistency of the existing unified encoder baseline evaluation. It does not rerun evaluation, train models, implement LeFFT, or run acoustic-response prediction.

## QA checks

{check_lines}

## Metric summary

{chr(10).join(summary_lines)}

## Bootstrap status

- Bootstrap used: `{payload['bootstrap']['bootstrap_used']}`
- Bootstrap iterations: `{payload['bootstrap']['bootstrap_iters']}`
- Finite CI values found: `{payload['bootstrap']['finite_ci_values']}`

## Label-permutation sanity check

{chr(10).join(permutation_lines)}

## Interpretation

Random and ImageNet ResNet-18 embeddings can score surprisingly high under stratified kNN because the train/validation split shares board, material, acquisition, and distribution signatures. Nearest-neighbor retrieval can exploit these local signatures without learning board/material-invariant modal structure. Their collapse under board-grouped and material-grouped evaluation shows that the stratified score is not sufficient evidence of encoder generalization.

v6.1 has the highest stratified kNN Macro-F1, but v6.2-A has the strongest board/material grouped robustness. For publication claims about frozen encoder generalization, grouped robustness should be prioritized over stratified-only performance.

Hierarchical v6.2 remains an internal-only comparison row until the source feature extraction is regenerated with correct `board` and `split_group` stored directly in the source NPZ.

## Recommendations

{rec_lines}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check unified encoder evaluation consistency.")
    parser.add_argument("--eval-dir", type=Path, default=Path("reports/encoder_baselines/eval_v001"))
    parser.add_argument("--features-dir", type=Path, default=Path("outputs/encoder_features_normalized_v001"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/encoder_baselines/eval_v001/qa"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks: list[dict[str, Any]] = []
    artifacts = load_eval_artifacts(args.eval_dir)
    bootstrap = check_bootstrap(artifacts, checks)
    convergence = check_convergence(args.eval_dir, artifacts, checks)
    metric = check_metric_consistency(artifacts, checks)
    feature = check_feature_consistency(args.features_dir, checks)
    permutation = check_label_permutation(artifacts, checks)
    recs = recommendations(bootstrap, convergence, feature)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "eval_dir": str(args.eval_dir),
        "features_dir": str(args.features_dir),
        "checks": checks,
        "bootstrap": bootstrap,
        "convergence": convergence,
        "metric_consistency": metric,
        "feature_consistency": feature,
        "label_permutation": permutation,
        "summary": artifacts["key_numbers"]["summary"],
        "recommendations": recs,
    }
    report_path = args.out_dir / "ENCODER_EVAL_QA_REPORT.md"
    json_path = args.out_dir / "encoder_eval_qa.json"
    write_report(report_path, payload)
    write_json(json_path, payload)
    print(f"[done] report={report_path}")
    print(f"[done] json={json_path}")
    for row in checks:
        print(f"[{row['status']}] {row['check']}: {row['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
