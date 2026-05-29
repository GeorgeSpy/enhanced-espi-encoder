#!/usr/bin/env python
"""Create kNN fixed-k and best-k diagnostic hardening tables.

This script reads existing eval_v002 outputs only. It does not rerun encoder
evaluation, modify feature dumps, train/fine-tune models, implement LeFFT, or
evaluate acoustic-response prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


FIXED_K_VALUES = [1, 5, 10]
ALL_K_VALUES = [1, 3, 5, 10, 20]
RECOMMENDED_K = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def pct(value: float | str) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "n/a"


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def extract_stratified_knn(knn_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in knn_rows:
        if row.get("split") != "train_to_val":
            continue
        if not row.get("method", "").startswith("knn_cosine_k"):
            continue
        output.append(
            {
                "protocol": "stratified_train_val",
                "encoder_id": row["encoder_id"],
                "encoder_name": row["encoder_name"],
                "method": row["method"],
                "k": int(float(row["k"])),
                "n_train": int(float(row["n_train"])),
                "n_test": int(float(row["n_test"])),
                "accuracy": safe_float(row["accuracy"]),
                "macro_recall": safe_float(row["macro_recall"]),
                "macro_f1": safe_float(row["macro_f1"]),
                "macro_f1_ci95_low": safe_float(row.get("macro_f1_ci95_low")),
                "macro_f1_ci95_high": safe_float(row.get("macro_f1_ci95_high")),
                "source": "knn_summary.csv",
            }
        )
    return output


def summarize_grouped_knn(rows: list[dict[str, str]], protocol: str, source: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        method = row.get("method", "")
        if not method.startswith("knn_cosine_k"):
            continue
        if row.get("k") in ("", None):
            continue
        grouped[(row["encoder_id"], row["encoder_name"], int(float(row["k"])))].append(row)

    output: list[dict[str, Any]] = []
    for (encoder_id, encoder_name, k_value), fold_rows in sorted(grouped.items()):
        scores = [safe_float(row["macro_f1"]) for row in fold_rows]
        worst = min(fold_rows, key=lambda row: safe_float(row["macro_f1"]))
        best = max(fold_rows, key=lambda row: safe_float(row["macro_f1"]))
        output.append(
            {
                "protocol": protocol,
                "encoder_id": encoder_id,
                "encoder_name": encoder_name,
                "method": f"knn_cosine_k{k_value}",
                "k": k_value,
                "n_groups": len(fold_rows),
                "mean_macro_f1": sum(scores) / len(scores) if scores else float("nan"),
                "worst_group": worst.get("held_out_group", ""),
                "worst_group_macro_f1": safe_float(worst.get("macro_f1")),
                "best_group": best.get("held_out_group", ""),
                "best_group_macro_f1": safe_float(best.get("macro_f1")),
                "source": source,
            }
        )
    return output


def build_fixed_k_summary(
    stratified_rows: list[dict[str, Any]],
    board_rows: list[dict[str, Any]],
    material_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in stratified_rows:
        if row["k"] in FIXED_K_VALUES:
            rows.append(
                {
                    "protocol": row["protocol"],
                    "encoder_id": row["encoder_id"],
                    "encoder_name": row["encoder_name"],
                    "k": row["k"],
                    "macro_f1": row["macro_f1"],
                    "mean_macro_f1": "",
                    "worst_group": "",
                    "worst_group_macro_f1": "",
                    "best_group": "",
                    "best_group_macro_f1": "",
                    "availability_note": "available",
                    "source": row["source"],
                }
            )
    for grouped_row in board_rows + material_rows:
        rows.append(
            {
                "protocol": grouped_row["protocol"],
                "encoder_id": grouped_row["encoder_id"],
                "encoder_name": grouped_row["encoder_name"],
                "k": grouped_row["k"],
                "macro_f1": "",
                "mean_macro_f1": grouped_row["mean_macro_f1"],
                "worst_group": grouped_row["worst_group"],
                "worst_group_macro_f1": grouped_row["worst_group_macro_f1"],
                "best_group": grouped_row["best_group"],
                "best_group_macro_f1": grouped_row["best_group_macro_f1"],
                "availability_note": "grouped kNN available only for this k in eval_v002",
                "source": grouped_row["source"],
            }
        )
    return rows


def best_stratified_by_encoder(stratified_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in stratified_rows:
        encoder_id = row["encoder_id"]
        if encoder_id not in best or row["macro_f1"] > best[encoder_id]["macro_f1"]:
            best[encoder_id] = row
    return best


def row_for_k(stratified_rows: list[dict[str, Any]], encoder_id: str, k_value: int) -> dict[str, Any] | None:
    for row in stratified_rows:
        if row["encoder_id"] == encoder_id and row["k"] == k_value:
            return row
    return None


def build_best_k_diagnostic(stratified_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best = best_stratified_by_encoder(stratified_rows)
    rows: list[dict[str, Any]] = []
    for encoder_id, best_row in sorted(best.items()):
        k10_row = row_for_k(stratified_rows, encoder_id, RECOMMENDED_K)
        k10_score = k10_row["macro_f1"] if k10_row else float("nan")
        rows.append(
            {
                "encoder_id": encoder_id,
                "encoder_name": best_row["encoder_name"],
                "best_k": best_row["k"],
                "best_stratified_macro_f1": best_row["macro_f1"],
                "fixed_k10_macro_f1": k10_score,
                "difference_best_minus_fixed_k10_pp": (best_row["macro_f1"] - k10_score) * 100.0,
                "best_method": best_row["method"],
                "fixed_k10_available": k10_row is not None,
            }
        )
    return rows


def grouped_k_availability(board_rows: list[dict[str, Any]], material_rows: list[dict[str, Any]]) -> dict[str, Any]:
    board_k = sorted({row["k"] for row in board_rows})
    material_k = sorted({row["k"] for row in material_rows})
    return {
        "board_grouped_available_k": board_k,
        "material_grouped_available_k": material_k,
        "board_grouped_by_k_complete": all(k in board_k for k in ALL_K_VALUES),
        "material_grouped_by_k_complete": all(k in material_k for k in ALL_K_VALUES),
        "grouped_by_k_note": (
            "eval_v002 stores grouped LOBO/LOMO kNN rows only for k=10; fixed-k sensitivity across k=1,3,5,20 is available for stratified train-to-val only."
        ),
    }


def winner_summary(rows: list[dict[str, Any]], protocol: str, score_field: str) -> dict[str, Any]:
    subset = [row for row in rows if row["protocol"] == protocol]
    if not subset:
        return {}
    best = max(subset, key=lambda row: safe_float(row[score_field]))
    return {
        "protocol": protocol,
        "winner_encoder_id": best["encoder_id"],
        "winner_encoder_name": best["encoder_name"],
        "winner_k": best["k"],
        "winner_score": safe_float(best[score_field]),
    }


def build_key_numbers(
    stratified_rows: list[dict[str, Any]],
    fixed_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    board_rows: list[dict[str, Any]],
    material_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    availability = grouped_k_availability(board_rows, material_rows)
    fixed_k_winners = {
        str(k): winner_summary(
            [
                row
                for row in fixed_rows
                if row["protocol"] == "stratified_train_val" and int(row["k"]) == k
            ],
            "stratified_train_val",
            "macro_f1",
        )
        for k in FIXED_K_VALUES
    }
    board_winner = winner_summary(
        [row for row in fixed_rows if row["protocol"] == "board_grouped"],
        "board_grouped",
        "mean_macro_f1",
    )
    material_winner = winner_summary(
        [row for row in fixed_rows if row["protocol"] == "material_grouped"],
        "material_grouped",
        "mean_macro_f1",
    )
    v62a_stratified = {
        str(row["k"]): row["macro_f1"]
        for row in stratified_rows
        if row["encoder_id"] == "v6_2_a"
    }
    v61_stratified = {
        str(row["k"]): row["macro_f1"]
        for row in stratified_rows
        if row["encoder_id"] == "v6_1"
    }
    return {
        "fixed_k_values_reported": FIXED_K_VALUES,
        "all_stratified_k_values_available": sorted({row["k"] for row in stratified_rows}),
        "recommended_primary_k": RECOMMENDED_K,
        "recommended_primary_k_reason": "k=10 is the development/default grouped kNN setting in eval_v002 and avoids selecting k solely by maximum stratified score.",
        "grouped_k_availability": availability,
        "fixed_k_stratified_winners": fixed_k_winners,
        "board_grouped_k10_winner": board_winner,
        "material_grouped_k10_winner": material_winner,
        "v6_2_a_stratified_by_k": v62a_stratified,
        "v6_1_stratified_by_k": v61_stratified,
        "best_k_diagnostic": best_rows,
    }


def render_report(
    fixed_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    board_rows: list[dict[str, Any]],
    material_rows: list[dict[str, Any]],
    key_numbers: dict[str, Any],
) -> str:
    fixed_fields = [
        "protocol",
        "encoder_name",
        "k",
        "macro_f1",
        "mean_macro_f1",
        "worst_group",
        "worst_group_macro_f1",
        "best_group",
        "best_group_macro_f1",
        "availability_note",
    ]
    best_fields = [
        "encoder_name",
        "best_k",
        "best_stratified_macro_f1",
        "fixed_k10_macro_f1",
        "difference_best_minus_fixed_k10_pp",
    ]
    grouped_fields = [
        "protocol",
        "encoder_name",
        "k",
        "mean_macro_f1",
        "worst_group",
        "worst_group_macro_f1",
        "best_group",
        "best_group_macro_f1",
        "n_groups",
    ]
    availability = key_numbers["grouped_k_availability"]
    v62a_k10 = key_numbers["v6_2_a_stratified_by_k"].get("10")
    v61_k10 = key_numbers["v6_1_stratified_by_k"].get("10")
    if v62a_k10 is not None and v61_k10 is not None:
        stratified_k10_line = (
            f"At fixed k=10, v6.1 remains above v6.2-A under stratified train-to-val evaluation "
            f"({pct(v61_k10)} vs {pct(v62a_k10)})."
        )
    else:
        stratified_k10_line = "Fixed k=10 stratified rows were not complete for v6.1 and v6.2-A."

    board_winner = key_numbers["board_grouped_k10_winner"]
    material_winner = key_numbers["material_grouped_k10_winner"]
    grouped_line = (
        f"At grouped k=10, {board_winner.get('winner_encoder_name', 'n/a')} is strongest for Board LOBO "
        f"({pct(board_winner.get('winner_score', 'nan'))}), and "
        f"{material_winner.get('winner_encoder_name', 'n/a')} is strongest for Material LOMO "
        f"({pct(material_winner.get('winner_score', 'nan'))})."
    )

    if availability["board_grouped_by_k_complete"] and availability["material_grouped_by_k_complete"]:
        grouped_note = "Grouped LOBO/LOMO by-k sensitivity is available for all requested k values."
    else:
        grouped_note = (
            "Grouped LOBO/LOMO by-k sensitivity is not available in eval_v002 beyond k=10. Therefore, grouped kNN metrics should be treated as fixed-k locked summaries, while best-k selection is used only as a stratified diagnostic sensitivity analysis."
        )

    return "\n".join(
        [
            "# kNN Protocol Hardening Report",
            "",
            "## Purpose",
            "",
            "This report separates fixed-k kNN results from best-k diagnostic selection using existing eval_v002 outputs only. No encoder evaluation is rerun.",
            "",
            "## Fixed-k Summary",
            "",
            markdown_table(fixed_rows, fixed_fields),
            "",
            "## Best-k Stratified Diagnostic",
            "",
            markdown_table(best_rows, best_fields),
            "",
            "## Grouped kNN Availability",
            "",
            f"- Stratified train-to-val kNN values available: `{key_numbers['all_stratified_k_values_available']}`.",
            f"- Board LOBO grouped kNN values available: `{availability['board_grouped_available_k']}`.",
            f"- Material LOMO grouped kNN values available: `{availability['material_grouped_available_k']}`.",
            f"- {availability['grouped_by_k_note']}",
            "",
            "## Board LOBO kNN Summary",
            "",
            markdown_table(board_rows, grouped_fields),
            "",
            "## Material LOMO kNN Summary",
            "",
            markdown_table(material_rows, grouped_fields),
            "",
            "## Reviewer-Facing Interpretation",
            "",
            f"- Recommended primary k for manuscript reporting: `k={RECOMMENDED_K}`.",
            f"- Rationale: {key_numbers['recommended_primary_k_reason']}",
            f"- {stratified_k10_line}",
            f"- {grouped_line}",
            f"- {grouped_note}",
            "- Best-k results should be labeled as diagnostic sensitivity analysis, not as the primary manuscript protocol.",
            "- Since the grouped conclusion uses the locked k=10 grouped rows, the v6.2-A grouped robustness conclusion is not based on post-hoc selection among grouped k values.",
        ]
    )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stratified_rows = extract_stratified_knn(read_csv(args.eval_dir / "knn_summary.csv"))
    board_rows = summarize_grouped_knn(read_csv(args.eval_dir / "lobo_board_summary.csv"), "board_grouped", "lobo_board_summary.csv")
    material_rows = summarize_grouped_knn(read_csv(args.eval_dir / "lomo_material_summary.csv"), "material_grouped", "lomo_material_summary.csv")
    fixed_rows = build_fixed_k_summary(stratified_rows, board_rows, material_rows)
    best_rows = build_best_k_diagnostic(stratified_rows)
    key_numbers = build_key_numbers(stratified_rows, fixed_rows, best_rows, board_rows, material_rows)

    write_csv(args.out_dir / "knn_fixed_k_summary.csv", fixed_rows)
    write_csv(args.out_dir / "knn_best_k_diagnostic.csv", best_rows)
    write_json(args.out_dir / "knn_protocol_key_numbers.json", key_numbers)
    (args.out_dir / "KNN_PROTOCOL_HARDENING_REPORT.md").write_text(
        render_report(fixed_rows, best_rows, board_rows, material_rows, key_numbers),
        encoding="utf-8",
    )

    print(f"[done] wrote kNN protocol hardening report to {args.out_dir}")
    print(f"[done] recommended_primary_k={RECOMMENDED_K}")
    print(f"[done] grouped_available_k board={key_numbers['grouped_k_availability']['board_grouped_available_k']} material={key_numbers['grouped_k_availability']['material_grouped_available_k']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
