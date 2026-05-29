#!/usr/bin/env python
"""Create paired v6.1 vs v6.2-A grouped-difference tables.

This script reads existing evaluation CSV/JSON outputs and grouped-fold support
tables. It does not rerun encoder evaluation, retrain models, or modify
embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


V61_ID = "v6_1"
V62A_ID = "v6_2_a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--composition-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def to_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def pp(delta: float) -> str:
    return f"{delta * 100:+.2f}"


def plain_pp(delta: float) -> float:
    return round(delta * 100, 6)


def winner(delta: float) -> str:
    if delta > 0:
        return "v6.2-A"
    if delta < 0:
        return "v6.1"
    return "tie"


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


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows}


def find_summary(rows: list[dict[str, str]], encoder_id: str) -> dict[str, str]:
    for row in rows:
        if row.get("encoder_id") == encoder_id:
            return row
    raise KeyError(f"Missing encoder summary row for {encoder_id}")


def rows_for_method(rows: list[dict[str, str]], encoder_id: str, method: str) -> dict[str, dict[str, str]]:
    selected = {}
    for row in rows:
        if row.get("encoder_id") == encoder_id and row.get("method") == method:
            selected[row["held_out_group"]] = row
    return selected


def protocol_interpretation(protocol: str, delta: float) -> str:
    if protocol == "Stratified kNN":
        return "v6.1 remains the stronger stratified embedding reference." if delta < 0 else "v6.2-A is stronger under stratified kNN."
    if protocol == "Board LOBO mean":
        return "v6.2-A has stronger board-grouped robustness." if delta > 0 else "v6.1 is stronger under board-grouped evaluation."
    if protocol == "Material LOMO mean":
        return "v6.2-A has stronger material-held-out robustness; interpret descriptively because only two materials are available." if delta > 0 else "v6.1 is stronger under material-held-out evaluation."
    return ""


def build_protocol_table(summary_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    v61 = find_summary(summary_rows, V61_ID)
    v62 = find_summary(summary_rows, V62A_ID)
    specs = [
        ("Stratified kNN", "stratified_best_knn_macro_f1"),
        ("Board LOBO mean", "best_lobo_mean_macro_f1"),
        ("Material LOMO mean", "best_lomo_mean_macro_f1"),
    ]
    rows = []
    for protocol, column in specs:
        v61_score = to_float(v61[column])
        v62_score = to_float(v62[column])
        delta = v62_score - v61_score
        rows.append(
            {
                "Protocol": protocol,
                "v6.1 Macro-F1": pct(v61_score),
                "v6.2-A Macro-F1": pct(v62_score),
                "Delta v6.2-A minus v6.1 (pp)": pp(delta),
                "Interpretation": protocol_interpretation(protocol, delta),
            }
        )
    return rows


def build_board_table(
    summary_rows: list[dict[str, str]],
    lobo_rows: list[dict[str, str]],
    board_support_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    v61 = find_summary(summary_rows, V61_ID)
    v62 = find_summary(summary_rows, V62A_ID)
    v61_method = v61["best_lobo_method"]
    v62_method = v62["best_lobo_method"]
    v61_by_group = rows_for_method(lobo_rows, V61_ID, v61_method)
    v62_by_group = rows_for_method(lobo_rows, V62A_ID, v62_method)
    support_by_board = index_by(board_support_rows, "held_out_board")
    rows = []
    for board in sorted(set(v61_by_group) & set(v62_by_group)):
        v61_score = to_float(v61_by_group[board]["macro_f1"])
        v62_score = to_float(v62_by_group[board]["macro_f1"])
        delta = v62_score - v61_score
        support = support_by_board.get(board, {})
        rows.append(
            {
                "held_out_board": board,
                "material": support.get("held_out_material", ""),
                "held_out_total": support.get("held_out_total", v62_by_group[board].get("n_test", "")),
                "v6.1 Macro-F1": pct(v61_score),
                "v6.2-A Macro-F1": pct(v62_score),
                "Delta v6.2-A minus v6.1 (pp)": pp(delta),
                "winner": winner(delta),
            }
        )
    return rows


def build_material_table(
    summary_rows: list[dict[str, str]],
    lomo_rows: list[dict[str, str]],
    material_support_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    v61 = find_summary(summary_rows, V61_ID)
    v62 = find_summary(summary_rows, V62A_ID)
    v61_method = v61["best_lomo_method"]
    v62_method = v62["best_lomo_method"]
    v61_by_group = rows_for_method(lomo_rows, V61_ID, v61_method)
    v62_by_group = rows_for_method(lomo_rows, V62A_ID, v62_method)
    support_by_material = index_by(material_support_rows, "held_out_material")
    rows = []
    for material in sorted(set(v61_by_group) & set(v62_by_group)):
        v61_score = to_float(v61_by_group[material]["macro_f1"])
        v62_score = to_float(v62_by_group[material]["macro_f1"])
        delta = v62_score - v61_score
        support = support_by_material.get(material, {})
        rows.append(
            {
                "held_out_material": material,
                "held_out_total": support.get("held_out_total", v62_by_group[material].get("n_test", "")),
                "v6.1 Macro-F1": pct(v61_score),
                "v6.2-A Macro-F1": pct(v62_score),
                "Delta v6.2-A minus v6.1 (pp)": pp(delta),
                "winner": winner(delta),
                "caution_note": "descriptive stress-test only; two material groups available",
            }
        )
    return rows


def numeric_delta(row: dict[str, Any]) -> float:
    return float(str(row["Delta v6.2-A minus v6.1 (pp)"]).replace("+", ""))


def summarize(board_rows: list[dict[str, Any]], material_rows: list[dict[str, Any]]) -> dict[str, Any]:
    board_deltas = [numeric_delta(row) for row in board_rows]
    material_deltas = [numeric_delta(row) for row in material_rows]
    return {
        "boards_total": len(board_rows),
        "boards_v62a_wins": sum(row["winner"] == "v6.2-A" for row in board_rows),
        "boards_v61_wins": sum(row["winner"] == "v6.1" for row in board_rows),
        "boards_ties": sum(row["winner"] == "tie" for row in board_rows),
        "mean_board_delta_pp": round(mean(board_deltas), 6) if board_deltas else None,
        "median_board_delta_pp": round(median(board_deltas), 6) if board_deltas else None,
        "min_board_delta_pp": round(min(board_deltas), 6) if board_deltas else None,
        "max_board_delta_pp": round(max(board_deltas), 6) if board_deltas else None,
        "materials_total": len(material_rows),
        "materials_v62a_wins": sum(row["winner"] == "v6.2-A" for row in material_rows),
        "materials_v61_wins": sum(row["winner"] == "v6.1" for row in material_rows),
        "materials_ties": sum(row["winner"] == "tie" for row in material_rows),
        "mean_material_delta_pp": round(mean(material_deltas), 6) if material_deltas else None,
        "median_material_delta_pp": round(median(material_deltas), 6) if material_deltas else None,
        "min_material_delta_pp": round(min(material_deltas), 6) if material_deltas else None,
        "max_material_delta_pp": round(max(material_deltas), 6) if material_deltas else None,
    }


def render_report(
    protocol_rows: list[dict[str, Any]],
    board_rows: list[dict[str, Any]],
    material_rows: list[dict[str, Any]],
    key_numbers: dict[str, Any],
) -> str:
    protocol_fields = [
        "Protocol",
        "v6.1 Macro-F1",
        "v6.2-A Macro-F1",
        "Delta v6.2-A minus v6.1 (pp)",
        "Interpretation",
    ]
    board_fields = [
        "held_out_board",
        "material",
        "held_out_total",
        "v6.1 Macro-F1",
        "v6.2-A Macro-F1",
        "Delta v6.2-A minus v6.1 (pp)",
        "winner",
    ]
    material_fields = [
        "held_out_material",
        "held_out_total",
        "v6.1 Macro-F1",
        "v6.2-A Macro-F1",
        "Delta v6.2-A minus v6.1 (pp)",
        "winner",
        "caution_note",
    ]
    return "\n".join(
        [
            "# Paired v6.1 vs v6.2-A Grouped Difference Tables",
            "",
            "## Purpose",
            "",
            "This report compares v6.1 and v6.2-A under the same frozen-embedding protocols using existing evaluation outputs only. It does not rerun encoder evaluation, retrain models, implement LeFFT, or evaluate acoustic-response prediction.",
            "",
            "## Protocol-Level Paired Difference",
            "",
            markdown_table(protocol_rows, protocol_fields),
            "",
            "## Board LOBO Fold-Level Paired Differences",
            "",
            markdown_table(board_rows, board_fields),
            "",
            "## Material LOMO Fold-Level Paired Differences",
            "",
            markdown_table(material_rows, material_fields),
            "",
            "## Summary Statistics",
            "",
            f"- Boards where v6.2-A wins: {key_numbers['boards_v62a_wins']} / {key_numbers['boards_total']}",
            f"- Boards where v6.1 wins: {key_numbers['boards_v61_wins']} / {key_numbers['boards_total']}",
            f"- Mean board delta: {key_numbers['mean_board_delta_pp']:+.2f} pp",
            f"- Median board delta: {key_numbers['median_board_delta_pp']:+.2f} pp",
            f"- Min board delta: {key_numbers['min_board_delta_pp']:+.2f} pp",
            f"- Max board delta: {key_numbers['max_board_delta_pp']:+.2f} pp",
            f"- Materials where v6.2-A wins: {key_numbers['materials_v62a_wins']} / {key_numbers['materials_total']}",
            f"- Materials where v6.1 wins: {key_numbers['materials_v61_wins']} / {key_numbers['materials_total']}",
            f"- Mean material delta: {key_numbers['mean_material_delta_pp']:+.2f} pp",
            "",
            "## Reviewer-Facing Interpretation",
            "",
            "v6.1 wins the stratified kNN protocol, confirming that it remains the stronger in-distribution stratified reference. v6.2-A wins the grouped board/material mean Macro-F1 protocols, supporting its selection as the primary reportable frozen ESPI encoder baseline for grouped representation studies.",
            "",
            "The board-level paired deltas indicate whether the grouped advantage is broad across held-out boards or concentrated in a small number of boards. The material-level deltas are useful descriptively, but they should be interpreted as material-held-out stress-test results because only two material groups are available.",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv(args.eval_dir / "encoder_baseline_summary.csv")
    lobo_rows = read_csv(args.eval_dir / "lobo_board_summary.csv")
    lomo_rows = read_csv(args.eval_dir / "lomo_material_summary.csv")
    board_support_rows = read_csv(args.composition_dir / "board_lobo_fold_support.csv")
    material_support_rows = read_csv(args.composition_dir / "material_lomo_fold_support.csv")

    protocol_rows = build_protocol_table(summary_rows)
    board_delta_rows = build_board_table(summary_rows, lobo_rows, board_support_rows)
    material_delta_rows = build_material_table(summary_rows, lomo_rows, material_support_rows)
    key_numbers = summarize(board_delta_rows, material_delta_rows)
    key_numbers["protocol_deltas"] = protocol_rows

    write_csv(
        args.out_dir / "paired_protocol_delta_v61_v62a.csv",
        protocol_rows,
        [
            "Protocol",
            "v6.1 Macro-F1",
            "v6.2-A Macro-F1",
            "Delta v6.2-A minus v6.1 (pp)",
            "Interpretation",
        ],
    )
    write_csv(
        args.out_dir / "paired_board_lobo_delta_v61_v62a.csv",
        board_delta_rows,
        [
            "held_out_board",
            "material",
            "held_out_total",
            "v6.1 Macro-F1",
            "v6.2-A Macro-F1",
            "Delta v6.2-A minus v6.1 (pp)",
            "winner",
        ],
    )
    write_csv(
        args.out_dir / "paired_material_lomo_delta_v61_v62a.csv",
        material_delta_rows,
        [
            "held_out_material",
            "held_out_total",
            "v6.1 Macro-F1",
            "v6.2-A Macro-F1",
            "Delta v6.2-A minus v6.1 (pp)",
            "winner",
            "caution_note",
        ],
    )
    (args.out_dir / "paired_v61_v62a_key_numbers.json").write_text(
        json.dumps(key_numbers, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "PAIRED_V61_V62A_GROUPED_DIFFERENCES.md").write_text(
        render_report(protocol_rows, board_delta_rows, material_delta_rows, key_numbers),
        encoding="utf-8",
    )

    print(f"[done] wrote paired v6.1 vs v6.2-A tables to {args.out_dir}")
    print(
        "[done] boards_v62a_wins={}/{} materials_v62a_wins={}/{}".format(
            key_numbers["boards_v62a_wins"],
            key_numbers["boards_total"],
            key_numbers["materials_v62a_wins"],
            key_numbers["materials_total"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
