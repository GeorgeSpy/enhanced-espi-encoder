#!/usr/bin/env python
"""Create publication-ready encoder baseline tables and figures.

This script is intentionally read-only with respect to feature dumps and model
artifacts. It consumes the already-computed eval_v002 CSV/JSON outputs and
produces Markdown/CSV tables plus lightweight SVG figures.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ENCODER_ORDER = [
    "random_resnet18",
    "imagenet_resnet18",
    "v6_1",
    "v6_2_a",
    "hierarchical_v6_2",
]

SHORT_NAMES = {
    "random_resnet18": "Random",
    "imagenet_resnet18": "ImageNet",
    "v6_1": "v6.1",
    "v6_2_a": "v6.2-A",
    "hierarchical_v6_2": "Hierarchical",
}

ROLES = {
    "random_resnet18": "Untrained texture/control encoder",
    "imagenet_resnet18": "Generic natural-image pretrained control encoder",
    "v6_1": "Reference frozen ResNet-18 ESPI baseline",
    "v6_2_a": "Primary reportable frozen ESPI encoder baseline",
    "hierarchical_v6_2": "Controlled physics-aware development branch",
}

INTERPRETATIONS = {
    "random_resnet18": "High stratified texture similarity but weak grouped robustness.",
    "imagenet_resnet18": "Generic features improve stratified retrieval but collapse under grouped OOD.",
    "v6_1": "Best stratified kNN, but weaker board/material grouped robustness than v6.2-A.",
    "v6_2_a": "Best grouped board/material robustness; primary reportable frozen encoder baseline.",
    "hierarchical_v6_2": "Technically valid embeddings, but not representation-robust in this checkpoint.",
}

EMBEDDING_POINTS = {
    "v6_1": "avgpool_pre_classifier",
    "v6_2_a": "MCDropoutClassifier.global_pool",
    "hierarchical_v6_2": "z_expert_prelogit / z_arcface_input",
}

ARCHITECTURE_ROLES = {
    "v6_1": "Reference frozen ResNet-18 baseline",
    "v6_2_a": "Official reportable baseline and frozen encoder candidate",
    "hierarchical_v6_2": "Physics-aware hierarchical phase2 expert branch",
}

PUBLICATION_DECISIONS = {
    "v6_1": "Strong lineage baseline; not grouped-robust leader.",
    "v6_2_a": "Primary reportable frozen ESPI encoder baseline.",
    "hierarchical_v6_2": "Internal-only for final tables until regenerated with source board/split_group metadata.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication tables and figures from eval_v002 outputs."
    )
    parser.add_argument(
        "--eval-dir",
        default="reports/encoder_baselines/eval_v002",
        help="Directory containing eval_v002 CSV/JSON outputs.",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/publication_assets/encoder_baseline_v001",
        help="Output directory for publication-ready assets.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_float(value: Any) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def pct(value: Any) -> str:
    number = to_float(value)
    if math.isnan(number):
        return "n/a"
    return f"{number * 100:.2f}%"


def pp(value: Any) -> str:
    number = to_float(value)
    if math.isnan(number):
        return "n/a"
    return f"{number:.2f} pp"


def ci_text(row: dict[str, Any], low_key: str, high_key: str) -> str:
    low = to_float(row.get(low_key))
    high = to_float(row.get(high_key))
    if math.isnan(low) or math.isnan(high):
        return "n/a"
    return f"[{pct(low)}, {pct(high)}]"


def sort_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_id = {row["encoder_id"]: row for row in rows}
    ordered = [by_id[encoder_id] for encoder_id in ENCODER_ORDER if encoder_id in by_id]
    extras = [row for row in rows if row.get("encoder_id") not in ENCODER_ORDER]
    return ordered + extras


def write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def markdown_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [str(row.get(header, "")).replace("\n", " ") for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_table(path: Path, title: str, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.write_text(f"# {title}\n\n{markdown_table(headers, rows)}", encoding="utf-8")


def table_1_rows(summary: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary:
        encoder_id = row["encoder_id"]
        rows.append(
            {
                "Encoder": row["encoder_name"],
                "Role": ROLES.get(encoder_id, row.get("model_name", "")),
                "Embedding dimension": row["embedding_dim"],
                "Stratified kNN Macro-F1": pct(row["stratified_best_knn_macro_f1"]),
                "Board LOBO Macro-F1": pct(row["best_lobo_mean_macro_f1"]),
                "Material LOMO Macro-F1": pct(row["best_lomo_mean_macro_f1"]),
                "Main interpretation": INTERPRETATIONS.get(encoder_id, ""),
            }
        )
    return rows


def table_2_rows(summary: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary:
        rows.append(
            {
                "Encoder": row["encoder_name"],
                "Board LOBO Macro-F1": pct(row["best_lobo_mean_macro_f1"]),
                "Board LOBO 95% CI": ci_text(
                    row,
                    "best_lobo_mean_macro_f1_ci95_low",
                    "best_lobo_mean_macro_f1_ci95_high",
                ),
                "Material LOMO Macro-F1": pct(row["best_lomo_mean_macro_f1"]),
                "Material LOMO 95% CI": ci_text(
                    row,
                    "best_lomo_mean_macro_f1_ci95_low",
                    "best_lomo_mean_macro_f1_ci95_high",
                ),
                "Worst board group": row.get("best_lobo_worst_group", "n/a"),
                "Worst board Macro-F1": pct(row.get("best_lobo_worst_group_macro_f1")),
                "Worst material group": row.get("best_lomo_worst_group", "n/a"),
                "Worst material Macro-F1": pct(row.get("best_lomo_worst_group_macro_f1")),
            }
        )
    return rows


def table_3_rows(summary: list[dict[str, str]]) -> list[dict[str, Any]]:
    keep = {"v6_1", "v6_2_a", "hierarchical_v6_2"}
    rows: list[dict[str, Any]] = []
    for row in summary:
        encoder_id = row["encoder_id"]
        if encoder_id not in keep:
            continue
        rows.append(
            {
                "Encoder": row["encoder_name"],
                "Architecture role": ARCHITECTURE_ROLES[encoder_id],
                "Embedding point": EMBEDDING_POINTS[encoder_id],
                "Embedding dimension": row["embedding_dim"],
                "Stratified kNN Macro-F1": pct(row["stratified_best_knn_macro_f1"]),
                "Board LOBO Macro-F1": pct(row["best_lobo_mean_macro_f1"]),
                "Material LOMO Macro-F1": pct(row["best_lomo_mean_macro_f1"]),
                "Publication decision": PUBLICATION_DECISIONS[encoder_id],
            }
        )
    return rows


def svg_text(x: float, y: float, text: str, size: int = 14, anchor: str = "middle", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">'
        f"{html.escape(text)}</text>"
    )


def svg_rect(x: float, y: float, width: float, height: float, fill: str, stroke: str = "none") -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="{fill}" stroke="{stroke}" />'
    )


def make_grouped_bar_svg(path: Path, summary: list[dict[str, str]]) -> None:
    width, height = 1240, 720
    margin_left, margin_right = 100, 40
    margin_top, margin_bottom = 80, 150
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    y0 = margin_top + plot_h
    series = [
        ("Stratified kNN", "stratified_best_knn_macro_f1", "#4C78A8"),
        ("Board LOBO", "best_lobo_mean_macro_f1", "#59A14F"),
        ("Material LOMO", "best_lomo_mean_macro_f1", "#F28E2B"),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        svg_rect(0, 0, width, height, "#ffffff"),
        svg_text(width / 2, 34, "Stratified vs Grouped Frozen-Embedding Macro-F1", 22, weight="bold"),
        svg_text(width / 2, 58, "Higher is better; grouped metrics test board/material exclusion from the reference set.", 13),
    ]
    for tick in range(0, 101, 20):
        y = y0 - (tick / 100.0) * plot_h
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#dddddd" />')
        parts.append(svg_text(margin_left - 12, y + 5, f"{tick}%", 12, "end"))
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{y0}" stroke="#333333" />')
    parts.append(f'<line x1="{margin_left}" y1="{y0}" x2="{width - margin_right}" y2="{y0}" stroke="#333333" />')
    group_w = plot_w / len(summary)
    bar_w = min(42, group_w / 5)
    for group_index, row in enumerate(summary):
        center = margin_left + group_w * (group_index + 0.5)
        for series_index, (_, key, color) in enumerate(series):
            value = max(0.0, min(1.0, to_float(row[key])))
            bar_h = value * plot_h
            x = center + (series_index - 1) * (bar_w + 6) - bar_w / 2
            y = y0 - bar_h
            parts.append(svg_rect(x, y, bar_w, bar_h, color))
            parts.append(svg_text(x + bar_w / 2, y - 6, f"{value * 100:.1f}", 10))
        parts.append(svg_text(center, y0 + 28, SHORT_NAMES.get(row["encoder_id"], row["encoder_name"]), 13))
    legend_x = margin_left + 10
    legend_y = height - 70
    for index, (label, _, color) in enumerate(series):
        x = legend_x + index * 220
        parts.append(svg_rect(x, legend_y, 18, 18, color))
        parts.append(svg_text(x + 26, legend_y + 14, label, 13, "start"))
    parts.append(svg_text(36, margin_top + plot_h / 2, "Macro-F1", 13, "middle"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_robustness_gain_svg(path: Path, summary: list[dict[str, str]]) -> None:
    by_id = {row["encoder_id"]: row for row in summary}
    v62a = by_id["v6_2_a"]
    v62a_avg = (to_float(v62a["best_lobo_mean_macro_f1"]) + to_float(v62a["best_lomo_mean_macro_f1"])) / 2
    comparisons = [
        encoder_id
        for encoder_id in ["v6_1", "imagenet_resnet18", "random_resnet18", "hierarchical_v6_2"]
        if encoder_id in by_id
    ]
    gains = []
    for encoder_id in comparisons:
        row = by_id[encoder_id]
        avg = (to_float(row["best_lobo_mean_macro_f1"]) + to_float(row["best_lomo_mean_macro_f1"])) / 2
        gains.append((encoder_id, (v62a_avg - avg) * 100.0))
    max_gain = max([gain for _, gain in gains] + [1.0])
    width, height = 1080, 520
    margin_left, margin_right = 230, 80
    margin_top, margin_bottom = 90, 70
    plot_w = width - margin_left - margin_right
    row_h = (height - margin_top - margin_bottom) / len(gains)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        svg_rect(0, 0, width, height, "#ffffff"),
        svg_text(width / 2, 36, "v6.2-A Grouped Robustness Advantage", 22, weight="bold"),
        svg_text(width / 2, 60, "Advantage in mean grouped Macro-F1: average(Board LOBO, Material LOMO).", 13),
    ]
    axis_x = margin_left
    axis_y = height - margin_bottom
    parts.append(f'<line x1="{axis_x}" y1="{margin_top}" x2="{axis_x}" y2="{axis_y}" stroke="#333333" />')
    parts.append(f'<line x1="{axis_x}" y1="{axis_y}" x2="{width - margin_right}" y2="{axis_y}" stroke="#333333" />')
    for tick in range(0, int(math.ceil(max_gain / 10.0) * 10) + 1, 10):
        x = axis_x + (tick / max_gain) * plot_w if max_gain > 0 else axis_x
        parts.append(f'<line x1="{x:.1f}" y1="{axis_y}" x2="{x:.1f}" y2="{axis_y + 6}" stroke="#333333" />')
        parts.append(svg_text(x, axis_y + 24, f"{tick}", 11))
    for index, (encoder_id, gain) in enumerate(gains):
        y = margin_top + index * row_h + row_h * 0.25
        bar_h = row_h * 0.45
        bar_w = (gain / max_gain) * plot_w if max_gain > 0 else 0
        parts.append(svg_text(margin_left - 18, y + bar_h * 0.7, SHORT_NAMES[encoder_id], 14, "end"))
        parts.append(svg_rect(axis_x, y, bar_w, bar_h, "#59A14F"))
        parts.append(svg_text(axis_x + bar_w + 8, y + bar_h * 0.68, pp(gain), 13, "start"))
    parts.append(svg_text(width / 2, height - 20, "v6.2-A grouped Macro-F1 advantage, percentage points", 13))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def wrapped_multiline_text(x: float, y: float, lines: Iterable[str], size: int = 13) -> list[str]:
    output = []
    for index, line in enumerate(lines):
        output.append(svg_text(x, y + index * (size + 5), line, size))
    return output


def make_decision_flow_svg(path: Path) -> None:
    width, height = 1200, 360
    box_w, box_h = 230, 110
    y = 130
    xs = [70, 345, 620, 895]
    boxes = [
        ("Random/ImageNet controls", ["Stratified texture signal", "fails grouped robustness"]),
        ("v6.1 lineage baseline", ["Best stratified kNN", "not grouped-robust leader"]),
        ("v6.2-A grouped robust encoder", ["Primary reportable frozen", "ESPI encoder baseline"]),
        ("Hierarchical controlled alternative", ["Technically valid", "not superior in phase2"]),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        svg_rect(0, 0, width, height, "#ffffff"),
        svg_text(width / 2, 38, "Encoder Baseline Decision Flow", 22, weight="bold"),
        svg_text(width / 2, 62, "Decision boundary: frozen representation evidence only; no acoustic-response or validated Physics-Aligned Encoder claim.", 13),
    ]
    for index, (title, lines) in enumerate(boxes):
        x = xs[index]
        fill = "#EAF2F8" if index != 2 else "#EAF6EA"
        stroke = "#4C78A8" if index != 2 else "#59A14F"
        parts.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="12" ry="12" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2" />'
        )
        parts.append(svg_text(x + box_w / 2, y + 28, title, 14, weight="bold"))
        parts.extend(wrapped_multiline_text(x + box_w / 2, y + 58, lines, 12))
        if index < len(boxes) - 1:
            x1 = x + box_w + 12
            x2 = xs[index + 1] - 12
            ym = y + box_h / 2
            parts.append(f'<line x1="{x1}" y1="{ym}" x2="{x2}" y2="{ym}" stroke="#666666" stroke-width="2" />')
            parts.append(f'<polygon points="{x2},{ym} {x2 - 10},{ym - 6} {x2 - 10},{ym + 6}" fill="#666666" />')
    parts.append(svg_text(width / 2, 300, "Publication decision: report v6.2-A as the primary frozen ESPI encoder baseline; keep hierarchical phase2 as controlled internal alternative.", 13))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_publication_summary(
    path: Path,
    summary: list[dict[str, str]],
    key_numbers: dict[str, Any],
    eval_dir: Path,
) -> None:
    by_id = {row["encoder_id"]: row for row in summary}
    v61 = by_id["v6_1"]
    v62a = by_id["v6_2_a"]
    bootstrap = key_numbers.get("bootstrap_used", "unknown")
    lines = [
        "# Publication Result Summary - Encoder Baseline v001",
        "",
        "## Source",
        f"- Evaluation directory: `{eval_dir}`",
        f"- Bootstrap enabled: `{bootstrap}`",
        f"- Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Main result",
        f"- v6.1 wins stratified kNN Macro-F1: {pct(v61['stratified_best_knn_macro_f1'])}.",
        f"- v6.2-A wins board LOBO Macro-F1: {pct(v62a['best_lobo_mean_macro_f1'])}.",
        f"- v6.2-A wins material LOMO Macro-F1: {pct(v62a['best_lomo_mean_macro_f1'])}.",
        "- v6.2-A remains the primary reportable frozen ESPI encoder baseline.",
        "",
        "## Interpretation boundary",
        "- v6.1 is the strongest stratified retrieval baseline, but v6.2-A is more robust under grouped board/material exclusion.",
        "- Random and ImageNet ResNet-18 controls show that stratified texture features alone are insufficient for grouped robustness.",
        "- Hierarchical v6.2 phase2 is technically valid but not superior in this frozen embedding evaluation.",
        "- No acoustic-response predictive value is claimed.",
        "- No validated Physics-Aligned Encoder claim is made.",
        "- LeFFT remains a future/ablation direction, not part of the current core claim.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_manuscript_map(repo_root: Path, out_dir: Path) -> None:
    map_path = repo_root / "docs" / "MANUSCRIPT_MAP.md"
    if not map_path.exists():
        return
    text = map_path.read_text(encoding="utf-8")
    rel_out = out_dir.as_posix()
    table_rows = [
        f"| Publication Table 1: Encoder baseline summary | `scripts/encoder/make_publication_tables_figures.py` | not required after eval_v002 | `reports/encoder_baselines/eval_v002/encoder_baseline_summary.csv` | `{rel_out}/TABLE_1_ENCODER_BASELINE_SUMMARY.md` | done-external |",
        f"| Publication Table 2: Grouped robustness with CI | `scripts/encoder/make_publication_tables_figures.py` | not required after eval_v002 | `reports/encoder_baselines/eval_v002/encoder_baseline_summary.csv` | `{rel_out}/TABLE_2_GROUPED_ROBUSTNESS_WITH_CI.md` | done-external |",
        f"| Publication Table 3: Controlled architecture comparison | `scripts/encoder/make_publication_tables_figures.py` | not required after eval_v002 | `reports/encoder_baselines/eval_v002/encoder_baseline_summary.csv` | `{rel_out}/TABLE_3_CONTROLLED_ARCHITECTURE_COMPARISON.md` | done-external |",
    ]
    if "Publication Table 1: Encoder baseline summary" not in text:
        marker = "| Table 5b: Publication-grade encoder baseline evaluation |"
        lines = text.splitlines()
        insert_at = None
        for index, line in enumerate(lines):
            if line.startswith(marker):
                insert_at = index + 1
                break
        if insert_at is None:
            insert_at = len(lines)
        lines[insert_at:insert_at] = table_rows
        text = "\n".join(lines) + "\n"
    figure_rows = [
        f"| Publication Fig. 1: Stratified vs grouped Macro-F1 | `scripts/encoder/make_publication_tables_figures.py` | `reports/encoder_baselines/eval_v002/encoder_baseline_summary.csv` | `{rel_out}/FIG_1_STRATIFIED_VS_GROUPED_MACRO_F1.svg` | done-external |",
        f"| Publication Fig. 2: Grouped robustness gain | `scripts/encoder/make_publication_tables_figures.py` | `reports/encoder_baselines/eval_v002/encoder_baseline_summary.csv` | `{rel_out}/FIG_2_GROUPED_ROBUSTNESS_GAIN.svg` | done-external |",
        f"| Publication Fig. 3: Encoder decision flow | `scripts/encoder/make_publication_tables_figures.py` | `reports/encoder_baselines/eval_v002/encoder_baseline_summary.csv` | `{rel_out}/FIG_3_ENCODER_DECISION_FLOW.svg` | done-external |",
    ]
    if "Publication Fig. 1: Stratified vs grouped Macro-F1" not in text:
        lines = text.splitlines()
        insert_at = None
        for index, line in enumerate(lines):
            if line.startswith("| Fig. 5: grouped evaluation bars |"):
                insert_at = index + 1
                break
        if insert_at is None:
            for index, line in enumerate(lines):
                if line.startswith("## Architecture References"):
                    insert_at = index
                    break
        if insert_at is None:
            insert_at = len(lines)
        lines[insert_at:insert_at] = figure_rows
        text = "\n".join(lines) + "\n"
    map_path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = sort_summary(read_csv_rows(eval_dir / "encoder_baseline_summary.csv"))
    key_numbers = read_json(eval_dir / "encoder_baseline_key_numbers.json")

    table_specs = [
        (
            "TABLE_1_ENCODER_BASELINE_SUMMARY",
            "Table 1 - Encoder Baseline Summary",
            [
                "Encoder",
                "Role",
                "Embedding dimension",
                "Stratified kNN Macro-F1",
                "Board LOBO Macro-F1",
                "Material LOMO Macro-F1",
                "Main interpretation",
            ],
            table_1_rows(summary),
        ),
        (
            "TABLE_2_GROUPED_ROBUSTNESS_WITH_CI",
            "Table 2 - Grouped Robustness with Bootstrap Confidence Intervals",
            [
                "Encoder",
                "Board LOBO Macro-F1",
                "Board LOBO 95% CI",
                "Material LOMO Macro-F1",
                "Material LOMO 95% CI",
                "Worst board group",
                "Worst board Macro-F1",
                "Worst material group",
                "Worst material Macro-F1",
            ],
            table_2_rows(summary),
        ),
        (
            "TABLE_3_CONTROLLED_ARCHITECTURE_COMPARISON",
            "Table 3 - Controlled Architecture Comparison",
            [
                "Encoder",
                "Architecture role",
                "Embedding point",
                "Embedding dimension",
                "Stratified kNN Macro-F1",
                "Board LOBO Macro-F1",
                "Material LOMO Macro-F1",
                "Publication decision",
            ],
            table_3_rows(summary),
        ),
    ]

    generated: list[str] = []
    for basename, title, headers, rows in table_specs:
        md_path = out_dir / f"{basename}.md"
        csv_path = out_dir / f"{basename}.csv"
        write_markdown_table(md_path, title, headers, rows)
        write_csv(csv_path, headers, rows)
        generated.extend([str(md_path), str(csv_path)])

    fig1 = out_dir / "FIG_1_STRATIFIED_VS_GROUPED_MACRO_F1.svg"
    fig2 = out_dir / "FIG_2_GROUPED_ROBUSTNESS_GAIN.svg"
    fig3 = out_dir / "FIG_3_ENCODER_DECISION_FLOW.svg"
    make_grouped_bar_svg(fig1, summary)
    make_robustness_gain_svg(fig2, summary)
    make_decision_flow_svg(fig3)
    generated.extend([str(fig1), str(fig2), str(fig3)])

    summary_path = out_dir / "PUBLICATION_RESULT_SUMMARY.md"
    write_publication_summary(summary_path, summary, key_numbers, eval_dir)
    generated.append(str(summary_path))

    update_manuscript_map(repo_root, out_dir)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "eval_dir": str(eval_dir),
        "out_dir": str(out_dir),
        "generated_files": generated,
        "source_summary_csv": str(eval_dir / "encoder_baseline_summary.csv"),
        "source_key_numbers_json": str(eval_dir / "encoder_baseline_key_numbers.json"),
        "claim_boundary": {
            "primary_reportable_frozen_encoder_baseline": "v6.2-A",
            "stratified_knn_leader": "v6.1",
            "no_acoustic_response_claim": True,
            "no_validated_physics_aligned_encoder_claim": True,
            "lefft_status": "future_ablation_direction",
        },
    }
    manifest_path = out_dir / "publication_assets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    generated.append(str(manifest_path))

    print(f"output_dir={out_dir}")
    print("generated_files:")
    for path in generated:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
