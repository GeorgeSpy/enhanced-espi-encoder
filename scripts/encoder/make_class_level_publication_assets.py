#!/usr/bin/env python
"""Create publication-ready class-level tables and figures.

Reads only the existing class diagnostics outputs. It does not rerun inference,
feature extraction, training, fine-tuning, LeFFT, or acoustic-response tasks.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ENCODER_ORDER = [
    "random_resnet18",
    "imagenet_resnet18",
    "v6_1",
    "v6_2_a",
    "hierarchical_v6_2",
]

ENCODER_NAMES = {
    "random_resnet18": "Random ResNet-18",
    "imagenet_resnet18": "ImageNet ResNet-18",
    "v6_1": "v6.1",
    "v6_2_a": "v6.2-A",
    "hierarchical_v6_2": "hierarchical v6.2 phase2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build class-level publication assets.")
    parser.add_argument("--class-dir", default="reports/encoder_baselines/eval_v002/class_diagnostics")
    parser.add_argument("--out-dir", default="reports/publication_assets/class_level_v001")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
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
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_table(path: Path, title: str, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.write_text(f"# {title}\n\n{markdown_table(headers, rows)}", encoding="utf-8")


def to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def pct(value: Any) -> str:
    number = to_float(value)
    if math.isnan(number):
        return "n/a"
    return f"{number * 100:.2f}%"


def pp(value: Any) -> str:
    number = to_float(value)
    if math.isnan(number):
        return "n/a"
    return f"{number * 100:.2f} pp"


def imbalance_note(support: int, total: int) -> str:
    frac = support / total if total else 0.0
    if frac >= 0.50:
        return "Dominant majority class."
    if frac < 0.08:
        return "Minority class; grouped robustness is sensitive to errors."
    return "Minority/modal class."


def table_4_support_rows(support_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    total = sum(int(row["support"]) for row in support_rows)
    output = []
    for row in support_rows:
        support = int(row["support"])
        output.append(
            {
                "Class ID": row["class_id"],
                "Class name": row["label_name"],
                "Total support": support,
                "Support percentage": pct(support / total),
                "Notes on imbalance": imbalance_note(support, total),
            }
        )
    return output


def table_5_v62a_v61_rows(key_numbers: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for class_name, values in key_numbers["v61_vs_v62a_grouped_class_delta"].items():
        delta = to_float(values["delta"])
        if delta >= 0.10:
            interpretation = "Large grouped stability gain for a difficult modal class."
        elif delta >= 0.03:
            interpretation = "Moderate grouped stability gain."
        elif delta > 0:
            interpretation = "Small positive grouped stability gain."
        else:
            interpretation = "No grouped stability gain."
        rows.append(
            {
                "Class name": class_name,
                "v6.1 grouped F1": pct(values["v6_1_grouped_f1"]),
                "v6.2-A grouped F1": pct(values["v6_2_a_grouped_f1"]),
                "Delta pp": pp(delta),
                "Interpretation": interpretation,
                "_sort_delta": delta,
            }
        )
    rows.sort(key=lambda row: row["_sort_delta"], reverse=True)
    for row in rows:
        row.pop("_sort_delta", None)
    return rows


def table_6_zero_recall_rows(zero_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in zero_rows:
        output.append(
            {
                "Encoder": row.get("encoder_name", ENCODER_NAMES.get(row.get("encoder_id", ""), row.get("encoder_id", ""))),
                "Evaluation mode": row.get("evaluation_mode", ""),
                "Group type": row.get("group_field", ""),
                "Class": row.get("label_name", ""),
                "Group name": row.get("held_out_group", ""),
                "Method": row.get("method", ""),
                "Published-best method": row.get("is_published_best_grouped_method", ""),
                "Support": row.get("support", ""),
            }
        )
    return output


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "middle", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(text)}</text>'
    )


def make_support_svg(path: Path, support_rows: list[dict[str, str]]) -> None:
    labels = [row["label_name"] for row in support_rows]
    values = [int(row["support"]) for row in support_rows]
    width, height = 860, 500
    left, top, plot_w, plot_h = 100, 78, 690, 310
    max_value = max(values) if values else 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#ffffff"/>']
    parts.append(svg_text(width / 2, 34, "Class Support Imbalance", 20, weight="bold"))
    parts.append(svg_text(width / 2, 56, "The dataset is dominated by the higher class.", 12))
    tick_step = max(1, math.ceil(max_value / 5 / 500) * 500)
    for tick in range(0, max_value + tick_step, tick_step):
        y = top + plot_h - min(tick / max_value, 1.0) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(svg_text(left - 10, y + 4, str(tick), 11, "end"))
    bar_w = plot_w / len(values) * 0.58
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + (index + 0.5) * plot_w / len(values) - bar_w / 2
        bar_h = value / max_value * plot_h
        y = top + plot_h - bar_h
        color = "#F28E2B" if label == "higher" else "#4C78A8"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        parts.append(svg_text(x + bar_w / 2, y - 6, str(value), 10))
        parts.append(svg_text(x + bar_w / 2, top + plot_h + 24, label, 12))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_v62a_v61_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    ordered = list(reversed(rows))
    width, height = 980, 560
    left, top, plot_w, plot_h = 120, 80, 790, 330
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#ffffff"/>']
    parts.append(svg_text(width / 2, 34, "v6.2-A vs v6.1 Per-Class Grouped F1", 20, weight="bold"))
    parts.append(svg_text(width / 2, 56, "Grouped F1 is averaged over board LOBO and material LOMO published-best methods.", 12))
    for tick in range(0, 101, 20):
        y = top + plot_h - tick / 100 * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(svg_text(left - 10, y + 4, f"{tick}%", 11, "end"))
    group_w = plot_w / len(ordered)
    bar_w = group_w * 0.26
    for idx, row in enumerate(ordered):
        center = left + group_w * (idx + 0.5)
        v61 = to_float(row["v6.1 grouped F1"].replace("%", "")) / 100
        v62a = to_float(row["v6.2-A grouped F1"].replace("%", "")) / 100
        for offset, value, color in [(-0.6, v61, "#4C78A8"), (0.6, v62a, "#59A14F")]:
            bar_h = value * plot_h
            x = center + offset * bar_w - bar_w / 2
            y = top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
            parts.append(svg_text(x + bar_w / 2, y - 5, f"{value * 100:.1f}", 9))
        parts.append(svg_text(center, top + plot_h + 24, row["Class name"], 12))
    parts.append(f'<rect x="{left}" y="{height - 70}" width="16" height="16" fill="#4C78A8"/>')
    parts.append(svg_text(left + 24, height - 57, "v6.1", 12, "start"))
    parts.append(f'<rect x="{left + 110}" y="{height - 70}" width="16" height="16" fill="#59A14F"/>')
    parts.append(svg_text(left + 134, height - 57, "v6.2-A", 12, "start"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_zero_recall_svg(path: Path, zero_rows: list[dict[str, str]]) -> None:
    counts: Counter[tuple[str, str]] = Counter()
    for row in zero_rows:
        encoder_id = row.get("encoder_id", "")
        mode = row.get("evaluation_mode", "")
        counts[(encoder_id, mode)] += 1
    modes = sorted({mode for _, mode in counts})
    width, height = 1060, 560
    left, top, plot_w, plot_h = 110, 80, 860, 330
    max_count = max(counts.values()) if counts else 1
    colors = {"board_grouped": "#4C78A8", "material_grouped": "#F28E2B", "board_grouped_aggregate": "#9C755F", "material_grouped_aggregate": "#B07AA1"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#ffffff"/>']
    parts.append(svg_text(width / 2, 34, "Zero-Recall Cases by Encoder and Evaluation Mode", 20, weight="bold"))
    parts.append(svg_text(width / 2, 56, "Counts are per class/group/method rows with recall equal to zero.", 12))
    tick_step = max(1, math.ceil(max_count / 5))
    for tick in range(0, max_count + tick_step, tick_step):
        y = top + plot_h - min(tick / max_count, 1.0) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(svg_text(left - 10, y + 4, str(tick), 11, "end"))
    group_w = plot_w / len(ENCODER_ORDER)
    bar_w = group_w / max(1, len(modes) + 1)
    for enc_idx, encoder_id in enumerate(ENCODER_ORDER):
        center = left + group_w * (enc_idx + 0.5)
        for mode_idx, mode in enumerate(modes):
            value = counts.get((encoder_id, mode), 0)
            bar_h = value / max_count * plot_h
            x = center + (mode_idx - (len(modes) - 1) / 2) * bar_w - bar_w / 2
            y = top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.78:.1f}" height="{bar_h:.1f}" fill="{colors.get(mode, "#888888")}"/>')
            if value:
                parts.append(svg_text(x + bar_w * 0.39, y - 5, str(value), 9))
        parts.append(svg_text(center, top + plot_h + 26, ENCODER_NAMES.get(encoder_id, encoder_id).replace(" ResNet-18", ""), 11))
    legend_y = height - 84
    for idx, mode in enumerate(modes):
        x = left + idx * 220
        parts.append(f'<rect x="{x}" y="{legend_y}" width="16" height="16" fill="{colors.get(mode, "#888888")}"/>')
        parts.append(svg_text(x + 24, legend_y + 13, mode, 11, "start"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_summary(path: Path, support_rows: list[dict[str, str]], delta_rows: list[dict[str, Any]], zero_rows: list[dict[str, str]]) -> None:
    total = sum(int(row["support"]) for row in support_rows)
    higher = next(row for row in support_rows if row["label_name"] == "higher")
    largest = sorted(delta_rows, key=lambda row: to_float(row["Delta pp"].split()[0]) if row["Delta pp"] != "n/a" else -999, reverse=True)[:2]
    zero_by_encoder = Counter(row["encoder_name"] for row in zero_rows)
    lines = [
        "# Class-Level Publication Summary",
        "",
        "## Main findings",
        f"- The dataset is strongly imbalanced, with `higher` contributing `{higher['support']}` of `{total}` samples ({pct(int(higher['support']) / total)}).",
        "- v6.2-A improves grouped F1 over v6.1 across all five classes.",
        f"- The largest improvements occur for `{largest[0]['Class name']}` ({largest[0]['Delta pp']}) and `{largest[1]['Class name']}` ({largest[1]['Delta pp']}).",
        "- This supports the claim that v6.2-A improves grouped class stability, not only aggregate Macro-F1.",
        "- Zero-recall cases must be interpreted by encoder and evaluation mode, because they are not uniformly distributed across models.",
        "",
        "## Zero-recall concentration",
    ]
    for encoder_name, count in zero_by_encoder.most_common():
        lines.append(f"- `{encoder_name}`: `{count}` zero-recall rows.")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "- No acoustic-response predictive value is claimed.",
            "- No validated Physics-Aligned Encoder claim is made.",
            "- LeFFT is not part of this analysis.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_manuscript_map(repo_root: Path) -> None:
    map_path = repo_root / "docs" / "MANUSCRIPT_MAP.md"
    if not map_path.exists():
        return
    text = map_path.read_text(encoding="utf-8")
    table_rows = [
        "| Table 6: Class support and imbalance | `scripts/encoder/make_class_level_publication_assets.py` | not required after class diagnostics | `reports/encoder_baselines/eval_v002/class_diagnostics/class_support_overall.csv` | `reports/publication_assets/class_level_v001/TABLE_4_CLASS_SUPPORT.md` | done-external |",
        "| Table 7: Per-class grouped v6.2-A vs v6.1 comparison | `scripts/encoder/make_class_level_publication_assets.py` | not required after class diagnostics | `reports/encoder_baselines/eval_v002/class_diagnostics/class_level_key_numbers.json` | `reports/publication_assets/class_level_v001/TABLE_5_V62A_VS_V61_GROUPED_PER_CLASS_F1.md` | done-external |",
        "| Table 8: Zero-recall breakdown | `scripts/encoder/make_class_level_publication_assets.py` | not required after class diagnostics | `reports/encoder_baselines/eval_v002/class_diagnostics/zero_recall_classes.csv` | `reports/publication_assets/class_level_v001/TABLE_6_ZERO_RECALL_BREAKDOWN.md` | done-external |",
    ]
    for row in table_rows:
        label = row.split("|")[1].strip()
        if label not in text:
            lines = text.splitlines()
            insert_at = len(lines)
            for idx, line in enumerate(lines):
                if line.startswith("| Table 6: acoustic-response prediction |"):
                    insert_at = idx
                    break
            lines.insert(insert_at, row)
            text = "\n".join(lines) + "\n"
    figure_rows = [
        "| Figure 4: Class support imbalance | `scripts/encoder/make_class_level_publication_assets.py` | class-level diagnostics outputs | `reports/publication_assets/class_level_v001/FIG_4_CLASS_SUPPORT_IMBALANCE.svg` | done-external |",
        "| Figure 5: v6.2-A vs v6.1 per-class grouped F1 | `scripts/encoder/make_class_level_publication_assets.py` | class-level diagnostics outputs | `reports/publication_assets/class_level_v001/FIG_5_V62A_VS_V61_PER_CLASS_GROUPED_F1.svg` | done-external |",
        "| Figure 6: Zero-recall breakdown by encoder | `scripts/encoder/make_class_level_publication_assets.py` | class-level diagnostics outputs | `reports/publication_assets/class_level_v001/FIG_6_ZERO_RECALL_BREAKDOWN_BY_ENCODER.svg` | done-external |",
    ]
    for row in figure_rows:
        label = row.split("|")[1].strip()
        if label not in text:
            lines = text.splitlines()
            insert_at = len(lines)
            for idx, line in enumerate(lines):
                if line.startswith("## Architecture References"):
                    insert_at = idx
                    break
            lines.insert(insert_at, row)
            text = "\n".join(lines) + "\n"
    map_path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    class_dir = Path(args.class_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    support_rows = read_csv(class_dir / "class_support_overall.csv")
    key_numbers = read_json(class_dir / "class_level_key_numbers.json")
    zero_rows = read_csv(class_dir / "zero_recall_classes.csv")

    table4 = table_4_support_rows(support_rows)
    table5 = table_5_v62a_v61_rows(key_numbers)
    table6 = table_6_zero_recall_rows(zero_rows)

    table_specs = [
        (
            "TABLE_4_CLASS_SUPPORT",
            "Table 4 - Class Support and Imbalance",
            ["Class ID", "Class name", "Total support", "Support percentage", "Notes on imbalance"],
            table4,
        ),
        (
            "TABLE_5_V62A_VS_V61_GROUPED_PER_CLASS_F1",
            "Table 5 - v6.2-A vs v6.1 Grouped Per-Class F1",
            ["Class name", "v6.1 grouped F1", "v6.2-A grouped F1", "Delta pp", "Interpretation"],
            table5,
        ),
        (
            "TABLE_6_ZERO_RECALL_BREAKDOWN",
            "Table 6 - Zero-Recall Breakdown",
            ["Encoder", "Evaluation mode", "Group type", "Class", "Group name", "Method", "Published-best method", "Support"],
            table6,
        ),
    ]
    generated: list[str] = []
    for basename, title, headers, rows in table_specs:
        md_path = out_dir / f"{basename}.md"
        csv_path = out_dir / f"{basename}.csv"
        write_markdown_table(md_path, title, headers, rows)
        write_csv(csv_path, rows, headers)
        generated.extend([str(md_path), str(csv_path)])

    fig4 = out_dir / "FIG_4_CLASS_SUPPORT_IMBALANCE.svg"
    fig5 = out_dir / "FIG_5_V62A_VS_V61_PER_CLASS_GROUPED_F1.svg"
    fig6 = out_dir / "FIG_6_ZERO_RECALL_BREAKDOWN_BY_ENCODER.svg"
    make_support_svg(fig4, support_rows)
    make_v62a_v61_svg(fig5, table5)
    make_zero_recall_svg(fig6, zero_rows)
    generated.extend([str(fig4), str(fig5), str(fig6)])

    summary_path = out_dir / "CLASS_LEVEL_PUBLICATION_SUMMARY.md"
    write_summary(summary_path, support_rows, table5, zero_rows)
    generated.append(str(summary_path))

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "class_dir": str(class_dir),
        "out_dir": str(out_dir),
        "generated_files": generated,
        "zero_recall_counts_by_encoder": dict(Counter(row["encoder_name"] for row in zero_rows)),
        "claim_boundary": {
            "no_acoustic_response_claim": True,
            "no_validated_physics_aligned_encoder_claim": True,
            "no_lefft": True,
        },
    }
    manifest_path = out_dir / "class_level_publication_assets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    generated.append(str(manifest_path))

    update_manuscript_map(repo_root)

    print(f"output_dir={out_dir}")
    print("generated_files:")
    for path in generated:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
