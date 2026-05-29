#!/usr/bin/env python
"""Build final manuscript tables from locked evaluation artifacts.

This script is intentionally read-only with respect to evaluation outputs and
feature dumps. It assembles journal-facing Markdown/CSV tables from existing
CSV/JSON/Markdown artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ENCODER_ORDER = [
    "random_resnet18",
    "imagenet_resnet18",
    "v6_1",
    "v6_2_a",
    "hierarchical_v6_2",
]

CONTROLLED_ORDER = ["v6_1", "v6_2_a", "hierarchical_v6_2"]

ROLES = {
    "random_resnet18": "Random generic visual-control encoder",
    "imagenet_resnet18": "ImageNet generic visual-control encoder",
    "v6_1": "Reference ESPI-specific frozen ResNet18-style baseline",
    "v6_2_a": "Primary image-derived frozen ESPI encoder baseline",
    "hierarchical_v6_2": "Controlled physics-aware architecture comparison",
}

INTERPRETATIONS = {
    "random_resnet18": "Non-trivial stratified texture baseline but weak grouped robustness.",
    "imagenet_resnet18": "Generic pretrained features remain insufficient under grouped evaluation.",
    "v6_1": "Strong stratified ESPI-specific representation, weaker grouped robustness than v6.2-A.",
    "v6_2_a": "Strongest image-derived frozen ESPI encoder under grouped board/material evaluation.",
    "hierarchical_v6_2": "Technically valid but not superior; internal comparison only pending metadata regeneration.",
}

EMBEDDING_POINTS = {
    "v6_1": "avgpool_pre_classifier",
    "v6_2_a": "MCDropoutClassifier.global_pool",
    "hierarchical_v6_2": "z_expert_prelogit / z_arcface_input",
}

ARCHITECTURE_ROLES = {
    "v6_1": "Reference frozen ResNet18-style ESPI baseline",
    "v6_2_a": "Official reportable image-derived frozen encoder baseline",
    "hierarchical_v6_2": "Physics-aware hierarchical phase2 expert comparison branch",
}

PUBLICATION_DECISIONS = {
    "v6_1": "Retain as strongest stratified ESPI-specific reference.",
    "v6_2_a": "Use as primary reportable image-derived frozen ESPI encoder baseline.",
    "hierarchical_v6_2": "Keep internal/controlled; not final-table ready until source metadata are regenerated.",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "not available"
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(value_float):
        return "not available"
    return f"{value_float * 100:.{digits}f}%"


def pp(value: Any, digits: int = 2, signed: bool = True) -> str:
    if value is None or value == "":
        return "not applicable"
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(value_float):
        return "not applicable"
    sign = "+" if signed and value_float > 0 else ""
    return f"{sign}{value_float:.{digits}f} pp"


def percent_string_to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip().replace("%", "")
    if not text:
        return float("nan")
    return float(text) / 100.0


def ci_string(low: Any, high: Any) -> str:
    if low is None or high is None:
        return "not available"
    try:
        low_float = float(low)
        high_float = float(high)
    except (TypeError, ValueError):
        return "not available"
    if pd.isna(low_float) or pd.isna(high_float):
        return "not available"
    return f"[{low_float * 100:.2f}%, {high_float * 100:.2f}%]"


def write_table(df: pd.DataFrame, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_base.with_suffix(".csv"), index=False)
    out_base.with_suffix(".md").write_text(to_markdown_table(df) + "\n", encoding="utf-8")


def markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def to_markdown_table(df: pd.DataFrame) -> str:
    headers = [markdown_cell(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(markdown_cell(row[column]) for column in df.columns) + " |")
    return "\n".join(lines)


def ordered(df: pd.DataFrame, id_col: str = "encoder_id") -> pd.DataFrame:
    order_map = {encoder_id: index for index, encoder_id in enumerate(ENCODER_ORDER)}
    return df.assign(_order=df[id_col].map(order_map)).sort_values("_order").drop(columns=["_order"])


def fixed_row(fixed: pd.DataFrame, encoder_id: str, protocol: str) -> pd.Series:
    rows = fixed[(fixed["encoder_id"] == encoder_id) & (fixed["protocol"] == protocol) & (fixed["k"].astype(str) == "10")]
    if rows.empty:
        raise ValueError(f"Missing fixed k=10 row for {encoder_id} / {protocol}")
    return rows.iloc[0]


def matching_ci(summary: pd.DataFrame, encoder_id: str, field_prefix: str) -> str:
    row = summary[summary["encoder_id"] == encoder_id]
    if row.empty:
        return "not available"
    row = row.iloc[0]
    method_field = f"best_{field_prefix}_method"
    if str(row.get(method_field, "")) != "knn_cosine_k10":
        return "not available for fixed k=10"
    return ci_string(row.get(f"best_{field_prefix}_mean_macro_f1_ci95_low"), row.get(f"best_{field_prefix}_mean_macro_f1_ci95_high"))


def table_1(summary: pd.DataFrame, fixed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for encoder_id in ENCODER_ORDER:
        srow = summary[summary["encoder_id"] == encoder_id].iloc[0]
        strat = fixed_row(fixed, encoder_id, "stratified_train_val")
        board = fixed_row(fixed, encoder_id, "board_grouped")
        material = fixed_row(fixed, encoder_id, "material_grouped")
        rows.append(
            {
                "Encoder": srow["encoder_name"],
                "Role": ROLES[encoder_id],
                "Embedding dimension": int(srow["embedding_dim"]),
                "Primary stratified kNN Macro-F1 at fixed k=10": pct(strat["macro_f1"]),
                "Board LOBO Macro-F1 at fixed k=10": pct(board["mean_macro_f1"]),
                "Material LOMO Macro-F1 at fixed k=10": pct(material["mean_macro_f1"]),
                "Main interpretation": INTERPRETATIONS[encoder_id],
            }
        )
    return pd.DataFrame(rows)


def table_2(summary: pd.DataFrame, fixed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for encoder_id in ENCODER_ORDER:
        srow = summary[summary["encoder_id"] == encoder_id].iloc[0]
        board = fixed_row(fixed, encoder_id, "board_grouped")
        material = fixed_row(fixed, encoder_id, "material_grouped")
        caution = "Material LOMO is descriptive because only two material groups are available."
        if encoder_id == "hierarchical_v6_2":
            caution += " Hierarchical row is internal-only pending source metadata regeneration."
        rows.append(
            {
                "Encoder": srow["encoder_name"],
                "Board LOBO Macro-F1": pct(board["mean_macro_f1"]),
                "Board LOBO 95% CI if available": matching_ci(summary, encoder_id, "lobo"),
                "Material LOMO Macro-F1": pct(material["mean_macro_f1"]),
                "Material LOMO 95% CI if available": matching_ci(summary, encoder_id, "lomo"),
                "Worst board": board["worst_group"],
                "Worst board Macro-F1": pct(board["worst_group_macro_f1"]),
                "Worst material": material["worst_group"],
                "Worst material Macro-F1": pct(material["worst_group_macro_f1"]),
                "Caution note": caution,
            }
        )
    return pd.DataFrame(rows)


def table_3(summary: pd.DataFrame, fixed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for encoder_id in CONTROLLED_ORDER:
        srow = summary[summary["encoder_id"] == encoder_id].iloc[0]
        strat = fixed_row(fixed, encoder_id, "stratified_train_val")
        board = fixed_row(fixed, encoder_id, "board_grouped")
        material = fixed_row(fixed, encoder_id, "material_grouped")
        rows.append(
            {
                "Encoder": srow["encoder_name"],
                "Architecture role": ARCHITECTURE_ROLES[encoder_id],
                "Embedding point": EMBEDDING_POINTS[encoder_id],
                "Embedding dimension": int(srow["embedding_dim"]),
                "Stratified fixed k=10 Macro-F1": pct(strat["macro_f1"]),
                "Board LOBO Macro-F1": pct(board["mean_macro_f1"]),
                "Material LOMO Macro-F1": pct(material["mean_macro_f1"]),
                "Publication decision": PUBLICATION_DECISIONS[encoder_id],
            }
        )
    return pd.DataFrame(rows)


def table_6(zero_recall: pd.DataFrame) -> pd.DataFrame:
    if zero_recall.empty:
        return pd.DataFrame(
            columns=[
                "Encoder",
                "Zero-recall rows",
                "Main affected protocol or condition if available",
                "Interpretation",
            ]
        )
    grouped = (
        zero_recall.groupby("Encoder")
        .agg(
            **{
                "Zero-recall rows": ("Encoder", "size"),
                "Main affected protocol or condition if available": (
                    "Evaluation mode",
                    lambda values: ", ".join(sorted(set(str(value) for value in values))),
                ),
            }
        )
        .reset_index()
    )
    all_encoders = ["Random ResNet-18", "ImageNet ResNet-18", "v6.1", "v6.2-A", "hierarchical v6.2 phase2"]
    grouped = pd.DataFrame({"Encoder": all_encoders}).merge(grouped, on="Encoder", how="left")
    grouped["Zero-recall rows"] = grouped["Zero-recall rows"].fillna(0).astype(int)
    grouped["Main affected protocol or condition if available"] = grouped["Main affected protocol or condition if available"].fillna("none")
    grouped["Interpretation"] = grouped.apply(
        lambda row: "No zero-recall rows in the generated class-level summary."
        if row["Zero-recall rows"] == 0
        else "Zero-recall cases indicate unstable grouped class recovery for this baseline.",
        axis=1,
    )
    return grouped


def best_condition_rows(fusion_summary: pd.DataFrame) -> pd.DataFrame:
    metric = fusion_summary["macro_f1"].where(fusion_summary["macro_f1"].notna(), fusion_summary["mean_macro_f1"])
    working = fusion_summary.assign(_metric=metric)
    rows = working.sort_values("_metric", ascending=False).groupby(["protocol", "condition"], as_index=False).first()
    return rows.drop(columns=["_metric"])


def get_condition_metric(best_rows: pd.DataFrame, protocol: str, condition: str) -> float:
    row = best_rows[(best_rows["protocol"] == protocol) & (best_rows["condition"] == condition)]
    if row.empty:
        raise ValueError(f"Missing fusion summary row for {protocol} / {condition}")
    row = row.iloc[0]
    value = row["macro_f1"] if pd.notna(row["macro_f1"]) else row["mean_macro_f1"]
    return float(value)


def table_7(fusion_summary: pd.DataFrame) -> pd.DataFrame:
    best_rows = best_condition_rows(fusion_summary)
    protocols = ["stratified_train_val", "board_grouped", "material_grouped"]
    labels = {
        "frequency_only": "frequency-only",
        "v6_2_a_embedding_only": "v6.2-A embedding-only",
        "v6_2_a_frequency_fusion": "frequency + v6.2-A",
    }
    values: dict[str, dict[str, float]] = {}
    for condition in labels:
        values[condition] = {protocol: get_condition_metric(best_rows, protocol, condition) for protocol in protocols}
    rows: list[dict[str, Any]] = []
    for condition, display in labels.items():
        strat = values[condition]["stratified_train_val"]
        board = values[condition]["board_grouped"]
        material = values[condition]["material_grouped"]
        if condition == "frequency_only":
            delta_freq = "reference"
            delta_embedding = "not applicable"
            interp = "Frequency_hz is the dominant metadata-only predictor for the current label protocol."
        elif condition == "v6_2_a_embedding_only":
            delta_freq = (
                f"Stratified {pp((strat - values['frequency_only']['stratified_train_val']) * 100)}, "
                f"Board {pp((board - values['frequency_only']['board_grouped']) * 100)}, "
                f"Material {pp((material - values['frequency_only']['material_grouped']) * 100)}"
            )
            delta_embedding = "reference"
            interp = "Image-derived v6.2-A embeddings remain the strongest frozen ESPI encoder baseline."
        else:
            delta_freq = (
                f"Stratified {pp((strat - values['frequency_only']['stratified_train_val']) * 100)}, "
                f"Board {pp((board - values['frequency_only']['board_grouped']) * 100)}, "
                f"Material {pp((material - values['frequency_only']['material_grouped']) * 100)}"
            )
            delta_embedding = (
                f"Stratified {pp((strat - values['v6_2_a_embedding_only']['stratified_train_val']) * 100)}, "
                f"Board {pp((board - values['v6_2_a_embedding_only']['board_grouped']) * 100)}, "
                f"Material {pp((material - values['v6_2_a_embedding_only']['material_grouped']) * 100)}"
            )
            interp = "Fusion improves over embedding-only but does not clearly exceed frequency-only except in Material LOMO."
        rows.append(
            {
                "Condition": display,
                "Stratified Macro-F1": pct(strat),
                "Board LOBO Macro-F1": pct(board),
                "Material LOMO Macro-F1": pct(material),
                "Delta vs frequency-only": delta_freq,
                "Delta vs embedding-only": delta_embedding,
                "Interpretation": interp,
            }
        )
    return pd.DataFrame(rows)


def table_8(best_diag: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    best_diag = ordered(best_diag)
    for _, row in best_diag.iterrows():
        rows.append(
            {
                "Encoder": row["encoder_name"],
                "Fixed k": 10,
                "Fixed-k stratified Macro-F1": pct(row["fixed_k10_macro_f1"]),
                "Best k": int(row["best_k"]),
                "Best-k stratified Macro-F1": pct(row["best_stratified_macro_f1"]),
                "Best-k minus fixed-k delta": pp(row["difference_best_minus_fixed_k10_pp"]),
                "Manuscript use": "Fixed k=10 is primary; best-k is diagnostic only.",
            }
        )
    return pd.DataFrame(rows)


def supplementary_s5(board_delta: pd.DataFrame, material_delta: pd.DataFrame) -> pd.DataFrame:
    board = board_delta.copy()
    board.insert(0, "Protocol", "Board LOBO")
    board = board.rename(columns={"held_out_board": "Held-out group", "material": "Material"})
    board["Caution note"] = "paired board fold"
    material = material_delta.copy()
    material.insert(0, "Protocol", "Material LOMO")
    material = material.rename(columns={"held_out_material": "Held-out group"})
    material["Material"] = material["Held-out group"]
    material["Caution note"] = material.get("caution_note", "descriptive stress-test only; two material groups available")
    keep = [
        "Protocol",
        "Held-out group",
        "Material",
        "held_out_total",
        "v6.1 Macro-F1",
        "v6.2-A Macro-F1",
        "Delta v6.2-A minus v6.1 (pp)",
        "winner",
        "Caution note",
    ]
    combined = pd.concat([board[keep], material[keep]], ignore_index=True)
    return combined.rename(
        columns={
            "held_out_total": "Held-out total",
            "winner": "Winner",
        }
    )


def supplementary_s6(fusion_summary: pd.DataFrame, per_class: pd.DataFrame) -> pd.DataFrame:
    best_rows = best_condition_rows(fusion_summary)
    conditions = ["frequency_only", "v6_2_a_embedding_only", "v6_2_a_frequency_fusion"]
    condition_display = {
        "frequency_only": "Frequency only",
        "v6_2_a_embedding_only": "v6.2-A embedding-only",
        "v6_2_a_frequency_fusion": "Frequency + v6.2-A",
    }
    rows: list[dict[str, Any]] = []
    for protocol in ["stratified_train_val", "board_grouped", "material_grouped"]:
        selected: dict[str, pd.DataFrame] = {}
        for condition in conditions:
            method_row = best_rows[(best_rows["protocol"] == protocol) & (best_rows["condition"] == condition)]
            if method_row.empty:
                continue
            method = method_row.iloc[0]["method"]
            subset = per_class[
                (per_class["protocol"] == protocol)
                & (per_class["condition"] == condition)
                & (per_class["method"] == method)
            ].copy()
            if protocol in {"board_grouped", "material_grouped"}:
                subset = subset.groupby(["class_id", "label_name"], as_index=False).agg({"f1": "mean"})
            selected[condition] = subset[["class_id", "label_name", "f1"]].rename(columns={"f1": condition})
        if not all(condition in selected for condition in conditions):
            continue
        merged = selected["frequency_only"].merge(selected["v6_2_a_embedding_only"], on=["class_id", "label_name"]).merge(
            selected["v6_2_a_frequency_fusion"], on=["class_id", "label_name"]
        )
        for _, row in merged.sort_values("class_id").iterrows():
            rows.append(
                {
                    "Protocol": protocol,
                    "Class ID": int(row["class_id"]),
                    "Class name": row["label_name"],
                    condition_display["frequency_only"]: pct(row["frequency_only"]),
                    condition_display["v6_2_a_embedding_only"]: pct(row["v6_2_a_embedding_only"]),
                    condition_display["v6_2_a_frequency_fusion"]: pct(row["v6_2_a_frequency_fusion"]),
                    "Fusion minus frequency-only": pp((float(row["v6_2_a_frequency_fusion"]) - float(row["frequency_only"])) * 100),
                    "Fusion minus embedding-only": pp((float(row["v6_2_a_frequency_fusion"]) - float(row["v6_2_a_embedding_only"])) * 100),
                }
            )
    return pd.DataFrame(rows)


def build_index(entries: list[dict[str, str]], out_dir: Path) -> None:
    index = pd.DataFrame(entries)
    index.to_csv(out_dir / "TABLE_INDEX.csv", index=False)
    lines = [
        "# Final Manuscript Table Index",
        "",
        "All tables are generated from locked evaluation, class-diagnostic, and methodological-hardening artifacts. No experiments are rerun by this script.",
        "",
        to_markdown_table(index),
        "",
    ]
    (out_dir / "TABLE_INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def update_manuscript_map(map_path: Path, out_dir: Path) -> None:
    marker = "Final manuscript table pack"
    line = (
        f"| {marker} | `scripts/encoder/make_final_manuscript_tables.py` | locked eval_v002/publication/hardening artifacts | "
        f"`{out_dir.as_posix()}/TABLE_INDEX.md` | done-external |"
    )
    if not map_path.exists():
        return
    text = map_path.read_text(encoding="utf-8")
    if marker in text:
        return
    insert = (
        "\n## Final Manuscript Tables\n\n"
        "| Item | Source script | Input artifact | Output/report | Status |\n"
        "|---|---|---|---|---|\n"
        f"{line}\n"
    )
    release_gate = "\n## Release Gate\n"
    if release_gate in text:
        text = text.replace(release_gate, insert + release_gate, 1)
    else:
        text = text.rstrip() + "\n" + insert
    map_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--publication-assets", required=True, type=Path)
    parser.add_argument("--hardening-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    eval_dir = args.eval_dir
    publication_assets = args.publication_assets
    hardening_dir = args.hardening_dir
    class_assets = publication_assets.parent / "class_level_v001"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = read_csv(eval_dir / "encoder_baseline_summary.csv")
    fixed = read_csv(hardening_dir / "knn_protocol" / "knn_fixed_k_summary.csv")
    best_diag = read_csv(hardening_dir / "knn_protocol" / "knn_best_k_diagnostic.csv")
    class_support = read_csv(class_assets / "TABLE_4_CLASS_SUPPORT.csv")
    class_f1 = read_csv(class_assets / "TABLE_5_V62A_VS_V61_GROUPED_PER_CLASS_F1.csv")
    zero_recall = read_csv(class_assets / "TABLE_6_ZERO_RECALL_BREAKDOWN.csv")
    fusion_summary = read_csv(hardening_dir / "frequency_embedding_fusion" / "frequency_embedding_fusion_summary.csv")
    fusion_per_class = read_csv(hardening_dir / "frequency_embedding_fusion" / "frequency_embedding_fusion_per_class.csv")
    board_delta = read_csv(hardening_dir / "paired_board_lobo_delta_v61_v62a.csv")
    material_delta = read_csv(hardening_dir / "paired_material_lomo_delta_v61_v62a.csv")

    table_sources: list[dict[str, str]] = []

    tables = {
        "TABLE_1_ENCODER_BASELINE_SUMMARY_FIXED_K10": (
            table_1(summary, fixed),
            "reports/encoder_baselines/eval_v002/encoder_baseline_summary.csv; reports/publication_assets/methodological_hardening_v001/knn_protocol/knn_fixed_k_summary.csv",
            "Main text Table 1",
        ),
        "TABLE_2_GROUPED_ROBUSTNESS_WITH_CI": (
            table_2(summary, fixed),
            "reports/encoder_baselines/eval_v002/encoder_baseline_summary.csv; reports/publication_assets/methodological_hardening_v001/knn_protocol/knn_fixed_k_summary.csv",
            "Main text Table 2",
        ),
        "TABLE_3_CONTROLLED_ARCHITECTURE_COMPARISON": (
            table_3(summary, fixed),
            "reports/encoder_baselines/eval_v002/encoder_baseline_summary.csv; reports/publication_assets/methodological_hardening_v001/knn_protocol/knn_fixed_k_summary.csv",
            "Main text Table 3",
        ),
        "TABLE_4_CLASS_SUPPORT_AND_IMBALANCE": (
            class_support,
            "reports/publication_assets/class_level_v001/TABLE_4_CLASS_SUPPORT.csv",
            "Main text Table 4",
        ),
        "TABLE_5_V62A_VS_V61_PER_CLASS_GROUPED_F1": (
            class_f1,
            "reports/publication_assets/class_level_v001/TABLE_5_V62A_VS_V61_GROUPED_PER_CLASS_F1.csv",
            "Main text Table 5",
        ),
        "TABLE_6_ZERO_RECALL_BREAKDOWN": (
            table_6(zero_recall),
            "reports/publication_assets/class_level_v001/TABLE_6_ZERO_RECALL_BREAKDOWN.csv",
            "Main text Table 6",
        ),
        "TABLE_7_FREQUENCY_INFORMATION_BUDGET": (
            table_7(fusion_summary),
            "reports/publication_assets/methodological_hardening_v001/frequency_embedding_fusion/frequency_embedding_fusion_summary.csv",
            "Main text Table 7",
        ),
        "TABLE_8_FIXED_K_VS_BEST_K_DIAGNOSTIC": (
            table_8(best_diag),
            "reports/publication_assets/methodological_hardening_v001/knn_protocol/knn_best_k_diagnostic.csv",
            "Main text Table 8",
        ),
        "SUPPLEMENTARY_TABLE_S1_BOARD_LEVEL_SUPPORT": (
            read_csv(hardening_dir / "dataset_composition_by_board.csv"),
            "reports/publication_assets/methodological_hardening_v001/dataset_composition_by_board.csv",
            "Supplementary Table S1",
        ),
        "SUPPLEMENTARY_TABLE_S2_MATERIAL_LEVEL_SUPPORT": (
            read_csv(hardening_dir / "dataset_composition_by_material.csv"),
            "reports/publication_assets/methodological_hardening_v001/dataset_composition_by_material.csv",
            "Supplementary Table S2",
        ),
        "SUPPLEMENTARY_TABLE_S3_CLASS_BY_BOARD_SUPPORT": (
            read_csv(hardening_dir / "class_by_board_support.csv"),
            "reports/publication_assets/methodological_hardening_v001/class_by_board_support.csv",
            "Supplementary Table S3",
        ),
        "SUPPLEMENTARY_TABLE_S4_CLASS_BY_MATERIAL_SUPPORT": (
            read_csv(hardening_dir / "class_by_material_support.csv"),
            "reports/publication_assets/methodological_hardening_v001/class_by_material_support.csv",
            "Supplementary Table S4",
        ),
        "SUPPLEMENTARY_TABLE_S5_PAIRED_BOARD_AND_MATERIAL_DELTAS_V61_V62A": (
            supplementary_s5(board_delta, material_delta),
            "reports/publication_assets/methodological_hardening_v001/paired_board_lobo_delta_v61_v62a.csv; reports/publication_assets/methodological_hardening_v001/paired_material_lomo_delta_v61_v62a.csv",
            "Supplementary Table S5",
        ),
        "SUPPLEMENTARY_TABLE_S6_FREQUENCY_FUSION_PER_CLASS_CHANGES": (
            supplementary_s6(fusion_summary, fusion_per_class),
            "reports/publication_assets/methodological_hardening_v001/frequency_embedding_fusion/frequency_embedding_fusion_per_class.csv",
            "Supplementary Table S6",
        ),
    }

    for name, (df, source, placement) in tables.items():
        out_base = out_dir / name
        write_table(df, out_base)
        table_sources.append(
            {
                "Table": placement,
                "Output Markdown": str(out_base.with_suffix(".md")),
                "Output CSV": str(out_base.with_suffix(".csv")),
                "Source artifact paths": source,
                "Notes": "Generated from locked artifacts; no experiments rerun.",
            }
        )

    build_index(table_sources, out_dir)
    update_manuscript_map(Path("docs") / "MANUSCRIPT_MAP.md", out_dir)

    manifest = {
        "out_dir": str(out_dir),
        "n_tables": len(tables),
        "main_tables": 8,
        "supplementary_tables": 6,
        "primary_knn_protocol": "fixed k=10",
        "best_k_use": "diagnostic only",
        "material_lomo_caution": "descriptive stress-test only; two material groups available",
        "outputs": table_sources,
    }
    (out_dir / "final_manuscript_tables_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote {len(tables)} tables to {out_dir}")
    print(f"[done] index: {out_dir / 'TABLE_INDEX.md'}")


if __name__ == "__main__":
    main()
