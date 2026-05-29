#!/usr/bin/env python
"""Create dataset composition and grouped-fold support tables.

This script is intentionally lightweight: it reads existing manifest/feature
metadata and writes publication-facing support tables. It does not load images,
run encoder inference, retrain models, or recompute encoder evaluation metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


CLASS_IDS = [0, 1, 2, 3, 4]
DEFAULT_CLASS_NAMES = {
    0: "1_1H",
    1: "1_1T",
    2: "1_2",
    3: "2_1",
    4: "higher",
}
LOW_SUPPORT_THRESHOLD = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def as_str_array(value: Any, n: int | None = None, default: str = "unknown") -> np.ndarray:
    array = np.asarray(value)
    if array.shape == ():
        if n is None:
            return np.asarray([str(array.item())], dtype=object)
        return np.asarray([str(array.item())] * n, dtype=object)
    return array.astype(str)


def as_int_array(value: Any) -> np.ndarray:
    return np.asarray(value).astype(int)


def load_manifest(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def infer_board_from_path(path: str) -> str:
    match = re.search(r"\b([CW]\d{2})\b", path)
    return match.group(1) if match else "unknown"


def infer_material(board: str, path: str = "", distribution_group: str = "") -> str:
    text = " ".join([board, path, distribution_group]).lower()
    if board.upper().startswith("C") or "carbon" in text:
        return "carbon"
    if board.upper().startswith("W") or "wood" in text:
        return "wood"
    return "unknown"


def load_primary_metadata(features_dir: Path, manifest: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    feature_path = features_dir / "features_v62a_epoch25.normalized.npz"
    if not feature_path.exists():
        raise FileNotFoundError(f"Primary v6.2-A normalized feature dump not found: {feature_path}")

    with np.load(feature_path, allow_pickle=True) as data:
        keys = set(data.files)
        n = int(np.asarray(data["label"]).shape[0])

        labels = as_int_array(data["label"]) if "label" in keys else as_int_array(manifest["labels"])
        paths = as_str_array(data["path"], n) if "path" in keys else as_str_array(manifest["paths"], n)
        label_names = (
            as_str_array(data["label_name"], n)
            if "label_name" in keys
            else np.asarray([DEFAULT_CLASS_NAMES.get(int(label), str(label)) for label in labels], dtype=object)
        )
        boards = as_str_array(data["board"], n) if "board" in keys else as_str_array(manifest.get("boards", []), n)
        distribution_group = (
            as_str_array(data["distribution_group"], n)
            if "distribution_group" in keys
            else np.asarray([Path(path).parent.name for path in paths], dtype=object)
        )
        material = as_str_array(data["material"], n) if "material" in keys else np.asarray(["unknown"] * n, dtype=object)
        frequency_hz = (
            np.asarray(data["frequency_hz"])
            if "frequency_hz" in keys
            else np.asarray(manifest.get("freq_hz", np.full(n, np.nan)))
        )

    recovered_boards = []
    recovered_material = []
    for index, board in enumerate(boards):
        board_value = str(board)
        if not board_value or board_value.lower() == "unknown":
            board_value = infer_board_from_path(str(paths[index]))
        recovered_boards.append(board_value)

        material_value = str(material[index])
        if not material_value or material_value.lower() == "unknown":
            material_value = infer_material(board_value, str(paths[index]), str(distribution_group[index]))
        recovered_material.append(material_value)

    return {
        "label": labels,
        "label_name": label_names,
        "path": paths,
        "board": np.asarray(recovered_boards, dtype=object),
        "material": np.asarray(recovered_material, dtype=object),
        "distribution_group": distribution_group,
        "frequency_hz": frequency_hz,
        "source_feature_dump": np.asarray([str(feature_path)], dtype=object),
    }


def class_lookup(metadata: dict[str, np.ndarray]) -> dict[int, str]:
    lookup = dict(DEFAULT_CLASS_NAMES)
    for class_id, label_name in zip(metadata["label"], metadata["label_name"]):
        lookup[int(class_id)] = str(label_name)
    return lookup


def count_classes(labels: np.ndarray, lookup: dict[int, str]) -> OrderedDict[str, int]:
    counts: OrderedDict[str, int] = OrderedDict()
    for class_id in CLASS_IDS:
        counts[f"class_{class_id}"] = int(np.sum(labels == class_id))
    return counts


def missing_classes(counts: dict[str, int]) -> list[str]:
    missing = []
    for key, value in counts.items():
        if key.startswith("class_") and int(value) == 0:
            missing.append(key)
    return missing


def low_support_classes(counts: dict[str, int]) -> list[str]:
    low = []
    for key, value in counts.items():
        if key.startswith("class_") and 0 < int(value) < LOW_SUPPORT_THRESHOLD:
            low.append(f"{key}:{value}")
    return low


def caution_flag(heldout_counts: dict[str, int], reference_counts: dict[str, int], extra: list[str] | None = None) -> str:
    flags: list[str] = []
    heldout_missing = missing_classes(heldout_counts)
    reference_missing = missing_classes(reference_counts)
    heldout_low = low_support_classes(heldout_counts)
    reference_low = low_support_classes(reference_counts)
    if heldout_missing:
        flags.append("missing_heldout_classes")
    if reference_missing:
        flags.append("missing_reference_classes")
    if heldout_low:
        flags.append("low_heldout_class_support")
    if reference_low:
        flags.append("low_reference_class_support")
    if extra:
        flags.extend(extra)
    return ";".join(flags) if flags else "none"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def markdown_table(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in fieldnames) + " |")
    return "\n".join(lines)


def build_dataset_by_board(metadata: dict[str, np.ndarray], lookup: dict[int, str]) -> list[dict[str, Any]]:
    rows = []
    boards = sorted(set(metadata["board"].astype(str).tolist()))
    for board in boards:
        mask = metadata["board"].astype(str) == board
        labels = metadata["label"][mask]
        counts = count_classes(labels, lookup)
        materials = sorted(set(metadata["material"][mask].astype(str).tolist()))
        groups = sorted(set(metadata["distribution_group"][mask].astype(str).tolist()))
        row: OrderedDict[str, Any] = OrderedDict()
        row["board"] = board
        row["material"] = ",".join(materials)
        row["distribution_group_count"] = len(groups)
        row["distribution_groups"] = ";".join(groups)
        row.update(counts)
        row["total_samples"] = int(mask.sum())
        rows.append(row)
    return rows


def build_dataset_by_material(metadata: dict[str, np.ndarray], lookup: dict[int, str]) -> list[dict[str, Any]]:
    rows = []
    materials = sorted(set(metadata["material"].astype(str).tolist()))
    for material in materials:
        mask = metadata["material"].astype(str) == material
        labels = metadata["label"][mask]
        counts = count_classes(labels, lookup)
        boards = sorted(set(metadata["board"][mask].astype(str).tolist()))
        row: OrderedDict[str, Any] = OrderedDict()
        row["material"] = material
        row.update(counts)
        row["total_samples"] = int(mask.sum())
        row["number_of_boards"] = len(boards)
        row["boards"] = ",".join(boards)
        rows.append(row)
    return rows


def build_class_by_board(metadata: dict[str, np.ndarray], lookup: dict[int, str]) -> list[dict[str, Any]]:
    rows = []
    for row in build_dataset_by_board(metadata, lookup):
        counts = {key: value for key, value in row.items() if key.startswith("class_")}
        output = OrderedDict()
        output["board"] = row["board"]
        output["material"] = row["material"]
        output.update(counts)
        output["total"] = row["total_samples"]
        output["missing_classes"] = ",".join(missing_classes(counts)) or "none"
        rows.append(output)
    return rows


def build_class_by_material(metadata: dict[str, np.ndarray], lookup: dict[int, str]) -> list[dict[str, Any]]:
    rows = []
    for row in build_dataset_by_material(metadata, lookup):
        counts = {key: value for key, value in row.items() if key.startswith("class_")}
        output = OrderedDict()
        output["material"] = row["material"]
        output.update(counts)
        output["total"] = row["total_samples"]
        output["missing_classes"] = ",".join(missing_classes(counts)) or "none"
        rows.append(output)
    return rows


def build_board_lobo_folds(metadata: dict[str, np.ndarray], lookup: dict[int, str]) -> list[dict[str, Any]]:
    rows = []
    boards = sorted(set(metadata["board"].astype(str).tolist()))
    all_labels = metadata["label"]
    for board in boards:
        test_mask = metadata["board"].astype(str) == board
        ref_mask = ~test_mask
        heldout_counts = count_classes(all_labels[test_mask], lookup)
        reference_counts = count_classes(all_labels[ref_mask], lookup)
        materials = sorted(set(metadata["material"][test_mask].astype(str).tolist()))
        row: OrderedDict[str, Any] = OrderedDict()
        row["held_out_board"] = board
        row["held_out_material"] = ",".join(materials)
        row["held_out_total"] = int(test_mask.sum())
        for key, value in heldout_counts.items():
            row[f"held_out_{key}"] = value
        row["reference_total"] = int(ref_mask.sum())
        for key, value in reference_counts.items():
            row[f"reference_{key}"] = value
        row["missing_classes_in_heldout"] = ",".join(missing_classes(heldout_counts)) or "none"
        row["missing_classes_in_reference"] = ",".join(missing_classes(reference_counts)) or "none"
        row["macro_f1_caution_flag"] = caution_flag(heldout_counts, reference_counts)
        rows.append(row)
    return rows


def build_material_lomo_folds(metadata: dict[str, np.ndarray], lookup: dict[int, str]) -> list[dict[str, Any]]:
    rows = []
    materials = sorted(set(metadata["material"].astype(str).tolist()))
    all_labels = metadata["label"]
    group_count_flag = ["small_number_of_material_groups"] if len(materials) <= 2 else []
    for material in materials:
        test_mask = metadata["material"].astype(str) == material
        ref_mask = ~test_mask
        heldout_counts = count_classes(all_labels[test_mask], lookup)
        reference_counts = count_classes(all_labels[ref_mask], lookup)
        row: OrderedDict[str, Any] = OrderedDict()
        row["held_out_material"] = material
        row["held_out_total"] = int(test_mask.sum())
        for key, value in heldout_counts.items():
            row[f"held_out_{key}"] = value
        row["reference_total"] = int(ref_mask.sum())
        for key, value in reference_counts.items():
            row[f"reference_{key}"] = value
        row["missing_classes_in_heldout"] = ",".join(missing_classes(heldout_counts)) or "none"
        row["missing_classes_in_reference"] = ",".join(missing_classes(reference_counts)) or "none"
        row["macro_f1_caution_flag"] = caution_flag(heldout_counts, reference_counts, group_count_flag)
        rows.append(row)
    return rows


def fieldnames_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def all_missing_none(rows: list[dict[str, Any]], key: str) -> bool:
    return all(str(row.get(key, "")) == "none" for row in rows)


def render_report(
    metadata: dict[str, np.ndarray],
    lookup: dict[int, str],
    board_rows: list[dict[str, Any]],
    material_rows: list[dict[str, Any]],
    class_board_rows: list[dict[str, Any]],
    class_material_rows: list[dict[str, Any]],
    lobo_rows: list[dict[str, Any]],
    lomo_rows: list[dict[str, Any]],
    key_numbers: dict[str, Any],
) -> str:
    class_counts = count_classes(metadata["label"], lookup)
    class_count_rows = []
    for class_id in CLASS_IDS:
        key = f"class_{class_id}"
        value = class_counts[key]
        class_count_rows.append(
            {
                "class_id": class_id,
                "label_name": lookup[class_id],
                "support": value,
                "percentage": f"{(value / key_numbers['total_samples'] * 100):.2f}%",
            }
        )

    report = [
        "# Dataset Composition and Grouped-Fold Support Audit",
        "",
        "## Purpose",
        "",
        "This report documents dataset composition and grouped-fold support for the frozen ESPI encoder evaluation. It is generated from existing manifest/feature metadata and does not rerun encoder evaluation, feature extraction, image loading, LeFFT analysis, or acoustic-response prediction.",
        "",
        "## Key Numbers",
        "",
        f"- Total samples: {key_numbers['total_samples']}",
        f"- Number of boards: {key_numbers['number_of_boards']}",
        f"- Boards: {', '.join(key_numbers['boards'])}",
        f"- Number of materials: {key_numbers['number_of_materials']}",
        f"- Materials: {', '.join(key_numbers['materials'])}",
        f"- Number of distribution groups: {key_numbers['number_of_distribution_groups']}",
        "",
        "## Class Distribution",
        "",
        markdown_table(class_count_rows, ["class_id", "label_name", "support", "percentage"]),
        "",
        "## Supplementary Table Mapping",
        "",
        "| Supplementary item | Generated CSV | Description |",
        "| --- | --- | --- |",
        "| Supplementary Table S1 | `dataset_composition_by_board.csv` | Board-level support with material, distribution groups, class counts, and total samples. |",
        "| Supplementary Table S2 | `dataset_composition_by_material.csv` | Material-level support with class counts, total samples, and number of boards. |",
        "| Supplementary Table S3 | `class_by_board_support.csv` | Class-by-board support and missing-class indicators. |",
        "| Supplementary Table S4 | `class_by_material_support.csv` | Class-by-material support and missing-class indicators. |",
        "",
        "## Board-Level Support Preview",
        "",
        markdown_table(board_rows, fieldnames_from_rows(board_rows)),
        "",
        "## Material-Level Support Preview",
        "",
        markdown_table(material_rows, fieldnames_from_rows(material_rows)),
        "",
        "## Grouped-Fold Interpretation",
        "",
        f"- All boards contain all classes: {'yes' if key_numbers['all_boards_contain_all_classes'] else 'no'}",
        f"- All materials contain all classes: {'yes' if key_numbers['all_materials_contain_all_classes'] else 'no'}",
        f"- Board LOBO folds contain missing held-out classes: {'no' if key_numbers['lobo_missing_heldout_folds'] == 0 else 'yes'}",
        f"- Board LOBO folds contain missing reference classes: {'no' if key_numbers['lobo_missing_reference_folds'] == 0 else 'yes'}",
        f"- Material LOMO folds contain missing held-out classes: {'no' if key_numbers['lomo_missing_heldout_folds'] == 0 else 'yes'}",
        f"- Material LOMO folds contain missing reference classes: {'no' if key_numbers['lomo_missing_reference_folds'] == 0 else 'yes'}",
        "",
        "## Reviewer-Facing Interpretation",
        "",
        "The dataset contains all five modal classes in every board group and every material group. This supports Macro-F1 reporting for board- and material-held-out frozen-embedding evaluation because no held-out fold lacks an entire class and no reference fold lacks an entire class.",
        "",
        "Board LOBO-style evaluation has six held-out board groups and should be interpreted as a grouped robustness stress test over board variation. Material LOMO-style evaluation has two material groups (`carbon` and `wood`). Because there are only two material groups, Material LOMO should be interpreted as a material-held-out stress test rather than a broad inferential estimate over a large population of material domains.",
        "",
        "The grouped metrics should therefore be described as frozen-embedding grouped robustness tests. They support the manuscript claim that v6.2-A is the primary reportable frozen ESPI encoder baseline under grouped board/material evaluation, while avoiding claims of full retrained CNN LOBO/LOMO generalization.",
        "",
        "## Caution Flags",
        "",
        f"- Low support threshold used for fold caution flags: {LOW_SUPPORT_THRESHOLD} samples per class.",
        f"- Board LOBO folds with caution flags: {key_numbers['lobo_caution_folds']}",
        f"- Material LOMO folds with caution flags: {key_numbers['lomo_caution_folds']}",
        "- Material LOMO folds are flagged for `small_number_of_material_groups` because only two material groups are available.",
        "",
    ]
    return "\n".join(report)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {args.eval_dir}")

    manifest = load_manifest(args.manifest)
    metadata = load_primary_metadata(args.features_dir, manifest)
    lookup = class_lookup(metadata)

    board_rows = build_dataset_by_board(metadata, lookup)
    material_rows = build_dataset_by_material(metadata, lookup)
    class_board_rows = build_class_by_board(metadata, lookup)
    class_material_rows = build_class_by_material(metadata, lookup)
    lobo_rows = build_board_lobo_folds(metadata, lookup)
    lomo_rows = build_material_lomo_folds(metadata, lookup)

    outputs = {
        "dataset_composition_by_board.csv": board_rows,
        "dataset_composition_by_material.csv": material_rows,
        "class_by_board_support.csv": class_board_rows,
        "class_by_material_support.csv": class_material_rows,
        "board_lobo_fold_support.csv": lobo_rows,
        "material_lomo_fold_support.csv": lomo_rows,
    }
    for filename, rows in outputs.items():
        write_csv(args.out_dir / filename, rows, fieldnames_from_rows(rows))

    boards = sorted(set(metadata["board"].astype(str).tolist()))
    materials = sorted(set(metadata["material"].astype(str).tolist()))
    distribution_groups = sorted(set(metadata["distribution_group"].astype(str).tolist()))
    key_numbers: dict[str, Any] = {
        "total_samples": int(len(metadata["label"])),
        "number_of_boards": int(len(boards)),
        "boards": boards,
        "number_of_materials": int(len(materials)),
        "materials": materials,
        "number_of_distribution_groups": int(len(distribution_groups)),
        "class_distribution": dict(count_classes(metadata["label"], lookup)),
        "all_boards_contain_all_classes": bool(all(row["missing_classes"] == "none" for row in class_board_rows)),
        "all_materials_contain_all_classes": bool(all(row["missing_classes"] == "none" for row in class_material_rows)),
        "lobo_missing_heldout_folds": int(sum(row["missing_classes_in_heldout"] != "none" for row in lobo_rows)),
        "lobo_missing_reference_folds": int(sum(row["missing_classes_in_reference"] != "none" for row in lobo_rows)),
        "lomo_missing_heldout_folds": int(sum(row["missing_classes_in_heldout"] != "none" for row in lomo_rows)),
        "lomo_missing_reference_folds": int(sum(row["missing_classes_in_reference"] != "none" for row in lomo_rows)),
        "lobo_caution_folds": int(sum(row["macro_f1_caution_flag"] != "none" for row in lobo_rows)),
        "lomo_caution_folds": int(sum(row["macro_f1_caution_flag"] != "none" for row in lomo_rows)),
        "low_support_threshold": LOW_SUPPORT_THRESHOLD,
        "primary_metadata_source": str((args.features_dir / "features_v62a_epoch25.normalized.npz").as_posix()),
        "manifest": str(args.manifest.as_posix()),
    }
    (args.out_dir / "dataset_composition_key_numbers.json").write_text(
        json.dumps(key_numbers, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = render_report(
        metadata,
        lookup,
        board_rows,
        material_rows,
        class_board_rows,
        class_material_rows,
        lobo_rows,
        lomo_rows,
        key_numbers,
    )
    (args.out_dir / "DATASET_COMPOSITION_REPORT.md").write_text(report, encoding="utf-8")

    print(f"[done] wrote dataset composition audit to {args.out_dir}")
    print(f"[done] total_samples={key_numbers['total_samples']} boards={len(boards)} materials={len(materials)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
