#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalize encoder feature dumps into a common internal comparison schema.

This script does not modify source NPZ files, does not run model inference,
does not train, and does not evaluate embeddings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUT_DIR = Path("outputs/encoder_features_normalized_v001")
DEFAULT_REPORT_DIR = Path("reports/encoder_baselines")
SCHEMA_VERSION = "encoder_feature_schema_v001"

NORMALIZED_FIELDS = [
    "embedding",
    "label",
    "label_name",
    "path",
    "board",
    "material",
    "frequency_hz",
    "split",
    "split_group",
    "distribution_group",
    "checkpoint_path",
    "checkpoint_sha256",
    "embedding_point",
    "model_name",
    "source_npz",
    "schema_version",
]

UNKNOWN_VALUES = {"", "unknown", "none", "nan", "null", "na"}


def scalar_to_str(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    if array.size == 1:
        return str(array.reshape(-1)[0])
    return str(array)


def as_str_array(value: np.ndarray | None, n_samples: int, default: str = "unknown") -> np.ndarray:
    if value is None:
        return np.array([default] * n_samples, dtype=object)
    array = np.asarray(value).astype(str)
    if array.shape == ():
        return np.array([str(array.item())] * n_samples, dtype=object)
    if len(array) != n_samples:
        raise ValueError(f"Expected metadata length {n_samples}, got {len(array)}")
    return array.astype(object)


def get_array(data: np.lib.npyio.NpzFile, aliases: list[str]) -> np.ndarray | None:
    for alias in aliases:
        if alias in data.files:
            return data[alias]
    return None


def get_scalar(data: np.lib.npyio.NpzFile, aliases: list[str], default: str = "unknown") -> str:
    for alias in aliases:
        if alias in data.files:
            return scalar_to_str(data[alias], default=default)
    return default


def is_unknown(value: Any) -> bool:
    return str(value).strip().lower() in UNKNOWN_VALUES


def unknown_fraction(values: np.ndarray) -> float:
    if len(values) == 0:
        return 1.0
    return float(np.mean([is_unknown(value) for value in values.astype(str)]))


def extract_board(*values: Any) -> str:
    joined = " ".join(str(value) for value in values if value is not None).upper()
    match = re.search(r"([CW]\d{2})", joined)
    return match.group(1) if match else "unknown"


def derive_material(board: str, *values: Any) -> str:
    joined = f"{board} " + " ".join(str(value) for value in values if value is not None)
    lower = joined.lower()
    if board.upper().startswith("C") or "carbon" in lower:
        return "carbon"
    if board.upper().startswith("W") or "wood" in lower:
        return "wood"
    return "unknown"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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


def load_manifest(manifest_path: Path | None) -> dict[str, Any]:
    if manifest_path is None or not manifest_path.exists():
        return {"exists": False, "path": str(manifest_path) if manifest_path else "", "by_path": {}}

    with np.load(str(manifest_path), allow_pickle=True) as data:
        paths = get_array(data, ["path", "paths", "image_path", "image_paths"])
        labels = get_array(data, ["label", "labels", "y"])
        boards = get_array(data, ["board", "boards"])
        freq = get_array(data, ["frequency_hz", "freq_hz", "frequency", "freq"])
        train_idx = get_array(data, ["train_idx"])
        val_idx = get_array(data, ["val_idx"])

        by_path: dict[str, dict[str, Any]] = {}
        if paths is not None:
            path_values = paths.astype(str)
            split = np.array(["unknown"] * len(path_values), dtype=object)
            if train_idx is not None:
                split[np.asarray(train_idx).astype(int)] = "train"
            if val_idx is not None:
                split[np.asarray(val_idx).astype(int)] = "val"
            for idx, path in enumerate(path_values):
                board_value = str(boards[idx]) if boards is not None else "unknown"
                board = extract_board(board_value, path)
                by_path[path] = {
                    "label": int(labels[idx]) if labels is not None else None,
                    "distribution_group": board_value,
                    "board": board,
                    "material": derive_material(board, board_value, path),
                    "frequency_hz": float(freq[idx]) if freq is not None else np.nan,
                    "split": str(split[idx]),
                    "split_group": board,
                }
    return {"exists": True, "path": str(manifest_path), "by_path": by_path}


def add_log(
    log_rows: list[dict[str, Any]],
    encoder: str,
    field: str,
    action: str,
    source: str,
    detail: str,
) -> None:
    log_rows.append(
        {
            "encoder": encoder,
            "field": field,
            "action": action,
            "source": source,
            "detail": detail,
        }
    )


def fill_from_manifest(
    values: np.ndarray,
    field: str,
    paths: np.ndarray,
    manifest: dict[str, Any],
    encoder: str,
    log_rows: list[dict[str, Any]],
) -> np.ndarray:
    if not manifest.get("exists"):
        return values
    by_path = manifest.get("by_path", {})
    output = values.copy()
    recovered = 0
    for idx, path in enumerate(paths.astype(str)):
        if not is_unknown(output[idx]):
            continue
        record = by_path.get(path)
        if record and record.get(field) is not None:
            output[idx] = record[field]
            recovered += 1
    if recovered:
        add_log(log_rows, encoder, field, "recovered", "manifest", f"Recovered {recovered} values from manifest path match.")
    return output


def normalize_feature_dump(
    encoder: str,
    source_path: Path,
    output_path: Path,
    manifest: dict[str, Any],
    model_name_default: str,
    final_publication_rule: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not source_path.exists():
        raise FileNotFoundError(f"Input feature dump not found: {source_path}")

    log_rows: list[dict[str, Any]] = []
    with np.load(str(source_path), allow_pickle=True) as data:
        original_fields = list(data.files)
        embedding = get_array(data, ["embedding", "embeddings"])
        if embedding is None:
            raise KeyError(f"No embedding/embeddings field in {source_path}")
        embedding = np.asarray(embedding).astype(np.float32)
        n_samples = int(embedding.shape[0])

        label = get_array(data, ["label", "labels", "y"])
        if label is None:
            raise KeyError(f"No label/labels field in {source_path}")
        label = np.asarray(label).astype(np.int64)

        label_name = as_str_array(get_array(data, ["label_name", "label_names", "class_name"]), n_samples)
        path = as_str_array(get_array(data, ["path", "paths", "image_path", "image_paths"]), n_samples, default="")
        distribution_group = as_str_array(get_array(data, ["distribution_group", "source_group", "group"]), n_samples)
        board = as_str_array(get_array(data, ["board", "boards"]), n_samples)
        material = as_str_array(get_array(data, ["material", "materials"]), n_samples)
        frequency_hz = get_array(data, ["frequency_hz", "freq_hz", "frequency", "freq"])
        if frequency_hz is None:
            frequency_hz = np.full(n_samples, np.nan, dtype=np.float32)
            add_log(log_rows, encoder, "frequency_hz", "unrecoverable", "source_npz", "frequency_hz was missing.")
        else:
            frequency_hz = np.asarray(frequency_hz).astype(np.float32)
        split = as_str_array(get_array(data, ["split", "splits"]), n_samples)
        split_group = as_str_array(get_array(data, ["split_group", "split_groups"]), n_samples)

        checkpoint_path = get_scalar(data, ["checkpoint_path"], default="unknown")
        checkpoint_sha256 = get_scalar(data, ["checkpoint_sha256"], default="unknown")
        embedding_point = get_scalar(data, ["embedding_point"], default="unknown")
        model_name = get_scalar(data, ["model_name"], default=model_name_default)
        if is_unknown(model_name):
            model_name = model_name_default
            add_log(log_rows, encoder, "model_name", "recovered", "default", model_name_default)

        if is_unknown(embedding_point):
            embedding_layer = get_scalar(data, ["embedding_layer"], default="unknown")
            if not is_unknown(embedding_layer):
                embedding_point = embedding_layer
                add_log(log_rows, encoder, "embedding_point", "recovered", "embedding_layer", embedding_layer)

        if is_unknown(checkpoint_sha256):
            if not is_unknown(checkpoint_path) and Path(checkpoint_path).exists():
                checkpoint_sha256 = sha256_file(Path(checkpoint_path))
                add_log(log_rows, encoder, "checkpoint_sha256", "recovered", "checkpoint_path", f"Computed SHA256 from {checkpoint_path}.")
            else:
                checkpoint_sha256 = "unknown"
                add_log(
                    log_rows,
                    encoder,
                    "checkpoint_sha256",
                    "unrecoverable",
                    "checkpoint_path",
                    f"Checkpoint file unavailable: {checkpoint_path}",
                )

    board = fill_from_manifest(board, "board", path, manifest, encoder, log_rows)
    material = fill_from_manifest(material, "material", path, manifest, encoder, log_rows)
    split = fill_from_manifest(split, "split", path, manifest, encoder, log_rows)
    split_group = fill_from_manifest(split_group, "split_group", path, manifest, encoder, log_rows)
    distribution_group = fill_from_manifest(distribution_group, "distribution_group", path, manifest, encoder, log_rows)

    recovered_board = 0
    recovered_material = 0
    recovered_split_group = 0
    recovered_distribution_group = 0

    for idx in range(n_samples):
        if is_unknown(distribution_group[idx]):
            distribution_group[idx] = Path(str(path[idx])).parent.name if str(path[idx]) else "unknown"
            if not is_unknown(distribution_group[idx]):
                recovered_distribution_group += 1
        if is_unknown(board[idx]):
            board[idx] = extract_board(distribution_group[idx], path[idx])
            if not is_unknown(board[idx]):
                recovered_board += 1
        if is_unknown(material[idx]):
            material[idx] = derive_material(str(board[idx]), distribution_group[idx], path[idx])
            if not is_unknown(material[idx]):
                recovered_material += 1
        if is_unknown(split_group[idx]):
            split_group[idx] = board[idx] if not is_unknown(board[idx]) else distribution_group[idx]
            if not is_unknown(split_group[idx]):
                recovered_split_group += 1

    if recovered_distribution_group:
        add_log(log_rows, encoder, "distribution_group", "recovered", "path_parent", f"Recovered {recovered_distribution_group} values.")
    if recovered_board:
        add_log(log_rows, encoder, "board", "recovered", "distribution_group_or_path", f"Recovered {recovered_board} values.")
    if recovered_material:
        add_log(log_rows, encoder, "material", "recovered", "board_path_distribution_group", f"Recovered {recovered_material} values.")
    if recovered_split_group:
        add_log(log_rows, encoder, "split_group", "recovered", "board", f"Recovered {recovered_split_group} values using board as conservative group.")

    source_npz = np.array(str(source_path), dtype=object)
    schema_version = np.array(SCHEMA_VERSION, dtype=object)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            embedding=embedding,
            label=label,
            label_name=label_name,
            path=path,
            board=board,
            material=material,
            frequency_hz=frequency_hz,
            split=split,
            split_group=split_group,
            distribution_group=distribution_group,
            checkpoint_path=np.array(checkpoint_path, dtype=object),
            checkpoint_sha256=np.array(checkpoint_sha256, dtype=object),
            embedding_point=np.array(embedding_point, dtype=object),
            model_name=np.array(model_name, dtype=object),
            source_npz=source_npz,
            schema_version=schema_version,
        )
    tmp_path.replace(output_path)

    missing_normalized = []
    for field_name, value in [
        ("embedding", embedding),
        ("label", label),
        ("label_name", label_name),
        ("path", path),
        ("board", board),
        ("material", material),
        ("frequency_hz", frequency_hz),
        ("split", split),
        ("split_group", split_group),
        ("distribution_group", distribution_group),
        ("checkpoint_path", np.array(checkpoint_path)),
        ("checkpoint_sha256", np.array(checkpoint_sha256)),
        ("embedding_point", np.array(embedding_point)),
        ("model_name", np.array(model_name)),
        ("source_npz", source_npz),
        ("schema_version", schema_version),
    ]:
        if value is None:
            missing_normalized.append(field_name)

    unrecoverable = [
        row["field"]
        for row in log_rows
        if row["action"] == "unrecoverable"
    ]
    unknown_counts = {
        "board_unknown": int(sum(is_unknown(value) for value in board)),
        "material_unknown": int(sum(is_unknown(value) for value in material)),
        "split_group_unknown": int(sum(is_unknown(value) for value in split_group)),
        "path_empty": int(sum(is_unknown(value) for value in path)),
    }
    acceptable_internal = not missing_normalized and unknown_counts["board_unknown"] == 0 and unknown_counts["split_group_unknown"] == 0
    acceptable_final = acceptable_internal and not unrecoverable and final_publication_rule == "final_ok"

    summary = {
        "encoder": encoder,
        "source_npz": str(source_path),
        "output_npz": str(output_path),
        "original_fields": original_fields,
        "normalized_fields": NORMALIZED_FIELDS,
        "missing_normalized_fields": missing_normalized,
        "recovered_fields": sorted({row["field"] for row in log_rows if row["action"] == "recovered"}),
        "unrecoverable_fields": sorted(set(unrecoverable)),
        "unknown_counts": unknown_counts,
        "n_samples": n_samples,
        "embedding_dim": int(embedding.shape[1]) if embedding.ndim == 2 else None,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "embedding_point": embedding_point,
        "model_name": model_name,
        "acceptable_for_internal_comparison": acceptable_internal,
        "acceptable_for_final_publication_tables": acceptable_final,
        "final_publication_rule": final_publication_rule,
    }
    return summary, log_rows


def report_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "encoder",
        "source_npz",
        "output_npz",
        "n_samples",
        "embedding_dim",
        "recovered_fields",
        "unrecoverable_fields",
        "acceptable_for_internal_comparison",
        "acceptable_for_final_publication_tables",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|---|---|---|---:|---:|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(
                str(row.get(column, "")).replace("\n", " ")
                for column in columns
            ) + " |"
        )
    return "\n".join(lines)


def write_report(path: Path, summaries: list[dict[str, Any]], log_rows: list[dict[str, Any]]) -> None:
    recovered_lines = "\n".join(
        f"- {row['encoder']} `{row['field']}`: {row['detail']} ({row['source']})"
        for row in log_rows
        if row["action"] == "recovered"
    ) or "- None"
    unrecoverable_lines = "\n".join(
        f"- {row['encoder']} `{row['field']}`: {row['detail']}"
        for row in log_rows
        if row["action"] == "unrecoverable"
    ) or "- None"
    text = f"""# Encoder Feature Schema Normalization

## Scope

This report documents normalization of existing encoder feature dumps into a common schema for internal baseline comparison. The process does not modify original NPZ files, does not run model inference, does not train, and does not evaluate embeddings.

## Normalized schema

`{', '.join(NORMALIZED_FIELDS)}`

## Summary

{report_table(summaries)}

## Recovered fields

{recovered_lines}

## Unrecoverable fields

{unrecoverable_lines}

## Final-publication rule

- v6.2-A normalized dump may be acceptable for final tables if checkpoint SHA is computed or the checkpoint path is documented.
- Hierarchical normalized dump is acceptable for internal comparison, but not final publication tables until extraction is regenerated with correct `board` and `split_group` directly stored in the source NPZ.

## Output files

{chr(10).join(f'- `{row["output_npz"]}`' for row in summaries)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize encoder feature dump schemas.")
    parser.add_argument("--v62a-features", type=Path, required=True)
    parser.add_argument("--hier-features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    summaries: list[dict[str, Any]] = []
    all_logs: list[dict[str, Any]] = []

    v62a_summary, v62a_logs = normalize_feature_dump(
        encoder="v6.2-A",
        source_path=args.v62a_features,
        output_path=args.out_dir / "features_v62a_epoch25.normalized.npz",
        manifest=manifest,
        model_name_default="v6.2-A MCDropoutClassifier",
        final_publication_rule="final_ok",
    )
    summaries.append(v62a_summary)
    all_logs.extend(v62a_logs)

    hier_summary, hier_logs = normalize_feature_dump(
        encoder="hierarchical_v6.2_phase2_expert",
        source_path=args.hier_features,
        output_path=args.out_dir / "features_hier_z_expert_prelogit.normalized.npz",
        manifest=manifest,
        model_name_default="EnhancedHybridPhysicsESPI_V6_2",
        final_publication_rule="regenerate_before_final",
    )
    summaries.append(hier_summary)
    all_logs.extend(hier_logs)

    report_dir = args.report_dir
    report_path = report_dir / "FEATURE_SCHEMA_NORMALIZATION.md"
    json_path = report_dir / "feature_schema_normalization.json"
    log_path = report_dir / "feature_schema_recovery_log.csv"
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "manifest": manifest["path"],
        "out_dir": str(args.out_dir),
        "summaries": summaries,
        "recovery_log": all_logs,
    }
    write_report(report_path, summaries, all_logs)
    write_json(json_path, payload)
    write_csv(log_path, all_logs)

    print(f"[done] report={report_path}")
    print(f"[done] key_numbers={json_path}")
    print(f"[done] recovery_log={log_path}")
    for summary in summaries:
        print(
            "[done] {encoder} output={output} internal={internal} final={final} recovered={recovered} unrecoverable={unrecoverable}".format(
                encoder=summary["encoder"],
                output=summary["output_npz"],
                internal=summary["acceptable_for_internal_comparison"],
                final=summary["acceptable_for_final_publication_tables"],
                recovered=",".join(summary["recovered_fields"]) or "none",
                unrecoverable=",".join(summary["unrecoverable_fields"]) or "none",
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
