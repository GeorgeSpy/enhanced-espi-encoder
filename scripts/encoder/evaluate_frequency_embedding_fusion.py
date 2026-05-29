#!/usr/bin/env python
"""Evaluate frequency-plus-embedding information budget controls.

This script evaluates saved metadata and frozen embeddings only. It does not
load raw images, run deep inference, train/fine-tune encoders, implement LeFFT,
or evaluate acoustic-response prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np


CLASS_IDS = [0, 1, 2, 3, 4]
CLASS_NAMES = {0: "1_1H", 1: "1_1T", 2: "1_2", 3: "2_1", 4: "higher"}
RANDOM_STATE = 42

FEATURE_FILES = {
    "v6_1": ("v6.1", "features_v61.normalized.npz"),
    "v6_2_a": ("v6.2-A", "features_v62a_epoch25.normalized.npz"),
    "random_resnet18": ("Random ResNet-18", "features_resnet18_random.normalized.npz"),
    "imagenet_resnet18": ("ImageNet ResNet-18", "features_resnet18_imagenet.normalized.npz"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--frequency-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--linear-max-iter", type=int, default=5000)
    parser.add_argument("--rf-trees", type=int, default=80)
    return parser.parse_args()


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


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


def collect_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    return fields


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = collect_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_feature_bundle(features_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    bundles: dict[str, dict[str, np.ndarray]] = {}
    feature_paths: dict[str, str] = {}
    for encoder_id, (_, filename) in FEATURE_FILES.items():
        path = features_dir / filename
        if path.exists():
            bundle = load_npz(path)
            if "embedding" not in bundle:
                raise KeyError(f"Missing embedding in {path}")
            bundles[encoder_id] = bundle
            feature_paths[encoder_id] = str(path)

    if "v6_2_a" not in bundles:
        raise FileNotFoundError(features_dir / FEATURE_FILES["v6_2_a"][1])
    reference = bundles["v6_2_a"]
    required = ["label", "frequency_hz", "board", "material", "split", "path"]
    missing = [key for key in required if key not in reference]
    if missing:
        raise KeyError(f"Missing required v6.2-A metadata keys: {missing}")

    metadata = {
        "label": np.asarray(reference["label"]).astype(int),
        "label_name": (
            np.asarray(reference["label_name"]).astype(str)
            if "label_name" in reference
            else np.asarray([CLASS_NAMES[int(item)] for item in reference["label"]], dtype=object)
        ),
        "frequency_hz": np.asarray(reference["frequency_hz"]).astype(np.float64),
        "board": np.asarray(reference["board"]).astype(str),
        "material": np.asarray(reference["material"]).astype(str),
        "split": np.asarray(reference["split"]).astype(str),
        "path": np.asarray(reference["path"]).astype(str),
    }

    embeddings: dict[str, np.ndarray] = {}
    for encoder_id, bundle in bundles.items():
        for key in ["label", "path", "board", "material", "split"]:
            if key in bundle and not np.array_equal(np.asarray(bundle[key]).astype(str), metadata[key].astype(str)):
                raise ValueError(f"Metadata mismatch for {encoder_id}: {key}")
        embeddings[encoder_id] = np.asarray(bundle["embedding"]).astype(np.float64)

    return metadata, embeddings, feature_paths


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    per_class: list[dict[str, Any]] = []
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    for class_id in CLASS_IDS:
        true_mask = y_true == class_id
        pred_mask = y_pred == class_id
        support = int(np.sum(true_mask))
        true_positive = int(np.sum(true_mask & pred_mask))
        false_positive = int(np.sum(~true_mask & pred_mask))
        false_negative = int(np.sum(true_mask & ~pred_mask))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "class_id": class_id,
                "label_name": CLASS_NAMES[class_id],
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {
        "accuracy": accuracy,
        "macro_recall": float(np.mean([item["recall"] for item in per_class])),
        "macro_f1": float(np.mean([item["f1"] for item in per_class])),
        "per_class": per_class,
    }


def standardize_train_test(train_x: np.ndarray, test_x: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std


def build_model_specs(linear_max_iter: int, rf_trees: int) -> list[dict[str, Any]]:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"scikit-learn is required for fusion evaluation: {exc}") from exc

    return [
        {
            "method": "balanced_logistic_regression",
            "description": "Balanced logistic regression on train-fold standardized inputs.",
            "parameters": {
                "class_weight": "balanced",
                "max_iter": linear_max_iter,
                "random_state": RANDOM_STATE,
                "preprocessing": "train/reference-fold standardization",
            },
            "factory": lambda: LogisticRegression(
                class_weight="balanced",
                max_iter=linear_max_iter,
                random_state=RANDOM_STATE,
            ),
        },
        {
            "method": "random_forest",
            "description": "Random forest control on train-fold standardized inputs.",
            "parameters": {
                "n_estimators": rf_trees,
                "class_weight": "balanced_subsample",
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "random_state": RANDOM_STATE,
                "preprocessing": "train/reference-fold standardization",
            },
            "factory": lambda: RandomForestClassifier(
                n_estimators=rf_trees,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
    ]


def build_conditions(embeddings: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = [
        {
            "condition": "frequency_only",
            "display_name": "Frequency only",
            "encoder_id": "",
            "condition_type": "metadata_only",
        }
    ]
    for encoder_id in ["v6_1", "v6_2_a", "random_resnet18", "imagenet_resnet18"]:
        if encoder_id not in embeddings:
            continue
        display = FEATURE_FILES[encoder_id][0]
        conditions.append(
            {
                "condition": f"{encoder_id}_embedding_only",
                "display_name": f"{display} embedding only",
                "encoder_id": encoder_id,
                "condition_type": "embedding_only",
            }
        )
        conditions.append(
            {
                "condition": f"{encoder_id}_frequency_fusion",
                "display_name": f"Frequency + {display} embedding",
                "encoder_id": encoder_id,
                "condition_type": "frequency_embedding_fusion",
            }
        )
    return conditions


def condition_matrix(
    condition: dict[str, Any],
    metadata: dict[str, np.ndarray],
    embeddings: dict[str, np.ndarray],
) -> np.ndarray:
    frequency = metadata["frequency_hz"].reshape(-1, 1).astype(np.float64)
    condition_type = condition["condition_type"]
    if condition_type == "metadata_only":
        return frequency
    embedding = embeddings[condition["encoder_id"]]
    if condition_type == "embedding_only":
        return embedding
    if condition_type == "frequency_embedding_fusion":
        return np.hstack([embedding, frequency])
    raise ValueError(f"Unknown condition type: {condition_type}")


def fit_predict(
    model_spec: dict[str, Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    train_features, test_features = standardize_train_test(train_x, test_x)
    model = model_spec["factory"]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train_features, train_y)
    return np.asarray(model.predict(test_features)).astype(int)


def append_metric_rows(
    summary_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    base: dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    metrics = metrics_from_predictions(y_true, y_pred)
    summary_rows.append(
        {
            **base,
            "accuracy": metrics["accuracy"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "n_test": int(len(y_true)),
        }
    )
    for class_row in metrics["per_class"]:
        per_class_rows.append({**base, **class_row})


def evaluate_stratified(
    metadata: dict[str, np.ndarray],
    embeddings: dict[str, np.ndarray],
    conditions: list[dict[str, Any]],
    model_specs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_mask = metadata["split"] == "train"
    test_mask = metadata["split"] == "val"
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("Expected split values `train` and `val` for stratified evaluation.")

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    labels = metadata["label"]
    for condition in conditions:
        matrix = condition_matrix(condition, metadata, embeddings)
        for spec in model_specs:
            pred = fit_predict(spec, matrix[train_mask], labels[train_mask], matrix[test_mask])
            append_metric_rows(
                summary_rows,
                per_class_rows,
                {
                    "protocol": "stratified_train_val",
                    "condition": condition["condition"],
                    "display_name": condition["display_name"],
                    "condition_type": condition["condition_type"],
                    "method": spec["method"],
                    "n_train": int(np.sum(train_mask)),
                },
                labels[test_mask],
                pred,
            )
    return summary_rows, per_class_rows


def evaluate_grouped(
    metadata: dict[str, np.ndarray],
    embeddings: dict[str, np.ndarray],
    conditions: list[dict[str, Any]],
    model_specs: list[dict[str, Any]],
    group_key: str,
    protocol: str,
    stress_test_note: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    labels = metadata["label"]
    groups = np.asarray(metadata[group_key]).astype(str)
    fold_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for condition in conditions:
        matrix = condition_matrix(condition, metadata, embeddings)
        for spec in model_specs:
            method_fold_rows: list[dict[str, Any]] = []
            for held_out in sorted(np.unique(groups).tolist()):
                test_mask = groups == held_out
                train_mask = ~test_mask
                pred = fit_predict(spec, matrix[train_mask], labels[train_mask], matrix[test_mask])
                metrics = metrics_from_predictions(labels[test_mask], pred)
                base = {
                    "protocol": protocol,
                    "condition": condition["condition"],
                    "display_name": condition["display_name"],
                    "condition_type": condition["condition_type"],
                    "method": spec["method"],
                    "held_out_group": held_out,
                    "n_train": int(np.sum(train_mask)),
                    "n_test": int(np.sum(test_mask)),
                    "stress_test_note": stress_test_note,
                }
                fold_row = {
                    **base,
                    "accuracy": metrics["accuracy"],
                    "macro_recall": metrics["macro_recall"],
                    "macro_f1": metrics["macro_f1"],
                }
                fold_rows.append(fold_row)
                method_fold_rows.append(fold_row)
                for class_row in metrics["per_class"]:
                    per_class_rows.append({**base, **class_row})

            f1_values = [float(row["macro_f1"]) for row in method_fold_rows]
            worst = min(method_fold_rows, key=lambda row: float(row["macro_f1"]))
            best = max(method_fold_rows, key=lambda row: float(row["macro_f1"]))
            summary_rows.append(
                {
                    "protocol": protocol,
                    "condition": condition["condition"],
                    "display_name": condition["display_name"],
                    "condition_type": condition["condition_type"],
                    "method": spec["method"],
                    "mean_macro_f1": float(np.mean(f1_values)),
                    "mean_accuracy": float(np.mean([float(row["accuracy"]) for row in method_fold_rows])),
                    "mean_macro_recall": float(np.mean([float(row["macro_recall"]) for row in method_fold_rows])),
                    "worst_group": worst["held_out_group"],
                    "worst_group_macro_f1": float(worst["macro_f1"]),
                    "best_group": best["held_out_group"],
                    "best_group_macro_f1": float(best["macro_f1"]),
                    "n_groups": len(method_fold_rows),
                    "stress_test_note": stress_test_note,
                }
            )
    return summary_rows, fold_rows, per_class_rows


def row_score(row: dict[str, Any]) -> float:
    if row.get("macro_f1") not in ("", None):
        return float(row["macro_f1"])
    return float(row["mean_macro_f1"])


def best_by_condition_protocol(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["condition"], row["protocol"])
        if key not in best or row_score(row) > row_score(best[key]):
            best[key] = row
    return best


def build_delta_rows(best: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    protocols = ["stratified_train_val", "board_grouped", "material_grouped"]
    for encoder_id in ["v6_1", "v6_2_a", "random_resnet18", "imagenet_resnet18"]:
        embedding_condition = f"{encoder_id}_embedding_only"
        fusion_condition = f"{encoder_id}_frequency_fusion"
        for protocol in protocols:
            frequency_row = best.get(("frequency_only", protocol))
            embedding_row = best.get((embedding_condition, protocol))
            fusion_row = best.get((fusion_condition, protocol))
            if not frequency_row or not embedding_row or not fusion_row:
                continue
            frequency_score = row_score(frequency_row)
            embedding_score = row_score(embedding_row)
            fusion_score = row_score(fusion_row)
            rows.append(
                {
                    "protocol": protocol,
                    "encoder_id": encoder_id,
                    "encoder_name": FEATURE_FILES[encoder_id][0],
                    "frequency_only_method": frequency_row["method"],
                    "frequency_only_macro_f1": frequency_score,
                    "embedding_only_method": embedding_row["method"],
                    "embedding_only_macro_f1": embedding_score,
                    "fusion_method": fusion_row["method"],
                    "fusion_macro_f1": fusion_score,
                    "delta_fusion_minus_frequency_pp": (fusion_score - frequency_score) * 100.0,
                    "delta_fusion_minus_embedding_pp": (fusion_score - embedding_score) * 100.0,
                }
            )
    return rows


def load_existing_encoder_summary(eval_dir: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(eval_dir / "encoder_baseline_summary.csv")
    return {row["encoder_id"]: row for row in rows if "encoder_id" in row}


def compact_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                "protocol": row["protocol"],
                "condition": row["condition"],
                "display_name": row["display_name"],
                "method": row["method"],
                "macro_f1": row.get("macro_f1", ""),
                "mean_macro_f1": row.get("mean_macro_f1", ""),
                "accuracy": row.get("accuracy", row.get("mean_accuracy", "")),
                "macro_recall": row.get("macro_recall", row.get("mean_macro_recall", "")),
                "worst_group": row.get("worst_group", ""),
                "worst_group_macro_f1": row.get("worst_group_macro_f1", ""),
                "best_group": row.get("best_group", ""),
                "best_group_macro_f1": row.get("best_group_macro_f1", ""),
                "n_groups": row.get("n_groups", ""),
            }
        )
    return output


def build_key_numbers(
    best: dict[tuple[str, str], dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    frequency_key_numbers: dict[str, Any],
    feature_paths: dict[str, str],
) -> dict[str, Any]:
    def score(condition: str, protocol: str) -> float:
        return row_score(best[(condition, protocol)])

    key_numbers: dict[str, Any] = {
        "feature_paths": feature_paths,
        "frequency_only_reference": frequency_key_numbers,
        "best_frequency_only": {
            protocol: {
                "method": best[("frequency_only", protocol)]["method"],
                "macro_f1": score("frequency_only", protocol),
            }
            for protocol in ["stratified_train_val", "board_grouped", "material_grouped"]
        },
        "v6_1_embedding_only": {
            protocol: {
                "method": best[("v6_1_embedding_only", protocol)]["method"],
                "macro_f1": score("v6_1_embedding_only", protocol),
            }
            for protocol in ["stratified_train_val", "board_grouped", "material_grouped"]
        },
        "v6_2_a_embedding_only": {
            protocol: {
                "method": best[("v6_2_a_embedding_only", protocol)]["method"],
                "macro_f1": score("v6_2_a_embedding_only", protocol),
            }
            for protocol in ["stratified_train_val", "board_grouped", "material_grouped"]
        },
        "v6_1_frequency_fusion": {
            protocol: {
                "method": best[("v6_1_frequency_fusion", protocol)]["method"],
                "macro_f1": score("v6_1_frequency_fusion", protocol),
            }
            for protocol in ["stratified_train_val", "board_grouped", "material_grouped"]
        },
        "v6_2_a_frequency_fusion": {
            protocol: {
                "method": best[("v6_2_a_frequency_fusion", protocol)]["method"],
                "macro_f1": score("v6_2_a_frequency_fusion", protocol),
            }
            for protocol in ["stratified_train_val", "board_grouped", "material_grouped"]
        },
        "deltas": delta_rows,
    }
    for row in delta_rows:
        if row["encoder_id"] == "v6_2_a":
            key_numbers[f"v62a_{row['protocol']}_delta_fusion_minus_frequency_pp"] = row[
                "delta_fusion_minus_frequency_pp"
            ]
            key_numbers[f"v62a_{row['protocol']}_delta_fusion_minus_embedding_pp"] = row[
                "delta_fusion_minus_embedding_pp"
            ]
        if row["encoder_id"] == "v6_1":
            key_numbers[f"v61_{row['protocol']}_delta_fusion_minus_frequency_pp"] = row[
                "delta_fusion_minus_frequency_pp"
            ]
            key_numbers[f"v61_{row['protocol']}_delta_fusion_minus_embedding_pp"] = row[
                "delta_fusion_minus_embedding_pp"
            ]
    return key_numbers


def render_report(
    model_specs: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    board_rows: list[dict[str, Any]],
    material_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    key_numbers: dict[str, Any],
    encoder_summary: dict[str, dict[str, str]],
) -> str:
    compact_rows = compact_summary_rows(summary_rows)
    summary_fields = [
        "protocol",
        "condition",
        "display_name",
        "method",
        "macro_f1",
        "mean_macro_f1",
        "worst_group",
        "worst_group_macro_f1",
        "best_group",
        "best_group_macro_f1",
        "n_groups",
    ]
    delta_fields = [
        "protocol",
        "encoder_name",
        "frequency_only_method",
        "frequency_only_macro_f1",
        "embedding_only_method",
        "embedding_only_macro_f1",
        "fusion_method",
        "fusion_macro_f1",
        "delta_fusion_minus_frequency_pp",
        "delta_fusion_minus_embedding_pp",
    ]
    board_fields = [
        "protocol",
        "condition",
        "method",
        "held_out_group",
        "accuracy",
        "macro_recall",
        "macro_f1",
        "n_test",
        "stress_test_note",
    ]
    model_lines = [
        f"- `{spec['method']}`: {spec['description']} Parameters: `{json.dumps(spec['parameters'])}`"
        for spec in model_specs
    ]
    condition_lines = [
        f"- `{condition['condition']}`: {condition['display_name']} ({condition['condition_type']})"
        for condition in conditions
    ]

    def score(section: str, protocol: str) -> float:
        return float(key_numbers[section][protocol]["macro_f1"])

    v62a_lines = []
    for protocol in ["stratified_train_val", "board_grouped", "material_grouped"]:
        freq_score = score("best_frequency_only", protocol)
        emb_score = score("v6_2_a_embedding_only", protocol)
        fusion_score = score("v6_2_a_frequency_fusion", protocol)
        v62a_lines.append(
            f"- `{protocol}`: frequency-only {pct(freq_score)}, v6.2-A embedding-only {pct(emb_score)}, "
            f"frequency + v6.2-A {pct(fusion_score)}; "
            f"fusion minus frequency {100.0 * (fusion_score - freq_score):+.2f} pp, "
            f"fusion minus embedding {100.0 * (fusion_score - emb_score):+.2f} pp."
        )

    v61_benefit = np.mean(
        [
            row["delta_fusion_minus_embedding_pp"]
            for row in delta_rows
            if row["encoder_id"] == "v6_1"
        ]
    )
    v62a_benefit = np.mean(
        [
            row["delta_fusion_minus_embedding_pp"]
            for row in delta_rows
            if row["encoder_id"] == "v6_2_a"
        ]
    )
    if v62a_benefit > v61_benefit:
        benefit_line = "v6.2-A benefits more from frequency fusion than v6.1 on average across the evaluated protocols."
    elif v61_benefit > v62a_benefit:
        benefit_line = "v6.1 benefits more from frequency fusion than v6.2-A on average across the evaluated protocols."
    else:
        benefit_line = "v6.1 and v6.2-A show equal average benefit from frequency fusion across the evaluated protocols."

    v62a_fusion_beats_frequency = any(
        row["encoder_id"] == "v6_2_a" and row["delta_fusion_minus_frequency_pp"] > 0.0 for row in delta_rows
    )
    if v62a_fusion_beats_frequency:
        interpretation = (
            "Frequency + v6.2-A improves over frequency-only in at least one protocol, indicating that the frozen ESPI "
            "embedding can add complementary image-derived information under that setting."
        )
    else:
        interpretation = (
            "Frequency-only remains stronger than frequency + v6.2-A across the evaluated protocols, indicating that "
            "the current five-class label protocol is strongly frequency-structured."
        )

    existing_rows = []
    for encoder_id in ["random_resnet18", "imagenet_resnet18", "v6_1", "v6_2_a"]:
        row = encoder_summary.get(encoder_id, {})
        if not row:
            continue
        existing_rows.append(
            {
                "encoder": row.get("encoder_name", encoder_id),
                "stratified_knn_macro_f1": row.get("stratified_best_knn_macro_f1", ""),
                "board_lobo_macro_f1": row.get("best_lobo_mean_macro_f1", ""),
                "material_lomo_macro_f1": row.get("best_lomo_mean_macro_f1", ""),
            }
        )

    return "\n".join(
        [
            "# Frequency + Embedding Information-Budget Comparison",
            "",
            "## Purpose",
            "",
            "This report evaluates whether frozen ESPI embeddings add predictive information beyond `frequency_hz`. It uses only normalized feature dumps and metadata. No raw images, deep-model inference, encoder fine-tuning, LeFFT evaluation, or acoustic-response prediction is performed.",
            "",
            "## Conditions",
            "",
            *condition_lines,
            "",
            "## Models and Preprocessing",
            "",
            *model_lines,
            "",
            "All input features, including `frequency_hz` and frozen embeddings, are standardized using the training/reference fold only and then applied to the held-out fold. Grouped held-out samples are never used to fit preprocessing.",
            "",
            "## Summary Metrics",
            "",
            markdown_table(compact_rows, summary_fields),
            "",
            "## Fusion Deltas",
            "",
            markdown_table(delta_rows, delta_fields),
            "",
            "## Board LOBO-Style Fold Scores",
            "",
            markdown_table(board_rows, board_fields),
            "",
            "## Material LOMO-Style Fold Scores",
            "",
            markdown_table(material_rows, board_fields),
            "",
            "## Existing Encoder-Only Reference Metrics",
            "",
            markdown_table(existing_rows, ["encoder", "stratified_knn_macro_f1", "board_lobo_macro_f1", "material_lomo_macro_f1"]),
            "",
            "## v6.2-A Information-Budget Interpretation",
            "",
            *v62a_lines,
            f"- {benefit_line}",
            f"- {interpretation}",
            "- Material LOMO remains a descriptive stress test because only two material groups are available.",
            "- The result refines, rather than expands, the manuscript claim: the main claim is image-derived frozen encoder evidence, not acoustic-response prediction.",
            "",
            "## Manuscript Patch Recommendation",
            "",
            "- Abstract: state that frequency metadata is a strong metadata-only predictor and that v6.2-A is the strongest image-derived frozen ESPI encoder baseline, not the strongest predictor overall.",
            "- Results: add a frequency-only and frequency-plus-embedding information-budget paragraph before the final encoder decision paragraph.",
            "- Discussion: explain that the five-class modal protocol is strongly frequency-structured; frozen ESPI embeddings should be interpreted as image-derived representation evidence rather than as proof that frequency metadata is insufficient.",
        ]
    )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metadata, embeddings, feature_paths = load_feature_bundle(args.features_dir)
    model_specs = build_model_specs(args.linear_max_iter, args.rf_trees)
    conditions = build_conditions(embeddings)

    strat_rows, strat_per_class = evaluate_stratified(metadata, embeddings, conditions, model_specs)
    board_summary, board_rows, board_per_class = evaluate_grouped(
        metadata,
        embeddings,
        conditions,
        model_specs,
        group_key="board",
        protocol="board_grouped",
        stress_test_note="none",
    )
    material_summary, material_rows, material_per_class = evaluate_grouped(
        metadata,
        embeddings,
        conditions,
        model_specs,
        group_key="material",
        protocol="material_grouped",
        stress_test_note="descriptive stress-test only; two material groups available",
    )

    summary_rows = strat_rows + board_summary + material_summary
    per_class_rows = strat_per_class + board_per_class + material_per_class
    best = best_by_condition_protocol(summary_rows)
    delta_rows = build_delta_rows(best)
    frequency_key_numbers = read_json(args.frequency_dir / "frequency_only_key_numbers.json")
    encoder_summary = load_existing_encoder_summary(args.eval_dir)
    key_numbers = build_key_numbers(best, delta_rows, frequency_key_numbers, feature_paths)

    write_csv(args.out_dir / "frequency_embedding_fusion_summary.csv", compact_summary_rows(summary_rows))
    write_csv(args.out_dir / "frequency_embedding_fusion_board_lobo.csv", board_rows)
    write_csv(args.out_dir / "frequency_embedding_fusion_material_lomo.csv", material_rows)
    write_csv(args.out_dir / "frequency_embedding_fusion_per_class.csv", per_class_rows)
    write_csv(args.out_dir / "frequency_embedding_fusion_deltas.csv", delta_rows)
    (args.out_dir / "frequency_embedding_fusion_key_numbers.json").write_text(
        json.dumps(key_numbers, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.out_dir / "FREQUENCY_EMBEDDING_FUSION_REPORT.md").write_text(
        render_report(model_specs, conditions, summary_rows, board_rows, material_rows, delta_rows, key_numbers, encoder_summary),
        encoding="utf-8",
    )

    print(f"[done] wrote frequency + embedding fusion report to {args.out_dir}")
    print(
        "[done] v62a_fusion stratified={:.6f} board={:.6f} material={:.6f}".format(
            key_numbers["v6_2_a_frequency_fusion"]["stratified_train_val"]["macro_f1"],
            key_numbers["v6_2_a_frequency_fusion"]["board_grouped"]["macro_f1"],
            key_numbers["v6_2_a_frequency_fusion"]["material_grouped"]["macro_f1"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
