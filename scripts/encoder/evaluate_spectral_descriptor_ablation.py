#!/usr/bin/env python
"""Evaluate deterministic spectral descriptor ablations.

This script uses saved normalized feature dumps and metadata only. It does not
load images, train deep models, fine-tune encoders, implement acoustic-response
prediction, or treat the descriptors as a trained LeFFT model.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np


CLASS_IDS = [0, 1, 2, 3, 4]
CLASS_NAMES = {0: "1_1H", 1: "1_1T", 2: "1_2", 3: "2_1", 4: "higher"}
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--frequency-dir", required=True, type=Path)
    parser.add_argument("--fusion-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--linear-max-iter", type=int, default=5000)
    parser.add_argument("--rf-trees", type=int, default=80)
    return parser.parse_args()


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def pp(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f} pp"


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(str(path), allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_feature_inputs(features_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    spectral_path = features_dir / "features_spectral_descriptors.normalized.npz"
    v62a_path = features_dir / "features_v62a_epoch25.normalized.npz"
    spectral = load_npz(spectral_path)
    v62a = load_npz(v62a_path)
    for name, bundle in [("spectral", spectral), ("v6.2-A", v62a)]:
        if "embedding" not in bundle:
            raise KeyError(f"Missing embedding in {name} feature dump")
    required = ["label", "label_name", "frequency_hz", "board", "material", "split", "path"]
    missing = [key for key in required if key not in v62a]
    if missing:
        raise KeyError(f"Missing required v6.2-A metadata keys: {missing}")

    metadata = {
        "label": np.asarray(v62a["label"]).astype(int),
        "label_name": np.asarray(v62a["label_name"]).astype(str),
        "frequency_hz": np.asarray(v62a["frequency_hz"]).astype(np.float64),
        "board": np.asarray(v62a["board"]).astype(str),
        "material": np.asarray(v62a["material"]).astype(str),
        "split": np.asarray(v62a["split"]).astype(str),
        "path": np.asarray(v62a["path"]).astype(str),
    }
    for key in ["label", "path", "board", "material", "split"]:
        if key in spectral and not np.array_equal(np.asarray(spectral[key]).astype(str), metadata[key].astype(str)):
            raise ValueError(f"Metadata mismatch between spectral and v6.2-A feature dumps: {key}")
    embeddings = {
        "spectral_descriptors": np.asarray(spectral["embedding"]).astype(np.float64),
        "v6_2_a": np.asarray(v62a["embedding"]).astype(np.float64),
    }
    feature_paths = {
        "spectral_descriptors": str(spectral_path),
        "v6_2_a": str(v62a_path),
    }
    return metadata, embeddings, feature_paths


def standardize_train_test(train_x: np.ndarray, test_x: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std


def build_model_specs(linear_max_iter: int, rf_trees: int) -> list[dict[str, Any]]:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"scikit-learn is required for spectral descriptor ablation: {exc}") from exc

    return [
        {
            "method": "balanced_logistic_regression",
            "description": "Balanced logistic regression on train/reference-fold standardized inputs.",
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
            "description": "Random forest control on train/reference-fold standardized inputs.",
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


def build_conditions() -> list[dict[str, str]]:
    return [
        {
            "condition": "frequency_only",
            "display_name": "Frequency only",
            "condition_type": "metadata_only",
        },
        {
            "condition": "spectral_descriptors_only",
            "display_name": "Spectral descriptors only",
            "condition_type": "spectral_only",
        },
        {
            "condition": "v6_2_a_embedding_only",
            "display_name": "v6.2-A embedding only",
            "condition_type": "v62a_only",
        },
        {
            "condition": "frequency_spectral_fusion",
            "display_name": "Frequency + spectral descriptors",
            "condition_type": "frequency_spectral",
        },
        {
            "condition": "v6_2_a_spectral_fusion",
            "display_name": "v6.2-A + spectral descriptors",
            "condition_type": "v62a_spectral",
        },
        {
            "condition": "v6_2_a_frequency_fusion",
            "display_name": "Frequency + v6.2-A",
            "condition_type": "frequency_v62a",
        },
        {
            "condition": "frequency_v6_2_a_spectral_fusion",
            "display_name": "Frequency + v6.2-A + spectral descriptors",
            "condition_type": "frequency_v62a_spectral",
        },
    ]


def condition_matrix(condition: dict[str, str], metadata: dict[str, np.ndarray], embeddings: dict[str, np.ndarray]) -> np.ndarray:
    frequency = metadata["frequency_hz"].reshape(-1, 1).astype(np.float64)
    spectral = embeddings["spectral_descriptors"]
    v62a = embeddings["v6_2_a"]
    condition_type = condition["condition_type"]
    if condition_type == "metadata_only":
        return frequency
    if condition_type == "spectral_only":
        return spectral
    if condition_type == "v62a_only":
        return v62a
    if condition_type == "frequency_spectral":
        return np.hstack([spectral, frequency])
    if condition_type == "v62a_spectral":
        return np.hstack([v62a, spectral])
    if condition_type == "frequency_v62a":
        return np.hstack([v62a, frequency])
    if condition_type == "frequency_v62a_spectral":
        return np.hstack([v62a, spectral, frequency])
    raise ValueError(f"Unknown condition type: {condition_type}")


def fit_predict(model_spec: dict[str, Any], train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    train_features, test_features = standardize_train_test(train_x, test_x)
    model = model_spec["factory"]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train_features, train_y)
    return np.asarray(model.predict(test_features)).astype(int)


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    per_class: list[dict[str, Any]] = []
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
        "macro_recall": float(np.mean([row["recall"] for row in per_class])),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "per_class": per_class,
    }


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
    conditions: list[dict[str, str]],
    model_specs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_mask = metadata["split"] == "train"
    test_mask = metadata["split"] == "val"
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("Expected split values `train` and `val` for stratified evaluation.")
    labels = metadata["label"]
    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
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
    conditions: list[dict[str, str]],
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


def delta_row(best: dict[tuple[str, str], dict[str, Any]], protocol: str, left: str, right: str, label: str) -> dict[str, Any]:
    left_score = row_score(best[(left, protocol)])
    right_score = row_score(best[(right, protocol)])
    return {
        "protocol": protocol,
        "comparison": label,
        "left_condition": left,
        "right_condition": right,
        "left_macro_f1": left_score,
        "right_macro_f1": right_score,
        "delta_left_minus_right_pp": (left_score - right_score) * 100.0,
    }


def build_delta_rows(best: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparisons = [
        ("spectral_descriptors_only", "frequency_only", "spectral-only minus frequency-only"),
        ("frequency_spectral_fusion", "frequency_only", "frequency+spectral minus frequency-only"),
        ("v6_2_a_spectral_fusion", "v6_2_a_embedding_only", "v6.2-A+spectral minus v6.2-A"),
        (
            "frequency_v6_2_a_spectral_fusion",
            "v6_2_a_frequency_fusion",
            "frequency+v6.2-A+spectral minus frequency+v6.2-A",
        ),
    ]
    for protocol in ["stratified_train_val", "board_grouped", "material_grouped"]:
        for left, right, label in comparisons:
            rows.append(delta_row(best, protocol, left, right, label))
    return rows


def zero_recall_rows(per_class_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in per_class_rows:
        if int(row.get("support", 0)) > 0 and float(row.get("recall", 0.0)) == 0.0:
            rows.append(row)
    return rows


def per_class_delta_rows(
    per_class_rows: list[dict[str, Any]],
    best: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup: dict[tuple[str, str, str, int], list[float]] = {}
    best_methods = {(condition, protocol): row["method"] for (condition, protocol), row in best.items()}
    for row in per_class_rows:
        condition = row["condition"]
        protocol = row["protocol"]
        if row["method"] != best_methods.get((condition, protocol)):
            continue
        key = (protocol, condition, row.get("held_out_group", ""), int(row["class_id"]))
        lookup.setdefault(key, []).append(float(row["f1"]))

    def mean_f1(protocol: str, condition: str, class_id: int) -> float:
        values = []
        for (key_protocol, key_condition, _group, key_class), f1_values in lookup.items():
            if key_protocol == protocol and key_condition == condition and key_class == class_id:
                values.extend(f1_values)
        return float(np.mean(values)) if values else float("nan")

    rows: list[dict[str, Any]] = []
    for protocol in ["stratified_train_val", "board_grouped", "material_grouped"]:
        for class_id in CLASS_IDS:
            freq = mean_f1(protocol, "frequency_only", class_id)
            spectral = mean_f1(protocol, "spectral_descriptors_only", class_id)
            v62a = mean_f1(protocol, "v6_2_a_embedding_only", class_id)
            freq_v62a = mean_f1(protocol, "v6_2_a_frequency_fusion", class_id)
            freq_v62a_spectral = mean_f1(protocol, "frequency_v6_2_a_spectral_fusion", class_id)
            rows.append(
                {
                    "protocol": protocol,
                    "class_id": class_id,
                    "label_name": CLASS_NAMES[class_id],
                    "frequency_only_f1": freq,
                    "spectral_only_f1": spectral,
                    "v6_2_a_embedding_only_f1": v62a,
                    "frequency_v6_2_a_f1": freq_v62a,
                    "frequency_v6_2_a_spectral_f1": freq_v62a_spectral,
                    "spectral_minus_frequency_pp": (spectral - freq) * 100.0,
                    "spectral_added_to_frequency_v6_2_a_pp": (freq_v62a_spectral - freq_v62a) * 100.0,
                    "minority_focus": CLASS_NAMES[class_id] in {"1_2", "2_1"},
                }
            )
    return rows


def score_line(best: dict[tuple[str, str], dict[str, Any]], condition: str) -> str:
    return (
        f"- `{condition}`: stratified {pct(row_score(best[(condition, 'stratified_train_val')]))}, "
        f"Board LOBO {pct(row_score(best[(condition, 'board_grouped')]))}, "
        f"Material LOMO {pct(row_score(best[(condition, 'material_grouped')]))}."
    )


def decision_text(delta_rows: list[dict[str, Any]]) -> str:
    added_rows = [
        row
        for row in delta_rows
        if row["comparison"] == "frequency+v6.2-A+spectral minus frequency+v6.2-A"
    ]
    positive = [row for row in added_rows if float(row["delta_left_minus_right_pp"]) > 0.25]
    if len(positive) >= 2:
        return "include in appendix/supplementary only; deterministic descriptors show limited complementary value but require matched presentation as a control, not a main claim"
    if positive:
        return "include in appendix/supplementary only; gains are protocol-specific and not sufficient for a main LeFFT claim"
    return "keep as future work / negative control; deterministic descriptors do not provide clear matched gains over the stronger controls"


def build_key_numbers(
    best: dict[tuple[str, str], dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    per_class_delta: list[dict[str, Any]],
    zero_rows: list[dict[str, Any]],
    feature_paths: dict[str, str],
    frequency_reference: dict[str, Any],
    fusion_reference: dict[str, Any],
) -> dict[str, Any]:
    conditions = [
        "frequency_only",
        "spectral_descriptors_only",
        "v6_2_a_embedding_only",
        "frequency_spectral_fusion",
        "v6_2_a_spectral_fusion",
        "v6_2_a_frequency_fusion",
        "frequency_v6_2_a_spectral_fusion",
    ]
    protocols = ["stratified_train_val", "board_grouped", "material_grouped"]
    condition_scores = {
        condition: {
            protocol: {
                "method": best[(condition, protocol)]["method"],
                "macro_f1": row_score(best[(condition, protocol)]),
            }
            for protocol in protocols
        }
        for condition in conditions
    }
    targeted_class_rows = [
        row for row in per_class_delta if row["label_name"] in {"1_2", "2_1"} and row["protocol"] == "material_grouped"
    ]
    return {
        "feature_paths": feature_paths,
        "frequency_reference": frequency_reference,
        "frequency_embedding_fusion_reference": fusion_reference,
        "condition_scores": condition_scores,
        "deltas": delta_rows,
        "minority_class_material_lomo_deltas": targeted_class_rows,
        "zero_recall_rows": len(zero_rows),
        "decision": decision_text(delta_rows),
        "claim_boundary": "Deterministic spectral / LeFFT-inspired descriptors only; not a trained LeFFT model; no LeFFT superiority claim; no acoustic-response claim.",
    }


def render_report(
    model_specs: list[dict[str, Any]],
    conditions: list[dict[str, str]],
    summary_rows: list[dict[str, Any]],
    board_rows: list[dict[str, Any]],
    material_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    per_class_delta: list[dict[str, Any]],
    zero_rows: list[dict[str, Any]],
    key_numbers: dict[str, Any],
) -> str:
    compact_rows = compact_summary_rows(summary_rows)
    summary_fields = [
        "protocol",
        "condition",
        "display_name",
        "method",
        "macro_f1",
        "mean_macro_f1",
        "accuracy",
        "macro_recall",
        "worst_group",
        "worst_group_macro_f1",
        "best_group",
        "best_group_macro_f1",
        "n_groups",
    ]
    fold_fields = [
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
    delta_fields = [
        "protocol",
        "comparison",
        "left_macro_f1",
        "right_macro_f1",
        "delta_left_minus_right_pp",
    ]
    per_class_fields = [
        "protocol",
        "label_name",
        "frequency_only_f1",
        "spectral_only_f1",
        "v6_2_a_embedding_only_f1",
        "frequency_v6_2_a_f1",
        "frequency_v6_2_a_spectral_f1",
        "spectral_minus_frequency_pp",
        "spectral_added_to_frequency_v6_2_a_pp",
        "minority_focus",
    ]
    condition_lines = [f"- `{row['condition']}`: {row['display_name']}." for row in conditions]
    model_lines = [
        f"- `{spec['method']}`: {spec['description']} Parameters: `{json.dumps(spec['parameters'])}`"
        for spec in model_specs
    ]
    best = {
        (condition, protocol): values
        for condition, protocols in key_numbers["condition_scores"].items()
        for protocol, values in protocols.items()
    }
    comparison_lines = [
        score_line_from_scores(key_numbers["condition_scores"], "frequency_only"),
        score_line_from_scores(key_numbers["condition_scores"], "spectral_descriptors_only"),
        score_line_from_scores(key_numbers["condition_scores"], "v6_2_a_embedding_only"),
        score_line_from_scores(key_numbers["condition_scores"], "frequency_spectral_fusion"),
        score_line_from_scores(key_numbers["condition_scores"], "v6_2_a_spectral_fusion"),
        score_line_from_scores(key_numbers["condition_scores"], "v6_2_a_frequency_fusion"),
        score_line_from_scores(key_numbers["condition_scores"], "frequency_v6_2_a_spectral_fusion"),
    ]
    targeted = [row for row in per_class_delta if row["label_name"] in {"1_2", "2_1"}]
    return "\n".join(
        [
            "# Spectral Descriptor Ablation Evaluation",
            "",
            "## Scope",
            "",
            "This report evaluates deterministic spectral / LeFFT-inspired descriptors under the same information-budget protocols used for frequency and v6.2-A controls. It uses saved normalized feature dumps only. No raw images, deep-model training, encoder fine-tuning, trained LeFFT model, or acoustic-response prediction is used.",
            "",
            "## Conditions",
            "",
            *condition_lines,
            "",
            "## Models and preprocessing",
            "",
            *model_lines,
            "",
            "All standardization is fitted only on the training/reference fold and then applied to the held-out fold. Grouped held-out samples are never used to fit preprocessing.",
            "",
            "## Summary metrics",
            "",
            markdown_table(compact_rows, summary_fields),
            "",
            "## Information-budget comparisons",
            "",
            *comparison_lines,
            "",
            "## Required deltas",
            "",
            markdown_table(delta_rows, delta_fields),
            "",
            "## Board LOBO-style fold scores",
            "",
            markdown_table(board_rows, fold_fields),
            "",
            "## Material LOMO-style fold scores",
            "",
            markdown_table(material_rows, fold_fields),
            "",
            "## Per-class changes",
            "",
            markdown_table(per_class_delta, per_class_fields),
            "",
            "## Minority-class focus: 1_2 and 2_1",
            "",
            markdown_table(targeted, per_class_fields),
            "",
            "## Zero-recall rows",
            "",
            f"- Zero-recall rows: `{len(zero_rows)}`.",
            "",
            "## Decision",
            "",
            f"- Recommendation: {key_numbers['decision']}.",
            "- Include in the main paper only if the table is framed as a deterministic spectral descriptor control.",
            "- Do not claim LeFFT superiority unless future matched experiments show clear, robust gains.",
            "",
            "## Claim boundary",
            "",
            "- Deterministic descriptors only.",
            "- Not a trained LeFFT model.",
            "- No LeFFT superiority claim from this evaluation.",
            "- No acoustic-response claim.",
            "- Material LOMO remains a descriptive stress test because only two material groups are available.",
        ]
    )


def score_line_from_scores(scores: dict[str, dict[str, dict[str, Any]]], condition: str) -> str:
    return (
        f"- `{condition}`: stratified {pct(scores[condition]['stratified_train_val']['macro_f1'])}, "
        f"Board LOBO {pct(scores[condition]['board_grouped']['macro_f1'])}, "
        f"Material LOMO {pct(scores[condition]['material_grouped']['macro_f1'])}."
    )


def update_manuscript_map(map_path: Path, report_path: Path) -> None:
    if not map_path.exists():
        return
    marker = "Spectral descriptor ablation evaluation"
    text = map_path.read_text(encoding="utf-8")
    if marker in text:
        return
    line = (
        f"| {marker} | `scripts/encoder/evaluate_spectral_descriptor_ablation.py` | normalized spectral/v6.2-A feature dumps | "
        f"`{report_path.as_posix()}` | done-external |"
    )
    section = (
        "\n## Spectral Descriptor Ablation Evaluation\n\n"
        "| Artifact | Source script | Input artifact | Output/report | Status |\n"
        "|---|---|---|---|---|\n"
        f"{line}\n"
    )
    release_gate = "\n## Release Gate\n"
    if release_gate in text:
        text = text.replace(release_gate, section + release_gate, 1)
    else:
        text = text.rstrip() + "\n" + section
    map_path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metadata, embeddings, feature_paths = load_feature_inputs(args.features_dir)
    model_specs = build_model_specs(args.linear_max_iter, args.rf_trees)
    conditions = build_conditions()

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
    zero_rows = zero_recall_rows(per_class_rows)
    per_class_delta = per_class_delta_rows(per_class_rows, best)

    frequency_reference = read_json(args.frequency_dir / "frequency_only_key_numbers.json")
    fusion_reference = read_json(args.fusion_dir / "frequency_embedding_fusion_key_numbers.json")
    key_numbers = build_key_numbers(best, delta_rows, per_class_delta, zero_rows, feature_paths, frequency_reference, fusion_reference)

    write_csv(args.out_dir / "spectral_descriptor_ablation_summary.csv", compact_summary_rows(summary_rows))
    write_csv(args.out_dir / "spectral_descriptor_board_lobo.csv", board_rows)
    write_csv(args.out_dir / "spectral_descriptor_material_lomo.csv", material_rows)
    write_csv(args.out_dir / "spectral_descriptor_per_class.csv", per_class_rows)
    write_csv(args.out_dir / "spectral_descriptor_ablation_deltas.csv", delta_rows)
    write_csv(args.out_dir / "spectral_descriptor_per_class_deltas.csv", per_class_delta)
    write_csv(args.out_dir / "spectral_descriptor_zero_recall_rows.csv", zero_rows)
    (args.out_dir / "spectral_descriptor_ablation_key_numbers.json").write_text(
        json.dumps(key_numbers, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path = args.out_dir / "SPECTRAL_DESCRIPTOR_ABLATION_REPORT.md"
    report_path.write_text(
        render_report(model_specs, conditions, summary_rows, board_rows, material_rows, delta_rows, per_class_delta, zero_rows, key_numbers),
        encoding="utf-8",
    )
    update_manuscript_map(Path("docs") / "MANUSCRIPT_MAP.md", report_path)

    print(f"[done] wrote spectral descriptor ablation report to {args.out_dir}")
    print(
        "[done] spectral_only stratified={:.6f} board={:.6f} material={:.6f}".format(
            key_numbers["condition_scores"]["spectral_descriptors_only"]["stratified_train_val"]["macro_f1"],
            key_numbers["condition_scores"]["spectral_descriptors_only"]["board_grouped"]["macro_f1"],
            key_numbers["condition_scores"]["spectral_descriptors_only"]["material_grouped"]["macro_f1"],
        )
    )
    print(f"[done] decision={key_numbers['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
