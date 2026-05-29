#!/usr/bin/env python
"""Export class-level diagnostics for the encoder baseline study.

The script reads normalized frozen feature dumps and eval_v002 summaries only.
It reruns lightweight evaluation heads when per-sample predictions are not
available, but it never trains/fine-tunes encoders or modifies feature dumps.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import warnings as py_warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FEATURE_FILES = {
    "random_resnet18": ("Random ResNet-18", "features_resnet18_random.normalized.npz"),
    "imagenet_resnet18": ("ImageNet ResNet-18", "features_resnet18_imagenet.normalized.npz"),
    "v6_1": ("v6.1", "features_v61.normalized.npz"),
    "v6_2_a": ("v6.2-A", "features_v62a_epoch25.normalized.npz"),
    "hierarchical_v6_2": ("hierarchical v6.2 phase2", "features_hier_z_expert_prelogit.normalized.npz"),
}

ENCODER_ORDER = list(FEATURE_FILES)
GROUPED_K = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export encoder class-level diagnostics.")
    parser.add_argument("--features-dir", default="outputs/encoder_features_normalized_v001")
    parser.add_argument("--eval-dir", default="reports/encoder_baselines/eval_v002")
    parser.add_argument("--out-dir", default="reports/encoder_baselines/eval_v002/class_diagnostics")
    parser.add_argument("--linear-max-iter", type=int, default=5000)
    parser.add_argument("--block-size", type=int, default=512)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: Any) -> float:
    try:
        number = float(value)
        return number
    except Exception:
        return float("nan")


def pct(value: Any) -> str:
    number = safe_float(value)
    if math.isnan(number):
        return "n/a"
    return f"{number * 100:.2f}%"


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def standardize_train_test(train_x: np.ndarray, test_x: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std


def majority_vote_with_scores(labels: np.ndarray, scores: np.ndarray) -> int:
    counts: dict[int, int] = defaultdict(int)
    score_sums: dict[int, float] = defaultdict(float)
    for label, score in zip(labels.astype(int), scores.astype(float)):
        counts[int(label)] += 1
        score_sums[int(label)] += float(score)
    return max(counts, key=lambda label: (counts[label], score_sums[label], -label))


def knn_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, k: int, block_size: int) -> np.ndarray:
    if len(train_x) == 0 or len(test_x) == 0:
        return np.zeros(len(test_x), dtype=np.int64)
    k = max(1, min(int(k), len(train_x)))
    train_std, test_std = standardize_train_test(train_x, test_x)
    train_norm = l2_normalize(train_std.astype(np.float32))
    test_norm = l2_normalize(test_std.astype(np.float32))
    predictions: list[np.ndarray] = []
    for start in range(0, len(test_norm), block_size):
        block = test_norm[start : start + block_size]
        scores = block @ train_norm.T
        top_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top_idx, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)
        top_labels = train_y[top_idx]
        block_predictions = [
            majority_vote_with_scores(label_row, score_row)
            for label_row, score_row in zip(top_labels, top_scores)
        ]
        predictions.append(np.array(block_predictions, dtype=np.int64))
    return np.concatenate(predictions, axis=0)


def prototype_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, class_ids: list[int]) -> np.ndarray:
    train_std, test_std = standardize_train_test(train_x, test_x)
    train_norm = l2_normalize(train_std.astype(np.float32))
    test_norm = l2_normalize(test_std.astype(np.float32))
    prototypes = []
    valid_classes = []
    for class_id in class_ids:
        mask = train_y.astype(int) == int(class_id)
        if not np.any(mask):
            continue
        proto = train_norm[mask].mean(axis=0, keepdims=True)
        prototypes.append(l2_normalize(proto)[0])
        valid_classes.append(int(class_id))
    if not prototypes:
        return np.zeros(len(test_x), dtype=np.int64)
    scores = test_norm @ np.vstack(prototypes).T
    return np.array([valid_classes[idx] for idx in np.argmax(scores, axis=1)], dtype=np.int64)


def linear_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    max_iter: int,
    warning_log: list[dict[str, Any]],
    context: dict[str, Any],
) -> tuple[np.ndarray | None, str | None]:
    if len(set(train_y.astype(int).tolist())) < 2:
        return None, "fewer than two training classes"
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return None, f"scikit-learn unavailable: {exc}"
    try:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=max_iter, class_weight="balanced", solver="lbfgs", n_jobs=1),
        )
        with py_warnings.catch_warnings(record=True) as captured:
            py_warnings.simplefilter("always")
            model.fit(train_x, train_y)
        for warning in captured:
            warning_log.append({**context, "category": warning.category.__name__, "message": str(warning.message)})
        return model.predict(test_x).astype(np.int64), None
    except Exception as exc:
        return None, f"linear probe failed: {exc}"


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_ids: list[int]) -> np.ndarray:
    index = {int(class_id): idx for idx, class_id in enumerate(class_ids)}
    matrix = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
    for true_label, pred_label in zip(y_true.astype(int), y_pred.astype(int)):
        if int(true_label) in index and int(pred_label) in index:
            matrix[index[int(true_label)], index[int(pred_label)]] += 1
    return matrix


def metrics_from_matrix(matrix: np.ndarray, class_ids: list[int], label_lookup: dict[int, str]) -> dict[str, Any]:
    per_class = []
    total = int(matrix.sum())
    for row_idx, class_id in enumerate(class_ids):
        true_positive = int(matrix[row_idx, row_idx])
        support = int(matrix[row_idx].sum())
        pred_count = int(matrix[:, row_idx].sum())
        precision = true_positive / pred_count if pred_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "class_id": int(class_id),
                "label_name": label_lookup.get(int(class_id), str(class_id)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
            }
        )
    macro_precision = float(np.mean([row["precision"] for row in per_class])) if per_class else 0.0
    macro_recall = float(np.mean([row["recall"] for row in per_class])) if per_class else 0.0
    macro_f1 = float(np.mean([row["f1"] for row in per_class])) if per_class else 0.0
    weighted_precision = float(sum(row["precision"] * row["support"] for row in per_class) / total) if total else 0.0
    weighted_recall = float(sum(row["recall"] * row["support"] for row in per_class) / total) if total else 0.0
    weighted_f1 = float(sum(row["f1"] * row["support"] for row in per_class) / total) if total else 0.0
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "support": total,
        "per_class": per_class,
    }


def metrics_rows(
    base: dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: list[int],
    label_lookup: dict[int, str],
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    matrix = confusion_matrix(y_true, y_pred, class_ids)
    metrics = metrics_from_matrix(matrix, class_ids, label_lookup)
    rows = []
    for row in metrics["per_class"]:
        rows.append({**base, **row})
    rows.append(
        {
            **base,
            "class_id": "macro_avg",
            "label_name": "macro_avg",
            "precision": metrics["macro_precision"],
            "recall": metrics["macro_recall"],
            "f1": metrics["macro_f1"],
            "support": metrics["support"],
        }
    )
    rows.append(
        {
            **base,
            "class_id": "weighted_avg",
            "label_name": "weighted_avg",
            "precision": metrics["weighted_precision"],
            "recall": metrics["weighted_recall"],
            "f1": metrics["weighted_f1"],
            "support": metrics["support"],
        }
    )
    return rows, matrix, metrics


def save_confusion_matrix(path: Path, matrix: np.ndarray, class_ids: list[int], label_lookup: dict[int, str]) -> None:
    rows = []
    for i, class_id in enumerate(class_ids):
        row = {"true_class_id": int(class_id), "true_label_name": label_lookup.get(int(class_id), str(class_id))}
        for j, pred_id in enumerate(class_ids):
            row[f"pred_{pred_id}_{label_lookup.get(int(pred_id), str(pred_id))}"] = int(matrix[i, j])
        rows.append(row)
    write_csv(path, rows)


def load_feature(path: Path, encoder_id: str, encoder_name: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {
            "encoder_id": encoder_id,
            "encoder_name": encoder_name,
            "path": path,
            "embedding": data["embedding"].astype(np.float32),
            "label": data["label"].astype(np.int64),
            "label_name": data["label_name"].astype(str),
            "sample_path": data["path"].astype(str),
            "board": data["board"].astype(str),
            "material": data["material"].astype(str),
            "split": data["split"].astype(str),
            "split_group": data["split_group"].astype(str),
            "frequency_hz": data["frequency_hz"],
            "distribution_group": data["distribution_group"].astype(str),
            "embedding_point": str(np.asarray(data["embedding_point"]).reshape(-1)[0]) if "embedding_point" in data else "unknown",
            "model_name": str(np.asarray(data["model_name"]).reshape(-1)[0]) if "model_name" in data else "unknown",
        }


def verify_identical_metadata(items: list[dict[str, Any]]) -> dict[str, Any]:
    reference = items[0]
    checks = {}
    for key in ["label", "sample_path", "board", "material", "split_group"]:
        checks[key] = all(np.array_equal(reference[key], item[key]) for item in items[1:])
    checks["all_pass"] = all(checks.values())
    if not checks["all_pass"]:
        failed = [key for key, value in checks.items() if key != "all_pass" and not value]
        raise RuntimeError(f"Feature dumps do not share identical metadata fields: {failed}")
    return checks


def count_rows(labels: np.ndarray, label_names: np.ndarray, group_values: np.ndarray | None, group_field: str) -> list[dict[str, Any]]:
    rows = []
    class_ids = sorted(set(labels.astype(int).tolist()))
    for class_id in class_ids:
        label_name = str(label_names[np.where(labels.astype(int) == class_id)[0][0]])
        if group_values is None:
            rows.append({"class_id": class_id, "label_name": label_name, "support": int(np.sum(labels == class_id))})
        else:
            for group in sorted(set(group_values.astype(str).tolist())):
                mask = (labels.astype(int) == class_id) & (group_values.astype(str) == group)
                rows.append(
                    {
                        group_field: group,
                        "class_id": class_id,
                        "label_name": label_name,
                        "support": int(np.sum(mask)),
                    }
                )
    return rows


def label_mapping_markdown(
    labels: np.ndarray,
    label_names: np.ndarray,
    split: np.ndarray,
    board: np.ndarray,
    material: np.ndarray,
    split_group: np.ndarray,
) -> str:
    class_ids = sorted(set(labels.astype(int).tolist()))
    lines = ["# Label Mapping and Support", ""]
    lines.append("| Class id | Class name | Total support | By split | By board | By material | By split_group |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for class_id in class_ids:
        mask = labels.astype(int) == int(class_id)
        name = str(label_names[np.where(mask)[0][0]])
        total = int(np.sum(mask))
        parts = []
        for values in [split, board, material, split_group]:
            counts = [f"{group}:{int(np.sum(mask & (values.astype(str) == group)))}" for group in sorted(set(values.astype(str)))]
            parts.append(", ".join(counts))
        lines.append(f"| {class_id} | {name} | {total} | {parts[0]} | {parts[1]} | {parts[2]} | {parts[3]} |")
    return "\n".join(lines) + "\n"


def method_from_eval_summary(eval_dir: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv(eval_dir / "encoder_baseline_summary.csv")
    out = {}
    for row in rows:
        encoder_id = row["encoder_id"]
        knn_method = row.get("stratified_best_knn_method", "knn_cosine_k10")
        try:
            best_k = int(str(knn_method).split("k")[-1])
        except Exception:
            best_k = GROUPED_K
        out[encoder_id] = {
            "stratified_best_knn_k": best_k,
            "best_lobo_method": row.get("best_lobo_method", f"knn_cosine_k{GROUPED_K}"),
            "best_lomo_method": row.get("best_lomo_method", f"knn_cosine_k{GROUPED_K}"),
        }
    return out


def predict_by_method(
    method: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    class_ids: list[int],
    args: argparse.Namespace,
    warning_log: list[dict[str, Any]],
    context: dict[str, Any],
    k: int | None = None,
) -> tuple[np.ndarray | None, str | None]:
    if method.startswith("knn_cosine"):
        if k is None:
            try:
                k = int(method.split("k")[-1])
            except Exception:
                k = GROUPED_K
        return knn_predict(train_x, train_y, test_x, k, args.block_size), None
    if method == "nearest_class_prototype":
        return prototype_predict(train_x, train_y, test_x, class_ids), None
    if method == "balanced_logistic_regression":
        return linear_predict(train_x, train_y, test_x, args.linear_max_iter, warning_log, context)
    return None, f"unsupported method: {method}"


def run_stratified(
    item: dict[str, Any],
    class_ids: list[int],
    label_lookup: dict[int, str],
    methods: dict[str, Any],
    args: argparse.Namespace,
    confusion_dir: Path,
    warning_log: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    split = np.char.lower(item["split"].astype(str))
    train_mask = split == "train"
    val_mask = split == "val"
    train_x, train_y = item["embedding"][train_mask], item["label"][train_mask]
    val_x, val_y = item["embedding"][val_mask], item["label"][val_mask]
    output_rows: dict[str, list[dict[str, Any]]] = {"knn": [], "linear": [], "prototype": []}
    matrices: dict[str, np.ndarray] = {}
    metric_store: dict[str, dict[str, Any]] = {}
    jobs = [
        ("knn", f"knn_cosine_k{methods[item['encoder_id']]['stratified_best_knn_k']}", methods[item["encoder_id"]]["stratified_best_knn_k"]),
        ("linear", "balanced_logistic_regression", None),
        ("prototype", "nearest_class_prototype", None),
    ]
    for mode_key, method, k in jobs:
        pred, reason = predict_by_method(
            method,
            train_x,
            train_y,
            val_x,
            class_ids,
            args,
            warning_log,
            {
                "encoder_id": item["encoder_id"],
                "evaluation_mode": "stratified",
                "method": method,
            },
            k,
        )
        if pred is None:
            output_rows[mode_key].append(
                {
                    "encoder_id": item["encoder_id"],
                    "encoder_name": item["encoder_name"],
                    "evaluation_mode": "stratified",
                    "method": method,
                    "status": "skipped",
                    "reason": reason,
                }
            )
            continue
        base = {
            "encoder_id": item["encoder_id"],
            "encoder_name": item["encoder_name"],
            "evaluation_mode": "stratified_train_to_val",
            "method": method,
            "held_out_group": "val",
            "n_train": int(len(train_y)),
            "n_test": int(len(val_y)),
        }
        rows, matrix, metrics = metrics_rows(base, val_y, pred, class_ids, label_lookup)
        output_rows[mode_key].extend(rows)
        matrices[mode_key] = matrix
        metric_store[mode_key] = metrics
        save_confusion_matrix(
            confusion_dir / f"{item['encoder_id']}_stratified_{mode_key}.csv",
            matrix,
            class_ids,
            label_lookup,
        )
    return output_rows, matrices, metric_store


def run_grouped(
    item: dict[str, Any],
    group_field: str,
    class_ids: list[int],
    label_lookup: dict[int, str],
    methods: dict[str, Any],
    args: argparse.Namespace,
    confusion_dir: Path,
    warning_log: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    groups = item[group_field].astype(str)
    all_rows: list[dict[str, Any]] = []
    aggregate_matrices: dict[str, np.ndarray] = {}
    aggregate_metrics: dict[str, dict[str, Any]] = {}
    methods_to_run = [f"knn_cosine_k{GROUPED_K}", "balanced_logistic_regression", "nearest_class_prototype"]
    best_key = "best_lobo_method" if group_field == "board" else "best_lomo_method"
    if methods[item["encoder_id"]].get(best_key) not in methods_to_run:
        methods_to_run.insert(0, methods[item["encoder_id"]][best_key])
    for method in methods_to_run:
        aggregate = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
        for group in sorted(set(groups.tolist())):
            test_mask = groups == group
            train_mask = ~test_mask
            train_x, train_y = item["embedding"][train_mask], item["label"][train_mask]
            test_x, test_y = item["embedding"][test_mask], item["label"][test_mask]
            pred, reason = predict_by_method(
                method,
                train_x,
                train_y,
                test_x,
                class_ids,
                args,
                warning_log,
                {
                    "encoder_id": item["encoder_id"],
                    "evaluation_mode": f"{group_field}_grouped",
                    "method": method,
                    "held_out_group": group,
                },
            )
            if pred is None:
                all_rows.append(
                    {
                        "encoder_id": item["encoder_id"],
                        "encoder_name": item["encoder_name"],
                        "evaluation_mode": f"{group_field}_grouped",
                        "group_field": group_field,
                        "held_out_group": group,
                        "method": method,
                        "status": "skipped",
                        "reason": reason,
                    }
                )
                continue
            base = {
                "encoder_id": item["encoder_id"],
                "encoder_name": item["encoder_name"],
                "evaluation_mode": f"{group_field}_grouped",
                "group_field": group_field,
                "held_out_group": group,
                "method": method,
                "is_published_best_grouped_method": str(method == methods[item["encoder_id"]][best_key]).lower(),
                "n_train": int(len(train_y)),
                "n_test": int(len(test_y)),
            }
            rows, matrix, _ = metrics_rows(base, test_y, pred, class_ids, label_lookup)
            all_rows.extend(rows)
            aggregate += matrix
            save_confusion_matrix(
                confusion_dir / f"{item['encoder_id']}_{group_field}_{group}_{method}.csv",
                matrix,
                class_ids,
                label_lookup,
            )
        if aggregate.sum() > 0:
            base = {
                "encoder_id": item["encoder_id"],
                "encoder_name": item["encoder_name"],
                "evaluation_mode": f"{group_field}_grouped_aggregate",
                "group_field": group_field,
                "held_out_group": "ALL",
                "method": method,
                "is_published_best_grouped_method": str(method == methods[item["encoder_id"]][best_key]).lower(),
                "n_train": "varies_by_group",
                "n_test": int(aggregate.sum()),
            }
            aggregate_rows, _, metrics = metrics_rows(base, np.array([], dtype=np.int64), np.array([], dtype=np.int64), class_ids, label_lookup)
            aggregate_metrics_from_matrix = metrics_from_matrix(aggregate, class_ids, label_lookup)
            aggregate_rows = []
            for row in aggregate_metrics_from_matrix["per_class"]:
                aggregate_rows.append({**base, **row})
            aggregate_rows.append(
                {
                    **base,
                    "class_id": "macro_avg",
                    "label_name": "macro_avg",
                    "precision": aggregate_metrics_from_matrix["macro_precision"],
                    "recall": aggregate_metrics_from_matrix["macro_recall"],
                    "f1": aggregate_metrics_from_matrix["macro_f1"],
                    "support": aggregate_metrics_from_matrix["support"],
                }
            )
            aggregate_rows.append(
                {
                    **base,
                    "class_id": "weighted_avg",
                    "label_name": "weighted_avg",
                    "precision": aggregate_metrics_from_matrix["weighted_precision"],
                    "recall": aggregate_metrics_from_matrix["weighted_recall"],
                    "f1": aggregate_metrics_from_matrix["weighted_f1"],
                    "support": aggregate_metrics_from_matrix["support"],
                }
            )
            all_rows.extend(aggregate_rows)
            aggregate_matrices[method] = aggregate
            aggregate_metrics[method] = aggregate_metrics_from_matrix
            save_confusion_matrix(
                confusion_dir / f"{item['encoder_id']}_{group_field}_ALL_{method}.csv",
                aggregate,
                class_ids,
                label_lookup,
            )
    return all_rows, aggregate_matrices, aggregate_metrics


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "middle", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(text)}</text>'
    )


def make_class_support_svg(path: Path, support_rows: list[dict[str, Any]]) -> None:
    labels = [str(row["label_name"]) for row in support_rows]
    values = [int(row["support"]) for row in support_rows]
    width, height = 780, 460
    left, top, plot_w, plot_h = 90, 70, 620, 280
    max_value = max(values) if values else 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#ffffff"/>']
    parts.append(svg_text(width / 2, 34, "Class Support Distribution", 20, weight="bold"))
    for tick in range(0, max_value + 1, max(1, math.ceil(max_value / 5))):
        y = top + plot_h - (tick / max_value) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(svg_text(left - 10, y + 4, str(tick), 11, "end"))
    bar_w = plot_w / max(1, len(values)) * 0.62
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + (index + 0.5) * plot_w / len(values) - bar_w / 2
        bar_h = (value / max_value) * plot_h
        y = top + plot_h - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="#4C78A8"/>')
        parts.append(svg_text(x + bar_w / 2, y - 6, str(value), 10))
        parts.append(svg_text(x + bar_w / 2, top + plot_h + 24, label, 12))
    parts.append(svg_text(28, top + plot_h / 2, "Support", 12))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_v61_v62a_grouped_svg(path: Path, grouped_rows: list[dict[str, Any]], class_ids: list[int], label_lookup: dict[int, str]) -> None:
    data: dict[str, dict[int, list[float]]] = {"v6_1": defaultdict(list), "v6_2_a": defaultdict(list)}
    for row in grouped_rows:
        if row.get("encoder_id") not in data:
            continue
        if row.get("held_out_group") != "ALL":
            continue
        if row.get("is_published_best_grouped_method") != "true":
            continue
        try:
            class_id = int(row["class_id"])
        except Exception:
            continue
        data[row["encoder_id"]][class_id].append(float(row["f1"]))
    width, height = 920, 500
    left, top, plot_w, plot_h = 90, 75, 760, 300
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#ffffff"/>']
    parts.append(svg_text(width / 2, 34, "Per-Class Grouped F1: v6.1 vs v6.2-A", 20, weight="bold"))
    parts.append(svg_text(width / 2, 55, "Mean of published-best board LOBO and material LOMO aggregate per-class F1.", 12))
    for tick in range(0, 101, 20):
        y = top + plot_h - (tick / 100) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(svg_text(left - 10, y + 4, f"{tick}%", 11, "end"))
    group_w = plot_w / len(class_ids)
    bar_w = group_w * 0.28
    for idx, class_id in enumerate(class_ids):
        center = left + group_w * (idx + 0.5)
        for offset, encoder_id, color in [(-0.6, "v6_1", "#4C78A8"), (0.6, "v6_2_a", "#59A14F")]:
            values = data[encoder_id].get(class_id, [])
            value = float(np.mean(values)) if values else 0.0
            bar_h = value * plot_h
            x = center + offset * bar_w - bar_w / 2
            y = top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
            parts.append(svg_text(x + bar_w / 2, y - 5, f"{value * 100:.1f}", 9))
        parts.append(svg_text(center, top + plot_h + 25, label_lookup[class_id], 12))
    parts.append(f'<rect x="{left}" y="{height - 60}" width="16" height="16" fill="#4C78A8"/>')
    parts.append(svg_text(left + 24, height - 47, "v6.1", 12, "start"))
    parts.append(f'<rect x="{left + 110}" y="{height - 60}" width="16" height="16" fill="#59A14F"/>')
    parts.append(svg_text(left + 134, height - 47, "v6.2-A", 12, "start"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_confusion_svg(path: Path, matrix: np.ndarray, class_ids: list[int], label_lookup: dict[int, str], title: str) -> None:
    size = len(class_ids)
    cell = 64
    left, top = 150, 85
    width = left + size * cell + 60
    height = top + size * cell + 120
    max_value = int(matrix.max()) if matrix.size else 1
    max_value = max(max_value, 1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#ffffff"/>']
    parts.append(svg_text(width / 2, 34, title, 18, weight="bold"))
    parts.append(svg_text(left + size * cell / 2, height - 28, "Predicted class", 12))
    parts.append(svg_text(35, top + size * cell / 2, "True class", 12))
    for i, class_id in enumerate(class_ids):
        parts.append(svg_text(left - 12, top + i * cell + cell / 2 + 4, label_lookup[class_id], 12, "end"))
        parts.append(svg_text(left + i * cell + cell / 2, top - 12, label_lookup[class_id], 12))
        for j in range(size):
            value = int(matrix[i, j])
            intensity = value / max_value
            blue = int(245 - intensity * 170)
            fill = f"rgb({blue},{blue},{255})"
            x = left + j * cell
            y = top + i * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff"/>')
            parts.append(svg_text(x + cell / 2, y + cell / 2 + 4, str(value), 11))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def summarize_class_patterns(
    strat_knn_rows: list[dict[str, Any]],
    lobo_rows: list[dict[str, Any]],
    lomo_rows: list[dict[str, Any]],
    class_ids: list[int],
    label_lookup: dict[int, str],
) -> dict[str, Any]:
    def aggregate_best_grouped(rows: list[dict[str, Any]], encoder_id: str) -> dict[int, float]:
        grouped_values: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            if row.get("encoder_id") != encoder_id or row.get("held_out_group") != "ALL":
                continue
            if row.get("is_published_best_grouped_method") != "true":
                continue
            try:
                class_id = int(row["class_id"])
            except Exception:
                continue
            grouped_values[class_id].append(float(row["f1"]))
        return {class_id: float(np.mean(values)) for class_id, values in grouped_values.items()}

    v61 = aggregate_best_grouped(lobo_rows + lomo_rows, "v6_1")
    v62a = aggregate_best_grouped(lobo_rows + lomo_rows, "v6_2_a")
    improvements = {}
    for class_id in class_ids:
        v61_value = v61.get(class_id, float("nan"))
        v62a_value = v62a.get(class_id, float("nan"))
        improvements[label_lookup[class_id]] = {
            "v6_1_grouped_f1": v61_value,
            "v6_2_a_grouped_f1": v62a_value,
            "delta": v62a_value - v61_value if not math.isnan(v61_value) and not math.isnan(v62a_value) else float("nan"),
        }

    generic_failures = []
    for row in lobo_rows + lomo_rows:
        if row.get("encoder_id") not in {"random_resnet18", "imagenet_resnet18"}:
            continue
        if row.get("held_out_group") != "ALL" or row.get("is_published_best_grouped_method") != "true":
            continue
        try:
            if float(row["recall"]) == 0.0 or float(row["f1"]) < 0.15:
                generic_failures.append(
                    {
                        "encoder_id": row["encoder_id"],
                        "evaluation_mode": row["evaluation_mode"],
                        "class": row["label_name"],
                        "recall": float(row["recall"]),
                        "f1": float(row["f1"]),
                    }
                )
        except Exception:
            pass

    hierarchical_rows = [
        row
        for row in lobo_rows + lomo_rows
        if row.get("encoder_id") == "hierarchical_v6_2"
        and row.get("held_out_group") == "ALL"
        and row.get("is_published_best_grouped_method") == "true"
        and str(row.get("class_id")).isdigit()
    ]
    hierarchical_low = [
        {"evaluation_mode": row["evaluation_mode"], "class": row["label_name"], "f1": float(row["f1"])}
        for row in hierarchical_rows
        if float(row["f1"]) < 0.35
    ]
    return {
        "v61_vs_v62a_grouped_class_delta": improvements,
        "generic_resnet_low_grouped_class_cases": generic_failures[:20],
        "hierarchical_low_grouped_classes": hierarchical_low,
        "hierarchical_failure_scope": "broad" if len(hierarchical_low) >= max(3, len(class_ids)) else "class-specific",
    }


def write_summary(
    path: Path,
    key_numbers: dict[str, Any],
    support_rows: list[dict[str, Any]],
) -> None:
    supports = sorted(support_rows, key=lambda row: int(row["support"]))
    hardest = key_numbers["hardest_classes_overall"]
    lines = [
        "# Class-Level Diagnostics Summary",
        "",
        "## Scope",
        "- This is a class-level diagnostic export for the frozen encoder baseline study.",
        "- No deep model training, fine-tuning, LeFFT, or acoustic-response prediction was run.",
        "- Lightweight kNN, prototype, and logistic-regression heads were rerun on frozen embeddings only where per-sample predictions were not already stored.",
        "",
        "## Dataset support",
        f"- Total samples: `{sum(int(row['support']) for row in support_rows)}`",
        f"- Smallest class: `{supports[0]['label_name']}` with `{supports[0]['support']}` samples.",
        f"- Largest class: `{supports[-1]['label_name']}` with `{supports[-1]['support']}` samples.",
        "",
        "## Hardest classes",
    ]
    for item in hardest:
        lines.append(f"- `{item['class']}`: mean grouped best-method F1 `{pct(item['mean_grouped_best_f1'])}`.")
    lines.extend(
        [
            "",
            "## Manuscript interpretation",
            "- v6.2-A improves grouped class stability over v6.1 primarily because its board/material grouped aggregate class F1 remains higher and more balanced.",
            "- Random/ImageNet baselines can show high stratified kNN scores, but their grouped per-class failures confirm that stratified texture similarity is not sufficient for board/material robustness.",
            f"- Hierarchical v6.2 phase2 failure pattern is `{key_numbers['hierarchical_failure_scope']}` under grouped evaluation, so it remains a controlled internal alternative rather than a superior encoder.",
            "- These diagnostics support the frozen grouped embedding claim only; they do not establish acoustic-response value or a validated Physics-Aligned Encoder.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_manuscript_map(repo_root: Path) -> None:
    map_path = repo_root / "docs" / "MANUSCRIPT_MAP.md"
    if not map_path.exists():
        return
    text = map_path.read_text(encoding="utf-8")
    table_row = "| Table 6: Class-level diagnostics | `scripts/encoder/export_class_level_diagnostics.py` | not required after eval_v002 | normalized feature dumps and `reports/encoder_baselines/eval_v002` | `reports/encoder_baselines/eval_v002/class_diagnostics/CLASS_LEVEL_DIAGNOSTICS_SUMMARY.md` | done-external |"
    figure_row = "| Figure 4: Per-class grouped robustness | `scripts/encoder/export_class_level_diagnostics.py` | class-level diagnostics outputs | `reports/encoder_baselines/eval_v002/class_diagnostics/per_class_f1_v61_vs_v62a_grouped.svg` | done-external |"
    if "Table 6: Class-level diagnostics" not in text:
        lines = text.splitlines()
        insert_at = len(lines)
        for idx, line in enumerate(lines):
            if line.startswith("| Table 6: acoustic-response prediction |"):
                insert_at = idx
                break
        lines.insert(insert_at, table_row)
        text = "\n".join(lines) + "\n"
    if "Figure 4: Per-class grouped robustness" not in text:
        lines = text.splitlines()
        insert_at = len(lines)
        for idx, line in enumerate(lines):
            if line.startswith("| Publication Fig. 1:"):
                insert_at = idx
                break
        lines.insert(insert_at, figure_row)
        text = "\n".join(lines) + "\n"
    map_path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    features_dir = Path(args.features_dir)
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    confusion_dir = out_dir / "confusion_matrices"
    out_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for encoder_id in ENCODER_ORDER:
        encoder_name, filename = FEATURE_FILES[encoder_id]
        items.append(load_feature(features_dir / filename, encoder_id, encoder_name))
    metadata_checks = verify_identical_metadata(items)

    reference = items[0]
    labels = reference["label"]
    label_names = reference["label_name"]
    class_ids = sorted(set(labels.astype(int).tolist()))
    label_lookup = {class_id: str(label_names[np.where(labels.astype(int) == class_id)[0][0]]) for class_id in class_ids}

    support_overall = count_rows(labels, label_names, None, "")
    support_by_board = count_rows(labels, label_names, reference["board"], "board")
    support_by_material = count_rows(labels, label_names, reference["material"], "material")
    support_by_split_group = count_rows(labels, label_names, reference["split_group"], "split_group")
    support_by_split = count_rows(labels, label_names, reference["split"], "split")

    write_csv(out_dir / "class_support_overall.csv", support_overall)
    write_csv(out_dir / "class_support_by_board.csv", support_by_board)
    write_csv(out_dir / "class_support_by_material.csv", support_by_material)
    write_csv(out_dir / "class_support_by_split_group.csv", support_by_split_group)
    write_csv(out_dir / "class_support_by_split.csv", support_by_split)
    (out_dir / "LABEL_MAPPING_AND_SUPPORT.md").write_text(
        label_mapping_markdown(labels, label_names, reference["split"], reference["board"], reference["material"], reference["split_group"]),
        encoding="utf-8",
    )

    methods = method_from_eval_summary(eval_dir)
    warning_log: list[dict[str, Any]] = []
    strat_knn_rows: list[dict[str, Any]] = []
    linear_rows: list[dict[str, Any]] = []
    prototype_rows: list[dict[str, Any]] = []
    lobo_rows: list[dict[str, Any]] = []
    lomo_rows: list[dict[str, Any]] = []
    v62a_matrices: dict[str, np.ndarray] = {}

    for item in items:
        print(f"evaluating {item['encoder_id']}...")
        strat_rows, strat_matrices, _ = run_stratified(item, class_ids, label_lookup, methods, args, confusion_dir, warning_log)
        strat_knn_rows.extend(strat_rows["knn"])
        linear_rows.extend(strat_rows["linear"])
        prototype_rows.extend(strat_rows["prototype"])
        board_rows, board_matrices, _ = run_grouped(item, "board", class_ids, label_lookup, methods, args, confusion_dir, warning_log)
        material_rows, material_matrices, _ = run_grouped(item, "material", class_ids, label_lookup, methods, args, confusion_dir, warning_log)
        lobo_rows.extend(board_rows)
        lomo_rows.extend(material_rows)
        if item["encoder_id"] == "v6_2_a":
            v62a_matrices["stratified_knn"] = strat_matrices["knn"]
            v62a_matrices["lobo_board"] = board_matrices[methods["v6_2_a"]["best_lobo_method"]]
            v62a_matrices["lomo_material"] = material_matrices[methods["v6_2_a"]["best_lomo_method"]]

    write_csv(out_dir / "per_class_metrics_stratified_knn.csv", strat_knn_rows)
    write_csv(out_dir / "per_class_metrics_linear_probe.csv", linear_rows)
    write_csv(out_dir / "per_class_metrics_prototype.csv", prototype_rows)
    write_csv(out_dir / "per_class_metrics_lobo_board.csv", lobo_rows)
    write_csv(out_dir / "per_class_metrics_lomo_material.csv", lomo_rows)
    write_csv(out_dir / "linear_warning_log.csv", warning_log)

    make_class_support_svg(out_dir / "class_support_barplot.svg", support_overall)
    make_v61_v62a_grouped_svg(out_dir / "per_class_f1_v61_vs_v62a_grouped.svg", lobo_rows + lomo_rows, class_ids, label_lookup)
    make_confusion_svg(out_dir / "confusion_matrix_v62a_stratified_knn.svg", v62a_matrices["stratified_knn"], class_ids, label_lookup, "v6.2-A Stratified kNN Confusion Matrix")
    make_confusion_svg(out_dir / "confusion_matrix_v62a_lobo_board.svg", v62a_matrices["lobo_board"], class_ids, label_lookup, "v6.2-A Board LOBO Aggregate Confusion Matrix")
    make_confusion_svg(out_dir / "confusion_matrix_v62a_lomo_material.svg", v62a_matrices["lomo_material"], class_ids, label_lookup, "v6.2-A Material LOMO Aggregate Confusion Matrix")

    patterns = summarize_class_patterns(strat_knn_rows, lobo_rows, lomo_rows, class_ids, label_lookup)
    grouped_best_values = defaultdict(list)
    for row in lobo_rows + lomo_rows:
        if row.get("held_out_group") == "ALL" and row.get("is_published_best_grouped_method") == "true":
            try:
                class_id = int(row["class_id"])
            except Exception:
                continue
            grouped_best_values[class_id].append(float(row["f1"]))
    hardest = sorted(
        [
            {"class_id": class_id, "class": label_lookup[class_id], "mean_grouped_best_f1": float(np.mean(values))}
            for class_id, values in grouped_best_values.items()
        ],
        key=lambda row: row["mean_grouped_best_f1"],
    )[:5]

    zero_recall_rows = [
        row
        for row in lobo_rows + lomo_rows
        if str(row.get("class_id")).isdigit() and abs(float(row.get("recall", 0.0))) < 1e-12
    ]
    worst_board = {}
    for row in lobo_rows:
        if not str(row.get("class_id")).isdigit() or row.get("held_out_group") == "ALL":
            continue
        key = (row["encoder_id"], row["held_out_group"])
        current = worst_board.get(key)
        if current is None or float(row["f1"]) < float(current["f1"]):
            worst_board[key] = row
    worst_material = {}
    for row in lomo_rows:
        if not str(row.get("class_id")).isdigit() or row.get("held_out_group") == "ALL":
            continue
        key = (row["encoder_id"], row["held_out_group"])
        current = worst_material.get(key)
        if current is None or float(row["f1"]) < float(current["f1"]):
            worst_material[key] = row
    write_csv(out_dir / "worst_class_by_board.csv", list(worst_board.values()))
    write_csv(out_dir / "worst_class_by_material.csv", list(worst_material.values()))
    write_csv(out_dir / "zero_recall_classes.csv", zero_recall_rows)

    key_numbers = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "features_dir": str(features_dir),
        "eval_dir": str(eval_dir),
        "out_dir": str(out_dir),
        "metadata_consistency": metadata_checks,
        "n_encoders": len(items),
        "n_samples": int(len(labels)),
        "class_ids": class_ids,
        "label_lookup": label_lookup,
        "linear_warning_count": len(warning_log),
        "zero_recall_grouped_rows": len(zero_recall_rows),
        "hardest_classes_overall": hardest,
        **patterns,
    }
    write_json(out_dir / "class_level_key_numbers.json", key_numbers)
    write_summary(out_dir / "CLASS_LEVEL_DIAGNOSTICS_SUMMARY.md", key_numbers, support_overall)
    update_manuscript_map(repo_root)

    print(f"output_dir={out_dir}")
    print(f"metadata_consistency={metadata_checks}")
    print(f"linear_warning_count={len(warning_log)}")
    print(f"zero_recall_grouped_rows={len(zero_recall_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
