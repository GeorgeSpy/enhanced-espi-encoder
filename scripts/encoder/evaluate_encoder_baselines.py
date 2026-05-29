#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified evaluation for normalized encoder feature dumps.

This script evaluates saved embeddings only. It does not run model inference,
does not train/fine-tune CNNs, does not implement LeFFT, and does not run
acoustic-response prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings as py_warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_FEATURES_DIR = Path("outputs/encoder_features_normalized_v001")
DEFAULT_OUT_DIR = Path("reports/encoder_baselines/eval_v002")

FEATURE_FILES = [
    ("random_resnet18", "Random ResNet-18", "features_resnet18_random.normalized.npz"),
    ("imagenet_resnet18", "ImageNet ResNet-18", "features_resnet18_imagenet.normalized.npz"),
    ("v6_1", "v6.1", "features_v61.normalized.npz"),
    ("v6_2_a", "v6.2-A", "features_v62a_epoch25.normalized.npz"),
    ("hierarchical_v6_2", "hierarchical v6.2 phase2", "features_hier_z_expert_prelogit.normalized.npz"),
]

KNN_VALUES = (1, 3, 5, 10, 20)
GROUPED_K = 10


def safe_float(value: Any) -> float | str:
    try:
        number = float(value)
    except Exception:
        return "nan"
    if math.isnan(number) or math.isinf(number):
        return "nan"
    return number


def percent(value: Any) -> str:
    if not isinstance(value, (float, int)):
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def scalar_to_str(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    if array.size == 1:
        return str(array.reshape(-1)[0])
    return str(array)


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


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def standardize_train_test(train_x: np.ndarray, test_x: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_ids: list[int]) -> np.ndarray:
    index = {class_id: idx for idx, class_id in enumerate(class_ids)}
    matrix = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
    for true_label, pred_label in zip(y_true.astype(int), y_pred.astype(int)):
        if int(true_label) in index and int(pred_label) in index:
            matrix[index[int(true_label)], index[int(pred_label)]] += 1
    return matrix


def metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: list[int],
    label_lookup: dict[int, str],
) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, class_ids)
    per_class = []
    for row_idx, class_id in enumerate(class_ids):
        true_positive = int(matrix[row_idx, row_idx])
        support = int(matrix[row_idx].sum())
        pred_count = int(matrix[:, row_idx].sum())
        precision = true_positive / pred_count if pred_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "class_id": int(class_id),
                "label_name": label_lookup.get(int(class_id), str(class_id)),
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    return {
        "accuracy": accuracy,
        "macro_recall": float(np.mean([item["recall"] for item in per_class])) if per_class else 0.0,
        "macro_f1": float(np.mean([item["f1"] for item in per_class])) if per_class else 0.0,
        "per_class": per_class,
    }


def bootstrap_macro_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: list[int],
    label_lookup: dict[int, str],
    rng: np.random.Generator,
    iters: int,
) -> tuple[float, float]:
    if iters <= 0 or len(y_true) == 0:
        return float("nan"), float("nan")
    scores = []
    n_samples = len(y_true)
    for _ in range(iters):
        sample_idx = rng.integers(0, n_samples, size=n_samples)
        metrics = metrics_from_predictions(y_true[sample_idx], y_pred[sample_idx], class_ids, label_lookup)
        scores.append(float(metrics["macro_f1"]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def bootstrap_mean_ci(values: list[float], rng: np.random.Generator | None, iters: int) -> tuple[float, float]:
    if rng is None or iters <= 0 or not values:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    means = []
    for _ in range(iters):
        sample = rng.choice(array, size=len(array), replace=True)
        means.append(float(np.mean(sample)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def majority_vote_with_scores(labels: np.ndarray, scores: np.ndarray) -> int:
    counts: dict[int, int] = defaultdict(int)
    score_sums: dict[int, float] = defaultdict(float)
    for label, score in zip(labels.astype(int), scores.astype(float)):
        counts[int(label)] += 1
        score_sums[int(label)] += float(score)
    return max(counts, key=lambda label: (counts[label], score_sums[label], -label))


def knn_predict_multi(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    k_values: Iterable[int],
    block_size: int,
) -> dict[int, np.ndarray]:
    if len(train_x) == 0 or len(test_x) == 0:
        return {}
    effective_k_values = sorted({int(k) for k in k_values if 1 <= int(k) <= len(train_x)})
    if not effective_k_values:
        return {}

    max_k = max(effective_k_values)
    train_std, test_std = standardize_train_test(train_x, test_x)
    train_norm = l2_normalize(train_std.astype(np.float32))
    test_norm = l2_normalize(test_std.astype(np.float32))
    predictions: dict[int, list[np.ndarray]] = {k: [] for k in effective_k_values}

    for start in range(0, len(test_norm), block_size):
        block = test_norm[start : start + block_size]
        scores = block @ train_norm.T
        top_idx = np.argpartition(-scores, kth=max_k - 1, axis=1)[:, :max_k]
        top_scores = np.take_along_axis(scores, top_idx, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)
        top_labels = train_y[top_idx]
        for k in effective_k_values:
            block_preds = [
                majority_vote_with_scores(label_row[:k], score_row[:k])
                for label_row, score_row in zip(top_labels, top_scores)
            ]
            predictions[k].append(np.array(block_preds, dtype=np.int64))
    return {k: np.concatenate(parts, axis=0) for k, parts in predictions.items()}


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
    local_pred = np.argmax(scores, axis=1)
    return np.array([valid_classes[idx] for idx in local_pred], dtype=np.int64)


def linear_probe_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    max_iter: int,
    warning_log: list[dict[str, Any]] | None = None,
    context: dict[str, Any] | None = None,
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
            LogisticRegression(
                max_iter=max_iter,
                class_weight="balanced",
                solver="lbfgs",
                n_jobs=1,
            ),
        )
        with py_warnings.catch_warnings(record=True) as captured:
            py_warnings.simplefilter("always")
            model.fit(train_x, train_y)
        if warning_log is not None:
            for warning in captured:
                warning_log.append(
                    {
                        **(context or {}),
                        "category": warning.category.__name__,
                        "message": str(warning.message),
                    }
                )
        return model.predict(test_x).astype(np.int64), None
    except Exception as exc:
        return None, f"linear probe failed: {exc}"


def metric_row(
    base: dict[str, Any],
    metrics: dict[str, Any],
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    class_ids: list[int] | None = None,
    label_lookup: dict[int, str] | None = None,
    rng: np.random.Generator | None = None,
    bootstrap_iters: int = 0,
) -> dict[str, Any]:
    row = dict(base)
    row.update(
        {
            "accuracy": safe_float(metrics["accuracy"]),
            "macro_recall": safe_float(metrics["macro_recall"]),
            "macro_f1": safe_float(metrics["macro_f1"]),
        }
    )
    if bootstrap_iters and y_true is not None and y_pred is not None and class_ids is not None and label_lookup is not None and rng is not None:
        low, high = bootstrap_macro_f1_ci(y_true, y_pred, class_ids, label_lookup, rng, bootstrap_iters)
        row["macro_f1_ci95_low"] = safe_float(low)
        row["macro_f1_ci95_high"] = safe_float(high)
    else:
        row["macro_f1_ci95_low"] = "nan"
        row["macro_f1_ci95_high"] = "nan"
    for item in metrics["per_class"]:
        class_id = int(item["class_id"])
        row[f"class_{class_id}_support"] = int(item["support"])
        row[f"class_{class_id}_recall"] = safe_float(item["recall"])
        row[f"class_{class_id}_f1"] = safe_float(item["f1"])
    return row


def load_feature_dump(path: Path, encoder_id: str, encoder_name: str) -> dict[str, Any]:
    with np.load(str(path), allow_pickle=True) as data:
        x = data["embedding"].astype(np.float32)
        labels = data["label"].astype(np.int64)
        label_names = data["label_name"].astype(str)
        return {
            "encoder_id": encoder_id,
            "encoder_name": encoder_name,
            "path": str(path),
            "embedding": x,
            "label": labels,
            "label_name": label_names,
            "file_path": data["path"].astype(str),
            "board": data["board"].astype(str),
            "material": data["material"].astype(str),
            "split": data["split"].astype(str),
            "split_group": data["split_group"].astype(str),
            "frequency_hz": data["frequency_hz"].astype(np.float32),
            "distribution_group": data["distribution_group"].astype(str),
            "model_name": scalar_to_str(data["model_name"]),
            "embedding_point": scalar_to_str(data["embedding_point"]),
            "checkpoint_sha256": scalar_to_str(data["checkpoint_sha256"]),
            "schema_version": scalar_to_str(data["schema_version"]),
        }


def quality_rows(item: dict[str, Any], class_ids: list[int]) -> list[dict[str, Any]]:
    x = item["embedding"]
    paths = item["file_path"]
    base = {
        "encoder_id": item["encoder_id"],
        "encoder_name": item["encoder_name"],
        "model_name": item["model_name"],
        "embedding_point": item["embedding_point"],
    }
    rows = [
        {**base, "section": "quality", "name": "sample_count", "value": int(len(x)), "status": "info"},
        {**base, "section": "quality", "name": "embedding_dim", "value": int(x.shape[1]), "status": "info"},
        {**base, "section": "quality", "name": "nan_count", "value": int(np.isnan(x).sum()), "status": "pass" if not np.isnan(x).any() else "fail"},
        {**base, "section": "quality", "name": "inf_count", "value": int(np.isinf(x).sum()), "status": "pass" if not np.isinf(x).any() else "fail"},
        {**base, "section": "quality", "name": "duplicate_paths", "value": int(len(paths) - len(set(paths.tolist()))), "status": "pass"},
    ]

    def add_counts(section: str, values: np.ndarray) -> None:
        for name, count in sorted(Counter(values.astype(str).tolist()).items()):
            rows.append({**base, "section": section, "name": name, "value": int(count), "status": "info"})

    add_counts("class_count", item["label"].astype(str))
    add_counts("board_count", item["board"])
    add_counts("material_count", item["material"])
    add_counts("split_group_count", item["split_group"])
    return rows


def evaluate_stratified(
    item: dict[str, Any],
    class_ids: list[int],
    label_lookup: dict[int, str],
    args: argparse.Namespace,
    rng: np.random.Generator,
    warning_log: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    x = item["embedding"]
    y = item["label"]
    split = np.char.lower(item["split"].astype(str))
    train_mask = split == "train"
    val_mask = split == "val"
    train_x, train_y = x[train_mask], y[train_mask]
    val_x, val_y = x[val_mask], y[val_mask]
    base = {
        "encoder_id": item["encoder_id"],
        "encoder_name": item["encoder_name"],
        "split": "train_to_val",
        "n_train": int(len(train_x)),
        "n_test": int(len(val_x)),
    }
    bootstrap_iters = args.bootstrap_iters if args.enable_bootstrap else 0

    knn_rows = []
    permutation_rows = []
    predictions = knn_predict_multi(train_x, train_y, val_x, KNN_VALUES, args.block_size)
    for k, y_pred in sorted(predictions.items()):
        metrics = metrics_from_predictions(val_y, y_pred, class_ids, label_lookup)
        knn_rows.append(
            metric_row(
                {**base, "method": f"knn_cosine_k{k}", "k": k},
                metrics,
                val_y,
                y_pred,
                class_ids,
                label_lookup,
                rng,
                bootstrap_iters,
            )
        )
    permuted_train_y = rng.permutation(train_y)
    perm_pred = knn_predict_multi(train_x, permuted_train_y, val_x, [GROUPED_K], args.block_size).get(GROUPED_K)
    if perm_pred is not None:
        perm_metrics = metrics_from_predictions(val_y, perm_pred, class_ids, label_lookup)
        permutation_rows.append(metric_row({**base, "method": f"label_permutation_knn_cosine_k{GROUPED_K}"}, perm_metrics))

    proto_pred = prototype_predict(train_x, train_y, val_x, class_ids)
    proto_metrics = metrics_from_predictions(val_y, proto_pred, class_ids, label_lookup)
    proto_rows = [
        metric_row(
            {**base, "method": "nearest_class_prototype"},
            proto_metrics,
            val_y,
            proto_pred,
            class_ids,
            label_lookup,
            rng,
            bootstrap_iters,
        )
    ]

    linear_rows = []
    linear_pred, reason = linear_probe_predict(
        train_x,
        train_y,
        val_x,
        args.linear_max_iter,
        warning_log,
        {
            "encoder_id": item["encoder_id"],
            "encoder_name": item["encoder_name"],
            "evaluation": "stratified",
            "method": "balanced_logistic_regression",
            "fold": "train_to_val",
        },
    )
    if linear_pred is not None:
        linear_metrics = metrics_from_predictions(val_y, linear_pred, class_ids, label_lookup)
        linear_rows.append(
            metric_row(
                {**base, "method": "balanced_logistic_regression"},
                linear_metrics,
                val_y,
                linear_pred,
                class_ids,
                label_lookup,
                rng,
                bootstrap_iters,
            )
        )
        perm_linear_pred, perm_reason = linear_probe_predict(
            train_x,
            permuted_train_y,
            val_x,
            args.linear_max_iter,
            warning_log,
            {
                "encoder_id": item["encoder_id"],
                "encoder_name": item["encoder_name"],
                "evaluation": "label_permutation",
                "method": "balanced_logistic_regression",
                "fold": "train_to_val",
            },
        )
        if perm_linear_pred is not None:
            perm_linear_metrics = metrics_from_predictions(val_y, perm_linear_pred, class_ids, label_lookup)
            permutation_rows.append(metric_row({**base, "method": "label_permutation_balanced_logistic_regression"}, perm_linear_metrics))
        elif perm_reason:
            permutation_rows.append({**base, "method": "label_permutation_balanced_logistic_regression", "status": "skipped", "reason": perm_reason})
    elif reason:
        linear_rows.append({**base, "method": "balanced_logistic_regression", "status": "skipped", "reason": reason})
    return knn_rows, proto_rows, linear_rows, permutation_rows


def evaluate_grouped(
    item: dict[str, Any],
    group_field: str,
    class_ids: list[int],
    label_lookup: dict[int, str],
    args: argparse.Namespace,
    warning_log: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    x = item["embedding"]
    y = item["label"]
    groups = item[group_field].astype(str)
    rows = []
    for group in sorted(set(groups.tolist())):
        test_mask = groups == group
        train_mask = ~test_mask
        train_x, train_y = x[train_mask], y[train_mask]
        test_x, test_y = x[test_mask], y[test_mask]
        base = {
            "encoder_id": item["encoder_id"],
            "encoder_name": item["encoder_name"],
            "group_field": group_field,
            "held_out_group": group,
            "n_train": int(len(train_x)),
            "n_test": int(len(test_x)),
        }
        knn_pred = knn_predict_multi(train_x, train_y, test_x, [GROUPED_K], args.block_size).get(GROUPED_K)
        if knn_pred is not None:
            rows.append(metric_row({**base, "method": f"knn_cosine_k{GROUPED_K}", "k": GROUPED_K}, metrics_from_predictions(test_y, knn_pred, class_ids, label_lookup)))
        proto_pred = prototype_predict(train_x, train_y, test_x, class_ids)
        rows.append(metric_row({**base, "method": "nearest_class_prototype"}, metrics_from_predictions(test_y, proto_pred, class_ids, label_lookup)))
        linear_pred, reason = linear_probe_predict(
            train_x,
            train_y,
            test_x,
            args.linear_max_iter,
            warning_log,
            {
                "encoder_id": item["encoder_id"],
                "encoder_name": item["encoder_name"],
                "evaluation": f"{group_field}_grouped",
                "method": "balanced_logistic_regression",
                "held_out_group": group,
            },
        )
        if linear_pred is not None:
            rows.append(metric_row({**base, "method": "balanced_logistic_regression"}, metrics_from_predictions(test_y, linear_pred, class_ids, label_lookup)))
        elif reason:
            rows.append({**base, "method": "balanced_logistic_regression", "status": "skipped", "reason": reason})
    return rows


def aggregate_grouped(
    rows: list[dict[str, Any]],
    rng: np.random.Generator | None = None,
    bootstrap_iters: int = 0,
) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if isinstance(row.get("macro_f1"), (int, float)):
            grouped[(str(row["encoder_id"]), str(row["method"]))].append(row)
    output = {}
    for (encoder_id, method), method_rows in grouped.items():
        f1 = np.array([float(row["macro_f1"]) for row in method_rows], dtype=np.float64)
        worst = min(method_rows, key=lambda row: float(row["macro_f1"]))
        best = max(method_rows, key=lambda row: float(row["macro_f1"]))
        ci_low, ci_high = bootstrap_mean_ci(f1.tolist(), rng, bootstrap_iters)
        output[f"{encoder_id}:{method}"] = {
            "encoder_id": encoder_id,
            "method": method,
            "n_groups": int(len(method_rows)),
            "mean_macro_f1": float(np.mean(f1)),
            "mean_macro_f1_ci95_low": safe_float(ci_low),
            "mean_macro_f1_ci95_high": safe_float(ci_high),
            "worst_group": str(worst["held_out_group"]),
            "worst_group_macro_f1": float(worst["macro_f1"]),
            "best_group": str(best["held_out_group"]),
            "best_group_macro_f1": float(best["macro_f1"]),
        }
    return output


def best_row(rows: list[dict[str, Any]], encoder_id: str, method_prefix: str | None = None) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("encoder_id") == encoder_id
        and isinstance(row.get("macro_f1"), (int, float))
        and (method_prefix is None or str(row.get("method", "")).startswith(method_prefix))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["macro_f1"]))


def summary_rows(
    features: list[dict[str, Any]],
    knn_rows: list[dict[str, Any]],
    linear_rows: list[dict[str, Any]],
    prototype_rows: list[dict[str, Any]],
    lobo_rows: list[dict[str, Any]],
    lomo_rows: list[dict[str, Any]],
    rng: np.random.Generator | None = None,
    bootstrap_iters: int = 0,
) -> list[dict[str, Any]]:
    lobo_agg = aggregate_grouped(lobo_rows, rng, bootstrap_iters)
    lomo_agg = aggregate_grouped(lomo_rows, rng, bootstrap_iters)
    rows = []
    for item in features:
        encoder_id = item["encoder_id"]
        best_knn = best_row(knn_rows, encoder_id, "knn_cosine")
        linear = best_row(linear_rows, encoder_id)
        proto = best_row(prototype_rows, encoder_id)
        lobo_best = max(
            [value for key, value in lobo_agg.items() if value["encoder_id"] == encoder_id],
            key=lambda value: value["mean_macro_f1"],
            default=None,
        )
        lomo_best = max(
            [value for key, value in lomo_agg.items() if value["encoder_id"] == encoder_id],
            key=lambda value: value["mean_macro_f1"],
            default=None,
        )
        rows.append(
            {
                "encoder_id": encoder_id,
                "encoder_name": item["encoder_name"],
                "model_name": item["model_name"],
                "embedding_dim": int(item["embedding"].shape[1]),
                "stratified_best_knn_method": best_knn.get("method") if best_knn else "",
                "stratified_best_knn_macro_f1": best_knn.get("macro_f1") if best_knn else "nan",
                "stratified_best_knn_macro_f1_ci95_low": best_knn.get("macro_f1_ci95_low") if best_knn else "nan",
                "stratified_best_knn_macro_f1_ci95_high": best_knn.get("macro_f1_ci95_high") if best_knn else "nan",
                "linear_probe_macro_f1": linear.get("macro_f1") if linear else "nan",
                "linear_probe_macro_f1_ci95_low": linear.get("macro_f1_ci95_low") if linear else "nan",
                "linear_probe_macro_f1_ci95_high": linear.get("macro_f1_ci95_high") if linear else "nan",
                "prototype_macro_f1": proto.get("macro_f1") if proto else "nan",
                "prototype_macro_f1_ci95_low": proto.get("macro_f1_ci95_low") if proto else "nan",
                "prototype_macro_f1_ci95_high": proto.get("macro_f1_ci95_high") if proto else "nan",
                "best_lobo_method": lobo_best.get("method") if lobo_best else "",
                "best_lobo_mean_macro_f1": lobo_best.get("mean_macro_f1") if lobo_best else "nan",
                "best_lobo_mean_macro_f1_ci95_low": lobo_best.get("mean_macro_f1_ci95_low") if lobo_best else "nan",
                "best_lobo_mean_macro_f1_ci95_high": lobo_best.get("mean_macro_f1_ci95_high") if lobo_best else "nan",
                "best_lobo_worst_group": lobo_best.get("worst_group") if lobo_best else "",
                "best_lobo_worst_group_macro_f1": lobo_best.get("worst_group_macro_f1") if lobo_best else "nan",
                "best_lomo_method": lomo_best.get("method") if lomo_best else "",
                "best_lomo_mean_macro_f1": lomo_best.get("mean_macro_f1") if lomo_best else "nan",
                "best_lomo_mean_macro_f1_ci95_low": lomo_best.get("mean_macro_f1_ci95_low") if lomo_best else "nan",
                "best_lomo_mean_macro_f1_ci95_high": lomo_best.get("mean_macro_f1_ci95_high") if lomo_best else "nan",
                "best_lomo_worst_group": lomo_best.get("worst_group") if lomo_best else "",
                "best_lomo_worst_group_macro_f1": lomo_best.get("worst_group_macro_f1") if lomo_best else "nan",
                "final_publication_note": "internal-only until regenerated" if encoder_id == "hierarchical_v6_2" else "usable",
            }
        )
    return rows


def plot_bars(out_dir: Path, summary: list[dict[str, Any]], grouped: bool) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    labels = [row["encoder_name"] for row in summary]
    if grouped:
        values = [float(row["best_lobo_mean_macro_f1"]) for row in summary]
        title = "Best board-grouped LOBO-style Macro-F1 by encoder"
        path = out_dir / "grouped_macro_f1_barplot.svg"
        ylabel = "Best LOBO mean Macro-F1"
    else:
        values = [float(row["stratified_best_knn_macro_f1"]) for row in summary]
        title = "Best stratified kNN Macro-F1 by encoder"
        path = out_dir / "stratified_macro_f1_barplot.svg"
        ylabel = "Best stratified kNN Macro-F1"
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(labels, values, color="#4C78A8")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.tick_params(axis="x", labelrotation=25)
    for idx, value in enumerate(values):
        axis.text(idx, value + 0.02, f"{100*value:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def write_report(
    path: Path,
    summary: list[dict[str, Any]],
    permutation_rows: list[dict[str, Any]],
    figure_paths: list[str],
    bootstrap_used: bool,
    linear_max_iter: int,
    convergence_warning_count: int,
) -> None:
    best = max(summary, key=lambda row: float(row["stratified_best_knn_macro_f1"]))
    lines = [
        "| Encoder | Dim | Best kNN Macro-F1 | Linear Macro-F1 | Prototype Macro-F1 | Best LOBO mean Macro-F1 | Best LOMO mean Macro-F1 | Note |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['encoder_name']} | {row['embedding_dim']} | {percent(row['stratified_best_knn_macro_f1'])} | {percent(row['linear_probe_macro_f1'])} | {percent(row['prototype_macro_f1'])} | {percent(row['best_lobo_mean_macro_f1'])} | {percent(row['best_lomo_mean_macro_f1'])} | {row['final_publication_note']} |"
        )
    perm_lines = [
        "| Encoder | Method | Macro-F1 |",
        "|---|---|---:|",
    ]
    for row in permutation_rows:
        if isinstance(row.get("macro_f1"), (int, float)):
            perm_lines.append(f"| {row['encoder_name']} | {row['method']} | {percent(row['macro_f1'])} |")
    fig_lines = "\n".join(f"- `{fig}`" for fig in figure_paths if fig) or "- Figure generation skipped."
    text = f"""# Unified Encoder Baseline Evaluation

## Scope

This report evaluates normalized encoder feature dumps under a shared frozen-representation protocol. It does not run model inference, train or fine-tune CNNs, implement LeFFT, or claim acoustic-response predictive value.

## Main comparison

{chr(10).join(lines)}

## Decision boundary

- v6.2-A is expected to remain the primary reportable frozen ESPI encoder if it outperforms random ResNet-18, ImageNet ResNet-18, v6.1, and hierarchical v6.2 under this matched protocol.
- The best stratified kNN result in this run is `{best['encoder_name']}` with Macro-F1 `{percent(best['stratified_best_knn_macro_f1'])}`.
- Primary encoder selection should prioritize grouped board/material robustness over stratified-only performance.
- Hierarchical v6.2 is internal-only for this table unless regenerated with correct `board` and `split_group` stored in the source NPZ.
- This evaluation does not support a validated Physics-Aligned Encoder claim.
- This evaluation does not support any acoustic-response predictive-value claim.

## Protocol

- Stratified train/validation: kNN cosine for k = 1, 3, 5, 10, 20; nearest class prototype; balanced logistic-regression linear probe.
- Grouped evaluation: board-grouped LOBO-style and material-grouped LOMO-style using kNN k=10, prototype, and balanced logistic regression.
- Sanity check: label-permutation test for kNN k=10 and linear probe.
- Linear probe max iterations: `{linear_max_iter}`.
- Captured linear convergence warnings: `{convergence_warning_count}`.
- Bootstrap 95% CI for stratified and grouped summary Macro-F1: `{bootstrap_used}`.

## Label-permutation sanity check

{chr(10).join(perm_lines)}

## Figures

{fig_lines}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate normalized encoder baseline feature dumps.")
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--linear-max-iter", type=int, default=5000)
    parser.add_argument("--max-iter", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-iters", type=int, default=200)
    parser.add_argument("--enable-bootstrap", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_iter is not None:
        args.linear_max_iter = args.max_iter
    if args.skip_bootstrap:
        args.enable_bootstrap = False
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    bootstrap_iters = args.bootstrap_iters if args.enable_bootstrap else 0

    features = []
    for encoder_id, encoder_name, filename in FEATURE_FILES:
        path = args.features_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing normalized feature dump: {path}")
        features.append(load_feature_dump(path, encoder_id, encoder_name))

    class_ids = sorted(set(features[0]["label"].astype(int).tolist()))
    label_lookup = {}
    for class_id in class_ids:
        mask = features[0]["label"].astype(int) == class_id
        label_lookup[class_id] = Counter(features[0]["label_name"][mask].tolist()).most_common(1)[0][0]

    quality = []
    knn_rows: list[dict[str, Any]] = []
    prototype_rows: list[dict[str, Any]] = []
    linear_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    lobo_rows: list[dict[str, Any]] = []
    lomo_rows: list[dict[str, Any]] = []
    linear_warning_log: list[dict[str, Any]] = []

    for item in features:
        print(f"[eval] {item['encoder_name']} quality")
        quality.extend(quality_rows(item, class_ids))
        print(f"[eval] {item['encoder_name']} stratified")
        k_rows, p_rows, l_rows, perm_rows = evaluate_stratified(item, class_ids, label_lookup, args, rng, linear_warning_log)
        knn_rows.extend(k_rows)
        prototype_rows.extend(p_rows)
        linear_rows.extend(l_rows)
        permutation_rows.extend(perm_rows)
        print(f"[eval] {item['encoder_name']} board grouped")
        lobo_rows.extend(evaluate_grouped(item, "board", class_ids, label_lookup, args, linear_warning_log))
        print(f"[eval] {item['encoder_name']} material grouped")
        lomo_rows.extend(evaluate_grouped(item, "material", class_ids, label_lookup, args, linear_warning_log))

    summary = summary_rows(features, knn_rows, linear_rows, prototype_rows, lobo_rows, lomo_rows, rng, bootstrap_iters)
    write_csv(args.out_dir / "quality_summary.csv", quality)
    write_csv(args.out_dir / "knn_summary.csv", knn_rows)
    write_csv(args.out_dir / "linear_probe_summary.csv", linear_rows)
    write_csv(args.out_dir / "prototype_summary.csv", prototype_rows)
    write_csv(args.out_dir / "lobo_board_summary.csv", lobo_rows)
    write_csv(args.out_dir / "lomo_material_summary.csv", lomo_rows)
    write_csv(args.out_dir / "label_permutation_summary.csv", permutation_rows)
    write_csv(args.out_dir / "encoder_baseline_summary.csv", summary)
    warning_path = args.out_dir / "linear_convergence_warnings.txt"
    if linear_warning_log:
        lines = []
        for row in linear_warning_log:
            lines.append(
                "[{category}] encoder={encoder_name} eval={evaluation} method={method} group={group}: {message}".format(
                    category=row.get("category", "Warning"),
                    encoder_name=row.get("encoder_name", ""),
                    evaluation=row.get("evaluation", ""),
                    method=row.get("method", ""),
                    group=row.get("held_out_group", row.get("fold", "")),
                    message=row.get("message", ""),
                )
            )
        warning_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        warning_path.write_text("No logistic-regression convergence warnings were captured.\n", encoding="utf-8")

    figures = [
        plot_bars(args.out_dir, summary, grouped=False),
        plot_bars(args.out_dir, summary, grouped=True),
    ]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "features_dir": str(args.features_dir),
        "out_dir": str(args.out_dir),
        "linear_max_iter": int(args.linear_max_iter),
        "bootstrap_used": bool(args.enable_bootstrap),
        "bootstrap_iters": int(bootstrap_iters),
        "summary": summary,
        "figures": [fig for fig in figures if fig],
        "linear_convergence_warnings": linear_warning_log,
        "linear_convergence_warning_count": len(linear_warning_log),
        "linear_convergence_warnings_path": str(warning_path),
        "hierarchical_publication_caveat": "internal-only unless regenerated with correct board/split_group stored in source NPZ",
    }
    write_json(args.out_dir / "encoder_baseline_key_numbers.json", payload)
    write_report(
        args.out_dir / "ENCODER_BASELINE_EVALUATION.md",
        summary,
        permutation_rows,
        [fig for fig in figures if fig],
        bool(args.enable_bootstrap),
        int(args.linear_max_iter),
        len(linear_warning_log),
    )

    print(f"[done] out_dir={args.out_dir}")
    for row in summary:
        print(
            "[done] {name}: kNN={knn:.6f} linear={linear:.6f} proto={proto:.6f} lobo={lobo:.6f} lomo={lomo:.6f}".format(
                name=row["encoder_name"],
                knn=float(row["stratified_best_knn_macro_f1"]),
                linear=float(row["linear_probe_macro_f1"]),
                proto=float(row["prototype_macro_f1"]),
                lobo=float(row["best_lobo_mean_macro_f1"]),
                lomo=float(row["best_lomo_mean_macro_f1"]),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
