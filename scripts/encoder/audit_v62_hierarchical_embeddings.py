#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit frozen hierarchical v6.2 embeddings.

This script only reads a saved hierarchical feature dump. It does not load raw
images, does not run GPU inference, and does not train a CNN.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_FEATURES = Path("outputs/hierarchical_embeddings_v001/features_hier_z_expert_prelogit.npz")
DEFAULT_OUT_DIR = Path("reports/hierarchical_encoder/audit_v001")

KNN_VALUES = (1, 3, 5, 10, 20)
GROUPED_K = 10
PCA_MAX_SAMPLES = 3000

DEFAULT_LABEL_NAMES = {
    0: "1_1H",
    1: "1_1T",
    2: "1_2",
    3: "2_1",
    4: "higher",
}

UNKNOWN_VALUES = {"", "unknown", "none", "nan", "null", "na"}


def warn(message: str, warnings: list[str]) -> None:
    warnings.append(message)
    print(f"[warn] {message}")


def first_existing(data: np.lib.npyio.NpzFile, names: Iterable[str]) -> str | None:
    for name in names:
        if name in data.files:
            return name
    return None


def scalar_to_str(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    if array.size == 1:
        return str(array.reshape(-1)[0])
    return str(array)


def as_str_array(values: np.ndarray | None, n_samples: int, default: str = "unknown") -> np.ndarray:
    if values is None:
        return np.array([default] * n_samples, dtype=object)
    array = np.asarray(values).astype(str)
    if array.shape == ():
        array = np.array([str(array.item())] * n_samples, dtype=object)
    if len(array) != n_samples:
        raise ValueError(f"Metadata length mismatch: expected {n_samples}, got {len(array)}")
    return array.astype(object)


def unknown_fraction(values: np.ndarray) -> float:
    if len(values) == 0:
        return 1.0
    unknown = [
        str(value).strip().lower() in UNKNOWN_VALUES
        for value in values.astype(str)
    ]
    return float(np.mean(unknown))


def extract_board_from_text(*values: Any) -> str:
    joined = " ".join(str(value) for value in values if value is not None).upper()
    match = re.search(r"([CW]\d{2})", joined)
    return match.group(1) if match else "unknown"


def derive_board(saved_board: np.ndarray, distribution_group: np.ndarray, paths: np.ndarray) -> tuple[np.ndarray, str]:
    if unknown_fraction(saved_board) < 0.5:
        recovered = []
        for board, group, path in zip(saved_board, distribution_group, paths):
            board_text = str(board).strip()
            if board_text.lower() not in UNKNOWN_VALUES:
                recovered.append(board_text.upper())
            else:
                recovered.append(extract_board_from_text(group, path))
        return np.array(recovered, dtype=object), "saved_with_missing_filled"

    recovered = [
        extract_board_from_text(group, path)
        for group, path in zip(distribution_group.astype(str), paths.astype(str))
    ]
    return np.array(recovered, dtype=object), "derived_from_distribution_group_or_path"


def derive_material(saved_material: np.ndarray, board: np.ndarray, paths: np.ndarray) -> tuple[np.ndarray, str]:
    recovered = []
    filled = 0
    for material, board_value, path_value in zip(saved_material, board, paths):
        material_text = str(material).strip().lower()
        if material_text not in UNKNOWN_VALUES:
            recovered.append(material_text)
            continue
        filled += 1
        joined = f"{board_value} {path_value}".lower()
        if str(board_value).upper().startswith("C") or "carbon" in joined:
            recovered.append("carbon")
        elif str(board_value).upper().startswith("W") or "wood" in joined:
            recovered.append("wood")
        else:
            recovered.append("unknown")
    source = "saved"
    if filled:
        source = "saved_with_board_path_fallback"
    if unknown_fraction(saved_material) >= 0.5:
        source = "derived_from_board_or_path"
    return np.array(recovered, dtype=object), source


def derive_domain(saved_domain: np.ndarray, distribution_group: np.ndarray, paths: np.ndarray) -> tuple[np.ndarray, str]:
    recovered = []
    filled = 0
    for domain, group, path in zip(saved_domain, distribution_group, paths):
        domain_text = str(domain).strip().lower()
        if domain_text not in UNKNOWN_VALUES:
            recovered.append(domain_text)
            continue
        filled += 1
        joined = f"{group} {path}".lower()
        if "pseudonoisy" in joined or "pseudo-noisy" in joined:
            recovered.append("pseudo_noisy")
        elif "averaged" in joined:
            recovered.append("averaged")
        else:
            recovered.append("clean")
    source = "saved" if not filled else "derived_from_distribution_group_or_path"
    return np.array(recovered, dtype=object), source


def derive_split_group(
    saved_split_group: np.ndarray,
    distribution_group: np.ndarray,
    board: np.ndarray,
) -> tuple[np.ndarray, str]:
    if unknown_fraction(saved_split_group) < 0.5:
        recovered = []
        for split_group, dist_group, board_value in zip(saved_split_group, distribution_group, board):
            split_text = str(split_group).strip()
            if split_text.lower() not in UNKNOWN_VALUES:
                recovered.append(split_text)
            elif str(dist_group).strip().lower() not in UNKNOWN_VALUES:
                recovered.append(str(dist_group))
            else:
                recovered.append(str(board_value))
        return np.array(recovered, dtype=object), "saved_with_missing_filled"

    if unknown_fraction(distribution_group) < 0.5:
        return distribution_group.astype(object), "fallback_distribution_group"
    return board.astype(object), "fallback_board"


def load_feature_dump(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Feature dump not found: {path}")

    with np.load(str(path), allow_pickle=True) as data:
        embedding_key = first_existing(data, ["embedding", "embeddings", "features", "x"])
        label_key = first_existing(data, ["label", "labels", "y", "targets"])
        if embedding_key is None or label_key is None:
            raise KeyError(f"NPZ must contain embeddings and labels. Keys found: {data.files}")

        embeddings = np.asarray(data[embedding_key]).astype(np.float32)
        labels = np.asarray(data[label_key]).astype(np.int64)
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")
        if len(labels) != len(embeddings):
            raise ValueError(f"Label length mismatch: labels={len(labels)}, embeddings={len(embeddings)}")

        n_samples = int(len(embeddings))
        label_name_key = first_existing(data, ["label_name", "label_names", "class_name", "class_names"])
        board_key = first_existing(data, ["board", "boards"])
        material_key = first_existing(data, ["material", "materials"])
        distribution_group_key = first_existing(data, ["distribution_group", "distribution_groups", "domain_group"])
        domain_key = first_existing(data, ["domain", "domains"])
        split_key = first_existing(data, ["split", "splits"])
        split_group_key = first_existing(data, ["split_group", "split_groups", "group", "groups"])
        path_key = first_existing(data, ["path", "paths", "file", "files"])
        frequency_key = first_existing(data, ["frequency_hz", "frequency", "freq_hz", "freq"])

        label_names = as_str_array(data[label_name_key], n_samples) if label_name_key else np.array(
            [DEFAULT_LABEL_NAMES.get(int(label), str(label)) for label in labels],
            dtype=object,
        )
        paths = as_str_array(data[path_key], n_samples, default="") if path_key else np.array([""] * n_samples, dtype=object)
        saved_board = as_str_array(data[board_key], n_samples) if board_key else np.array(["unknown"] * n_samples, dtype=object)
        saved_material = as_str_array(data[material_key], n_samples) if material_key else np.array(["unknown"] * n_samples, dtype=object)
        distribution_group = (
            as_str_array(data[distribution_group_key], n_samples)
            if distribution_group_key
            else np.array(["unknown"] * n_samples, dtype=object)
        )
        saved_domain = as_str_array(data[domain_key], n_samples) if domain_key else np.array(["unknown"] * n_samples, dtype=object)
        split = as_str_array(data[split_key], n_samples) if split_key else np.array(["unknown"] * n_samples, dtype=object)
        saved_split_group = (
            as_str_array(data[split_group_key], n_samples)
            if split_group_key
            else np.array(["unknown"] * n_samples, dtype=object)
        )

        if frequency_key:
            frequency = np.asarray(data[frequency_key]).astype(np.float32)
        else:
            frequency = np.full(n_samples, np.nan, dtype=np.float32)

        board, board_source = derive_board(saved_board, distribution_group, paths)
        material, material_source = derive_material(saved_material, board, paths)
        domain, domain_source = derive_domain(saved_domain, distribution_group, paths)
        split_group, split_group_source = derive_split_group(saved_split_group, distribution_group, board)

        if embedding_key != "embedding":
            warn(f"Using embedding key '{embedding_key}'.", warnings)
        if label_key != "label":
            warn(f"Using label key '{label_key}'.", warnings)
        if board_source != "saved_with_missing_filled" and board_source != "saved":
            warn(f"Board metadata was recovered using {board_source}.", warnings)
        if material_source != "saved":
            warn(f"Material metadata was recovered using {material_source}.", warnings)
        if domain_source != "saved":
            warn(f"Domain metadata was recovered using {domain_source}.", warnings)
        if split_group_source != "saved_with_missing_filled" and split_group_source != "saved":
            warn(f"Split-group metadata uses {split_group_source}.", warnings)

        checkpoint_sha256 = scalar_to_str(data["checkpoint_sha256"]) if "checkpoint_sha256" in data.files else "unknown"
        embedding_point = scalar_to_str(data["embedding_point"]) if "embedding_point" in data.files else "unknown"
        model_name = scalar_to_str(data["model_name"]) if "model_name" in data.files else "unknown"
        keys = list(data.files)

    return {
        "features_path": str(path),
        "keys": keys,
        "embeddings": embeddings,
        "labels": labels,
        "label_names": label_names,
        "board": board,
        "material": material,
        "distribution_group": distribution_group,
        "domain": domain,
        "frequency_hz": frequency,
        "paths": paths,
        "split": split,
        "split_group": split_group,
        "metadata_sources": {
            "board": board_source,
            "material": material_source,
            "domain": domain_source,
            "split_group": split_group_source,
        },
        "checkpoint_sha256": checkpoint_sha256,
        "embedding_point": embedding_point,
        "model_name": model_name,
    }


def l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), eps)
    return values / norms


def standardize_train_test(
    train_x: np.ndarray,
    test_x: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_ids: list[int]) -> np.ndarray:
    index = {class_id: idx for idx, class_id in enumerate(class_ids)}
    matrix = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
    for true_label, pred_label in zip(y_true.astype(int), y_pred.astype(int)):
        if true_label in index and pred_label in index:
            matrix[index[true_label], index[pred_label]] += 1
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
        support = int(matrix[row_idx, :].sum())
        pred_count = int(matrix[:, row_idx].sum())
        precision = true_positive / pred_count if pred_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "class_id": class_id,
                "label_name": label_lookup.get(class_id, str(class_id)),
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
        "confusion_matrix": matrix,
    }


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
        top_indices = np.argpartition(-scores, kth=max_k - 1, axis=1)[:, :max_k]
        top_scores = np.take_along_axis(scores, top_indices, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top_indices = np.take_along_axis(top_indices, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)
        top_labels = train_y[top_indices]
        for k in effective_k_values:
            block_preds = [
                majority_vote_with_scores(label_row[:k], score_row[:k])
                for label_row, score_row in zip(top_labels, top_scores)
            ]
            predictions[k].append(np.array(block_preds, dtype=np.int64))

    return {k: np.concatenate(parts, axis=0) for k, parts in predictions.items()}


def prototype_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    class_ids: list[int],
) -> np.ndarray:
    train_std, test_std = standardize_train_test(train_x, test_x)
    train_norm = l2_normalize(train_std.astype(np.float32))
    test_norm = l2_normalize(test_std.astype(np.float32))

    prototypes = []
    valid_classes = []
    for class_id in class_ids:
        mask = train_y.astype(int) == int(class_id)
        if not np.any(mask):
            continue
        prototype = train_norm[mask].mean(axis=0, keepdims=True)
        prototypes.append(l2_normalize(prototype)[0])
        valid_classes.append(int(class_id))

    if not prototypes:
        return np.zeros(len(test_x), dtype=np.int64)

    prototype_matrix = np.vstack(prototypes)
    scores = test_norm @ prototype_matrix.T
    local_predictions = np.argmax(scores, axis=1)
    return np.array([valid_classes[idx] for idx in local_predictions], dtype=np.int64)


def linear_probe_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    max_iter: int,
) -> tuple[np.ndarray | None, str | None]:
    if len(set(train_y.astype(int).tolist())) < 2:
        return None, "fewer than two classes in the reference fold"
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - depends on local environment
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
        model.fit(train_x, train_y)
        return model.predict(test_x).astype(np.int64), None
    except Exception as exc:  # pragma: no cover - depends on numerical convergence
        return None, f"linear probe failed: {exc}"


def safe_float(value: Any) -> float | str:
    try:
        output = float(value)
    except Exception:
        return "nan"
    if math.isnan(output) or math.isinf(output):
        return "nan"
    return output


def metric_row(
    base: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    row = dict(base)
    row.update(
        {
            "accuracy": safe_float(metrics["accuracy"]),
            "macro_recall": safe_float(metrics["macro_recall"]),
            "macro_f1": safe_float(metrics["macro_f1"]),
        }
    )
    for item in metrics["per_class"]:
        class_id = int(item["class_id"])
        row[f"class_{class_id}_support"] = int(item["support"])
        row[f"class_{class_id}_recall"] = safe_float(item["recall"])
        row[f"class_{class_id}_f1"] = safe_float(item["f1"])
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def counts_to_rows(section: str, values: np.ndarray) -> list[dict[str, Any]]:
    counts = Counter(values.astype(str).tolist())
    return [
        {
            "section": section,
            "name": key,
            "value": int(value),
            "status": "info",
            "details": "",
        }
        for key, value in sorted(counts.items(), key=lambda item: str(item[0]))
    ]


def quality_report(data: dict[str, Any], class_ids: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x = data["embeddings"]
    labels = data["labels"]
    paths = data["paths"].astype(str)
    split = data["split"].astype(str)
    split_group = data["split_group"].astype(str)

    duplicate_paths = int(len(paths) - len(set(paths.tolist())))
    nan_count = int(np.isnan(x).sum())
    inf_count = int(np.isinf(x).sum())
    nonzero_variance = bool(np.any(np.var(x, axis=0) > 0.0))
    consistent_shape = bool(x.ndim == 2 and x.shape[0] == len(labels))
    metadata_lengths_match = all(
        len(data[key]) == len(x)
        for key in ["label_names", "board", "material", "distribution_group", "domain", "frequency_hz", "paths", "split", "split_group"]
    )

    train_paths = set(paths[split == "train"].tolist())
    val_paths = set(paths[split == "val"].tolist())
    train_val_path_overlap = len(train_paths.intersection(val_paths))
    train_groups = set(split_group[split == "train"].tolist())
    val_groups = set(split_group[split == "val"].tolist())
    train_val_group_overlap = len(train_groups.intersection(val_groups))

    rows = [
        {"section": "quality", "name": "n_samples", "value": int(len(x)), "status": "info", "details": ""},
        {"section": "quality", "name": "embedding_dim", "value": int(x.shape[1]), "status": "info", "details": ""},
        {"section": "quality", "name": "embedding_nan_count", "value": nan_count, "status": "pass" if nan_count == 0 else "fail", "details": ""},
        {"section": "quality", "name": "embedding_inf_count", "value": inf_count, "status": "pass" if inf_count == 0 else "fail", "details": ""},
        {"section": "quality", "name": "duplicate_path_count", "value": duplicate_paths, "status": "pass" if duplicate_paths == 0 else "warn", "details": ""},
        {"section": "quality", "name": "nonzero_variance", "value": str(nonzero_variance).lower(), "status": "pass" if nonzero_variance else "fail", "details": ""},
        {"section": "quality", "name": "consistent_embedding_shape", "value": str(consistent_shape).lower(), "status": "pass" if consistent_shape else "fail", "details": ""},
        {"section": "quality", "name": "metadata_lengths_match", "value": str(metadata_lengths_match).lower(), "status": "pass" if metadata_lengths_match else "fail", "details": ""},
        {"section": "leakage_check", "name": "train_val_path_overlap_count", "value": train_val_path_overlap, "status": "pass" if train_val_path_overlap == 0 else "fail", "details": ""},
        {
            "section": "leakage_check",
            "name": "train_val_split_group_overlap_count",
            "value": train_val_group_overlap,
            "status": "warn" if train_val_group_overlap else "pass",
            "details": "The saved train/val split is stratified; grouped evaluations below explicitly hold out groups.",
        },
    ]

    class_counts = Counter(labels.astype(int).tolist())
    for class_id in class_ids:
        rows.append(
            {
                "section": "class_count",
                "name": f"{class_id}:{data['label_lookup'].get(class_id, str(class_id))}",
                "value": int(class_counts.get(class_id, 0)),
                "status": "info",
                "details": "",
            }
        )
    rows.extend(counts_to_rows("board_count", data["board"]))
    rows.extend(counts_to_rows("material_count", data["material"]))
    rows.extend(counts_to_rows("split_count", data["split"]))
    rows.extend(counts_to_rows("split_group_count", data["split_group"]))

    key_numbers = {
        "n_samples": int(len(x)),
        "embedding_dim": int(x.shape[1]),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "duplicate_paths": duplicate_paths,
        "nonzero_variance": nonzero_variance,
        "consistent_embedding_shape": consistent_shape,
        "metadata_lengths_match": metadata_lengths_match,
        "train_val_path_overlap_count": train_val_path_overlap,
        "train_val_split_group_overlap_count": train_val_group_overlap,
    }
    return rows, key_numbers


def stratified_evaluation(
    data: dict[str, Any],
    class_ids: list[int],
    out_dir: Path,
    warnings: list[str],
    block_size: int,
    max_iter: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    x = data["embeddings"]
    y = data["labels"]
    split = np.char.lower(data["split"].astype(str))
    train_mask = split == "train"
    val_mask = split == "val"
    if not np.any(train_mask) or not np.any(val_mask):
        warn("Train/val split not available; stratified evaluation skipped.", warnings)
        return [], [], [], {}

    train_x = x[train_mask]
    train_y = y[train_mask]
    val_x = x[val_mask]
    val_y = y[val_mask]
    label_lookup = data["label_lookup"]
    metadata = {
        "split": "train_to_val",
        "n_train": int(len(train_x)),
        "n_test": int(len(val_x)),
    }

    knn_rows = []
    predictions = knn_predict_multi(train_x, train_y, val_x, KNN_VALUES, block_size=block_size)
    for k, y_pred in sorted(predictions.items()):
        metrics = metrics_from_predictions(val_y, y_pred, class_ids, label_lookup)
        knn_rows.append(metric_row({**metadata, "method": f"knn_cosine_k{k}", "k": k}, metrics))

    prototype_rows = []
    y_pred_proto = prototype_predict(train_x, train_y, val_x, class_ids)
    metrics_proto = metrics_from_predictions(val_y, y_pred_proto, class_ids, label_lookup)
    prototype_rows.append(metric_row({**metadata, "method": "nearest_class_prototype"}, metrics_proto))

    linear_rows = []
    y_pred_linear, skipped_reason = linear_probe_predict(train_x, train_y, val_x, max_iter=max_iter)
    if y_pred_linear is None:
        warn(f"Stratified linear probe skipped: {skipped_reason}", warnings)
    else:
        metrics_linear = metrics_from_predictions(val_y, y_pred_linear, class_ids, label_lookup)
        linear_rows.append(metric_row({**metadata, "method": "balanced_logistic_regression"}, metrics_linear))

    key_numbers = {
        "stratified_best_knn": best_row(knn_rows),
        "stratified_prototype": prototype_rows[0] if prototype_rows else None,
        "stratified_linear_probe": linear_rows[0] if linear_rows else None,
    }
    return knn_rows, prototype_rows, linear_rows, key_numbers


def grouped_evaluation(
    data: dict[str, Any],
    group_values: np.ndarray,
    group_field: str,
    class_ids: list[int],
    warnings: list[str],
    block_size: int,
    max_iter: int,
) -> list[dict[str, Any]]:
    x = data["embeddings"]
    y = data["labels"]
    label_lookup = data["label_lookup"]
    groups = group_values.astype(str)
    unique_groups = sorted(
        group
        for group in set(groups.tolist())
        if group.strip().lower() not in UNKNOWN_VALUES
    )

    rows: list[dict[str, Any]] = []
    if len(unique_groups) < 2:
        warn(f"{group_field} grouped evaluation skipped: fewer than two usable groups.", warnings)
        return rows

    for group in unique_groups:
        test_mask = groups == group
        train_mask = ~test_mask
        train_x = x[train_mask]
        train_y = y[train_mask]
        test_x = x[test_mask]
        test_y = y[test_mask]
        if len(test_x) == 0 or len(train_x) == 0:
            continue

        train_values = ",".join(sorted(set(groups[train_mask].tolist())))
        base = {
            "group_field": group_field,
            "held_out_group": group,
            "train_groups": train_values,
            "n_train": int(len(train_x)),
            "n_test": int(len(test_x)),
        }

        y_pred_knn = knn_predict_multi(train_x, train_y, test_x, [GROUPED_K], block_size=block_size).get(GROUPED_K)
        if y_pred_knn is not None:
            metrics = metrics_from_predictions(test_y, y_pred_knn, class_ids, label_lookup)
            rows.append(metric_row({**base, "method": f"knn_cosine_k{GROUPED_K}", "k": GROUPED_K}, metrics))

        y_pred_proto = prototype_predict(train_x, train_y, test_x, class_ids)
        metrics = metrics_from_predictions(test_y, y_pred_proto, class_ids, label_lookup)
        rows.append(metric_row({**base, "method": "nearest_class_prototype"}, metrics))

        y_pred_linear, skipped_reason = linear_probe_predict(train_x, train_y, test_x, max_iter=max_iter)
        if y_pred_linear is None:
            warn(f"{group_field}={group} linear probe skipped: {skipped_reason}", warnings)
        else:
            metrics = metrics_from_predictions(test_y, y_pred_linear, class_ids, label_lookup)
            rows.append(metric_row({**base, "method": "balanced_logistic_regression"}, metrics))

    return rows


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    metric_rows = [row for row in rows if isinstance(row.get("macro_f1"), (int, float))]
    if not metric_rows:
        return None
    return max(metric_rows, key=lambda row: float(row["macro_f1"]))


def aggregate_group_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if isinstance(row.get("macro_f1"), (int, float)):
            by_method[str(row["method"])].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for method, method_rows in by_method.items():
        f1_values = np.array([float(row["macro_f1"]) for row in method_rows], dtype=np.float64)
        recalls = np.array([float(row["macro_recall"]) for row in method_rows], dtype=np.float64)
        accuracies = np.array([float(row["accuracy"]) for row in method_rows], dtype=np.float64)
        worst = min(method_rows, key=lambda row: float(row["macro_f1"]))
        best = max(method_rows, key=lambda row: float(row["macro_f1"]))
        summary[method] = {
            "n_groups": int(len(method_rows)),
            "mean_accuracy": float(np.mean(accuracies)),
            "mean_macro_recall": float(np.mean(recalls)),
            "mean_macro_f1": float(np.mean(f1_values)),
            "min_macro_f1": float(np.min(f1_values)),
            "max_macro_f1": float(np.max(f1_values)),
            "worst_group": str(worst.get("held_out_group", "")),
            "best_group": str(best.get("held_out_group", "")),
        }
    return summary


def stratified_sample_indices(labels: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_samples = len(labels)
    if n_samples <= max_samples:
        return np.arange(n_samples, dtype=np.int64)

    selected: list[np.ndarray] = []
    class_ids = sorted(set(labels.astype(int).tolist()))
    remaining = max_samples
    for idx, class_id in enumerate(class_ids):
        class_indices = np.flatnonzero(labels.astype(int) == class_id)
        if len(class_indices) == 0:
            continue
        classes_left = len(class_ids) - idx
        proportional = int(round(max_samples * len(class_indices) / n_samples))
        min_take = 1 if len(class_indices) else 0
        take = max(min_take, proportional)
        take = min(take, len(class_indices), remaining - max(0, classes_left - 1))
        if take > 0:
            selected.append(rng.choice(class_indices, size=take, replace=False))
            remaining -= take
    if not selected:
        return np.arange(min(max_samples, n_samples), dtype=np.int64)

    output = np.unique(np.concatenate(selected)).astype(np.int64)
    if len(output) > max_samples:
        output = rng.choice(output, size=max_samples, replace=False).astype(np.int64)
    output.sort()
    return output


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    components = vt[:2].T
    return x @ components


def save_pca_plot(
    x: np.ndarray,
    color_values: np.ndarray,
    title: str,
    output_path: Path,
    warnings: list[str],
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        warn(f"PCA plot skipped for {title}: matplotlib unavailable: {exc}", warnings)
        return False

    coords = pca_2d(x)
    values = color_values.astype(str)
    unique_values = sorted(set(values.tolist()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 6), dpi=140)
    cmap = plt.get_cmap("tab10")
    for idx, value in enumerate(unique_values):
        mask = values == value
        axis.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=7,
            alpha=0.72,
            label=value,
            color=cmap(idx % 10),
            linewidths=0,
        )
    axis.set_title(title)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.legend(loc="best", fontsize=7, markerscale=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def generate_pca_diagnostics(
    data: dict[str, Any],
    out_dir: Path,
    warnings: list[str],
    max_samples: int,
    seed: int,
) -> dict[str, str]:
    indices = stratified_sample_indices(data["labels"], max_samples=max_samples, seed=seed)
    x = data["embeddings"][indices]
    label_display = np.array(
        [
            f"{int(label)}:{data['label_lookup'].get(int(label), str(label))}"
            for label in data["labels"][indices]
        ],
        dtype=object,
    )
    outputs = {
        "pca_by_class": out_dir / "pca_by_class_z_expert_prelogit.png",
        "pca_by_board": out_dir / "pca_by_board_z_expert_prelogit.png",
        "pca_by_material": out_dir / "pca_by_material_z_expert_prelogit.png",
    }
    saved = {}
    if save_pca_plot(x, label_display, "Hierarchical v6.2 z_expert_prelogit PCA by class", outputs["pca_by_class"], warnings):
        saved["pca_by_class"] = str(outputs["pca_by_class"])
    if save_pca_plot(x, data["board"][indices], "Hierarchical v6.2 z_expert_prelogit PCA by board", outputs["pca_by_board"], warnings):
        saved["pca_by_board"] = str(outputs["pca_by_board"])
    if save_pca_plot(x, data["material"][indices], "Hierarchical v6.2 z_expert_prelogit PCA by material", outputs["pca_by_material"], warnings):
        saved["pca_by_material"] = str(outputs["pca_by_material"])
    return saved


def format_percent(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def method_summary_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "| Method | Result |\n|---|---|\n| n/a | skipped |\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        parts = []
        for column in columns:
            value = row.get(column, "")
            if column in {"accuracy", "macro_recall", "macro_f1"} and isinstance(value, (int, float)):
                parts.append(format_percent(value))
            else:
                parts.append(str(value))
        lines.append("| " + " | ".join(parts) + " |")
    return "\n".join(lines) + "\n"


def aggregate_summary_table(summary: dict[str, dict[str, Any]]) -> str:
    if not summary:
        return "| Method | Groups | Mean Macro-F1 | Worst group | Worst Macro-F1 | Best group | Best Macro-F1 |\n|---|---:|---:|---|---:|---|---:|\n| n/a | 0 | n/a | n/a | n/a | n/a | n/a |\n"
    lines = [
        "| Method | Groups | Mean Macro-F1 | Worst group | Worst Macro-F1 | Best group | Best Macro-F1 |",
        "|---|---:|---:|---|---:|---|---:|",
    ]
    for method, item in sorted(summary.items()):
        lines.append(
            "| {method} | {n_groups} | {mean} | {worst_group} | {min_f1} | {best_group} | {max_f1} |".format(
                method=method,
                n_groups=item["n_groups"],
                mean=format_percent(item["mean_macro_f1"]),
                worst_group=item["worst_group"],
                min_f1=format_percent(item["min_macro_f1"]),
                best_group=item["best_group"],
                max_f1=format_percent(item["max_macro_f1"]),
            )
        )
    return "\n".join(lines) + "\n"


def write_summary(
    out_path: Path,
    data: dict[str, Any],
    quality: dict[str, Any],
    stratified: dict[str, Any],
    grouped: dict[str, dict[str, dict[str, Any]]],
    pca_outputs: dict[str, str],
    warnings: list[str],
) -> None:
    best_knn = stratified.get("stratified_best_knn")
    prototype = stratified.get("stratified_prototype")
    linear = stratified.get("stratified_linear_probe")

    stratified_rows = []
    if best_knn:
        stratified_rows.append(best_knn)
    if prototype:
        stratified_rows.append(prototype)
    if linear:
        stratified_rows.append(linear)

    warning_lines = "\n".join(f"- {item}" for item in warnings) if warnings else "- None"
    pca_lines = "\n".join(f"- `{path}`" for path in pca_outputs.values()) if pca_outputs else "- PCA plots were skipped."
    source_lines = "\n".join(
        f"- `{key}`: `{value}`"
        for key, value in data["metadata_sources"].items()
    )

    text = f"""# Hierarchical v6.2 Frozen Embedding Audit v001

## Scope
This report audits the already extracted hierarchical v6.2 frozen embeddings from `z_expert_prelogit / z_arcface_input`.
It is a frozen grouped embedding evaluation over saved features only. It does not train a new CNN, does not reload raw images, and does not validate a final Physics-Aligned Encoder.

## Source artifact
- Feature dump: `{data['features_path']}`
- Model name: `{data['model_name']}`
- Embedding point: `{data['embedding_point']}`
- Checkpoint SHA256: `{data['checkpoint_sha256']}`
- Samples: `{quality['n_samples']}`
- Embedding dimension: `{quality['embedding_dim']}`

## Metadata recovery
{source_lines}

## Quality checks
- NaN count: `{quality['nan_count']}`
- Inf count: `{quality['inf_count']}`
- Duplicate paths: `{quality['duplicate_paths']}`
- Nonzero variance: `{quality['nonzero_variance']}`
- Metadata lengths match embeddings: `{quality['metadata_lengths_match']}`
- Train/val path overlap count: `{quality['train_val_path_overlap_count']}`
- Train/val split-group overlap count: `{quality['train_val_split_group_overlap_count']}`

## Stratified train-to-validation evaluation
{method_summary_table(stratified_rows, ['method', 'accuracy', 'macro_recall', 'macro_f1'])}

## Leave-split-group evaluation
{aggregate_summary_table(grouped.get('leave_split_group', {}))}

## Board-grouped LOBO-style evaluation
{aggregate_summary_table(grouped.get('board_lobo', {}))}

## Material-grouped LOMO-style evaluation
{aggregate_summary_table(grouped.get('material_lomo', {}))}

## PCA diagnostics
{pca_lines}

## Interpretation
The hierarchical v6.2 feature dump exposes a valid frozen embedding point through `outputs["embeddings"]`, used here as `z_expert_prelogit / z_arcface_input`.
The stratified and grouped metrics quantify whether this saved embedding space remains useful when the reference set excludes split groups, boards, or materials.
This is not a comparison with v6.2-A and it is not a claim of a validated Physics-Aligned Encoder.

## Caveats
- Grouped evaluations operate on frozen embeddings from an already trained hierarchical checkpoint.
- They test representation usefulness under excluded-reference settings, not a new CNN training run with held-out boards or materials.
- Metadata fields `board` and `split_group` in the feature dump may require recovery from `distribution_group` and `path`; the metadata sources are reported above.

## Warnings
{warning_lines}
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit frozen hierarchical v6.2 embeddings.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES, help="Saved hierarchical NPZ feature dump.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output report directory.")
    parser.add_argument("--block-size", type=int, default=512, help="Block size for kNN matrix products.")
    parser.add_argument("--max-iter", type=int, default=1000, help="Logistic regression max_iter.")
    parser.add_argument("--pca-max-samples", type=int, default=PCA_MAX_SAMPLES, help="Maximum stratified samples for PCA plots.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for PCA sampling.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    data = load_feature_dump(args.features, warnings)
    labels = data["labels"]
    class_ids = sorted(set(labels.astype(int).tolist()))
    label_lookup: dict[int, str] = {}
    for class_id in class_ids:
        names = data["label_names"][labels.astype(int) == class_id].astype(str)
        if len(names):
            label_lookup[class_id] = Counter(names.tolist()).most_common(1)[0][0]
        else:
            label_lookup[class_id] = DEFAULT_LABEL_NAMES.get(class_id, str(class_id))
    data["label_lookup"] = label_lookup

    quality_rows_data, quality_numbers = quality_report(data, class_ids)
    write_csv(out_dir / "quality_report.csv", quality_rows_data)

    print("[audit] running stratified train->val evaluations")
    knn_rows, prototype_rows, linear_rows, stratified_numbers = stratified_evaluation(
        data,
        class_ids,
        out_dir,
        warnings,
        block_size=args.block_size,
        max_iter=args.max_iter,
    )
    write_csv(out_dir / "knn_report.csv", knn_rows)
    write_csv(out_dir / "prototype_report.csv", prototype_rows)
    write_csv(out_dir / "linear_probe_report.csv", linear_rows)

    print("[audit] running leave-split-group evaluation")
    leave_split_rows = grouped_evaluation(
        data,
        data["split_group"],
        "split_group",
        class_ids,
        warnings,
        block_size=args.block_size,
        max_iter=args.max_iter,
    )
    write_csv(out_dir / "leave_split_group_out_report.csv", leave_split_rows)

    print("[audit] running board-grouped LOBO-style evaluation")
    board_rows = grouped_evaluation(
        data,
        data["board"],
        "board",
        class_ids,
        warnings,
        block_size=args.block_size,
        max_iter=args.max_iter,
    )
    write_csv(out_dir / "lobo_board_summary.csv", board_rows)

    print("[audit] running material-grouped LOMO-style evaluation")
    material_rows = grouped_evaluation(
        data,
        data["material"],
        "material",
        class_ids,
        warnings,
        block_size=args.block_size,
        max_iter=args.max_iter,
    )
    write_csv(out_dir / "lomo_material_summary.csv", material_rows)

    print("[audit] generating PCA diagnostics")
    pca_outputs = generate_pca_diagnostics(
        data,
        out_dir,
        warnings,
        max_samples=args.pca_max_samples,
        seed=args.seed,
    )

    grouped_numbers = {
        "leave_split_group": aggregate_group_rows(leave_split_rows),
        "board_lobo": aggregate_group_rows(board_rows),
        "material_lomo": aggregate_group_rows(material_rows),
    }

    key_numbers = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "features_path": str(args.features),
        "out_dir": str(out_dir),
        "model_name": data["model_name"],
        "embedding_point": data["embedding_point"],
        "checkpoint_sha256": data["checkpoint_sha256"],
        "npz_keys": data["keys"],
        "metadata_sources": data["metadata_sources"],
        "quality": quality_numbers,
        "stratified": stratified_numbers,
        "grouped": grouped_numbers,
        "pca_outputs": pca_outputs,
        "warnings": warnings,
    }
    write_json(out_dir / "hierarchical_audit_key_numbers.json", key_numbers)
    write_summary(
        out_dir / "HIERARCHICAL_ENCODER_AUDIT_SUMMARY.md",
        data,
        quality_numbers,
        stratified_numbers,
        grouped_numbers,
        pca_outputs,
        warnings,
    )

    print(f"[done] output_dir={out_dir}")
    best_knn = stratified_numbers.get("stratified_best_knn")
    if best_knn:
        print(f"[done] stratified_best_knn={best_knn['method']} macro_f1={best_knn['macro_f1']:.6f}")
    if linear_rows:
        print(f"[done] stratified_linear_probe macro_f1={linear_rows[0]['macro_f1']:.6f}")
    for name, summary in grouped_numbers.items():
        for method, item in sorted(summary.items()):
            print(f"[done] {name} {method} mean_macro_f1={item['mean_macro_f1']:.6f} min_macro_f1={item['min_macro_f1']:.6f}")
    if warnings:
        print("[done] warnings:")
        for item in warnings:
            print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
