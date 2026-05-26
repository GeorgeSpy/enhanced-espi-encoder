#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight grouped generalization evaluation for frozen v6.2-A embeddings.

This script uses only the saved feature dump. It does not load raw images,
does not use GPU, does not retrain any CNN, and does not duplicate embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURES = ROOT / "features_v62a_epoch25.npz"
DEFAULT_OUT = ROOT / "embedding_audit_v001" / "grouped_generalization_v001"
SMOKE_OUT = ROOT / "embedding_audit_v001" / "grouped_generalization_smoke"

LABEL_NAMES = {
    0: "1_1H",
    1: "1_1T",
    2: "1_2",
    3: "2_1",
    4: "higher",
}


def warn(message: str, warnings: list[str]) -> None:
    warnings.append(message)
    print(f"[warn] {message}")


def first_existing(data: np.lib.npyio.NpzFile, names: list[str]) -> str | None:
    for name in names:
        if name in data.files:
            return name
    return None


def as_str_array(values: np.ndarray, n_samples: int, default: str = "unknown") -> np.ndarray:
    if values is None:
        return np.array([default] * n_samples, dtype=object)
    return values.astype(str)


def derive_material(board: np.ndarray, path: np.ndarray) -> np.ndarray:
    out = []
    for board_value, path_value in zip(board.astype(str), path.astype(str)):
        lower = f"{board_value} {path_value}".lower()
        if "carbon" in lower or board_value.upper().startswith("C"):
            out.append("carbon")
        elif "wood" in lower or board_value.upper().startswith("W"):
            out.append("wood")
        else:
            out.append("unknown")
    return np.array(out, dtype=object)


def load_feature_dump(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Feature dump not found: {path}")

    data = np.load(str(path), allow_pickle=True)

    emb_key = first_existing(data, ["embeddings", "embedding", "features", "x"])
    label_key = first_existing(data, ["label", "labels", "y", "targets"])
    if emb_key is None or label_key is None:
        raise KeyError(f"NPZ must contain embeddings and labels. Keys found: {data.files}")

    embeddings = data[emb_key].astype(np.float32)
    labels = data[label_key].astype(np.int64)
    n_samples = int(embeddings.shape[0])

    if len(labels) != n_samples:
        raise ValueError(f"Label length mismatch: embeddings={n_samples}, labels={len(labels)}")

    board_key = first_existing(data, ["board", "boards", "split_group", "group"])
    material_key = first_existing(data, ["material", "materials"])
    domain_key = first_existing(data, ["domain", "domains"])
    path_key = first_existing(data, ["path", "paths"])
    freq_key = first_existing(data, ["frequency_hz", "frequency", "freq_hz", "freq"])

    path_values = as_str_array(data[path_key], n_samples) if path_key else np.array([""] * n_samples, dtype=object)
    board = as_str_array(data[board_key], n_samples) if board_key else np.array(["unknown"] * n_samples, dtype=object)
    material = as_str_array(data[material_key], n_samples) if material_key else derive_material(board, path_values)
    domain = as_str_array(data[domain_key], n_samples) if domain_key else np.array(["unknown"] * n_samples, dtype=object)
    frequency = data[freq_key].astype(np.float32) if freq_key else np.full(n_samples, np.nan, dtype=np.float32)

    checkpoint = str(data["checkpoint_path"]) if "checkpoint_path" in data.files else "unknown"
    embedding_layer = str(data["embedding_layer"]) if "embedding_layer" in data.files else "unknown"

    missing = list(data["load_missing"].astype(str)) if "load_missing" in data.files else []
    unexpected = list(data["load_unexpected"].astype(str)) if "load_unexpected" in data.files else []

    if emb_key != "embeddings":
        warn(f"Using embedding key '{emb_key}'", warnings)
    if label_key != "label":
        warn(f"Using label key '{label_key}'", warnings)
    if material_key is None:
        warn("Material key missing; material was derived from board/path heuristics.", warnings)

    return {
        "features_path": str(path),
        "embeddings": embeddings,
        "labels": labels,
        "board": board,
        "material": material,
        "domain": domain,
        "path": path_values,
        "frequency": frequency,
        "checkpoint": checkpoint,
        "embedding_layer": embedding_layer,
        "load_missing": missing,
        "load_unexpected": unexpected,
        "keys": list(data.files),
    }


def smoke_subset_indices(board: np.ndarray, material: np.ndarray, max_per_board: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for board_value in sorted(set(board.astype(str).tolist())):
        idx = np.flatnonzero(board.astype(str) == board_value)
        if len(idx) == 0:
            continue
        take = min(max_per_board, len(idx))
        selected.append(rng.choice(idx, size=take, replace=False))

    if not selected:
        for material_value in sorted(set(material.astype(str).tolist())):
            idx = np.flatnonzero(material.astype(str) == material_value)
            take = min(max_per_board, len(idx))
            selected.append(rng.choice(idx, size=take, replace=False))

    out = np.unique(np.concatenate(selected)).astype(np.int64)
    out.sort()
    return out


def standardize_train_test(train_x: np.ndarray, test_x: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)
    return x / denom


def majority_vote_with_scores(labels: np.ndarray, scores: np.ndarray) -> int:
    counts: dict[int, int] = defaultdict(int)
    score_sums: dict[int, float] = defaultdict(float)
    for label, score in zip(labels.astype(int), scores.astype(float)):
        counts[int(label)] += 1
        score_sums[int(label)] += float(score)
    return max(counts, key=lambda label: (counts[label], score_sums[label], -label))


def knn_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    k: int,
    block_size: int,
) -> np.ndarray:
    if len(train_x) == 0 or len(test_x) == 0:
        return np.array([], dtype=np.int64)

    train_std, test_std = standardize_train_test(train_x, test_x)
    train_norm = l2_normalize(train_std.astype(np.float32))
    test_norm = l2_normalize(test_std.astype(np.float32))
    effective_k = max(1, min(int(k), len(train_norm)))

    preds: list[np.ndarray] = []
    for start in range(0, len(test_norm), block_size):
        block = test_norm[start : start + block_size]
        scores = block @ train_norm.T
        top_idx = np.argpartition(-scores, kth=effective_k - 1, axis=1)[:, :effective_k]
        top_scores = np.take_along_axis(scores, top_idx, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)
        block_preds = [
            majority_vote_with_scores(train_y[row_idx], score_row)
            for row_idx, score_row in zip(top_idx, top_scores)
        ]
        preds.append(np.array(block_preds, dtype=np.int64))

    return np.concatenate(preds, axis=0)


def prototype_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, n_classes: int) -> np.ndarray:
    train_std, test_std = standardize_train_test(train_x, test_x)
    train_norm = l2_normalize(train_std.astype(np.float32))
    test_norm = l2_normalize(test_std.astype(np.float32))

    prototypes = []
    class_ids = []
    for class_id in range(n_classes):
        mask = train_y == class_id
        if not np.any(mask):
            continue
        proto = train_norm[mask].mean(axis=0, keepdims=True)
        prototypes.append(l2_normalize(proto)[0])
        class_ids.append(class_id)

    if not prototypes:
        return np.zeros(len(test_x), dtype=np.int64)

    proto_matrix = np.vstack(prototypes)
    scores = test_norm @ proto_matrix.T
    local_pred = np.argmax(scores, axis=1)
    return np.array([class_ids[idx] for idx in local_pred], dtype=np.int64)


def linear_probe_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    warnings: list[str],
) -> tuple[np.ndarray | None, str | None]:
    if len(set(train_y.astype(int).tolist())) < 2:
        return None, "linear_probe skipped: fewer than 2 train classes"

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        reason = f"linear_probe skipped: scikit-learn unavailable ({exc})"
        warn(reason, warnings)
        return None, reason

    try:
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=1),
        )
        clf.fit(train_x, train_y)
        return clf.predict(test_x).astype(np.int64), None
    except Exception as exc:
        reason = f"linear_probe skipped: fit/predict failed ({exc})"
        warn(reason, warnings)
        return None, reason


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= true_label < n_classes and 0 <= pred_label < n_classes:
            cm[true_label, pred_label] += 1
    return cm


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> tuple[np.ndarray, dict[str, Any]]:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    per_class = []
    active_recalls = []
    active_f1s = []

    for class_id in range(n_classes):
        tp = int(cm[class_id, class_id])
        support = int(cm[class_id].sum())
        pred_count = int(cm[:, class_id].sum())
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "class": class_id,
                "label_name": LABEL_NAMES.get(class_id, str(class_id)),
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        if support > 0:
            active_recalls.append(recall)
            active_f1s.append(f1)

    return cm, {
        "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
        "macro_recall": float(np.mean(active_recalls)) if active_recalls else 0.0,
        "macro_f1": float(np.mean(active_f1s)) if active_f1s else 0.0,
        "per_class": per_class,
        "n_samples": int(cm.sum()),
        "n_active_classes": int(sum(1 for row in per_class if row["support"] > 0)),
    }


def sanitize(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("_") or "unknown"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix(path: Path, cm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, cm, fmt="%d", delimiter=",")


def metric_row(
    evaluation: str,
    group_name: str,
    method: str,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "evaluation": evaluation,
        "group": group_name,
        "method": method,
        "n_samples": metrics["n_samples"],
        "n_active_classes": metrics["n_active_classes"],
        "accuracy": metrics["accuracy"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
    }
    for class_info in metrics["per_class"]:
        class_id = int(class_info["class"])
        row[f"support_{class_id}_{LABEL_NAMES.get(class_id, str(class_id))}"] = int(class_info["support"])
    if extra:
        row.update(extra)
    return row


def evaluate_fold(
    evaluation: str,
    group_name: str,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    out_dir: Path,
    k: int,
    block_size: int,
    warnings: list[str],
    skipped: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_x = embeddings[train_mask]
    train_y = labels[train_mask]
    test_x = embeddings[test_mask]
    test_y = labels[test_mask]

    if len(train_x) == 0 or len(test_x) == 0:
        reason = f"{evaluation}/{group_name}: empty train or test fold"
        warn(reason, warnings)
        skipped.append({"evaluation": evaluation, "group": group_name, "method": "all", "reason": reason})
        return rows

    method_predictions: list[tuple[str, np.ndarray | None, str | None]] = [
        (f"knn_cosine_k{k}", knn_predict(train_x, train_y, test_x, k=k, block_size=block_size), None),
        ("nearest_class_prototype_cosine", prototype_predict(train_x, train_y, test_x, n_classes=n_classes), None),
    ]

    linear_pred, linear_reason = linear_probe_predict(train_x, train_y, test_x, warnings)
    method_predictions.append(("linear_probe_logistic_regression", linear_pred, linear_reason))

    for method, pred, reason in method_predictions:
        if pred is None:
            skipped.append({"evaluation": evaluation, "group": group_name, "method": method, "reason": reason or "unknown"})
            continue

        cm, metrics = metrics_from_predictions(test_y, pred, n_classes)
        cm_name = f"{sanitize(evaluation)}_{sanitize(group_name)}_{sanitize(method)}.csv"
        cm_path = out_dir / "confusion_matrices" / cm_name
        write_confusion_matrix(cm_path, cm)
        rows.append(
            metric_row(
                evaluation=evaluation,
                group_name=group_name,
                method=method,
                metrics=metrics,
                extra={
                    "train_samples": int(len(train_x)),
                    "confusion_matrix": str(cm_path),
                },
            )
        )

    return rows


def aggregate_by_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method in sorted({str(row["method"]) for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        macro_f1 = np.array([float(row["macro_f1"]) for row in selected], dtype=np.float64)
        macro_recall = np.array([float(row["macro_recall"]) for row in selected], dtype=np.float64)
        accuracy = np.array([float(row["accuracy"]) for row in selected], dtype=np.float64)
        out.append(
            {
                "method": method,
                "n_groups": len(selected),
                "accuracy_mean": float(accuracy.mean()),
                "accuracy_min": float(accuracy.min()),
                "accuracy_max": float(accuracy.max()),
                "macro_recall_mean": float(macro_recall.mean()),
                "macro_recall_min": float(macro_recall.min()),
                "macro_recall_max": float(macro_recall.max()),
                "macro_f1_mean": float(macro_f1.mean()),
                "macro_f1_min": float(macro_f1.min()),
                "macro_f1_max": float(macro_f1.max()),
            }
        )
    return out


def best_worst(rows: list[dict[str, Any]], method: str | None = None) -> dict[str, Any]:
    selected = rows
    if method is not None:
        selected = [row for row in rows if row["method"] == method]
    if not selected:
        return {"best": None, "worst": None}
    return {
        "best": max(selected, key=lambda row: float(row["macro_f1"])),
        "worst": min(selected, key=lambda row: float(row["macro_f1"])),
    }


def pct(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_summary(
    path: Path,
    features: dict[str, Any],
    out_dir: Path,
    lobo_rows: list[dict[str, Any]],
    lomo_rows: list[dict[str, Any]],
    lobo_agg: list[dict[str, Any]],
    lomo_agg: list[dict[str, Any]],
    skipped: list[dict[str, str]],
    warnings: list[str],
    smoke_test: bool,
) -> None:
    methods = sorted({row["method"] for row in [*lobo_rows, *lomo_rows]})
    best_lobo = best_worst(lobo_rows, method="knn_cosine_k10")
    best_lomo = best_worst(lomo_rows, method="knn_cosine_k10")

    lobo_table = markdown_table(
        ["Method", "Groups", "Mean Accuracy", "Mean Macro-F1", "Min Macro-F1", "Max Macro-F1"],
        [
            [
                row["method"],
                str(row["n_groups"]),
                pct(row["accuracy_mean"]),
                pct(row["macro_f1_mean"]),
                pct(row["macro_f1_min"]),
                pct(row["macro_f1_max"]),
            ]
            for row in lobo_agg
        ],
    )
    lomo_table = markdown_table(
        ["Method", "Directions", "Mean Accuracy", "Mean Macro-F1", "Min Macro-F1", "Max Macro-F1"],
        [
            [
                row["method"],
                str(row["n_groups"]),
                pct(row["accuracy_mean"]),
                pct(row["macro_f1_mean"]),
                pct(row["macro_f1_min"]),
                pct(row["macro_f1_max"]),
            ]
            for row in lomo_agg
        ],
    )

    def row_ref(row: dict[str, Any] | None) -> str:
        if not row:
            return "N/A"
        return f"`{row['group']}` / `{row['method']}`: Macro-F1 `{pct(row['macro_f1'])}`, Accuracy `{pct(row['accuracy'])}`"

    lines = [
        "# Grouped Encoder Evaluation Summary",
        "",
        "## Scope",
        "",
        "This is a **frozen grouped embedding evaluation** for the **v6.2-A frozen ESPI encoder candidate**. It uses only the saved embeddings and metadata. It is **not a validated Physics-Aligned Encoder yet** and does not retrain any CNN.",
        "",
        "## Source feature dump",
        f"- Feature dump: `{features['features_path']}`",
        f"- Samples: `{len(features['labels'])}`",
        f"- Embedding dimension: `{features['embeddings'].shape[1]}`",
        f"- Checkpoint: `{features['checkpoint']}`",
        f"- Embedding layer: `{features['embedding_layer']}`",
        f"- Smoke test: `{smoke_test}`",
        "",
        "## Methods used",
        "",
        "\n".join(f"- `{method}`" for method in methods) if methods else "- No methods completed.",
        "",
        "All folds standardize features using only the reference/train fold and then apply the transform to the held-out test fold.",
        "",
        "## Board-level LOBO-style summary",
        "",
        lobo_table,
        "",
        f"- Best board result: {row_ref(best_lobo['best'])}",
        f"- Worst board result: {row_ref(best_lobo['worst'])}",
        "",
        "## Material-level LOMO-style summary",
        "",
        lomo_table,
        "",
        f"- Best material direction: {row_ref(best_lomo['best'])}",
        f"- Worst material direction: {row_ref(best_lomo['worst'])}",
        "",
        "## Automatic interpretation",
        "",
        "This evaluation tests whether the already extracted v6.2-A embedding space remains useful when the reference set excludes a board/material. It is stricter than a stratified retrieval/probe audit, but it is still not a new CNN training run.",
    ]

    if lobo_agg:
        best_macro = max(float(row["macro_f1_mean"]) for row in lobo_agg)
        if best_macro < 0.75:
            lines.append("- Board-grouped Macro-F1 drops substantially, indicating distribution-shift sensitivity and motivating contrastive/group-aware fine-tuning.")
        else:
            lines.append("- Board-grouped Macro-F1 remains relatively high, supporting the frozen ESPI encoder candidate framing.")

    if lomo_agg:
        best_macro = max(float(row["macro_f1_mean"]) for row in lomo_agg)
        if best_macro < 0.75:
            lines.append("- Material-grouped Macro-F1 is weak, indicating material shift sensitivity.")
        else:
            lines.append("- Material-grouped Macro-F1 remains relatively high, supporting cross-material embedding usefulness.")

    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "This is a strict grouped evaluation over frozen embeddings, but not a new CNN training run. It tests whether the already extracted v6.2-A embedding space remains useful when the reference set excludes a board/material.",
        ]
    )

    if skipped:
        lines.extend(["", "## Skipped methods", ""])
        lines.extend(f"- `{row['evaluation']}` / `{row['group']}` / `{row['method']}`: {row['reason']}" for row in skipped)

    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {message}" for message in warnings)

    path.write_text("\n".join(lines), encoding="utf-8")


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    warnings: list[str] = []
    skipped: list[dict[str, str]] = []

    features = load_feature_dump(args.features, warnings)

    if args.smoke_test:
        idx = smoke_subset_indices(features["board"], features["material"], args.smoke_per_board)
        for key in ["embeddings", "labels", "board", "material", "domain", "path", "frequency"]:
            features[key] = features[key][idx]

    embeddings = features["embeddings"]
    labels = features["labels"]
    board = features["board"].astype(str)
    material = features["material"].astype(str)
    n_classes = int(labels.max()) + 1 if len(labels) else 0
    out_dir = SMOKE_OUT if args.smoke_test else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    lobo_rows: list[dict[str, Any]] = []
    for board_value in sorted(value for value in set(board.tolist()) if value and value.lower() != "unknown"):
        test_mask = board == board_value
        train_mask = ~test_mask
        lobo_rows.extend(
            evaluate_fold(
                evaluation="lobo_board",
                group_name=board_value,
                train_mask=train_mask,
                test_mask=test_mask,
                embeddings=embeddings,
                labels=labels,
                n_classes=n_classes,
                out_dir=out_dir,
                k=args.k,
                block_size=args.block_size,
                warnings=warnings,
                skipped=skipped,
            )
        )

    lomo_rows: list[dict[str, Any]] = []
    material_values = {value.lower() for value in material.tolist()}
    directions = []
    if "wood" in material_values and "carbon" in material_values:
        directions = [("wood_to_carbon", "wood", "carbon"), ("carbon_to_wood", "carbon", "wood")]
    else:
        reason = f"LOMO skipped: expected wood and carbon, found {sorted(material_values)}"
        warn(reason, warnings)
        skipped.append({"evaluation": "lomo_material", "group": "all", "method": "all", "reason": reason})

    material_lower = np.array([value.lower() for value in material.tolist()])
    for direction_name, train_material, test_material in directions:
        train_mask = material_lower == train_material
        test_mask = material_lower == test_material
        lomo_rows.extend(
            evaluate_fold(
                evaluation="lomo_material",
                group_name=direction_name,
                train_mask=train_mask,
                test_mask=test_mask,
                embeddings=embeddings,
                labels=labels,
                n_classes=n_classes,
                out_dir=out_dir,
                k=args.k,
                block_size=args.block_size,
                warnings=warnings,
                skipped=skipped,
            )
        )

    lobo_agg = aggregate_by_method(lobo_rows)
    lomo_agg = aggregate_by_method(lomo_rows)

    write_csv(out_dir / "lobo_grouped_summary.csv", lobo_rows)
    write_csv(out_dir / "lomo_summary.csv", lomo_rows)
    write_csv(out_dir / "lobo_grouped_aggregate.csv", lobo_agg)
    write_csv(out_dir / "lomo_aggregate.csv", lomo_agg)

    key_numbers = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_feature_dump": str(args.features),
        "output_dir": str(out_dir),
        "smoke_test": bool(args.smoke_test),
        "n_samples": int(len(labels)),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "n_classes": n_classes,
        "boards": sorted(set(board.tolist())),
        "materials": sorted(set(material_lower.tolist())),
        "methods": sorted({row["method"] for row in [*lobo_rows, *lomo_rows]}),
        "lobo_aggregate": lobo_agg,
        "lomo_aggregate": lomo_agg,
        "best_worst_lobo_knn": best_worst(lobo_rows, method=f"knn_cosine_k{args.k}"),
        "best_worst_lomo_knn": best_worst(lomo_rows, method=f"knn_cosine_k{args.k}"),
        "skipped": skipped,
        "warnings": warnings,
        "wording": {
            "use": [
                "frozen grouped embedding evaluation",
                "v6.2-A frozen ESPI encoder candidate",
                "not a validated Physics-Aligned Encoder yet",
            ],
            "avoid": [
                "true validated encoder",
                "final LOBO proof",
                "metamaterials-ready encoder",
            ],
        },
    }

    (out_dir / "grouped_encoder_key_numbers.json").write_text(
        json.dumps(key_numbers, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_summary(
        path=out_dir / "GROUPED_ENCODER_EVAL_SUMMARY.md",
        features=features,
        out_dir=out_dir,
        lobo_rows=lobo_rows,
        lomo_rows=lomo_rows,
        lobo_agg=lobo_agg,
        lomo_agg=lomo_agg,
        skipped=skipped,
        warnings=warnings,
        smoke_test=args.smoke_test,
    )

    return key_numbers


def print_requested_summary(key_numbers: dict[str, Any]) -> None:
    print(f"Output directory: {key_numbers['output_dir']}")

    print("LOBO mean macro F1 per method:")
    for row in key_numbers.get("lobo_aggregate", []):
        print(f"  {row['method']}: {pct(row['macro_f1_mean'])}")

    print("LOMO macro F1 per direction/method:")
    out_dir = Path(key_numbers["output_dir"])
    lomo_path = out_dir / "lomo_summary.csv"
    if lomo_path.exists() and lomo_path.stat().st_size > 0:
        with lomo_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                print(f"  {row['group']} / {row['method']}: {pct(row['macro_f1'])}")
    else:
        print("  none")

    skipped = key_numbers.get("skipped", [])
    print("Skipped methods:")
    if skipped:
        for row in skipped:
            print(f"  {row['evaluation']} / {row['group']} / {row['method']}: {row['reason']}")
    else:
        print("  none")

    warnings = key_numbers.get("warnings", [])
    if warnings:
        print("Warnings:")
        for message in warnings:
            print(f"  {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate grouped generalization using saved frozen embeddings only.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-per-board", type=int, default=30)
    args = parser.parse_args()

    key_numbers = run_evaluation(args)
    print_requested_summary(key_numbers)


if __name__ == "__main__":
    main()
