#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit frozen v6.2-A embeddings.

Inputs:
  features_v62a_epoch25.npz from extract_v62_embeddings.py

Outputs:
  embedding_audit_v001/
    knn_report.csv
    linear_probe_report.csv
    prototype_report.csv
    lobo_retrieval_report.csv
    distance_report.csv
    pca_by_class.png / pca_by_board.png / pca_by_material.png when matplotlib is available
    EMBEDDING_AUDIT_SUMMARY.md
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


LABEL_NAMES = {
    0: "1_1H",
    1: "1_1T",
    2: "1_2",
    3: "2_1",
    4: "higher",
}


def normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= true_label < n_classes and 0 <= pred_label < n_classes:
            cm[true_label, pred_label] += 1
    return cm


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    per_class = []
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
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    accuracy = float(np.trace(cm) / max(cm.sum(), 1))
    return {
        "accuracy": accuracy,
        "macro_recall": float(np.mean([row["recall"] for row in per_class])),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "per_class": per_class,
        "cm": cm,
    }


def majority_vote(top_labels: np.ndarray) -> np.ndarray:
    preds = []
    for row in top_labels:
        counts = Counter(row.tolist())
        preds.append(max(counts.items(), key=lambda item: (item[1], -item[0]))[0])
    return np.array(preds, dtype=np.int64)


def knn_predict_cosine(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    k_values: Iterable[int],
    block_size: int = 512,
) -> dict[int, np.ndarray]:
    if len(train_x) == 0 or len(test_x) == 0:
        return {}
    k_values = sorted({int(k) for k in k_values if int(k) <= len(train_x)})
    if not k_values:
        return {}
    max_k = max(k_values)
    predictions = {k: [] for k in k_values}
    train_x = normalize(train_x.astype(np.float32))
    test_x = normalize(test_x.astype(np.float32))

    for start in range(0, len(test_x), block_size):
        block = test_x[start : start + block_size]
        scores = block @ train_x.T
        top_idx = np.argpartition(-scores, kth=max_k - 1, axis=1)[:, :max_k]
        top_scores = np.take_along_axis(scores, top_idx, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_labels = train_y[top_idx]
        for k in k_values:
            predictions[k].append(majority_vote(top_labels[:, :k]))

    return {k: np.concatenate(parts, axis=0) for k, parts in predictions.items()}


def prototype_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    train_x = normalize(train_x.astype(np.float32))
    test_x = normalize(test_x.astype(np.float32))
    prototypes = []
    valid_classes = []
    for class_id in range(n_classes):
        mask = train_y == class_id
        if np.any(mask):
            proto = normalize(train_x[mask].mean(axis=0, keepdims=True))[0]
            prototypes.append(proto)
            valid_classes.append(class_id)
    prototype_matrix = np.vstack(prototypes)
    scores = test_x @ prototype_matrix.T
    pred_local = np.argmax(scores, axis=1)
    return np.array([valid_classes[idx] for idx in pred_local], dtype=np.int64), prototype_matrix


def distance_report(x: np.ndarray, y: np.ndarray, n_classes: int) -> list[dict[str, object]]:
    x = normalize(x.astype(np.float32))
    rows = []
    prototypes = {}
    for class_id in range(n_classes):
        mask = y == class_id
        if not np.any(mask):
            continue
        class_x = x[mask]
        proto = normalize(class_x.mean(axis=0, keepdims=True))[0]
        prototypes[class_id] = proto
        cosine = class_x @ proto
        rows.append(
            {
                "class": class_id,
                "label_name": LABEL_NAMES.get(class_id, str(class_id)),
                "support": int(mask.sum()),
                "mean_cosine_to_prototype": float(np.mean(cosine)),
                "mean_cosine_distance_to_prototype": float(np.mean(1.0 - cosine)),
            }
        )

    for class_id, proto in prototypes.items():
        other_scores = [
            float(proto @ other_proto)
            for other_id, other_proto in prototypes.items()
            if other_id != class_id
        ]
        row = next(item for item in rows if item["class"] == class_id)
        row["max_other_prototype_cosine"] = max(other_scores) if other_scores else float("nan")
        row["nearest_other_class"] = max(
            ((other_id, float(proto @ other_proto)) for other_id, other_proto in prototypes.items() if other_id != class_id),
            key=lambda item: item[1],
            default=(-1, float("nan")),
        )[0]
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def support_rows(
    labels: np.ndarray,
    split: np.ndarray,
    split_group: np.ndarray,
    material: np.ndarray,
    domain: np.ndarray,
    n_classes: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add_counts(scope: str, value: str, mask: np.ndarray) -> None:
        counts = np.bincount(labels[mask].astype(int), minlength=n_classes)
        row: dict[str, object] = {
            "scope": scope,
            "value": value,
            "n_samples": int(mask.sum()),
        }
        for class_id in range(n_classes):
            row[f"class_{class_id}_{LABEL_NAMES.get(class_id, str(class_id))}"] = int(counts[class_id])
        rows.append(row)

    add_counts("all", "all", np.ones(len(labels), dtype=bool))
    for value in sorted(set(split.tolist())):
        add_counts("split", value, split == value)
    for value in sorted(set(split_group.tolist())):
        add_counts("split_group", value, split_group == value)
    for value in sorted(set(material.tolist())):
        add_counts("material", value, material == value)
    for value in sorted(set(domain.tolist())):
        add_counts("domain", value, domain == value)
    return rows


def quality_rows(data: np.lib.npyio.NpzFile, x: np.ndarray) -> list[dict[str, object]]:
    paths = data["path"].astype(str) if "path" in data.files else np.array([], dtype=str)
    duplicate_paths = int(len(paths) - len(set(paths.tolist()))) if len(paths) else 0
    return [
        {"check": "n_samples", "value": int(len(x)), "status": "info"},
        {"check": "embedding_dim", "value": int(x.shape[1]), "status": "info"},
        {"check": "embedding_nan_count", "value": int(np.isnan(x).sum()), "status": "pass" if not np.isnan(x).any() else "fail"},
        {"check": "embedding_inf_count", "value": int(np.isinf(x).sum()), "status": "pass" if not np.isinf(x).any() else "fail"},
        {"check": "duplicate_path_count", "value": duplicate_paths, "status": "pass" if duplicate_paths == 0 else "warn"},
    ]


def linear_probe(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, n_classes: int) -> dict[str, object] | None:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=1, multi_class="auto"),
    )
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    return metrics_from_predictions(test_y, pred.astype(np.int64), n_classes)


def pca_2d(x: np.ndarray, max_points: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    if len(x) > max_points:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(x), size=max_points, replace=False))
    else:
        idx = np.arange(len(x))
    xs = x[idx].astype(np.float32)
    xs = xs - xs.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(xs, full_matrices=False)
    coords = xs @ vt[:2].T
    return coords, idx


def write_pca_plots(out_dir: Path, x: np.ndarray, metadata: dict[str, np.ndarray]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    coords, idx = pca_2d(x)
    outputs = []
    for field in ["label_name", "board", "material", "domain"]:
        values = metadata[field][idx].astype(str)
        unique_values = sorted(set(values.tolist()))
        value_to_id = {value: i for i, value in enumerate(unique_values)}
        colors = np.array([value_to_id[value] for value in values])

        plt.figure(figsize=(8, 6), dpi=150)
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=colors, s=5, alpha=0.75, cmap="tab20")
        handles, _ = scatter.legend_elements(num=min(len(unique_values), 20))
        plt.legend(handles, unique_values[: len(handles)], loc="best", fontsize=7)
        plt.title(f"PCA of frozen v6.2-A embeddings by {field}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        path = out_dir / f"pca_by_{field}.png"
        plt.savefig(path)
        plt.close()
        outputs.append(str(path))
    return outputs


def append_metric_row(rows: list[dict[str, object]], evaluation: str, method: str, metrics: dict[str, object], extra: dict[str, object] | None = None) -> None:
    row = {
        "evaluation": evaluation,
        "method": method,
        "accuracy": metrics["accuracy"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "n_samples": int(metrics["cm"].sum()),
    }
    if extra:
        row.update(extra)
    rows.append(row)


def aggregate_metric_rows(rows: list[dict[str, object]], evaluation: str) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        selected = [row for row in rows if str(row["method"]) == method]
        if not selected:
            continue
        accuracies = np.array([float(row["accuracy"]) for row in selected], dtype=np.float64)
        macro_f1s = np.array([float(row["macro_f1"]) for row in selected], dtype=np.float64)
        macro_recalls = np.array([float(row["macro_recall"]) for row in selected], dtype=np.float64)
        output.append(
            {
                "evaluation": evaluation,
                "method": method,
                "n_groups": len(selected),
                "accuracy_mean": float(accuracies.mean()),
                "accuracy_min": float(accuracies.min()),
                "accuracy_max": float(accuracies.max()),
                "macro_recall_mean": float(macro_recalls.mean()),
                "macro_recall_min": float(macro_recalls.min()),
                "macro_recall_max": float(macro_recalls.max()),
                "macro_f1_mean": float(macro_f1s.mean()),
                "macro_f1_min": float(macro_f1s.min()),
                "macro_f1_max": float(macro_f1s.max()),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen v6.2-A embeddings.")
    parser.add_argument("--features", type=Path, default=Path("features_v62a_epoch25.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("embedding_audit_v001"))
    parser.add_argument("--k", default="1,3,5,10")
    parser.add_argument("--skip-linear", action="store_true")
    parser.add_argument("--block-size", type=int, default=512)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(str(args.features), allow_pickle=True)
    x = normalize(data["embeddings"].astype(np.float32))
    y = data["label"].astype(np.int64)
    split = data["split"].astype(str)
    split_group = data["split_group"].astype(str)
    n_classes = int(np.max(y)) + 1
    k_values = [int(part.strip()) for part in args.k.split(",") if part.strip()]

    metadata = {
        "label_name": data["label_name"].astype(str),
        "board": data["board"].astype(str),
        "material": data["material"].astype(str),
        "domain": data["domain"].astype(str),
    }
    quality = quality_rows(data, x)
    supports = support_rows(y, split, split_group, metadata["material"], metadata["domain"], n_classes)

    knn_rows: list[dict[str, object]] = []
    prototype_rows: list[dict[str, object]] = []
    linear_rows: list[dict[str, object]] = []
    lobo_rows: list[dict[str, object]] = []

    train_mask = split == "train"
    val_mask = split == "val"
    if np.any(train_mask) and np.any(val_mask):
        knn_preds = knn_predict_cosine(x[train_mask], y[train_mask], x[val_mask], k_values, args.block_size)
        for k_value, preds in knn_preds.items():
            metrics = metrics_from_predictions(y[val_mask], preds, n_classes)
            append_metric_row(knn_rows, "train_to_val", f"knn_cosine_k{k_value}", metrics, {"k": k_value})

        proto_pred, _ = prototype_predict(x[train_mask], y[train_mask], x[val_mask], n_classes)
        proto_metrics = metrics_from_predictions(y[val_mask], proto_pred, n_classes)
        append_metric_row(prototype_rows, "train_to_val", "nearest_class_prototype_cosine", proto_metrics)

        if not args.skip_linear:
            linear_metrics = linear_probe(x[train_mask], y[train_mask], x[val_mask], y[val_mask], n_classes)
            if linear_metrics is not None:
                append_metric_row(linear_rows, "train_to_val", "logistic_regression_balanced", linear_metrics)

    for group in sorted(set(split_group.tolist())):
        test_mask = split_group == group
        train_group_mask = ~test_mask
        if int(test_mask.sum()) < n_classes or int(train_group_mask.sum()) < n_classes:
            continue
        knn_preds = knn_predict_cosine(x[train_group_mask], y[train_group_mask], x[test_mask], k_values, args.block_size)
        for k_value, preds in knn_preds.items():
            metrics = metrics_from_predictions(y[test_mask], preds, n_classes)
            append_metric_row(
                lobo_rows,
                "leave_split_group_out",
                f"knn_cosine_k{k_value}",
                metrics,
                {"heldout_group": group, "k": k_value},
            )

        proto_pred, _ = prototype_predict(x[train_group_mask], y[train_group_mask], x[test_mask], n_classes)
        proto_metrics = metrics_from_predictions(y[test_mask], proto_pred, n_classes)
        append_metric_row(
            lobo_rows,
            "leave_split_group_out",
            "nearest_class_prototype_cosine",
            proto_metrics,
            {"heldout_group": group, "k": ""},
        )

    dist_rows = distance_report(x, y, n_classes)

    write_csv(args.out_dir / "knn_report.csv", knn_rows)
    write_csv(args.out_dir / "linear_probe_report.csv", linear_rows)
    write_csv(args.out_dir / "prototype_report.csv", prototype_rows)
    write_csv(args.out_dir / "lobo_retrieval_report.csv", lobo_rows)
    write_csv(args.out_dir / "leave_split_group_out_report.csv", lobo_rows)
    write_csv(args.out_dir / "leave_split_group_out_aggregate.csv", aggregate_metric_rows(lobo_rows, "leave_split_group_out"))
    write_csv(args.out_dir / "distance_report.csv", dist_rows)
    write_csv(args.out_dir / "class_supports_by_split.csv", supports)
    write_csv(args.out_dir / "embedding_quality_report.csv", quality)
    plot_paths = write_pca_plots(args.out_dir, x, metadata)

    summary = {
        "features": str(args.features),
        "n_samples": int(len(x)),
        "embedding_dim": int(x.shape[1]),
        "n_classes": n_classes,
        "generalization_caveat": (
            "leave_split_group_out removes the held-out group only from the retrieval/probe reference set; "
            "the frozen v6.2-A checkpoint itself was trained on the original stratified split, so this is "
            "post-hoc representation-geometry evidence, not a true LOBO-trained encoder result."
        ),
        "split_counts": dict(Counter(split.tolist())),
        "split_group_counts": dict(Counter(split_group.tolist())),
        "knn_rows": len(knn_rows),
        "prototype_rows": len(prototype_rows),
        "linear_probe_rows": len(linear_rows),
        "lobo_rows": len(lobo_rows),
        "quality": quality,
        "plots": plot_paths,
    }
    (args.out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# v6.2-A Encoder Evidence Audit",
        "",
        "## 1. Source checkpoint",
        f"- Checkpoint: `{data['checkpoint_path'].item() if 'checkpoint_path' in data.files else 'unknown'}`",
        "- Baseline reference: `96.18% Accuracy / 90.97% Macro-F1`",
        f"- Embedding layer: `{data['embedding_layer'].item() if 'embedding_layer' in data.files else 'unknown'}`",
        f"- Embedding dimension: `{x.shape[1]}`",
        "",
        "## 2. Dataset / manifest",
        f"- Features: `{args.features}`",
        f"- Manifest: `{data['manifest_path'].item() if 'manifest_path' in data.files else 'unknown'}`",
        f"- Samples: `{len(x)}`",
        f"- Split counts: `{dict(Counter(split.tolist()))}`",
        f"- Split group definition: `board`, falling back to source group when board cannot be parsed",
        "",
        "## 3. Embedding extraction audit",
        f"- NaN count: `{next(row['value'] for row in quality if row['check'] == 'embedding_nan_count')}`",
        f"- Inf count: `{next(row['value'] for row in quality if row['check'] == 'embedding_inf_count')}`",
        f"- Duplicate paths: `{next(row['value'] for row in quality if row['check'] == 'duplicate_path_count')}`",
        "- Class supports: `class_supports_by_split.csv`",
        "",
        "## 4. Frozen embedding evaluation",
        "- `knn_report.csv`: train-to-val kNN on frozen embeddings",
        "- `linear_probe_report.csv`: logistic-regression linear probe when scikit-learn is available",
        "- `prototype_report.csv`: nearest class prototype on train-to-val",
        "- `leave_split_group_out_report.csv`: leave-`split_group`-out retrieval/prototype audit",
        "- `leave_split_group_out_aggregate.csv`: grouped mean/min/max summary by method",
        "- `distance_report.csv`: intra-class prototype compactness and nearest prototype separation",
        "",
        "Important caveat: the frozen v6.2-A checkpoint was trained on the original stratified split. The leave-`split_group`-out report removes groups from the retrieval/probe reference set only; it is post-hoc representation-geometry evidence, not a true LOBO-trained encoder result.",
        "",
        "## 5. Grouped structure diagnostics",
        "- PCA plots are diagnostic only, not proof of generalization.",
        "- Use board/material/source-group coloring to check whether embeddings encode grouped dataset signatures.",
        "",
        "## 6. Interpretation Gate",
        "- Strong encoder evidence requires LOBO retrieval/prototype macro-F1 to remain non-collapsed.",
        "- If PCA/UMAP clusters primarily by board/material instead of label, the representation is group-biased.",
        "",
        "## 7. Next step",
        "- If grouped retrieval collapses, run supervised contrastive or group-aware fine-tuning before claiming progress toward a Physics-Aligned Encoder.",
    ]
    if not linear_rows and not args.skip_linear:
        lines.append("- Linear probe skipped because scikit-learn is not available.")
    if plot_paths:
        lines.append("")
        lines.append("## Plots")
        for plot_path in plot_paths:
            lines.append(f"- `{plot_path}`")

    (args.out_dir / "EMBEDDING_AUDIT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[audit] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
