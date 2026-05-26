#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a lightweight English technical report for Encoder Evidence v001.

By default this script only reads existing Markdown/CSV/JSON audit artifacts.
It does not load images, does not use GPU, and does not rerun feature
extraction or training. With --make-plots it reads the existing NPZ once,
samples at most 3000 embeddings, and writes PCA-only diagnostic PNGs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
AUDIT_DIR = ROOT / "embedding_audit_v001"

FEATURE_SUMMARY = ROOT / "features_v62a_epoch25.FEATURE_DUMP_SUMMARY.md"
FEATURES_NPZ = ROOT / "features_v62a_epoch25.npz"
METADATA_CSV = ROOT / "features_v62a_epoch25.metadata.csv"
AUDIT_SUMMARY = AUDIT_DIR / "EMBEDDING_AUDIT_SUMMARY.md"
AUDIT_JSON = AUDIT_DIR / "audit_summary.json"
KNN_CSV = AUDIT_DIR / "knn_report.csv"
LINEAR_CSV = AUDIT_DIR / "linear_probe_report.csv"
PROTOTYPE_CSV = AUDIT_DIR / "prototype_report.csv"
GROUP_AGG_CSV = AUDIT_DIR / "leave_split_group_out_aggregate.csv"
GROUP_REPORT_CSV = AUDIT_DIR / "leave_split_group_out_report.csv"
QUALITY_CSV = AUDIT_DIR / "embedding_quality_report.csv"

REPORT_PATH = AUDIT_DIR / "FROZEN_ENCODER_TECHNICAL_REPORT.md"
KEY_NUMBERS_PATH = AUDIT_DIR / "encoder_report_key_numbers.json"
PCA_PLOT_PATHS = {
    "class": AUDIT_DIR / "pca_by_class.png",
    "board": AUDIT_DIR / "pca_by_board.png",
    "material": AUDIT_DIR / "pca_by_material.png",
}


def warn(message: str, warnings: list[str]) -> None:
    warnings.append(message)
    print(f"[warn] {message}")


def read_text(path: Path, warnings: list[str]) -> str:
    if not path.exists():
        warn(f"Missing file: {path}", warnings)
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.exists():
        warn(f"Missing file: {path}", warnings)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warn(f"Failed to parse JSON {path}: {exc}", warnings)
        return {}


def read_csv(path: Path, warnings: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        warn(f"Missing file: {path}", warnings)
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except Exception as exc:
        warn(f"Failed to parse CSV {path}: {exc}", warnings)
        return []


def read_csv_header(path: Path, warnings: list[str]) -> list[str]:
    if not path.exists():
        warn(f"Missing file: {path}", warnings)
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            return next(reader, [])
    except Exception as exc:
        warn(f"Failed to read CSV header {path}: {exc}", warnings)
        return []


def parse_backtick_int(text: str, label: str) -> int | None:
    match = re.search(rf"- {re.escape(label)}:\s*`?([0-9]+)`?", text)
    return int(match.group(1)) if match else None


def parse_load_audit(text: str) -> dict[str, int | None]:
    return {
        "missing_keys": parse_backtick_int(text, "Missing keys"),
        "unexpected_keys": parse_backtick_int(text, "Unexpected keys"),
    }


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def pct(value: Any) -> str:
    return f"{100.0 * to_float(value):.2f}%"


def best_row(rows: list[dict[str, str]], metric: str = "macro_f1", method_contains: str | None = None) -> dict[str, str]:
    selected = rows
    if method_contains is not None:
        selected = [row for row in rows if method_contains in row.get("method", "")]
    if not selected:
        return {}
    return max(selected, key=lambda row: to_float(row.get(metric, "0")))


def quality_map(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {row.get("check", ""): row.get("value", "") for row in rows}


def metadata_fields(header: list[str]) -> list[str]:
    preferred = [
        "sample_id",
        "label",
        "label_name",
        "board",
        "source_group",
        "material",
        "frequency_hz",
        "domain",
        "path",
        "split",
        "split_group",
    ]
    existing = [field for field in preferred if field in header or field == "embedding"]
    extras = [field for field in header if field not in existing]
    return existing + extras


def method_label(row: dict[str, str]) -> str:
    method = row.get("method", "unknown")
    if method.startswith("knn_cosine_k"):
        return f"kNN cosine k={method.removeprefix('knn_cosine_k')}"
    if method == "logistic_regression_balanced":
        return "Linear probe - balanced logistic regression"
    if method == "nearest_class_prototype_cosine":
        return "Nearest class prototype - cosine"
    return method


def compact_metric(row: dict[str, str], aggregate: bool = False) -> dict[str, Any]:
    if not row:
        return {}
    if aggregate:
        return {
            "method": row.get("method"),
            "method_label": method_label(row),
            "n_groups": to_int(row.get("n_groups")),
            "accuracy_mean": to_float(row.get("accuracy_mean")),
            "accuracy_min": to_float(row.get("accuracy_min")),
            "accuracy_max": to_float(row.get("accuracy_max")),
            "macro_recall_mean": to_float(row.get("macro_recall_mean")),
            "macro_f1_mean": to_float(row.get("macro_f1_mean")),
            "macro_f1_min": to_float(row.get("macro_f1_min")),
            "macro_f1_max": to_float(row.get("macro_f1_max")),
        }
    return {
        "method": row.get("method"),
        "method_label": method_label(row),
        "accuracy": to_float(row.get("accuracy")),
        "macro_recall": to_float(row.get("macro_recall")),
        "macro_f1": to_float(row.get("macro_f1")),
        "n_samples": to_int(row.get("n_samples")),
        "k": row.get("k", ""),
    }


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    sep = ["---"] * len(header)
    body = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    body.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(body)


def stratified_sample_indices(labels: Any, max_samples: int = 3000, seed: int = 42) -> Any:
    import numpy as np

    labels = np.asarray(labels)
    n_samples = len(labels)
    if n_samples <= max_samples:
        return np.arange(n_samples, dtype=np.int64)

    rng = np.random.default_rng(seed)
    unique_labels, counts = np.unique(labels, return_counts=True)
    allocations: dict[Any, int] = {}
    remainders: list[tuple[float, Any]] = []

    for label, count in zip(unique_labels, counts):
        exact = max_samples * (int(count) / n_samples)
        allocated = max(1, int(exact))
        allocated = min(allocated, int(count))
        allocations[label] = allocated
        remainders.append((exact - int(exact), label))

    while sum(allocations.values()) > max_samples:
        candidates = [
            (allocations[label], label)
            for label in unique_labels
            if allocations[label] > 1
        ]
        if not candidates:
            break
        _, label_to_reduce = max(candidates)
        allocations[label_to_reduce] -= 1

    while sum(allocations.values()) < max_samples:
        grew = False
        for _, label in sorted(remainders, reverse=True):
            label_count = int(counts[list(unique_labels).index(label)])
            if allocations[label] < label_count:
                allocations[label] += 1
                grew = True
                if sum(allocations.values()) == max_samples:
                    break
        if not grew:
            break

    selected_parts = []
    for label in unique_labels:
        class_indices = np.flatnonzero(labels == label)
        n_take = allocations[label]
        selected_parts.append(rng.choice(class_indices, size=n_take, replace=False))
    selected = np.concatenate(selected_parts)
    selected.sort()
    return selected.astype(np.int64)


def pca_2d(embeddings: Any) -> Any:
    import numpy as np

    x = np.asarray(embeddings, dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def make_pca_plots(warnings: list[str], max_samples: int = 3000) -> dict[str, Any]:
    if not FEATURES_NPZ.exists():
        warn(f"Missing file: {FEATURES_NPZ}", warnings)
        return {"enabled": True, "created": {}, "sampled": 0, "total": 0}

    try:
        import numpy as np
    except Exception as exc:
        warn(f"Failed to import numpy for PCA plots: {exc}", warnings)
        return {"enabled": True, "created": {}, "sampled": 0, "total": 0}

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        warn(f"Failed to import matplotlib for PCA plots: {exc}", warnings)
        return {"enabled": True, "created": {}, "sampled": 0, "total": 0}

    try:
        data = np.load(str(FEATURES_NPZ), allow_pickle=True)
    except Exception as exc:
        warn(f"Failed to load NPZ for PCA plots: {exc}", warnings)
        return {"enabled": True, "created": {}, "sampled": 0, "total": 0}

    if "embeddings" not in data.files or "label" not in data.files:
        warn("NPZ missing required arrays: embeddings and/or label", warnings)
        return {"enabled": True, "created": {}, "sampled": 0, "total": 0}

    labels = data["label"].astype(int)
    selected = stratified_sample_indices(labels, max_samples=max_samples)
    coords = pca_2d(data["embeddings"][selected])
    total_samples = int(len(labels))
    sampled = int(len(selected))

    label_names = (
        data["label_name"].astype(str)
        if "label_name" in data.files
        else np.array([str(label) for label in labels], dtype=str)
    )
    board = (
        data["board"].astype(str)
        if "board" in data.files
        else np.array(["unknown"] * total_samples, dtype=str)
    )
    material = (
        data["material"].astype(str)
        if "material" in data.files
        else np.array(["unknown"] * total_samples, dtype=str)
    )

    plot_specs = {
        "class": label_names[selected],
        "board": board[selected],
        "material": material[selected],
    }

    created: dict[str, str] = {}
    for field, values in plot_specs.items():
        unique_values = sorted(set(values.tolist()))
        value_to_color = {value: idx for idx, value in enumerate(unique_values)}
        colors = np.array([value_to_color[value] for value in values])

        plt.figure(figsize=(8, 6), dpi=150)
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=colors, s=6, alpha=0.75, cmap="tab20")
        handles, _ = scatter.legend_elements(num=min(len(unique_values), 20))
        if handles:
            plt.legend(handles, unique_values[: len(handles)], loc="best", fontsize=7)
        plt.title(f"PCA of frozen v6.2-A embeddings by {field}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()

        out_path = PCA_PLOT_PATHS[field]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        created[field] = str(out_path)

    return {
        "enabled": True,
        "source_npz": str(FEATURES_NPZ),
        "created": created,
        "sampled": sampled,
        "total": total_samples,
        "method": "PCA 2D via numpy.linalg.svd",
    }


def build_report(key: dict[str, Any]) -> str:
    metadata = ", ".join(f"`{field}`" for field in key["metadata_fields"])
    load_audit = key["load_audit"]
    quality = key["quality_checks"]
    pca_plots = key.get("pca_plots", {})
    plot_lines = ""
    if pca_plots.get("created"):
        plot_items = "\n".join(
            f"- `{path}`"
            for path in pca_plots["created"].values()
        )
        plot_lines = (
            "\n\nOptional PCA diagnostics were generated with CPU-only SVD "
            f"on `{pca_plots.get('sampled')}` of `{pca_plots.get('total')}` embeddings:\n"
            f"{plot_items}"
        )

    eval_rows = [
        ["Evaluation", "Method", "Accuracy", "Macro Recall", "Macro-F1", "Comment"],
        [
            "Stratified train-to-val",
            key["knn_best"].get("method_label", "N/A"),
            pct(key["knn_best"].get("accuracy")),
            pct(key["knn_best"].get("macro_recall")),
            pct(key["knn_best"].get("macro_f1")),
            "Best kNN result over frozen embeddings.",
        ],
        [
            "Stratified train-to-val",
            key["linear_probe"].get("method_label", "N/A"),
            pct(key["linear_probe"].get("accuracy")),
            pct(key["linear_probe"].get("macro_recall")),
            pct(key["linear_probe"].get("macro_f1")),
            "Tests linear availability of representation information.",
        ],
        [
            "Stratified train-to-val",
            key["prototype"].get("method_label", "N/A"),
            pct(key["prototype"].get("accuracy")),
            pct(key["prototype"].get("macro_recall")),
            pct(key["prototype"].get("macro_f1")),
            "Tests whether classes form clean prototype centers.",
        ],
        [
            "Leave-split-group-out",
            key["group_knn_best"].get("method_label", "N/A"),
            pct(key["group_knn_best"].get("accuracy_mean")),
            pct(key["group_knn_best"].get("macro_recall_mean")),
            pct(key["group_knn_best"].get("macro_f1_mean")),
            f"Mean over {key['group_knn_best'].get('n_groups', 0)} groups.",
        ],
        [
            "Leave-split-group-out",
            "Minimum grouped Macro-F1",
            "-",
            "-",
            pct(key["minimum_grouped_macro_f1"]),
            "Worst group for the best aggregate kNN method.",
        ],
    ]

    return f"""# Encoder Evidence v001 - Frozen v6.2-A Embeddings

## 1. Source checkpoint and scope

This report summarizes **post-hoc representation-geometry evidence** from the frozen embedding space of the **v6.2-A reportable baseline**. The source is the epoch-25 checkpoint `{key['checkpoint']}`, which is associated with the internal reference result `96.18% Accuracy / 90.97% Macro-F1`.

No new classifier was trained, no new feature extraction is implied by this report, and the result must not be presented as **true LOBO generalization proof** or as a fully validated Physics-Aligned Encoder.

The technical objective is narrow: evaluate whether frozen pre-head embeddings from v6.2-A retain useful modal structure after removing the final classifier head.

## 2. Feature extraction audit

- Samples: `{key['total_samples']}`
- Embedding dimension: `{key['embedding_dimension']}`
- Embedding layer: `MCDropoutClassifier.global_pool`
- Load audit: missing keys `{load_audit.get('missing_keys')}`, unexpected keys `{load_audit.get('unexpected_keys')}`
- Metadata fields: {metadata}
- NaN embeddings: `{quality.get('embedding_nan_count', 'N/A')}`
- Inf embeddings: `{quality.get('embedding_inf_count', 'N/A')}`
- Duplicate paths: `{quality.get('duplicate_path_count', 'N/A')}`
- The report generator does not reload raw images, run GPU inference, retrain the CNN, or duplicate the embedding array.

## 3. Frozen embedding evaluation

{markdown_table(eval_rows)}
{plot_lines}

## 4. Interpretation

The frozen v6.2-A embedding space shows strong post-hoc modal geometry. The best kNN result and the linear probe are close to or above the classifier reference Macro-F1 of the v6.2-A reportable baseline. This supports using v6.2-A as a **frozen ESPI encoder candidate**.

The prototype result is lower but still informative: it indicates that class information exists in the embedding space, while pure centroid geometry is not equally strong for all classes.

## 5. Caveat

This is **not true LOBO/LOMO encoder proof**. The frozen checkpoint was trained on the original stratified split. The leave-split-group audit removes groups from the retrieval/probe reference set, but the frozen encoder itself was not trained with the target board or material excluded.

The result should therefore be framed as **post-hoc representation-geometry evidence**, not as definitive LOBO/LOMO generalization proof.

## 6. Next step

The next strict validation step is one of the following:

- true LOBO/LOMO encoder training, where the target board or material is excluded during training,
- supervised contrastive or domain-aware fine-tuning, so samples from the same modal class but different boards/materials are pulled together while frequency-adjacent hard negatives remain separable.

## 7. Appendix-ready claim

Frozen pre-head embeddings extracted from the v6.2-A reportable baseline retain strong classification and modal information without retraining the CNN or using the final classifier head. kNN, prototype, and linear-probe results provide post-hoc representation-geometry evidence that v6.2-A can be used as a **frozen ESPI encoder candidate**. Because the checkpoint was trained on the original stratified split, the grouped retrieval/probe results are not yet true LOBO/LOMO generalization proof. Strict validation requires LOBO/LOMO encoder training or supervised contrastive/domain-aware fine-tuning.
"""


def main() -> None:
    global AUDIT_DIR
    global FEATURE_SUMMARY
    global FEATURES_NPZ
    global METADATA_CSV
    global AUDIT_SUMMARY
    global AUDIT_JSON
    global KNN_CSV
    global LINEAR_CSV
    global PROTOTYPE_CSV
    global GROUP_AGG_CSV
    global GROUP_REPORT_CSV
    global QUALITY_CSV
    global REPORT_PATH
    global KEY_NUMBERS_PATH
    global PCA_PLOT_PATHS

    parser = argparse.ArgumentParser(description="Generate an English encoder technical report from existing audit artifacts.")
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR, help="Directory containing audit CSV/JSON/Markdown outputs.")
    parser.add_argument("--feature-summary", type=Path, default=FEATURE_SUMMARY)
    parser.add_argument("--features", type=Path, default=FEATURES_NPZ)
    parser.add_argument("--metadata", type=Path, default=METADATA_CSV)
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    parser.add_argument("--key-numbers", type=Path, default=KEY_NUMBERS_PATH)
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Optionally read the existing NPZ, sample <=3000 embeddings, and save PCA-only PNG diagnostics.",
    )
    args = parser.parse_args()

    AUDIT_DIR = args.audit_dir
    FEATURE_SUMMARY = args.feature_summary
    FEATURES_NPZ = args.features
    METADATA_CSV = args.metadata
    AUDIT_SUMMARY = AUDIT_DIR / "EMBEDDING_AUDIT_SUMMARY.md"
    AUDIT_JSON = AUDIT_DIR / "audit_summary.json"
    KNN_CSV = AUDIT_DIR / "knn_report.csv"
    LINEAR_CSV = AUDIT_DIR / "linear_probe_report.csv"
    PROTOTYPE_CSV = AUDIT_DIR / "prototype_report.csv"
    GROUP_AGG_CSV = AUDIT_DIR / "leave_split_group_out_aggregate.csv"
    GROUP_REPORT_CSV = AUDIT_DIR / "leave_split_group_out_report.csv"
    QUALITY_CSV = AUDIT_DIR / "embedding_quality_report.csv"
    REPORT_PATH = args.out
    KEY_NUMBERS_PATH = args.key_numbers
    PCA_PLOT_PATHS = {
        "class": AUDIT_DIR / "pca_by_class.png",
        "board": AUDIT_DIR / "pca_by_board.png",
        "material": AUDIT_DIR / "pca_by_material.png",
    }

    warnings: list[str] = []

    feature_text = read_text(FEATURE_SUMMARY, warnings)
    audit_text = read_text(AUDIT_SUMMARY, warnings)
    audit_json = read_json(AUDIT_JSON, warnings)

    knn_rows = read_csv(KNN_CSV, warnings)
    linear_rows = read_csv(LINEAR_CSV, warnings)
    prototype_rows = read_csv(PROTOTYPE_CSV, warnings)
    group_agg_rows = read_csv(GROUP_AGG_CSV, warnings)
    group_rows = read_csv(GROUP_REPORT_CSV, warnings)
    quality_rows = read_csv(QUALITY_CSV, warnings)
    metadata_header = read_csv_header(METADATA_CSV, warnings)

    total_samples = parse_backtick_int(feature_text, "Samples") or to_int(audit_json.get("n_samples"))
    embedding_dimension = parse_backtick_int(feature_text, "Embedding dim") or to_int(audit_json.get("embedding_dim"))
    load_audit = parse_load_audit(feature_text)

    checkpoint_match = re.search(r"- Checkpoint:\s*`([^`]+)`", feature_text or audit_text)
    checkpoint = checkpoint_match.group(1) if checkpoint_match else "unknown"

    knn_best = compact_metric(best_row(knn_rows, metric="macro_f1", method_contains="knn"))
    linear_probe = compact_metric(best_row(linear_rows, metric="macro_f1"))
    prototype = compact_metric(best_row(prototype_rows, metric="macro_f1"))
    group_knn_best = compact_metric(best_row(group_agg_rows, metric="macro_f1_mean", method_contains="knn"), aggregate=True)

    minimum_grouped_macro_f1 = group_knn_best.get("macro_f1_min")
    if minimum_grouped_macro_f1 is None and group_rows:
        minimum_grouped_macro_f1 = min(to_float(row.get("macro_f1")) for row in group_rows if "knn" in row.get("method", ""))

    pca_plots = make_pca_plots(warnings) if args.make_plots else {"enabled": False, "created": {}}

    key_numbers = {
        "source_files": {
            "feature_summary": str(FEATURE_SUMMARY),
            "audit_summary": str(AUDIT_SUMMARY),
            "knn_report": str(KNN_CSV),
            "linear_probe_report": str(LINEAR_CSV),
            "prototype_report": str(PROTOTYPE_CSV),
            "leave_split_group_out_aggregate": str(GROUP_AGG_CSV),
            "embedding_quality_report": str(QUALITY_CSV),
            "features_npz_for_optional_plots": str(FEATURES_NPZ),
        },
        "checkpoint": checkpoint,
        "total_samples": total_samples,
        "embedding_dimension": embedding_dimension,
        "embedding_layer": "MCDropoutClassifier.global_pool",
        "load_audit": load_audit,
        "quality_checks": quality_map(quality_rows),
        "metadata_fields": metadata_fields(metadata_header),
        "embedding_storage": "embeddings array in features_v62a_epoch25.npz; not duplicated by this report generator",
        "knn_best": knn_best,
        "linear_probe": linear_probe,
        "prototype": prototype,
        "group_knn_best": group_knn_best,
        "minimum_grouped_macro_f1": minimum_grouped_macro_f1,
        "pca_plots": pca_plots,
        "warnings": warnings,
        "wording_guardrails": {
            "use": [
                "post-hoc representation-geometry evidence",
                "frozen ESPI encoder candidate",
                "v6.2-A reportable baseline",
            ],
            "avoid": [
                "true LOBO generalization proof",
                "validated Physics-Aligned Encoder",
                "hierarchical v6.2 for the 96.18% result",
            ],
        },
    }

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    KEY_NUMBERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    KEY_NUMBERS_PATH.write_text(json.dumps(key_numbers, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_PATH.write_text(build_report(key_numbers), encoding="utf-8")

    print(f"Generated report: {REPORT_PATH}")
    print(f"Generated key numbers: {KEY_NUMBERS_PATH}")
    if pca_plots.get("created"):
        for field, path in pca_plots["created"].items():
            print(f"Generated PCA plot ({field}): {path}")
    print(json.dumps(key_numbers, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
