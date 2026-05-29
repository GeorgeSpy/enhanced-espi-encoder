#!/usr/bin/env python
"""Evaluate frequency_hz-only controls under matched grouped protocols.

This script uses only metadata from an existing normalized feature dump. It
does not use ESPI embeddings, load images, train deep models, implement LeFFT,
or evaluate acoustic-response prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CLASS_IDS = [0, 1, 2, 3, 4]
CLASS_NAMES = {0: "1_1H", 1: "1_1T", 2: "1_2", 3: "2_1", 4: "higher"}
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--composition-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


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


def load_metadata(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        required = ["label", "frequency_hz", "board", "material", "split"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"Missing required feature metadata keys: {missing}")
        label = np.asarray(data["label"]).astype(int)
        label_name = (
            np.asarray(data["label_name"]).astype(str)
            if "label_name" in data.files
            else np.asarray([CLASS_NAMES[int(item)] for item in label], dtype=object)
        )
        return {
            "label": label,
            "label_name": label_name,
            "frequency_hz": np.asarray(data["frequency_hz"]).astype(float),
            "board": np.asarray(data["board"]).astype(str),
            "material": np.asarray(data["material"]).astype(str),
            "split": np.asarray(data["split"]).astype(str),
        }


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    per_class = []
    correct = int(np.sum(y_true == y_pred))
    for class_id in CLASS_IDS:
        true_mask = y_true == class_id
        pred_mask = y_pred == class_id
        support = int(np.sum(true_mask))
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum(~true_mask & pred_mask))
        fn = int(np.sum(true_mask & ~pred_mask))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
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
        "accuracy": correct / len(y_true) if len(y_true) else 0.0,
        "macro_recall": float(np.mean([item["recall"] for item in per_class])),
        "macro_f1": float(np.mean([item["f1"] for item in per_class])),
        "per_class": per_class,
    }


def standardize_train_test(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = float(np.mean(train_x))
    std = float(np.std(train_x))
    if std == 0.0:
        std = 1.0
    return ((train_x - mean) / std).reshape(-1, 1), ((test_x - mean) / std).reshape(-1, 1)


class FrequencyBinMajority:
    def __init__(self, n_bins: int = 20) -> None:
        self.n_bins = n_bins
        self.edges: np.ndarray | None = None
        self.bin_labels: dict[int, int] = {}
        self.global_label = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FrequencyBinMajority":
        x = np.asarray(x).reshape(-1)
        quantiles = np.linspace(0, 100, self.n_bins + 1)
        edges = np.percentile(x, quantiles)
        edges = np.unique(edges)
        if len(edges) <= 2:
            edges = np.linspace(float(np.min(x)), float(np.max(x)), min(self.n_bins + 1, max(2, len(np.unique(x)))))
        self.edges = edges
        self.global_label = Counter(y.tolist()).most_common(1)[0][0]
        bins = np.searchsorted(edges[1:-1], x, side="right")
        for bin_id in sorted(set(bins.tolist())):
            mask = bins == bin_id
            self.bin_labels[int(bin_id)] = Counter(y[mask].tolist()).most_common(1)[0][0]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.edges is None:
            raise RuntimeError("FrequencyBinMajority is not fitted")
        x = np.asarray(x).reshape(-1)
        bins = np.searchsorted(self.edges[1:-1], x, side="right")
        return np.asarray([self.bin_labels.get(int(bin_id), self.global_label) for bin_id in bins], dtype=int)


def build_models() -> list[dict[str, Any]]:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"scikit-learn is required for frequency-only baselines: {exc}") from exc

    return [
        {
            "method": "frequency_logistic_regression",
            "description": "Balanced logistic regression on standardized frequency_hz.",
            "model": LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE),
            "standardize": True,
            "parameters": {"class_weight": "balanced", "max_iter": 5000, "random_state": RANDOM_STATE},
        },
        {
            "method": "frequency_random_forest",
            "description": "Random forest on raw frequency_hz.",
            "model": RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                n_jobs=-1,
            ),
            "standardize": False,
            "parameters": {
                "n_estimators": 300,
                "class_weight": "balanced_subsample",
                "min_samples_leaf": 2,
                "random_state": RANDOM_STATE,
            },
        },
        {
            "method": "frequency_bin_majority_diagnostic",
            "description": "Diagnostic majority-class frequency-bin baseline fitted on reference frequency_hz only.",
            "model": FrequencyBinMajority(n_bins=20),
            "standardize": False,
            "parameters": {"n_bins": 20, "diagnostic": True},
        },
    ]


def fit_predict(model_spec: dict[str, Any], train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    model = model_spec["model"]
    if model_spec["standardize"]:
        train_features, test_features = standardize_train_test(train_x, test_x)
    else:
        train_features = train_x.reshape(-1, 1)
        test_features = test_x.reshape(-1, 1)
    model.fit(train_features, train_y)
    return np.asarray(model.predict(test_features)).astype(int)


def add_metric_rows(
    rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    base: dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    metrics = metrics_from_predictions(y_true, y_pred)
    row = dict(base)
    row.update(
        {
            "accuracy": metrics["accuracy"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "n_test": int(len(y_true)),
        }
    )
    rows.append(row)
    for item in metrics["per_class"]:
        per_class = dict(base)
        per_class.update(item)
        per_class_rows.append(per_class)
    return row


def evaluate_stratified(metadata: dict[str, np.ndarray], model_specs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    train_mask = metadata["split"] == "train"
    test_mask = metadata["split"] == "val"
    for spec in model_specs:
        pred = fit_predict(spec, metadata["frequency_hz"][train_mask], metadata["label"][train_mask], metadata["frequency_hz"][test_mask])
        add_metric_rows(
            rows,
            per_class_rows,
            {"protocol": "stratified_train_val", "method": spec["method"], "held_out_group": "val", "stress_test_note": "none"},
            metadata["label"][test_mask],
            pred,
        )
    return rows, per_class_rows


def evaluate_grouped(
    metadata: dict[str, np.ndarray],
    model_specs: list[dict[str, Any]],
    group_field: str,
    stress_note: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    groups = sorted(set(metadata[group_field].tolist()))
    for group in groups:
        test_mask = metadata[group_field] == group
        train_mask = ~test_mask
        for spec in model_specs:
            pred = fit_predict(spec, metadata["frequency_hz"][train_mask], metadata["label"][train_mask], metadata["frequency_hz"][test_mask])
            add_metric_rows(
                rows,
                per_class_rows,
                {
                    "protocol": f"{group_field}_grouped",
                    "method": spec["method"],
                    "held_out_group": group,
                    "stress_test_note": stress_note,
                },
                metadata["label"][test_mask],
                pred,
            )
    return rows, per_class_rows


def aggregate_grouped(rows: list[dict[str, Any]], protocol: str) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["protocol"] == protocol:
            by_method[row["method"]].append(row)
    output = []
    for method, method_rows in sorted(by_method.items()):
        f1_values = [float(row["macro_f1"]) for row in method_rows]
        worst = min(method_rows, key=lambda item: float(item["macro_f1"]))
        best = max(method_rows, key=lambda item: float(item["macro_f1"]))
        output.append(
            {
                "protocol": protocol,
                "method": method,
                "mean_macro_f1": float(np.mean(f1_values)),
                "worst_group": worst["held_out_group"],
                "worst_group_macro_f1": float(worst["macro_f1"]),
                "best_group": best["held_out_group"],
                "best_group_macro_f1": float(best["macro_f1"]),
                "n_groups": len(method_rows),
            }
        )
    return output


def best_by_protocol(summary_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        protocol = row["protocol"]
        if protocol not in best or row_macro_f1(row) > row_macro_f1(best[protocol]):
            best[protocol] = row
    return best


def row_macro_f1(row: dict[str, Any]) -> float:
    if row.get("macro_f1") not in ("", None):
        return float(row["macro_f1"])
    return float(row["mean_macro_f1"])


def load_comparison_rows(eval_dir: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(eval_dir / "encoder_baseline_summary.csv")
    return {row["encoder_id"]: row for row in rows}


def compare_against_encoders(freq_best: dict[str, dict[str, Any]], encoder_rows: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    mapping = [
        ("random_resnet18", "Random ResNet-18"),
        ("imagenet_resnet18", "ImageNet ResNet-18"),
        ("v6_1", "v6.1"),
        ("v6_2_a", "v6.2-A"),
    ]
    protocol_columns = {
        "stratified_train_val": "stratified_best_knn_macro_f1",
        "board_grouped": "best_lobo_mean_macro_f1",
        "material_grouped": "best_lomo_mean_macro_f1",
    }
    rows = []
    for protocol, column in protocol_columns.items():
        freq_row = freq_best[protocol]
        freq_score = row_macro_f1(freq_row)
        for encoder_id, encoder_name in mapping:
            encoder_score = float(encoder_rows[encoder_id][column])
            rows.append(
                {
                    "protocol": protocol,
                    "frequency_method": freq_row["method"],
                    "frequency_macro_f1": freq_score,
                    "comparison_encoder": encoder_name,
                    "comparison_macro_f1": encoder_score,
                    "delta_frequency_minus_encoder_pp": (freq_score - encoder_score) * 100,
                }
            )
    return rows


def render_report(
    model_specs: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    board_rows: list[dict[str, Any]],
    material_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    key_numbers: dict[str, Any],
) -> str:
    summary_fields = ["protocol", "method", "macro_f1", "mean_macro_f1", "worst_group", "worst_group_macro_f1", "best_group", "best_group_macro_f1", "n_groups"]
    board_fields = ["protocol", "method", "held_out_group", "accuracy", "macro_recall", "macro_f1", "n_test", "stress_test_note"]
    comparison_fields = ["protocol", "frequency_method", "frequency_macro_f1", "comparison_encoder", "comparison_macro_f1", "delta_frequency_minus_encoder_pp"]
    model_lines = []
    for spec in model_specs:
        model_lines.append(f"- `{spec['method']}`: {spec['description']} Parameters: `{json.dumps(spec['parameters'])}`")
    v62a_best_grouped_macro_f1 = max(key_numbers["v62a_board_lobo_macro_f1"], key_numbers["v62a_material_lomo_macro_f1"])
    if key_numbers["best_overall_frequency_macro_f1"] >= v62a_best_grouped_macro_f1:
        strength_line = "Frequency metadata is a strong predictor and should be discussed as carrying substantial modal information."
    elif key_numbers["best_overall_frequency_macro_f1"] >= 0.75:
        strength_line = "Frequency metadata is informative, but it does not replace the grouped ESPI embedding evidence because v6.2-A remains stronger in the grouped protocols."
    else:
        strength_line = "Frequency metadata alone is weak to moderate relative to ESPI embeddings, supporting the need for image-derived ESPI representations."

    return "\n".join(
        [
            "# Frequency-Only Baseline Report",
            "",
            "## Purpose",
            "",
            "This report evaluates whether `frequency_hz` alone predicts the five ESPI modal labels under the same stratified, board-grouped, and material-grouped protocols used for frozen encoder evaluation. This is a metadata leakage/control baseline, not the main model.",
            "",
            "## Models",
            "",
            *model_lines,
            "",
            "## Summary Metrics",
            "",
            markdown_table(summary_rows, summary_fields),
            "",
            "## Board LOBO-Style Frequency-Only Results",
            "",
            markdown_table(board_rows, board_fields),
            "",
            "## Material LOMO-Style Frequency-Only Results",
            "",
            markdown_table(material_rows, board_fields),
            "",
            "## Comparison Against Existing Encoder Baselines",
            "",
            markdown_table(comparison_rows, comparison_fields),
            "",
            "## Reviewer-Facing Interpretation",
            "",
            f"- Best frequency-only stratified Macro-F1: {pct(key_numbers['best_stratified_frequency_macro_f1'])}.",
            f"- Best frequency-only Board LOBO mean Macro-F1: {pct(key_numbers['best_board_frequency_macro_f1'])}.",
            f"- Best frequency-only Material LOMO mean Macro-F1: {pct(key_numbers['best_material_frequency_macro_f1'])}.",
            f"- v6.2-A Board LOBO mean Macro-F1: {pct(key_numbers['v62a_board_lobo_macro_f1'])}.",
            f"- v6.2-A Material LOMO mean Macro-F1: {pct(key_numbers['v62a_material_lomo_macro_f1'])}.",
            f"- Interpretation: {strength_line}",
            "- Material LOMO remains a descriptive stress test because only two material groups are available.",
            "- This control does not use ESPI image-derived embeddings and does not modify the central frozen encoder claim.",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.features)
    model_specs = build_models()

    strat_rows, strat_per_class = evaluate_stratified(metadata, model_specs)
    board_rows, board_per_class = evaluate_grouped(metadata, model_specs, "board", "none")
    material_rows, material_per_class = evaluate_grouped(
        metadata,
        model_specs,
        "material",
        "descriptive stress-test only; two material groups available",
    )
    board_agg = aggregate_grouped(board_rows, "board_grouped")
    material_agg = aggregate_grouped(material_rows, "material_grouped")

    summary_rows: list[dict[str, Any]] = []
    for row in strat_rows:
        summary_rows.append(
            {
                "protocol": row["protocol"],
                "method": row["method"],
                "macro_f1": row["macro_f1"],
                "mean_macro_f1": "",
                "worst_group": "",
                "worst_group_macro_f1": "",
                "best_group": "",
                "best_group_macro_f1": "",
                "n_groups": "",
            }
        )
    for row in board_agg + material_agg:
        summary_rows.append(
            {
                "protocol": row["protocol"],
                "method": row["method"],
                "macro_f1": "",
                "mean_macro_f1": row["mean_macro_f1"],
                "worst_group": row["worst_group"],
                "worst_group_macro_f1": row["worst_group_macro_f1"],
                "best_group": row["best_group"],
                "best_group_macro_f1": row["best_group_macro_f1"],
                "n_groups": row["n_groups"],
            }
        )

    best = best_by_protocol(summary_rows)
    encoder_rows = load_comparison_rows(args.eval_dir)
    comparison_rows = compare_against_encoders(best, encoder_rows)

    per_class_rows = strat_per_class + board_per_class + material_per_class
    key_numbers = {
        "n_samples": int(len(metadata["label"])),
        "frequency_min_hz": float(np.min(metadata["frequency_hz"])),
        "frequency_max_hz": float(np.max(metadata["frequency_hz"])),
        "methods": [spec["method"] for spec in model_specs],
        "best_stratified_frequency_method": best["stratified_train_val"]["method"],
        "best_stratified_frequency_macro_f1": float(best["stratified_train_val"]["macro_f1"]),
        "best_board_frequency_method": best["board_grouped"]["method"],
        "best_board_frequency_macro_f1": float(best["board_grouped"]["mean_macro_f1"]),
        "best_material_frequency_method": best["material_grouped"]["method"],
        "best_material_frequency_macro_f1": float(best["material_grouped"]["mean_macro_f1"]),
        "v61_stratified_macro_f1": float(encoder_rows["v6_1"]["stratified_best_knn_macro_f1"]),
        "v62a_stratified_macro_f1": float(encoder_rows["v6_2_a"]["stratified_best_knn_macro_f1"]),
        "v62a_board_lobo_macro_f1": float(encoder_rows["v6_2_a"]["best_lobo_mean_macro_f1"]),
        "v62a_material_lomo_macro_f1": float(encoder_rows["v6_2_a"]["best_lomo_mean_macro_f1"]),
        "best_overall_frequency_macro_f1": max(
            float(best["stratified_train_val"]["macro_f1"]),
            float(best["board_grouped"]["mean_macro_f1"]),
            float(best["material_grouped"]["mean_macro_f1"]),
        ),
        "material_lomo_caution": "descriptive stress-test only; two material groups available",
    }

    write_csv(args.out_dir / "frequency_only_summary.csv", summary_rows, fieldnames(summary_rows))
    write_csv(args.out_dir / "frequency_only_board_lobo.csv", board_rows, fieldnames(board_rows))
    write_csv(args.out_dir / "frequency_only_material_lomo.csv", material_rows, fieldnames(material_rows))
    write_csv(args.out_dir / "frequency_only_per_class.csv", per_class_rows, fieldnames(per_class_rows))
    write_csv(args.out_dir / "frequency_only_encoder_comparison.csv", comparison_rows, fieldnames(comparison_rows))
    (args.out_dir / "frequency_only_key_numbers.json").write_text(
        json.dumps(key_numbers, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "FREQUENCY_ONLY_BASELINE_REPORT.md").write_text(
        render_report(model_specs, summary_rows, board_rows, material_rows, comparison_rows, key_numbers),
        encoding="utf-8",
    )

    print(f"[done] wrote frequency-only baseline report to {args.out_dir}")
    print(
        "[done] best_stratified={:.6f} best_board={:.6f} best_material={:.6f}".format(
            key_numbers["best_stratified_frequency_macro_f1"],
            key_numbers["best_board_frequency_macro_f1"],
            key_numbers["best_material_frequency_macro_f1"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
