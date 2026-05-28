import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


def load_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)

    def get_first(candidates, required=True):
        for k in candidates:
            if k in keys:
                return data[k]
        if required:
            raise KeyError(f"None of the candidate keys found: {candidates}. Available keys: {sorted(keys)}")
        return None

    X = get_first(["embeddings", "features", "X", "z", "embedding"])
    y = get_first(["labels", "y", "label"])
    paths = get_first(["paths", "path"], required=False)
    boards = get_first(["boards", "board", "board_id", "split_group"], required=False)
    materials = get_first(["materials", "material", "material_id"], required=False)
    splits = get_first(["splits", "split", "split_label"], required=False)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    if boards is None:
        raise KeyError("Board identifiers not found in NPZ.")
    if materials is None:
        raise KeyError("Material identifiers not found in NPZ.")
    if splits is None:
        raise KeyError("Split identifiers not found in NPZ.")

    boards = np.asarray(boards).astype(str)
    materials = np.asarray(materials).astype(str)
    splits = np.asarray(splits).astype(str)

    return X, y, boards, materials, splits


def macro_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def fit_transform_fold_pca(X_train, X_test, n_components):
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)

    return X_train_p, X_test_p, float(np.sum(pca.explained_variance_ratio_))


def eval_knn_fold(X_train, y_train, X_test, y_test, n_components, k):
    X_train_p, X_test_p, evr = fit_transform_fold_pca(X_train, X_test, n_components)

    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
    clf.fit(X_train_p, y_train)
    pred = clf.predict(X_test_p)

    out = macro_metrics(y_test, pred)
    out["explained_variance_ratio_sum"] = evr
    return out


def detect_stratified_masks(splits):
    s = np.char.lower(splits.astype(str))

    train_values = {"train", "training", "tr"}
    val_values = {"val", "valid", "validation", "test", "te"}

    train_mask = np.isin(s, list(train_values))
    test_mask = np.isin(s, list(val_values))

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        unique = sorted(set(s.tolist()))
        raise ValueError(
            f"Could not infer train/validation split from split labels. Unique split labels: {unique}"
        )

    return train_mask, test_mask


def run_stratified(X, y, splits, n_components, k):
    train_mask, test_mask = detect_stratified_masks(splits)

    res = eval_knn_fold(
        X[train_mask], y[train_mask],
        X[test_mask], y[test_mask],
        n_components=n_components,
        k=k
    )
    res.update({
        "protocol": "stratified",
        "heldout_group": "validation",
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_components": int(n_components),
        "k": int(k),
    })
    return [res]


def run_grouped(X, y, groups, protocol_name, n_components, k):
    rows = []
    unique_groups = sorted(set(groups.tolist()))

    for g in unique_groups:
        test_mask = groups == g
        train_mask = ~test_mask

        res = eval_knn_fold(
            X[train_mask], y[train_mask],
            X[test_mask], y[test_mask],
            n_components=n_components,
            k=k
        )
        res.update({
            "protocol": protocol_name,
            "heldout_group": str(g),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_components": int(n_components),
            "k": int(k),
        })
        rows.append(res)

    return rows


def summarize(rows):
    df = pd.DataFrame(rows)
    summary_rows = []

    for (protocol, n_components, k), sub in df.groupby(["protocol", "n_components", "k"]):
        summary_rows.append({
            "protocol": protocol,
            "n_components": int(n_components),
            "k": int(k),
            "n_folds": int(len(sub)),
            "macro_f1_mean": float(sub["macro_f1"].mean()),
            "macro_f1_std": float(sub["macro_f1"].std(ddof=0)) if len(sub) > 1 else 0.0,
            "balanced_accuracy_mean": float(sub["balanced_accuracy"].mean()),
            "accuracy_mean": float(sub["accuracy"].mean()),
            "weighted_f1_mean": float(sub["weighted_f1"].mean()),
            "explained_variance_ratio_sum_mean": float(sub["explained_variance_ratio_sum"].mean()),
            "worst_group": str(sub.sort_values("macro_f1").iloc[0]["heldout_group"]),
            "worst_group_macro_f1": float(sub.sort_values("macro_f1").iloc[0]["macro_f1"]),
            "best_group": str(sub.sort_values("macro_f1", ascending=False).iloc[0]["heldout_group"]),
            "best_group_macro_f1": float(sub.sort_values("macro_f1", ascending=False).iloc[0]["macro_f1"]),
        })

    return pd.DataFrame(summary_rows)


def df_to_markdown(df):
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for _, row in df.iterrows():
        row_strs = []
        for h in headers:
            val = row[h]
            if isinstance(val, float):
                row_strs.append(f"{val:.6f}")
            else:
                row_strs.append(str(val))
        lines.append("| " + " | ".join(row_strs) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--components", nargs="+", type=int, default=[512, 256, 128])
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    X, y, boards, materials, splits = load_npz(args.features)

    all_rows = []
    for nc in args.components:
        if nc >= X.shape[1]:
            print(f"[SKIP] n_components={nc} >= input_dim={X.shape[1]}")
            continue

        print(f"[RUN] PCA-{nc}, k={args.k}")

        all_rows.extend(run_stratified(X, y, splits, nc, args.k))
        all_rows.extend(run_grouped(X, y, boards, "board_lobo", nc, args.k))
        all_rows.extend(run_grouped(X, y, materials, "material_lomo", nc, args.k))

    fold_df = pd.DataFrame(all_rows)
    summary_df = summarize(all_rows)

    fold_csv = args.outdir / "v62a_dimension_matched_fold_results.csv"
    summary_csv = args.outdir / "v62a_dimension_matched_summary.csv"
    summary_md = args.outdir / "V62A_DIMENSION_MATCHED_SUMMARY.md"
    json_path = args.outdir / "v62a_dimension_matched_summary.json"

    fold_df.to_csv(fold_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "feature_file": str(args.features),
            "input_shape": list(X.shape),
            "k": args.k,
            "components": args.components,
            "summary": summary_df.to_dict(orient="records"),
        }, f, indent=2)

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# v6.2-A Dimension-Matched PCA Evaluation\n\n")
        f.write("Fold-local PCA was fitted only on each training/reference fold and applied to the corresponding held-out fold.\n\n")
        f.write(f"- Input feature file: `{args.features}`\n")
        f.write(f"- Input shape: `{X.shape}`\n")
        f.write(f"- kNN: cosine, fixed k={args.k}, distance-weighted\n\n")
        f.write("## Summary\n\n")
        f.write(df_to_markdown(summary_df))
        f.write("\n\n")
        f.write("## Files\n\n")
        f.write(f"- `{fold_csv}`\n")
        f.write(f"- `{summary_csv}`\n")
        f.write(f"- `{json_path}`\n")

    print(f"[DONE] Wrote: {summary_md}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
