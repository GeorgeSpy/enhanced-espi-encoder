import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    df2 = df.copy()
    for c in df2.columns:
        if pd.api.types.is_float_dtype(df2[c]):
            df2[c] = df2[c].map(lambda x: f"{x:.4f}")
        else:
            df2[c] = df2[c].astype(str)

    headers = list(df2.columns)
    rows = df2.values.tolist()

    widths = []
    for i, h in enumerate(headers):
        max_cell = max([len(str(r[i])) for r in rows], default=0)
        widths.append(max(len(str(h)), max_cell))

    def fmt_row(vals):
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    out = []
    out.append(fmt_row(headers))
    out.append("| " + " | ".join("-" * w for w in widths) + " |")
    for r in rows:
        out.append(fmt_row(r))
    return "\n".join(out)


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

    X = get_first(["embedding", "embeddings", "features", "X", "z"])
    y = get_first(["labels", "y", "label"])
    freq = get_first(["frequency_hz", "freq_hz", "frequency", "freq", "frequencies"])
    boards = get_first(["boards", "board", "board_id", "split_group"])
    materials = get_first(["materials", "material", "material_id"])
    splits = get_first(["splits", "split", "split_label"])

    label_names = get_first(["label_names", "class_names", "classes"], required=False)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    freq = np.asarray(freq, dtype=np.float32).reshape(-1, 1)
    boards = np.asarray(boards).astype(str)
    materials = np.asarray(materials).astype(str)
    splits = np.asarray(splits).astype(str)

    n = X.shape[0]
    for name, arr in [
        ("labels", y),
        ("frequency", freq),
        ("boards", boards),
        ("materials", materials),
        ("splits", splits),
    ]:
        if len(arr) != n:
            raise ValueError(f"Length mismatch for {name}: {len(arr)} vs embeddings {n}")

    if label_names is not None:
        label_names = np.asarray(label_names).astype(str)
    else:
        uniq = sorted(np.unique(y).tolist())
        label_names = np.asarray([str(u) for u in uniq])

    return X, y, freq, boards, materials, splits, label_names


def label_to_name(label, label_names):
    try:
        i = int(label)
        if 0 <= i < len(label_names):
            return str(label_names[i])
    except Exception:
        pass
    return str(label)


def detect_stratified_masks(splits):
    s = np.char.lower(splits.astype(str))

    train_values = {"train", "training", "tr"}
    test_values = {"val", "valid", "validation", "test", "te"}

    train_mask = np.isin(s, list(train_values))
    test_mask = np.isin(s, list(test_values))

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        unique = sorted(set(s.tolist()))
        raise ValueError(
            f"Could not infer train/validation split from split labels. Unique split labels: {unique}"
        )

    return train_mask, test_mask


def get_features(feature_kind, X, freq):
    if feature_kind == "freq":
        return freq
    if feature_kind == "embed":
        return X
    if feature_kind == "fusion":
        return np.concatenate([freq, X], axis=1)
    raise ValueError(feature_kind)


def fit_predict_lr(X_train, y_train, X_test, max_iter):
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=None,
        random_state=42,
    )
    clf.fit(X_train_s, y_train)
    return clf.predict(X_test_s)


def fit_predict_rf(X_train, y_train, X_test, n_estimators):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",
        max_features="sqrt",
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def evaluate_one_fold(X, y, freq, train_mask, test_mask, family, feature_kind, max_iter, rf_trees):
    X_all = get_features(feature_kind, X, freq)
    X_train = X_all[train_mask]
    X_test = X_all[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    if family == "lr":
        pred = fit_predict_lr(X_train, y_train, X_test, max_iter=max_iter)
    elif family == "rf":
        pred = fit_predict_rf(X_train, y_train, X_test, n_estimators=rf_trees)
    else:
        raise ValueError(family)

    return y_test, pred


def protocol_folds(protocol, boards, materials, splits):
    if protocol == "stratified":
        train_mask, test_mask = detect_stratified_masks(splits)
        return [("validation", train_mask, test_mask)]

    if protocol == "board_lobo":
        rows = []
        for g in sorted(set(boards.tolist())):
            test_mask = boards == g
            train_mask = ~test_mask
            rows.append((str(g), train_mask, test_mask))
        return rows

    if protocol == "material_lomo":
        rows = []
        for g in sorted(set(materials.tolist())):
            test_mask = materials == g
            train_mask = ~test_mask
            rows.append((str(g), train_mask, test_mask))
        return rows

    raise ValueError(protocol)


def aggregate_predictions(X, y, freq, boards, materials, splits, family, protocol, feature_kind, max_iter, rf_trees):
    y_all = []
    pred_all = []
    group_all = []
    fold_metric_rows = []

    for heldout, train_mask, test_mask in protocol_folds(protocol, boards, materials, splits):
        y_test, pred = evaluate_one_fold(
            X, y, freq,
            train_mask, test_mask,
            family=family,
            feature_kind=feature_kind,
            max_iter=max_iter,
            rf_trees=rf_trees,
        )

        y_all.append(y_test)
        pred_all.append(pred)
        group_all.append(np.asarray([heldout] * len(y_test)).astype(str))

        m = metrics(y_test, pred)
        m.update({
            "family": family,
            "protocol": protocol,
            "feature_kind": feature_kind,
            "heldout_group": heldout,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        })
        fold_metric_rows.append(m)

    return (
        np.concatenate(y_all),
        np.concatenate(pred_all),
        np.concatenate(group_all),
        fold_metric_rows,
    )


def correction_summary(y_true, pred_freq, pred_fusion, label_names, protocol, family):
    freq_wrong = pred_freq != y_true
    fusion_wrong = pred_fusion != y_true

    corrected = freq_wrong & (pred_fusion == y_true)
    degraded = (pred_freq == y_true) & fusion_wrong
    both_wrong = freq_wrong & fusion_wrong
    both_correct = (pred_freq == y_true) & (pred_fusion == y_true)

    n = len(y_true)
    freq_errors = int(freq_wrong.sum())
    fusion_errors = int(fusion_wrong.sum())
    corrected_n = int(corrected.sum())
    degraded_n = int(degraded.sum())

    return {
        "family": family,
        "protocol": protocol,
        "n": n,
        "freq_errors": freq_errors,
        "fusion_errors": fusion_errors,
        "corrected_by_fusion": corrected_n,
        "degraded_by_fusion": degraded_n,
        "both_wrong": int(both_wrong.sum()),
        "both_correct": int(both_correct.sum()),
        "net_corrections": corrected_n - degraded_n,
        "correction_rate_of_freq_errors": corrected_n / freq_errors if freq_errors else 0.0,
        "degradation_rate_of_freq_correct": degraded_n / int((~freq_wrong).sum()) if int((~freq_wrong).sum()) else 0.0,
        "freq_macro_f1": float(f1_score(y_true, pred_freq, average="macro", zero_division=0)),
        "fusion_macro_f1": float(f1_score(y_true, pred_fusion, average="macro", zero_division=0)),
        "delta_macro_f1_pp": 100.0 * (
            float(f1_score(y_true, pred_fusion, average="macro", zero_division=0))
            - float(f1_score(y_true, pred_freq, average="macro", zero_division=0))
        ),
    }


def per_class_correction(y_true, pred_freq, pred_fusion, label_names, protocol, family):
    rows = []
    for c in sorted(np.unique(y_true).tolist()):
        mask = y_true == c
        if mask.sum() == 0:
            continue

        freq_wrong = pred_freq[mask] != y_true[mask]
        fusion_wrong = pred_fusion[mask] != y_true[mask]
        corrected = freq_wrong & (pred_fusion[mask] == y_true[mask])
        degraded = (pred_freq[mask] == y_true[mask]) & fusion_wrong

        freq_f1 = f1_score(y_true, pred_freq, labels=[c], average="macro", zero_division=0)
        fusion_f1 = f1_score(y_true, pred_fusion, labels=[c], average="macro", zero_division=0)

        rows.append({
            "family": family,
            "protocol": protocol,
            "class_id": int(c) if str(c).isdigit() else str(c),
            "class_name": label_to_name(c, label_names),
            "support": int(mask.sum()),
            "freq_errors": int(freq_wrong.sum()),
            "fusion_errors": int(fusion_wrong.sum()),
            "corrected_by_fusion": int(corrected.sum()),
            "degraded_by_fusion": int(degraded.sum()),
            "net_corrections": int(corrected.sum()) - int(degraded.sum()),
            "correction_rate_of_freq_errors": float(corrected.sum() / freq_wrong.sum()) if freq_wrong.sum() else 0.0,
            "freq_class_f1": float(freq_f1),
            "fusion_class_f1": float(fusion_f1),
            "delta_class_f1_pp": 100.0 * float(fusion_f1 - freq_f1),
        })
    return rows


def correction_pairs(y_true, pred_freq, pred_fusion, label_names, protocol, family):
    mask = (pred_freq != y_true) & (pred_fusion == y_true)
    rows = []
    for t, pf, pfu in zip(y_true[mask], pred_freq[mask], pred_fusion[mask]):
        rows.append({
            "family": family,
            "protocol": protocol,
            "true_id": int(t) if str(t).isdigit() else str(t),
            "true_name": label_to_name(t, label_names),
            "freq_pred_id": int(pf) if str(pf).isdigit() else str(pf),
            "freq_pred_name": label_to_name(pf, label_names),
            "fusion_pred_id": int(pfu) if str(pfu).isdigit() else str(pfu),
            "fusion_pred_name": label_to_name(pfu, label_names),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "family", "protocol", "true_id", "true_name",
            "freq_pred_id", "freq_pred_name",
            "fusion_pred_id", "fusion_pred_name",
            "count",
        ])

    df = pd.DataFrame(rows)
    return (
        df.groupby([
            "family", "protocol", "true_id", "true_name",
            "freq_pred_id", "freq_pred_name",
            "fusion_pred_id", "fusion_pred_name",
        ])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--families", nargs="+", default=["lr"], choices=["lr", "rf"])
    parser.add_argument("--protocols", nargs="+", default=["stratified", "board_lobo", "material_lomo"])
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--rf-trees", type=int, default=300)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    X, y, freq, boards, materials, splits, label_names = load_npz(args.features)

    fold_metrics = []
    overall_metrics = []
    correction_rows = []
    class_rows = []
    pair_dfs = []

    for family in args.families:
        print(f"[FAMILY] {family}")

        for protocol in args.protocols:
            print(f"  [PROTOCOL] {protocol}")

            preds = {}
            y_ref = None
            group_ref = None

            for feature_kind in ["freq", "embed", "fusion"]:
                y_true, pred, groups, fold_rows = aggregate_predictions(
                    X, y, freq, boards, materials, splits,
                    family=family,
                    protocol=protocol,
                    feature_kind=feature_kind,
                    max_iter=args.max_iter,
                    rf_trees=args.rf_trees,
                )

                preds[feature_kind] = pred
                y_ref = y_true
                group_ref = groups
                fold_metrics.extend(fold_rows)

                m = metrics(y_true, pred)
                m.update({
                    "family": family,
                    "protocol": protocol,
                    "feature_kind": feature_kind,
                    "n": int(len(y_true)),
                })
                overall_metrics.append(m)

            correction_rows.append(
                correction_summary(
                    y_ref,
                    preds["freq"],
                    preds["fusion"],
                    label_names,
                    protocol,
                    family,
                )
            )

            class_rows.extend(
                per_class_correction(
                    y_ref,
                    preds["freq"],
                    preds["fusion"],
                    label_names,
                    protocol,
                    family,
                )
            )

            pair_dfs.append(
                correction_pairs(
                    y_ref,
                    preds["freq"],
                    preds["fusion"],
                    label_names,
                    protocol,
                    family,
                )
            )

    fold_df = pd.DataFrame(fold_metrics)
    overall_df = pd.DataFrame(overall_metrics)
    correction_df = pd.DataFrame(correction_rows)
    class_df = pd.DataFrame(class_rows)
    pair_df = pd.concat(pair_dfs, ignore_index=True) if pair_dfs else pd.DataFrame()

    fold_path = args.outdir / "frequency_residual_fold_metrics.csv"
    overall_path = args.outdir / "frequency_residual_overall_metrics.csv"
    correction_path = args.outdir / "frequency_residual_correction_summary.csv"
    class_path = args.outdir / "frequency_residual_per_class_corrections.csv"
    pair_path = args.outdir / "frequency_residual_corrected_error_pairs.csv"
    json_path = args.outdir / "frequency_residual_key_numbers.json"
    md_path = args.outdir / "FREQUENCY_RESIDUAL_CORRECTION_SUMMARY.md"

    fold_df.to_csv(fold_path, index=False)
    overall_df.to_csv(overall_path, index=False)
    correction_df.to_csv(correction_path, index=False)
    class_df.to_csv(class_path, index=False)
    pair_df.to_csv(pair_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "features": str(args.features),
            "input_shape": list(X.shape),
            "families": args.families,
            "protocols": args.protocols,
            "overall_metrics": overall_df.to_dict(orient="records"),
            "correction_summary": correction_df.to_dict(orient="records"),
        }, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Frequency-Residual / Error-Correction Analysis\n\n")
        f.write("This analysis compares `frequency_hz`-only prediction against `frequency_hz + v6.2-A embedding` fusion.\n\n")
        f.write("The central question is whether image-derived v6.2-A embeddings correct errors made by the frequency-only metadata baseline.\n\n")
        f.write(f"- Feature file: `{args.features}`\n")
        f.write(f"- Input embedding shape: `{X.shape}`\n")
        f.write(f"- Families: `{args.families}`\n")
        f.write(f"- Protocols: `{args.protocols}`\n\n")

        f.write("## Overall Metrics\n\n")
        f.write(df_to_markdown(overall_df.sort_values(["family", "protocol", "feature_kind"])))
        f.write("\n\n")

        f.write("## Frequency-only vs Frequency+Embedding Correction Summary\n\n")
        f.write(df_to_markdown(correction_df.sort_values(["family", "protocol"])))
        f.write("\n\n")

        f.write("## Per-Class Corrections\n\n")
        key_class = class_df.sort_values(["family", "protocol", "delta_class_f1_pp"], ascending=[True, True, False])
        f.write(df_to_markdown(key_class))
        f.write("\n\n")

        f.write("## Most Frequent Corrected Error Pairs\n\n")
        top_pairs = pair_df.sort_values("count", ascending=False).head(30) if not pair_df.empty else pair_df
        f.write(df_to_markdown(top_pairs))
        f.write("\n\n")

        f.write("## Output Files\n\n")
        for p in [fold_path, overall_path, correction_path, class_path, pair_path, json_path]:
            f.write(f"- `{p}`\n")

    print(f"[DONE] {md_path}")
    print(correction_df.to_string(index=False))


if __name__ == "__main__":
    main()
