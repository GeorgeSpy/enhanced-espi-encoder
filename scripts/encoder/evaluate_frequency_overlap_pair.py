import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix


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
        widths.append(max(len(str(h)), max([len(str(r[i])) for r in rows], default=0)))

    def fmt(vals):
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    out = [fmt(headers), "| " + " | ".join("-" * w for w in widths) + " |"]
    out += [fmt(r) for r in rows]
    return "\n".join(out)


def load_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    keys = set(d.files)

    def get(cands, required=True):
        for k in cands:
            if k in keys:
                return d[k]
        if required:
            raise KeyError(f"Missing keys {cands}. Available: {sorted(keys)}")
        return None

    X = np.asarray(get(["embedding", "embeddings", "features", "X", "z"]), dtype=np.float32)
    y = np.asarray(get(["labels", "y", "label"]))
    freq = np.asarray(get(["frequency_hz", "freq_hz", "frequency", "freq", "frequencies"]), dtype=np.float32)
    boards = np.asarray(get(["boards", "board", "board_id", "split_group"])).astype(str)
    materials = np.asarray(get(["materials", "material", "material_id"])).astype(str)
    splits = np.asarray(get(["splits", "split", "split_label"])).astype(str)
    label_names = get(["label_names", "class_names", "classes"], required=False)
    if label_names is not None:
        label_names = np.asarray(label_names).astype(str)
    else:
        label_names = np.asarray([str(i) for i in sorted(np.unique(y).tolist())])

    return X, y, freq.reshape(-1), boards, materials, splits, label_names


def label_name(label, label_names):
    i = int(label)
    if 0 <= i < len(label_names):
        return str(label_names[i])
    return str(label)


def stratified_masks(splits):
    s = np.char.lower(splits.astype(str))
    train = np.isin(s, ["train", "training", "tr"])
    test = np.isin(s, ["val", "valid", "validation", "test", "te"])
    if train.sum() == 0 or test.sum() == 0:
        raise RuntimeError(f"Could not infer split labels. Found: {sorted(set(s.tolist()))}")
    return train, test


def protocol_folds(protocol, boards, materials, splits):
    if protocol == "stratified":
        tr, te = stratified_masks(splits)
        return [("validation", tr, te)]
    if protocol == "board_lobo":
        out = []
        for g in sorted(set(boards.tolist())):
            te = boards == g
            tr = ~te
            out.append((g, tr, te))
        return out
    if protocol == "material_lomo":
        out = []
        for g in sorted(set(materials.tolist())):
            te = materials == g
            tr = ~te
            out.append((g, tr, te))
        return out
    raise ValueError(protocol)


def get_features(kind, X, freq):
    if kind == "freq":
        return freq.reshape(-1, 1)
    if kind == "embed":
        return X
    if kind == "fusion":
        return np.concatenate([freq.reshape(-1, 1), X], axis=1)
    raise ValueError(kind)


def fit_predict(kind, family, Xtr, ytr, Xte, max_iter, rf_trees, k):
    if family == "knn":
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
        clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
        clf.fit(Xtr, ytr)
        return clf.predict(Xte)

    if family == "lr":
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
        clf = LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
        clf.fit(Xtr, ytr)
        return clf.predict(Xte)

    if family == "rf":
        clf = RandomForestClassifier(
            n_estimators=rf_trees,
            class_weight="balanced_subsample",
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(Xtr, ytr)
        return clf.predict(Xte)

    raise ValueError(family)


def metrics(ytrue, ypred):
    return {
        "accuracy": float(accuracy_score(ytrue, ypred)),
        "balanced_accuracy": float(balanced_accuracy_score(ytrue, ypred)),
        "macro_f1": float(f1_score(ytrue, ypred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(ytrue, ypred, average="weighted", zero_division=0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--labels", nargs=2, type=int, default=[2, 3])
    ap.add_argument("--overlap-only", action="store_true")
    ap.add_argument("--families", nargs="+", default=["lr", "rf"], choices=["knn", "lr", "rf"])
    ap.add_argument("--protocols", nargs="+", default=["stratified", "board_lobo", "material_lomo"])
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--rf-trees", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    X, y, freq, boards, materials, splits, label_names = load_npz(args.features)

    labels = np.asarray(args.labels)
    base_mask = np.isin(y, labels)

    freq_ranges = []
    for lab in labels:
        f = freq[y == lab]
        freq_ranges.append((lab, float(np.min(f)), float(np.max(f)), int(len(f))))

    overlap_min = max(r[1] for r in freq_ranges)
    overlap_max = min(r[2] for r in freq_ranges)

    if args.overlap_only:
        mask = base_mask & (freq >= overlap_min) & (freq <= overlap_max)
    else:
        mask = base_mask

    X2 = X[mask]
    y2 = y[mask]
    freq2 = freq[mask]
    boards2 = boards[mask]
    materials2 = materials[mask]
    splits2 = splits[mask]

    rows = []
    fold_rows = []
    cm_rows = []

    for protocol in args.protocols:
        for family in args.families:
            for feature_kind in ["freq", "embed", "fusion"]:
                y_all = []
                pred_all = []

                for heldout, tr, te in protocol_folds(protocol, boards2, materials2, splits2):
                    if tr.sum() == 0 or te.sum() == 0:
                        continue
                    if len(np.unique(y2[tr])) < 2 or len(np.unique(y2[te])) < 2:
                        continue

                    feats = get_features(feature_kind, X2, freq2)
                    pred = fit_predict(
                        feature_kind,
                        family,
                        feats[tr],
                        y2[tr],
                        feats[te],
                        args.max_iter,
                        args.rf_trees,
                        args.k,
                    )

                    m = metrics(y2[te], pred)
                    m.update({
                        "protocol": protocol,
                        "family": family,
                        "feature_kind": feature_kind,
                        "heldout_group": str(heldout),
                        "n_train": int(tr.sum()),
                        "n_test": int(te.sum()),
                    })
                    fold_rows.append(m)

                    y_all.append(y2[te])
                    pred_all.append(pred)

                if y_all:
                    yt = np.concatenate(y_all)
                    yp = np.concatenate(pred_all)
                    m = metrics(yt, yp)
                    m.update({
                        "protocol": protocol,
                        "family": family,
                        "feature_kind": feature_kind,
                        "n": int(len(yt)),
                    })
                    rows.append(m)

                    cm = confusion_matrix(yt, yp, labels=labels)
                    for i, true_lab in enumerate(labels):
                        for j, pred_lab in enumerate(labels):
                            cm_rows.append({
                                "protocol": protocol,
                                "family": family,
                                "feature_kind": feature_kind,
                                "true_id": int(true_lab),
                                "true_name": label_name(true_lab, label_names),
                                "pred_id": int(pred_lab),
                                "pred_name": label_name(pred_lab, label_names),
                                "count": int(cm[i, j]),
                            })

    overall = pd.DataFrame(rows).sort_values(["protocol", "family", "feature_kind"])
    folds = pd.DataFrame(fold_rows).sort_values(["protocol", "family", "feature_kind", "heldout_group"])
    cms = pd.DataFrame(cm_rows)

    overall_path = args.outdir / "overlap_pair_overall_metrics.csv"
    folds_path = args.outdir / "overlap_pair_fold_metrics.csv"
    cm_path = args.outdir / "overlap_pair_confusion_counts.csv"
    json_path = args.outdir / "overlap_pair_key_numbers.json"
    md_path = args.outdir / "OVERLAP_PAIR_ANALYSIS_SUMMARY.md"

    overall.to_csv(overall_path, index=False)
    folds.to_csv(folds_path, index=False)
    cms.to_csv(cm_path, index=False)

    info = {
        "features": str(args.features),
        "labels": args.labels,
        "label_names": [label_name(x, label_names) for x in labels],
        "overlap_only": bool(args.overlap_only),
        "freq_ranges": [
            {
                "label": int(lab),
                "label_name": label_name(lab, label_names),
                "freq_min": fmin,
                "freq_max": fmax,
                "n": n,
            }
            for lab, fmin, fmax, n in freq_ranges
        ],
        "overlap_min": overlap_min,
        "overlap_max": overlap_max,
        "n_selected": int(mask.sum()),
        "overall": overall.to_dict(orient="records"),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Frequency-Overlap Pair Analysis\n\n")
        f.write("This analysis isolates the target modal pair and evaluates whether image-derived embeddings help when frequency information is ambiguous.\n\n")
        f.write(f"- Labels: `{args.labels}` / `{info['label_names']}`\n")
        f.write(f"- Overlap only: `{args.overlap_only}`\n")
        f.write(f"- Selected samples: `{int(mask.sum())}`\n")
        f.write(f"- Frequency overlap range: `{overlap_min:.4f}` to `{overlap_max:.4f}` Hz\n\n")

        f.write("## Frequency ranges by class\n\n")
        f.write(df_to_markdown(pd.DataFrame(info["freq_ranges"])))
        f.write("\n\n")

        f.write("## Overall metrics\n\n")
        f.write(df_to_markdown(overall))
        f.write("\n\n")

        f.write("## Fold metrics\n\n")
        f.write(df_to_markdown(folds))
        f.write("\n\n")

        f.write("## Output files\n\n")
        for p in [overall_path, folds_path, cm_path, json_path]:
            f.write(f"- `{p}`\n")

    print(f"[DONE] {md_path}")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
