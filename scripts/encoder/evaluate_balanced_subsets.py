import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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
    paths = np.asarray(get(["paths", "path"])).astype(str)
    freq = np.asarray(get(["frequency_hz", "freq_hz", "frequency", "freq", "frequencies"]), dtype=np.float32).reshape(-1)
    boards = np.asarray(get(["boards", "board", "board_id", "split_group"])).astype(str)
    materials = np.asarray(get(["materials", "material", "material_id"])).astype(str)
    splits = np.asarray(get(["splits", "split", "split_label"])).astype(str)
    label_names = get(["label_names", "class_names", "classes"], required=False)

    if label_names is not None:
        label_names = np.asarray(label_names).astype(str)
    else:
        label_names = np.asarray([str(i) for i in sorted(np.unique(y).tolist())])

    return {
        "X": X,
        "y": y,
        "paths": paths,
        "freq": freq,
        "boards": boards,
        "materials": materials,
        "splits": splits,
        "label_names": label_names,
    }


def align_by_paths(ref, other):
    ref_paths = ref["paths"].astype(str)
    other_paths = other["paths"].astype(str)

    index = {p: i for i, p in enumerate(other_paths.tolist())}
    order = []
    missing = []
    for p in ref_paths:
        if p not in index:
            missing.append(p)
        else:
            order.append(index[p])

    if missing:
        raise RuntimeError(f"Missing {len(missing)} paths in second feature file. Example: {missing[:3]}")

    order = np.asarray(order, dtype=int)

    aligned = {}
    for k, v in other.items():
        if k == "label_names":
            aligned[k] = v
        elif len(v) == len(other_paths):
            aligned[k] = v[order]
        else:
            aligned[k] = v

    return aligned


def check_alignment(a, b):
    for k in ["paths", "y", "boards", "materials", "splits"]:
        if not np.array_equal(a[k].astype(str), b[k].astype(str)):
            raise RuntimeError(f"Alignment mismatch in field: {k}")


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
        return [(g, boards != g, boards == g) for g in sorted(set(boards.tolist()))]
    if protocol == "material_lomo":
        return [(g, materials != g, materials == g) for g in sorted(set(materials.tolist()))]
    raise ValueError(protocol)


def metrics(ytrue, ypred):
    return {
        "accuracy": float(accuracy_score(ytrue, ypred)),
        "balanced_accuracy": float(balanced_accuracy_score(ytrue, ypred)),
        "macro_f1": float(f1_score(ytrue, ypred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(ytrue, ypred, average="weighted", zero_division=0)),
    }


def eval_knn(X, y, train_mask, test_mask, k=10):
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr = scaler.fit_transform(X[train_mask])
    Xte = scaler.transform(X[test_mask])

    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
    clf.fit(Xtr, y[train_mask])
    pred = clf.predict(Xte)
    return metrics(y[test_mask], pred)


def eval_lr(X, y, train_mask, test_mask, max_iter=5000):
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr = scaler.fit_transform(X[train_mask])
    Xte = scaler.transform(X[test_mask])

    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(Xtr, y[train_mask])
    pred = clf.predict(Xte)
    return metrics(y[test_mask], pred)


def sample_group_indices(df, group_cols, rng):
    counts = df.groupby(group_cols).size()
    min_n = int(counts.min())

    selected = []
    for _, sub in df.groupby(group_cols):
        chosen = rng.choice(sub.index.to_numpy(), size=min_n, replace=False)
        selected.extend(chosen.tolist())

    return np.asarray(sorted(selected), dtype=int), min_n


def make_freq_bins(freq, width):
    return np.floor(freq / width).astype(int)


def make_subset_indices(kind, y, materials, freq, seed=42, freq_bin_width=25.0, min_cell=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "idx": np.arange(len(y)),
        "label": y,
        "material": materials,
        "freq": freq,
    }).set_index("idx")

    if kind == "class_balanced":
        idx, min_n = sample_group_indices(df, ["label"], rng)
        meta = {"strategy": kind, "min_per_class": min_n}
        return idx, meta

    if kind == "class_material_balanced":
        idx, min_n = sample_group_indices(df, ["label", "material"], rng)
        meta = {"strategy": kind, "min_per_class_material_cell": min_n}
        return idx, meta

    if kind == "class_material_freqmatched":
        df = df.copy()
        df["freq_bin"] = make_freq_bins(df["freq"].to_numpy(), freq_bin_width)

        selected = []
        material_values = sorted(df["material"].unique().tolist())

        # Match materials inside each label x frequency-bin cell.
        for (lab, fb), sub in df.groupby(["label", "freq_bin"]):
            mats_present = sorted(sub["material"].unique().tolist())
            if mats_present != material_values:
                continue

            counts = sub.groupby("material").size()
            if counts.min() < min_cell:
                continue

            n = int(counts.min())
            for mat, sm in sub.groupby("material"):
                chosen = rng.choice(sm.index.to_numpy(), size=n, replace=False)
                selected.extend(chosen.tolist())

        if not selected:
            raise RuntimeError("No samples selected for class_material_freqmatched. Try larger --freq-bin-width or smaller --min-cell.")

        selected = np.asarray(sorted(selected), dtype=int)

        # Final class balancing after material/frequency matching.
        tmp = pd.DataFrame({
            "idx": selected,
            "label": y[selected],
        }).set_index("idx")

        final_idx, min_class = sample_group_indices(tmp, ["label"], rng)

        meta = {
            "strategy": kind,
            "freq_bin_width": freq_bin_width,
            "min_cell": min_cell,
            "n_after_freq_material_matching": int(len(selected)),
            "min_per_class_after_final_balance": int(min_class),
        }
        return final_idx, meta

    raise ValueError(kind)


def summarize_subset(name, idx, y, materials, boards, freq):
    df = pd.DataFrame({
        "label": y[idx],
        "material": materials[idx],
        "board": boards[idx],
        "freq": freq[idx],
    })
    rows = []
    for lab, sub in df.groupby("label"):
        rows.append({
            "subset": name,
            "label": lab,
            "n": int(len(sub)),
            "freq_min": float(sub["freq"].min()),
            "freq_max": float(sub["freq"].max()),
            "freq_mean": float(sub["freq"].mean()),
            "wood_n": int((sub["material"] == "wood").sum()),
            "carbon_n": int((sub["material"] == "carbon").sum()),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v61", required=True, type=Path)
    ap.add_argument("--v62a", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--strategies", nargs="+", default=["class_balanced", "class_material_balanced", "class_material_freqmatched"])
    ap.add_argument("--protocols", nargs="+", default=["stratified", "board_lobo", "material_lomo"])
    ap.add_argument("--freq-bin-width", type=float, default=25.0)
    ap.add_argument("--min-cell", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-iter", type=int, default=5000)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    d62 = load_npz(args.v62a)
    d61 = load_npz(args.v61)
    d61 = align_by_paths(d62, d61)
    check_alignment(d62, d61)

    y = d62["y"]
    freq = d62["freq"]
    boards = d62["boards"]
    materials = d62["materials"]
    splits = d62["splits"]

    all_results = []
    all_folds = []
    subset_rows = []
    subset_meta = {}

    for strategy in args.strategies:
        idx, meta = make_subset_indices(
            strategy,
            y,
            materials,
            freq,
            seed=args.seed,
            freq_bin_width=args.freq_bin_width,
            min_cell=args.min_cell,
        )
        subset_meta[strategy] = meta
        subset_rows.append(summarize_subset(strategy, idx, y, materials, boards, freq))

        print(f"[SUBSET] {strategy}: n={len(idx)} meta={meta}")

        # Slice data
        y_s = y[idx]
        freq_s = freq[idx].reshape(-1, 1)
        boards_s = boards[idx]
        materials_s = materials[idx]
        splits_s = splits[idx]
        X61_s = d61["X"][idx]
        X62_s = d62["X"][idx]

        feature_sets = {
            "frequency_only_lr": ("lr", freq_s),
            "v61_knn": ("knn", X61_s),
            "v62a_knn": ("knn", X62_s),
            "frequency_plus_v62a_lr": ("lr", np.concatenate([freq_s, X62_s], axis=1)),
        }

        for protocol in args.protocols:
            for name, (head, Xuse) in feature_sets.items():
                fold_metrics = []
                for heldout, tr, te in protocol_folds(protocol, boards_s, materials_s, splits_s):
                    if tr.sum() == 0 or te.sum() == 0:
                        continue
                    if len(np.unique(y_s[tr])) < 2 or len(np.unique(y_s[te])) < 2:
                        continue

                    if head == "knn":
                        m = eval_knn(Xuse, y_s, tr, te, k=args.k)
                    else:
                        m = eval_lr(Xuse, y_s, tr, te, max_iter=args.max_iter)

                    row = dict(m)
                    row.update({
                        "subset": strategy,
                        "protocol": protocol,
                        "method": name,
                        "heldout_group": str(heldout),
                        "n_train": int(tr.sum()),
                        "n_test": int(te.sum()),
                    })
                    fold_metrics.append(row)
                    all_folds.append(row)

                if fold_metrics:
                    fdf = pd.DataFrame(fold_metrics)
                    summary = {
                        "subset": strategy,
                        "protocol": protocol,
                        "method": name,
                        "n_folds": int(len(fdf)),
                        "macro_f1_mean": float(fdf["macro_f1"].mean()),
                        "macro_f1_std": float(fdf["macro_f1"].std(ddof=0)) if len(fdf) > 1 else 0.0,
                        "balanced_accuracy_mean": float(fdf["balanced_accuracy"].mean()),
                        "accuracy_mean": float(fdf["accuracy"].mean()),
                        "weighted_f1_mean": float(fdf["weighted_f1"].mean()),
                        "worst_group": str(fdf.sort_values("macro_f1").iloc[0]["heldout_group"]),
                        "worst_group_macro_f1": float(fdf.sort_values("macro_f1").iloc[0]["macro_f1"]),
                        "best_group": str(fdf.sort_values("macro_f1", ascending=False).iloc[0]["heldout_group"]),
                        "best_group_macro_f1": float(fdf.sort_values("macro_f1", ascending=False).iloc[0]["macro_f1"]),
                    }
                    all_results.append(summary)

    results = pd.DataFrame(all_results).sort_values(["subset", "protocol", "method"])
    folds = pd.DataFrame(all_folds).sort_values(["subset", "protocol", "method", "heldout_group"])
    subset_summary = pd.concat(subset_rows, ignore_index=True)

    results_path = args.outdir / "balanced_subset_summary.csv"
    folds_path = args.outdir / "balanced_subset_fold_metrics.csv"
    subset_path = args.outdir / "balanced_subset_composition.csv"
    json_path = args.outdir / "balanced_subset_key_numbers.json"
    md_path = args.outdir / "BALANCED_SUBSET_EVALUATION_SUMMARY.md"

    results.to_csv(results_path, index=False)
    folds.to_csv(folds_path, index=False)
    subset_summary.to_csv(subset_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "v61": str(args.v61),
            "v62a": str(args.v62a),
            "strategies": args.strategies,
            "subset_meta": subset_meta,
            "results": results.to_dict(orient="records"),
        }, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Balanced Subset Evaluation\n\n")
        f.write("This evaluation tests whether the v6.2-A grouped advantage persists under class/material/frequency balancing.\n\n")
        f.write(f"- v6.1 features: `{args.v61}`\n")
        f.write(f"- v6.2-A features: `{args.v62a}`\n")
        f.write(f"- kNN: fixed k={args.k}\n")
        f.write(f"- Frequency-bin width for matched subset: `{args.freq_bin_width}` Hz\n")
        f.write(f"- Min cell for frequency-matched subset: `{args.min_cell}`\n\n")

        f.write("## Subset Composition\n\n")
        f.write(df_to_markdown(subset_summary))
        f.write("\n\n")

        f.write("## Summary Metrics\n\n")
        f.write(df_to_markdown(results))
        f.write("\n\n")

        f.write("## Fold Metrics\n\n")
        f.write(df_to_markdown(folds))
        f.write("\n\n")

        f.write("## Output Files\n\n")
        for p in [results_path, folds_path, subset_path, json_path]:
            f.write(f"- `{p}`\n")

    print(f"[DONE] {md_path}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
