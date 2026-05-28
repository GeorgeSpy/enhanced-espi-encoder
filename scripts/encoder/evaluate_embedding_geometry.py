import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score

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

def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return X / norms

def compute_embedding_geometry(X, y, boards, materials):
    print("Normalizing embeddings...")
    X_norm = l2_normalize(X)

    print("Computing full pairwise Cosine distance matrix...")
    # Cosine distance = 1.0 - Cosine similarity
    # Since inputs are L2 normalized, similarity is just X_norm @ X_norm.T
    sims = np.clip(X_norm @ X_norm.T, -1.0, 1.0)
    D = 1.0 - sims

    print("Calculating silhouette score and Davies-Bouldin index...")
    # Silhouette score using cosine distance
    # Sklearn has a silhouette score that can take precomputed distance matrix or pre-normalized features
    sil = float(silhouette_score(D, y, metric="precomputed"))
    db = float(davies_bouldin_score(X_norm, y))

    print("Calculating intra/inter-class distance ratios...")
    classes = sorted(np.unique(y).tolist())
    intra_inter_ratios = {}
    per_class_metrics = {}

    for c in classes:
        mask_c = (y == c)
        n_c = int(mask_c.sum())
        if n_c < 2:
            continue

        # Intra-class average pairwise distance (excluding self-distance on diagonal)
        sub_D = D[mask_c][:, mask_c]
        intra_mean = float(sub_D[~np.eye(n_c, dtype=bool)].mean())

        # Inter-class average pairwise distance to all other classes
        inter_mean = float(D[mask_c][:, ~mask_c].mean())

        ratio = intra_mean / inter_mean if inter_mean > 0 else 0.0
        intra_inter_ratios[c] = ratio
        per_class_metrics[c] = {
            "class_id": c,
            "support": n_c,
            "intra_mean": intra_mean,
            "inter_mean": inter_mean,
            "ratio": ratio,
        }

    macro_ratio = float(np.mean(list(intra_inter_ratios.values())))

    print("Calculating nearest-neighbor consistency rates...")
    # Exclude self by setting diagonal to infinity
    D_self_excl = D.copy()
    np.fill_diagonal(D_self_excl, np.inf)

    # 1. Standard 1-NN Same-Class Rate
    nn_idx = np.argmin(D_self_excl, axis=1)
    same_class_nn = float((y[nn_idx] == y).mean())

    # 2. Cross-Board Same-Class NN Rate
    print("Computing cross-board NN consistency...")
    D_cb = D.copy()
    same_board_mask = (boards[:, None] == boards[None, :])
    D_cb[same_board_mask] = np.inf
    cb_nn_idx = np.argmin(D_cb, axis=1)
    cross_board_nn = float((y[cb_nn_idx] == y).mean())

    # 3. Cross-Material Same-Class NN Rate
    print("Computing cross-material NN consistency...")
    D_cm = D.copy()
    same_material_mask = (materials[:, None] == materials[None, :])
    D_cm[same_material_mask] = np.inf
    cm_nn_idx = np.argmin(D_cm, axis=1)
    cross_material_nn = float((y[cm_nn_idx] == y).mean())

    return {
        "silhouette": sil,
        "davies_bouldin": db,
        "intra_inter_ratio_macro": macro_ratio,
        "same_class_nn_rate": same_class_nn,
        "cross_board_nn_rate": cross_board_nn,
        "cross_material_nn_rate": cross_material_nn,
        "per_class": per_class_metrics,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v61", required=True, type=Path)
    ap.add_argument("--v62a", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Loading v6.2-A features...")
    d62 = load_npz(args.v62a)
    print("Loading v6.1 features...")
    d61 = load_npz(args.v61)

    print("Aligning features by path...")
    d61 = align_by_paths(d62, d61)
    check_alignment(d62, d61)

    y = d62["y"]
    boards = d62["boards"]
    materials = d62["materials"]
    label_names = d62["label_names"]

    print("\n--- Running Geometry Audit on v6.1 ---")
    g61 = compute_embedding_geometry(d61["X"], y, boards, materials)

    print("\n--- Running Geometry Audit on v6.2-A ---")
    g62 = compute_embedding_geometry(d62["X"], y, boards, materials)

    print("\n--- Collating Global Results ---")
    global_df = pd.DataFrame([
        {
            "Metric": "Silhouette Score (Cosine, higher is better)",
            "v6.1 Reference": g61["silhouette"],
            "v6.2-A Encoder": g62["silhouette"],
            "Delta": g62["silhouette"] - g61["silhouette"],
        },
        {
            "Metric": "Davies-Bouldin Index (L2-norm, lower is better)",
            "v6.1 Reference": g61["davies_bouldin"],
            "v6.2-A Encoder": g62["davies_bouldin"],
            "Delta": g62["davies_bouldin"] - g61["davies_bouldin"],
        },
        {
            "Metric": "Intra/Inter Distance Ratio (Cosine, lower is better)",
            "v6.1 Reference": g61["intra_inter_ratio_macro"],
            "v6.2-A Encoder": g62["intra_inter_ratio_macro"],
            "Delta": g62["intra_inter_ratio_macro"] - g61["intra_inter_ratio_macro"],
        },
        {
            "Metric": "Same-Class 1-NN Consistency Rate (higher is better)",
            "v6.1 Reference": g61["same_class_nn_rate"],
            "v6.2-A Encoder": g62["same_class_nn_rate"],
            "Delta": g62["same_class_nn_rate"] - g61["same_class_nn_rate"],
        },
        {
            "Metric": "Cross-Board Same-Class NN Rate (higher is better)",
            "v6.1 Reference": g61["cross_board_nn_rate"],
            "v6.2-A Encoder": g62["cross_board_nn_rate"],
            "Delta": g62["cross_board_nn_rate"] - g61["cross_board_nn_rate"],
        },
        {
            "Metric": "Cross-Material Same-Class NN Rate (higher is better)",
            "v6.1 Reference": g61["cross_material_nn_rate"],
            "v6.2-A Encoder": g62["cross_material_nn_rate"],
            "Delta": g62["cross_material_nn_rate"] - g61["cross_material_nn_rate"],
        }
    ])

    print(global_df.to_string(index=False))

    print("\n--- Collating Per-Class Metrics ---")
    per_class_rows = []
    for c in sorted(g61["per_class"].keys()):
        class_name = label_names[c] if c < len(label_names) else str(c)
        r61 = g61["per_class"][c]
        r62 = g62["per_class"][c]
        per_class_rows.append({
            "Class ID": c,
            "Class Name": class_name,
            "Support": r61["support"],
            "v6.1 Intra-Dist": r61["intra_mean"],
            "v6.2-A Intra-Dist": r62["intra_mean"],
            "v6.1 Ratio": r61["ratio"],
            "v6.2-A Ratio": r62["ratio"],
            "Ratio Delta": r62["ratio"] - r61["ratio"],
        })
    per_class_df = pd.DataFrame(per_class_rows)
    print(per_class_df.to_string(index=False))

    # Save to CSV/JSON/MD
    global_path = args.outdir / "embedding_geometry_global.csv"
    per_class_path = args.outdir / "embedding_geometry_per_class.csv"
    json_path = args.outdir / "embedding_geometry_data.json"
    md_path = args.outdir / "EMBEDDING_GEOMETRY_SUMMARY.md"

    global_df.to_csv(global_path, index=False)
    per_class_df.to_csv(per_class_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "v61_path": str(args.v61),
            "v62a_path": str(args.v62a),
            "global_metrics": {
                "v61": {k: v for k, v in g61.items() if k != "per_class"},
                "v62a": {k: v for k, v in g62.items() if k != "per_class"},
            },
            "per_class_metrics": {
                "v61": g61["per_class"],
                "v62a": g62["per_class"],
            }
        }, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Embedding Geometry & Nearest-Neighbor Consistency Analysis\n\n")
        f.write("This report presents a direct quantitative analysis of the frozen embedding geometry for **v6.1** and **v6.2-A**.\n\n")
        f.write(f"- v6.1 features: `{args.v61}`\n")
        f.write(f"- v6.2-A features: `{args.v62a}`\n\n")

        f.write("## Global Geometry & Generalization Metrics\n\n")
        f.write("The global metrics assess the cohesion, separation, and generalization (cross-specimen and cross-material consistency) of the representation spaces:\n\n")
        f.write(df_to_markdown(global_df))
        f.write("\n\n")

        f.write("## Per-Class Intra/Inter Cosine Distance Ratio\n\n")
        f.write("Lower Intra/Inter ratios indicate tighter class clusters relative to the background distribution. A negative Delta indicates a better-contained class cluster in v6.2-A:\n\n")
        f.write(df_to_markdown(per_class_df))
        f.write("\n\n")

        f.write("## Output Files\n\n")
        f.write(f"- `{global_path}`\n")
        f.write(f"- `{per_class_path}`\n")
        f.write(f"- `{json_path}`\n")

    print(f"\n[DONE] Saved summary report to: {md_path}")

if __name__ == "__main__":
    main()
