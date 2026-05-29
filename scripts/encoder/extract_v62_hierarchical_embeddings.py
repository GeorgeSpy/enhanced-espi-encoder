#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Frozen hierarchical v6.2 embedding extraction with safe output handling."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from enhanced_espi_encoder import ESPIv62HierarchicalEncoder


DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "hierarchical_embeddings_v001"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "hierarchical_encoder"
DEFAULT_MANIFEST = REPO_ROOT / "artifacts" / "manifests" / "manifest_v1_5class.npz"
FEATURE_FILE = "features_hier_z_expert_prelogit.npz"
SMALL_REPORT_FILE = "EXTRACTION_SMALL_REPORT.md"
SMALL_SUMMARY_FILE = "extraction_small_summary.json"
FULL_REPORT_FILE = "EXTRACTION_FULL_REPORT.md"
FULL_SUMMARY_FILE = "extraction_full_summary.json"
LABEL_NAMES = {
    0: "1_1H",
    1: "1_1T",
    2: "1_2",
    3: "2_1",
    4: "higher",
    5: "noise_unknown",
}


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (REPO_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    data = np.load(str(path), allow_pickle=True)
    return {
        "paths": data["paths"].astype(str),
        "labels": data["labels"].astype(np.int64),
        "boards": data["boards"].astype(str) if "boards" in data else None,
        "freq_hz": data["freq_hz"].astype(np.float32) if "freq_hz" in data else None,
        "train_idx": data["train_idx"].astype(np.int64) if "train_idx" in data else np.array([], dtype=np.int64),
        "val_idx": data["val_idx"].astype(np.int64) if "val_idx" in data else np.array([], dtype=np.int64),
    }


def selected_indices(n_samples: int, max_samples: int, full: bool) -> np.ndarray:
    if full:
        return np.arange(n_samples, dtype=np.int64)
    return np.arange(min(max_samples, n_samples), dtype=np.int64)


def source_group_from_path(path_text: str) -> str:
    path = Path(path_text)
    for part in path.parts:
        if re.match(r"^[CW]\d{2}_ESPI_", part, flags=re.IGNORECASE):
            return part
    return path.parent.name if path.parent.name else "unknown"


def board_from_path(path_text: str, fallback: str | None = None) -> str:
    fallback_text = str(fallback or "")
    for text in (fallback_text, path_text):
        match = re.search(r"\b([CW]\d{2})\b", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return "unknown"


def material_from_board(board: str, path_text: str) -> str:
    lower = path_text.lower()
    if board.startswith("C") or "carbon" in lower:
        return "carbon"
    if board.startswith("W") or "wood" in lower:
        return "wood"
    return "unknown"


def frequency_from_path(path_text: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)Hz", path_text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else float("nan")


def split_array(n_samples: int, train_idx: np.ndarray, val_idx: np.ndarray) -> np.ndarray:
    split = np.full(n_samples, "unknown", dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    return split


def image_to_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - float(array.mean())) / (float(array.std()) + 1e-6)
    return torch.from_numpy(array).unsqueeze(0)


def batched(items: list[Path], batch_size: int) -> list[list[Path]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def quality_checks(embeddings: np.ndarray, metadata: dict[str, np.ndarray]) -> dict[str, Any]:
    paths = metadata["path"].astype(str)
    variances = np.var(embeddings, axis=0) if len(embeddings) else np.array([0.0])
    expected_len = embeddings.shape[0]
    metadata_lengths = {key: int(len(value)) for key, value in metadata.items()}
    return {
        "embedding_shape": list(embeddings.shape),
        "nan_count": int(np.isnan(embeddings).sum()),
        "inf_count": int(np.isinf(embeddings).sum()),
        "no_nan": bool(not np.isnan(embeddings).any()),
        "no_inf": bool(not np.isinf(embeddings).any()),
        "nonzero_variance": bool(np.nanmax(variances) > 0.0 and float(np.nanmean(variances)) > 0.0),
        "consistent_embedding_shape": bool(embeddings.ndim == 2 and embeddings.shape[0] == expected_len and embeddings.shape[1] > 0),
        "duplicate_path_count": int(len(paths) - len(set(paths.tolist()))),
        "no_duplicate_paths": bool(len(paths) == len(set(paths.tolist()))),
        "metadata_lengths": metadata_lengths,
        "metadata_length_equals_embedding_length": bool(all(length == expected_len for length in metadata_lengths.values())),
        "mean_feature_variance": float(np.nanmean(variances)),
        "max_feature_variance": float(np.nanmax(variances)),
    }


def atomic_savez(output_path: Path, **arrays: Any) -> Path:
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return output_path


def write_reports(report_dir: Path, summary: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = FULL_REPORT_FILE if summary["full"] else SMALL_REPORT_FILE
    summary_file = FULL_SUMMARY_FILE if summary["full"] else SMALL_SUMMARY_FILE
    (report_dir / summary_file).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    checks = summary["quality_checks"]
    title = "Full" if summary["full"] else "Small-Sample"
    lines = [
        f"# Hierarchical v6.2 {title} Embedding Extraction",
        "",
        "## Scope",
        "",
        "This report summarizes a frozen hierarchical v6.2 embedding extraction run. It does not train. Full extraction is only enabled when `--full` is explicitly used.",
        "",
        "## Inputs",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Checkpoint SHA256: `{summary['checkpoint_sha256']}`",
        f"- Manifest: `{summary['manifest']}`",
        f"- Output NPZ: `{summary['output_npz']}`",
        f"- Device: `{summary['device']}`",
        f"- Full extraction: `{summary['full']}`",
        f"- Max samples argument: `{summary['max_samples_arg']}`",
        f"- Samples extracted: `{summary['n_samples']}`",
        f"- Embedding point: `{summary['embedding_point']}`",
        f"- Model name: `{summary['model_name']}`",
        f"- Created UTC: `{summary['created_utc']}`",
        "",
        "## Quality Checks",
        "",
        f"- Overall status: **{summary['status']}**",
        f"- Embedding shape: `{checks['embedding_shape']}`",
        f"- No NaN: `{checks['no_nan']}` (`{checks['nan_count']}` NaN values)",
        f"- No Inf: `{checks['no_inf']}` (`{checks['inf_count']}` Inf values)",
        f"- Nonzero variance: `{checks['nonzero_variance']}`",
        f"- Consistent embedding shape: `{checks['consistent_embedding_shape']}`",
        f"- Duplicate paths: `{checks['duplicate_path_count']}`",
        f"- Metadata length equals embedding length: `{checks['metadata_length_equals_embedding_length']}`",
        "",
        "## Notes",
        "",
        "- `z_expert_prelogit` and `z_arcface_input` are extracted from `outputs[\"embeddings\"]`.",
        "- `z_physics_fusion` is intentionally not exposed or extracted in this run.",
        "- This is a small-sample extraction by default; full extraction requires `--full`.",
    ]
    (report_dir / report_file).write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract frozen hierarchical v6.2 embeddings.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args()

    checkpoint = resolve_path(args.checkpoint)
    manifest_path = resolve_path(args.manifest)
    out_dir = resolve_path(args.out_dir)
    report_dir = DEFAULT_REPORT_DIR
    output_npz = out_dir / FEATURE_FILE

    if output_npz.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {output_npz}")
    if not args.full and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive unless --full is used.")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    indices = selected_indices(len(manifest["paths"]), int(args.max_samples), bool(args.full))
    split_all = split_array(len(manifest["paths"]), manifest["train_idx"], manifest["val_idx"])
    dependency_path = manifest_path.parent.parent
    encoder = ESPIv62HierarchicalEncoder(
        checkpoint=checkpoint,
        device=args.device,
        dependency_paths=[dependency_path],
        strict=True,
    )

    selected_paths_text = manifest["paths"][indices].astype(str)
    selected_paths = [Path(path) for path in selected_paths_text]
    missing_paths = [str(path) for path in selected_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing image paths; first missing entries: {missing_paths[:5]}")

    embeddings: list[np.ndarray] = []
    batch_paths_list = batched(selected_paths, int(args.batch_size))
    total_batches = len(batch_paths_list)
    print(
        f"[hier-extract] mode={'full' if args.full else 'small'} "
        f"samples={len(selected_paths)} batches={total_batches} batch_size={args.batch_size}"
    )
    with torch.no_grad():
        for batch_idx, batch_paths in enumerate(batch_paths_list, start=1):
            batch = torch.stack([image_to_tensor(path, int(args.image_size)) for path in batch_paths], dim=0)
            outputs = encoder.encode(batch)
            z = outputs["z_expert_prelogit"]
            if z is None:
                raise RuntimeError("Encoder did not return z_expert_prelogit.")
            embeddings.append(z.detach().cpu().numpy().astype(np.float32))
            if batch_idx == 1 or batch_idx == total_batches or batch_idx % max(1, math.ceil(total_batches / 10)) == 0:
                print(f"[hier-extract] batch {batch_idx}/{total_batches}")

    embedding = np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=np.float32)
    labels = manifest["labels"][indices].astype(np.int64)
    fallback_boards = manifest["boards"][indices].astype(str) if manifest["boards"] is not None else np.array([""] * len(indices))
    distribution_group = np.array([source_group_from_path(path) for path in selected_paths_text], dtype=object)
    board = np.array([board_from_path(path, fallback) for path, fallback in zip(selected_paths_text, fallback_boards)], dtype=object)
    material = np.array([material_from_board(board_id, path) for board_id, path in zip(board, selected_paths_text)], dtype=object)
    frequency = (
        manifest["freq_hz"][indices].astype(np.float32)
        if manifest["freq_hz"] is not None
        else np.array([frequency_from_path(path) for path in selected_paths_text], dtype=np.float32)
    )
    split = split_all[indices]
    split_group = board.astype(object)
    label_name = np.array([LABEL_NAMES.get(int(label), f"class_{int(label)}") for label in labels], dtype=object)
    checkpoint_sha256 = sha256_file(checkpoint)
    embedding_point = "z_expert_prelogit / z_arcface_input from outputs['embeddings']"
    model_name = "EnhancedHybridPhysicsESPI_V6_2"

    metadata = {
        "label": labels,
        "label_name": label_name,
        "board": board,
        "material": material,
        "distribution_group": distribution_group,
        "frequency_hz": frequency,
        "path": selected_paths_text.astype(object),
        "split": split.astype(object),
        "split_group": split_group,
    }
    checks = quality_checks(embedding, metadata)
    status = "pass" if all(
        [
            checks["no_nan"],
            checks["no_inf"],
            checks["nonzero_variance"],
            checks["consistent_embedding_shape"],
            checks["no_duplicate_paths"],
            checks["metadata_length_equals_embedding_length"],
        ]
    ) else "fail"

    atomic_savez(
        output_npz,
        embedding=embedding,
        label=labels,
        label_name=label_name,
        board=board,
        material=material,
        distribution_group=distribution_group,
        frequency_hz=frequency,
        path=selected_paths_text.astype(object),
        split=split.astype(object),
        split_group=split_group,
        checkpoint_sha256=np.array(checkpoint_sha256, dtype=object),
        embedding_point=np.array(embedding_point, dtype=object),
        model_name=np.array(model_name, dtype=object),
    )

    summary = {
        "created_utc": utc_now(),
        "status": status,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "manifest": str(manifest_path),
        "output_npz": str(output_npz),
        "device": str(encoder.device),
        "full": bool(args.full),
        "max_samples_arg": int(args.max_samples),
        "n_samples": int(embedding.shape[0]),
        "embedding_dim": int(embedding.shape[1]) if embedding.ndim == 2 and embedding.shape[0] else 0,
        "embedding_point": embedding_point,
        "model_name": model_name,
        "quality_checks": checks,
        "metadata_fields": list(metadata.keys()),
        "notes": [
            "No training was run.",
            "No gradients were computed.",
            "Full extraction is disabled by default and requires --full.",
            "z_physics_fusion is intentionally not exposed yet.",
        ],
    }
    write_reports(report_dir, summary)

    print(f"[hier-extract] wrote {output_npz}")
    print(f"[hier-extract] wrote {report_dir / (FULL_REPORT_FILE if args.full else SMALL_REPORT_FILE)}")
    print(f"[hier-extract] wrote {report_dir / (FULL_SUMMARY_FILE if args.full else SMALL_SUMMARY_FILE)}")
    print(f"[hier-extract] status: {status}")
    print(f"[hier-extract] embedding shape: {list(embedding.shape)}")
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
