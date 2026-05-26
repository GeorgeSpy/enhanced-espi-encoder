#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract frozen v6.2-A pre-head embeddings.

Default inputs are repository-relative placeholders for reviewer-safe runs:
  - configs/v6_2/config.antigravity.5class.yaml
  - artifacts/manifests/manifest_v1_5class.npz
  - artifacts/checkpoints/checkpoint_epoch25_20260211_035150.pt

The v6.2 helper package can be supplied with --package-root or by setting
ESPI_V62_PACKAGE_ROOT. It is intentionally not hard-coded to a local machine.

The embedding is captured immediately before the fully connected classifier head:
MCDropoutClassifier.global_pool -> flatten -> L2-normalize.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE_ROOT = Path(os.environ.get("ESPI_V62_PACKAGE_ROOT", "external/v6_2_fixed_package"))
DEFAULT_CONFIG = REPO_ROOT / "configs" / "v6_2" / "config.antigravity.5class.yaml"
DEFAULT_MANIFEST = REPO_ROOT / "artifacts" / "manifests" / "manifest_v1_5class.npz"
DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "checkpoint_epoch25_20260211_035150.pt"
DEFAULT_OUT = REPO_ROOT / "outputs" / "features_v62a_epoch25.npz"


LABEL_NAMES = {
    0: "1_1H",
    1: "1_1T",
    2: "1_2",
    3: "2_1",
    4: "higher",
}


def import_v62_helpers(package_root: Path) -> tuple[Any, Any, Any, Any]:
    package_root = package_root.expanduser().resolve()
    if not package_root.exists():
        raise FileNotFoundError(
            "v6.2 helper package not found. Pass --package-root or set "
            f"ESPI_V62_PACKAGE_ROOT. Tried: {package_root}"
        )
    sys.path.insert(0, str(package_root))
    from generate_val_report import build_model_from_config, load_checkpoint_into_model, load_yaml
    from enhanced_espi_pipeline_clean_FIXED import FilteredDataset, make_windows_safe_loader

    return build_model_from_config, load_checkpoint_into_model, load_yaml, (FilteredDataset, make_windows_safe_loader)


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = np.load(str(path), allow_pickle=True)
    data = {
        "paths": manifest["paths"].astype(str),
        "labels": manifest["labels"].astype(np.int64),
        "train_idx": manifest["train_idx"].astype(np.int64) if "train_idx" in manifest else np.array([], dtype=np.int64),
        "val_idx": manifest["val_idx"].astype(np.int64) if "val_idx" in manifest else np.array([], dtype=np.int64),
        "boards_raw": manifest["boards"].astype(str) if "boards" in manifest else None,
        "freq_hz": manifest["freq_hz"].astype(np.float32) if "freq_hz" in manifest else None,
        "energy": manifest["energy"].astype(np.float32) if "energy" in manifest else None,
        "meta_json": str(manifest["meta_json"]) if "meta_json" in manifest else "{}",
    }
    return data


def parse_source_group(path_text: str, fallback: str | None = None) -> str:
    parts = Path(path_text).parts
    for part in parts:
        if re.match(r"^[CW]\d{2}_ESPI_", part, flags=re.IGNORECASE):
            return part
    return fallback or "unknown"


def parse_board_id(source_group: str, path_text: str) -> str:
    match = re.search(r"\b([CW]\d{2})\b", source_group, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([CW]\d{2})_ESPI_", path_text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "unknown"


def parse_material(board_id: str, path_text: str) -> str:
    lower = path_text.lower()
    if "carbon" in lower or board_id.startswith("C"):
        return "carbon"
    if "wood" in lower or board_id.startswith("W"):
        return "wood"
    return "unknown"


def parse_domain(source_group: str, path_text: str) -> str:
    lower = f"{source_group} {path_text}".lower()
    if "pseudonoisy" in lower or "pseudo-noisy" in lower:
        return "pseudo-noisy"
    if "averaged" in lower:
        return "averaged"
    return "clean"


def parse_frequency(path_text: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)Hz", path_text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else float("nan")


def split_lookup(n_samples: int, train_idx: np.ndarray, val_idx: np.ndarray) -> np.ndarray:
    split = np.full(n_samples, "unknown", dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    return split


def selected_indices(args: argparse.Namespace, manifest: dict[str, Any]) -> np.ndarray:
    n_samples = len(manifest["paths"])
    if args.split == "all":
        indices = np.arange(n_samples, dtype=np.int64)
    elif args.split == "train":
        indices = manifest["train_idx"]
    elif args.split == "val":
        indices = manifest["val_idx"]
    else:
        raise ValueError(f"Unsupported split: {args.split}")

    if args.max_samples and args.max_samples > 0:
        indices = indices[: args.max_samples]
    return indices.astype(np.int64)


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.norm(x, dim=1, keepdim=True), min=eps)


def write_metadata_csv(path: Path, metadata: dict[str, np.ndarray]) -> None:
    columns = [
        "sample_id",
        "label",
        "label_name",
        "board",
        "source_group",
        "material",
        "frequency_hz",
        "domain",
        "split",
        "split_group",
        "path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row_idx in range(len(metadata["sample_id"])):
            writer.writerow({col: metadata[col][row_idx] for col in columns})


def write_summary(path: Path, args: argparse.Namespace, out_npz: Path, n_samples: int, embedding_dim: int, missing: list[str], unexpected: list[str]) -> None:
    lines = [
        "# v6.2-A Frozen Embedding Dump",
        "",
        "## Inputs",
        f"- Config: `{args.config}`",
        f"- Manifest: `{args.manifest}`",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- v6.2 package root: `{args.package_root}`",
        "",
        "## Output",
        f"- Features: `{out_npz}`",
        f"- Samples: `{n_samples}`",
        f"- Embedding dim: `{embedding_dim}`",
        f"- Embedding layer: `MCDropoutClassifier.global_pool` pre-FC",
        f"- L2 normalized: `{not args.no_normalize}`",
        "",
        "## Load Audit",
        f"- Missing keys: `{len(missing)}`",
        f"- Unexpected keys: `{len(unexpected)}`",
        "",
        "## Created",
        f"- UTC: `{datetime.now(timezone.utc).isoformat()}`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen v6.2-A pre-head embeddings.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=DEFAULT_PACKAGE_ROOT,
        help="Directory containing generate_val_report.py and enhanced_espi_pipeline_clean_FIXED.py.",
    )
    parser.add_argument("--split", choices=["all", "train", "val"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=0, help="Smoke-test limit; 0 means no limit.")
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    build_model_from_config, load_checkpoint_into_model, load_yaml, dataset_helpers = import_v62_helpers(args.package_root)
    FilteredDataset, make_windows_safe_loader = dataset_helpers

    cfg = load_yaml(args.config)
    manifest = load_manifest(args.manifest)
    indices = selected_indices(args, manifest)
    split_all = split_lookup(len(manifest["paths"]), manifest["train_idx"], manifest["val_idx"])

    eval_cfg = dict(cfg)
    eval_cfg["data"] = dict(cfg.get("data", {}))
    eval_cfg["data"]["use_cache_train"] = False
    eval_cfg["data"]["use_cache_val"] = False

    dataset = FilteredDataset(manifest["paths"], manifest["labels"], eval_cfg, None)
    dataset.augment = False
    dataset.advanced_augment = False
    subset = torch.utils.data.Subset(dataset, indices)

    batch_size = args.batch_size or int(cfg.get("training", {}).get("eval_batch_size", 8))
    loader = make_windows_safe_loader(subset, batch_size=batch_size, shuffle=False)

    model = build_model_from_config(cfg)
    missing, unexpected, _ = load_checkpoint_into_model(model, args.checkpoint)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model.to(device)
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def capture_global_pool(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured["embedding"] = output.detach().flatten(1)

    hook_handle = model.classifier.global_pool.register_forward_hook(capture_global_pool)

    embeddings: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []

    use_amp = bool(cfg.get("training", {}).get("amp", False)) and device.type == "cuda"

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                images = batch["image"].to(device, non_blocking=False)
                captured.clear()
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(images, return_uncertainty=False, use_bce=False)
                else:
                    outputs = model(images, return_uncertainty=False, use_bce=False)

                logits = outputs["logits"].detach().float().cpu()
                embedding = captured["embedding"].float()
                if not args.no_normalize:
                    embedding = l2_normalize(embedding)
                embeddings.append(embedding.cpu().numpy().astype(np.float32))
                logits_all.append(logits.numpy().astype(np.float32))
                preds_all.append(torch.argmax(logits, dim=1).numpy().astype(np.int64))

                if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
                    print(f"[extract] batch {batch_idx + 1}/{len(loader)}")
    finally:
        hook_handle.remove()

    embedding_matrix = np.concatenate(embeddings, axis=0)
    logits_matrix = np.concatenate(logits_all, axis=0)
    pred_vector = np.concatenate(preds_all, axis=0)

    selected_paths = manifest["paths"][indices].astype(str)
    selected_labels = manifest["labels"][indices].astype(np.int64)
    boards_raw = manifest["boards_raw"]
    selected_fallback_groups = boards_raw[indices].astype(str) if boards_raw is not None else np.array(["unknown"] * len(indices), dtype=object)
    freq_hz = manifest["freq_hz"][indices].astype(np.float32) if manifest["freq_hz"] is not None else np.array([parse_frequency(p) for p in selected_paths], dtype=np.float32)
    energy = manifest["energy"][indices].astype(np.float32) if manifest["energy"] is not None else np.full(len(indices), np.nan, dtype=np.float32)

    source_group = np.array([parse_source_group(path, fallback) for path, fallback in zip(selected_paths, selected_fallback_groups)], dtype=object)
    board = np.array([parse_board_id(group, path) for group, path in zip(source_group, selected_paths)], dtype=object)
    material = np.array([parse_material(board_id, path) for board_id, path in zip(board, selected_paths)], dtype=object)
    domain = np.array([parse_domain(group, path) for group, path in zip(source_group, selected_paths)], dtype=object)
    label_name = np.array([LABEL_NAMES.get(int(label), str(int(label))) for label in selected_labels], dtype=object)
    split = split_all[indices].astype(object)
    split_group = board.copy()
    split_group[split_group == "unknown"] = source_group[split_group == "unknown"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        embeddings=embedding_matrix,
        logits=logits_matrix,
        preds=pred_vector,
        sample_id=indices.astype(np.int64),
        label=selected_labels,
        label_name=label_name,
        board=board,
        source_group=source_group,
        material=material,
        frequency_hz=freq_hz,
        energy=energy,
        domain=domain,
        path=selected_paths.astype(object),
        split=split,
        split_group=split_group,
        manifest_path=str(args.manifest),
        checkpoint_path=str(args.checkpoint),
        config_path=str(args.config),
        created_utc=datetime.now(timezone.utc).isoformat(),
        embedding_layer="MCDropoutClassifier.global_pool.pre_fc",
        normalized=not args.no_normalize,
        load_missing=np.array(list(missing), dtype=object),
        load_unexpected=np.array(list(unexpected), dtype=object),
    )

    metadata = {
        "sample_id": indices.astype(object),
        "label": selected_labels.astype(object),
        "label_name": label_name,
        "board": board,
        "source_group": source_group,
        "material": material,
        "frequency_hz": freq_hz.astype(object),
        "domain": domain,
        "split": split,
        "split_group": split_group,
        "path": selected_paths.astype(object),
    }
    write_metadata_csv(args.out.with_suffix(".metadata.csv"), metadata)
    write_summary(
        args.out.with_suffix(".FEATURE_DUMP_SUMMARY.md"),
        args,
        args.out,
        n_samples=embedding_matrix.shape[0],
        embedding_dim=embedding_matrix.shape[1],
        missing=list(missing),
        unexpected=list(unexpected),
    )

    print(f"[extract] saved {args.out}")
    print(f"[extract] embeddings={embedding_matrix.shape} missing={len(missing)} unexpected={len(unexpected)}")


if __name__ == "__main__":
    main()
