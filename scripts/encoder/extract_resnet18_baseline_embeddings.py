#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract random and ImageNet ResNet-18 baseline embeddings for encoder comparison.

This script performs deterministic eval-mode feature extraction only. It does
not train, fine-tune, evaluate, implement LeFFT, or run acoustic-response tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


DEFAULT_OUT_DIR = Path("outputs/encoder_features_normalized_v001")
DEFAULT_REPORT_DIR = Path("reports/encoder_baselines")
SCHEMA_VERSION = "encoder_feature_schema_v001"

LABEL_NAMES = {
    0: "1_1H",
    1: "1_1T",
    2: "1_2",
    3: "2_1",
    4: "higher",
}

VARIANT_OUTPUTS = {
    "random": "features_resnet18_random.normalized.npz",
    "imagenet": "features_resnet18_imagenet.normalized.npz",
}


def set_deterministic(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def extract_board(*values: Any) -> str:
    joined = " ".join(str(value) for value in values if value is not None).upper()
    match = re.search(r"([CW]\d{2})", joined)
    return match.group(1) if match else "unknown"


def derive_material(board: str, *values: Any) -> str:
    joined = f"{board} " + " ".join(str(value) for value in values if value is not None)
    lower = joined.lower()
    if board.upper().startswith("C") or "carbon" in lower:
        return "carbon"
    if board.upper().startswith("W") or "wood" in lower:
        return "wood"
    return "unknown"


def resolve_image_path(path_text: str, image_root: Path) -> Path:
    original = Path(path_text)
    if original.exists():
        return original
    if original.is_absolute():
        relative_parts = original.parts[1:]
        for idx, part in enumerate(relative_parts):
            if str(part).lower() == "data":
                candidate = image_root.joinpath(*relative_parts[idx + 1 :])
                if candidate.exists():
                    return candidate
        candidate = image_root / original.name
        if candidate.exists():
            return candidate
        return original
    candidate = image_root / original
    if candidate.exists():
        return candidate
    return original


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with np.load(str(path), allow_pickle=True) as data:
        paths = data["paths"].astype(str) if "paths" in data.files else data["path"].astype(str)
        labels = data["labels"].astype(np.int64) if "labels" in data.files else data["label"].astype(np.int64)
        groups = data["boards"].astype(str) if "boards" in data.files else np.array(["unknown"] * len(paths), dtype=object)
        frequency = (
            data["freq_hz"].astype(np.float32)
            if "freq_hz" in data.files
            else data["frequency_hz"].astype(np.float32)
            if "frequency_hz" in data.files
            else np.full(len(paths), np.nan, dtype=np.float32)
        )
        split = np.array(["unknown"] * len(paths), dtype=object)
        if "train_idx" in data.files:
            split[data["train_idx"].astype(np.int64)] = "train"
        if "val_idx" in data.files:
            split[data["val_idx"].astype(np.int64)] = "val"
    return {
        "paths": paths,
        "labels": labels,
        "distribution_group": groups.astype(object),
        "frequency_hz": frequency,
        "split": split,
    }


def selected_indices(n_samples: int, max_samples: int, full: bool) -> np.ndarray:
    if full:
        return np.arange(n_samples, dtype=np.int64)
    limit = max(1, min(max_samples, n_samples))
    return np.arange(limit, dtype=np.int64)


class ESPIDataset(Dataset):
    def __init__(
        self,
        manifest: dict[str, Any],
        indices: np.ndarray,
        image_root: Path,
        transform: Any,
        input_mode: str,
    ) -> None:
        self.manifest = manifest
        self.indices = indices.astype(np.int64)
        self.image_root = image_root
        self.transform = transform
        self.input_mode = input_mode

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        source_idx = int(self.indices[item])
        path_text = str(self.manifest["paths"][source_idx])
        image_path = resolve_image_path(path_text, self.image_root)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path} (source path: {path_text})")
        if self.input_mode == "rgb":
            image = Image.open(image_path).convert("L").convert("RGB")
        else:
            image = Image.open(image_path).convert("L")
        return self.transform(image), source_idx


def build_model_and_transform(variant: str, device: torch.device) -> tuple[nn.Module, Any, str, str, str]:
    resize = transforms.Resize((256, 256))
    if variant == "random":
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        input_mode = "grayscale"
        transform = transforms.Compose(
            [
                resize,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        preprocessing = "grayscale 1-channel input, resized to 256x256, normalized with mean=0.5/std=0.5; conv1 adapted to 1 input channel"
        checkpoint_path = "torchvision:none"
        model_name = "torchvision_resnet18_random"
    elif variant == "imagenet":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        input_mode = "rgb"
        transform = transforms.Compose(
            [
                resize,
                transforms.ToTensor(),
                transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
            ]
        )
        preprocessing = "grayscale ESPI image repeated to RGB, resized to 256x256, ImageNet mean/std normalization"
        checkpoint_path = "torchvision:ResNet18_Weights.IMAGENET1K_V1"
        model_name = "torchvision_resnet18_imagenet"
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model, transform, input_mode, preprocessing, checkpoint_path, model_name


def build_metadata(manifest: dict[str, Any], indices: np.ndarray, image_root: Path) -> dict[str, np.ndarray]:
    paths = manifest["paths"][indices].astype(str)
    labels = manifest["labels"][indices].astype(np.int64)
    distribution_group = manifest["distribution_group"][indices].astype(str)
    frequency_hz = manifest["frequency_hz"][indices].astype(np.float32)
    split = manifest["split"][indices].astype(object)

    board_values = []
    material_values = []
    split_group_values = []
    label_names = []
    for label, path_text, group in zip(labels, paths, distribution_group):
        board = extract_board(group, path_text)
        material = derive_material(board, group, path_text)
        board_values.append(board)
        material_values.append(material)
        split_group_values.append(board)
        label_names.append(LABEL_NAMES.get(int(label), str(int(label))))

    return {
        "label": labels,
        "label_name": np.array(label_names, dtype=object),
        "path": paths.astype(object),
        "board": np.array(board_values, dtype=object),
        "material": np.array(material_values, dtype=object),
        "frequency_hz": frequency_hz,
        "split": split,
        "split_group": np.array(split_group_values, dtype=object),
        "distribution_group": distribution_group.astype(object),
    }


def quality_checks(embedding: np.ndarray, metadata: dict[str, np.ndarray]) -> dict[str, Any]:
    paths = metadata["path"].astype(str)
    metadata_lengths_match = all(len(value) == len(embedding) for value in metadata.values())
    return {
        "sample_count": int(len(embedding)),
        "embedding_shape": [int(dim) for dim in embedding.shape],
        "nan_count": int(np.isnan(embedding).sum()),
        "inf_count": int(np.isinf(embedding).sum()),
        "duplicate_paths": int(len(paths) - len(set(paths.tolist()))),
        "nonzero_variance": bool(np.any(np.var(embedding, axis=0) > 0.0)),
        "metadata_lengths_match": bool(metadata_lengths_match),
    }


def quality_passed(quality: dict[str, Any]) -> bool:
    return (
        quality["sample_count"] > 0
        and len(quality["embedding_shape"]) == 2
        and quality["embedding_shape"][1] == 512
        and quality["nan_count"] == 0
        and quality["inf_count"] == 0
        and quality["duplicate_paths"] == 0
        and quality["nonzero_variance"]
        and quality["metadata_lengths_match"]
    )


def extract_variant(
    variant: str,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    indices: np.ndarray,
) -> dict[str, Any]:
    device = torch.device(args.device)
    model, transform, input_mode, preprocessing, checkpoint_path, model_name = build_model_and_transform(variant, device)
    dataset = ESPIDataset(manifest, indices, args.image_root, transform, input_mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    embeddings: list[np.ndarray] = []
    seen_indices: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_idx, (images, source_indices) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=False)
            features = model(images).detach().cpu().numpy().astype(np.float32)
            embeddings.append(features)
            seen_indices.append(source_indices.numpy().astype(np.int64))
            if batch_idx == 1 or batch_idx % 10 == 0:
                print(f"[{variant}] batch={batch_idx} samples={sum(len(x) for x in seen_indices)}")

    embedding = np.concatenate(embeddings, axis=0)
    output_indices = np.concatenate(seen_indices, axis=0)
    metadata = build_metadata(manifest, output_indices, args.image_root)
    quality = quality_checks(embedding, metadata)

    out_path = args.out_dir / VARIANT_OUTPUTS[variant]
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            embedding=embedding,
            label=metadata["label"],
            label_name=metadata["label_name"],
            path=metadata["path"],
            board=metadata["board"],
            material=metadata["material"],
            frequency_hz=metadata["frequency_hz"],
            split=metadata["split"],
            split_group=metadata["split_group"],
            distribution_group=metadata["distribution_group"],
            checkpoint_path=np.array(checkpoint_path, dtype=object),
            checkpoint_sha256=np.array("not_applicable", dtype=object),
            embedding_point=np.array("avgpool_pre_fc", dtype=object),
            model_name=np.array(model_name, dtype=object),
            source_npz=np.array("generated_from_manifest_images", dtype=object),
            schema_version=np.array(SCHEMA_VERSION, dtype=object),
        )
    tmp_path.replace(out_path)

    acceptable = quality_passed(quality)
    return {
        "variant": variant,
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": "not_applicable",
        "embedding_point": "avgpool_pre_fc",
        "preprocessing_strategy": preprocessing,
        "output_npz": str(out_path),
        "full": bool(args.full),
        "max_samples": int(args.max_samples),
        "quality_checks": quality,
        "acceptable_for_internal_comparison": acceptable,
        "acceptable_for_final_publication_tables": acceptable,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, args: argparse.Namespace, summaries: list[dict[str, Any]]) -> None:
    rows = []
    for item in summaries:
        q = item["quality_checks"]
        rows.append(
            "| {variant} | {strategy} | {samples} | {dim} | {nan} | {inf} | {dups} | {variance} | {internal} | {final} | `{out}` |".format(
                variant=item["variant"],
                strategy=item["preprocessing_strategy"],
                samples=q["sample_count"],
                dim=q["embedding_shape"][1] if len(q["embedding_shape"]) == 2 else "n/a",
                nan=q["nan_count"],
                inf=q["inf_count"],
                dups=q["duplicate_paths"],
                variance=q["nonzero_variance"],
                internal=item["acceptable_for_internal_comparison"],
                final=item["acceptable_for_final_publication_tables"],
                out=item["output_npz"],
            )
        )
    text = f"""# ResNet-18 Baseline Embedding Extraction

## Scope

This report documents deterministic eval-mode embedding extraction for random and ImageNet-pretrained ResNet-18 baselines. No training, fine-tuning, evaluation, LeFFT, or acoustic-response prediction was performed.

## Inputs

- Manifest: `{args.manifest}`
- Image root: `{args.image_root}`
- Output directory: `{args.out_dir}`
- Device: `{args.device}`
- Batch size: `{args.batch_size}`
- Full extraction: `{args.full}`
- Max samples when not full: `{args.max_samples}`

## Preprocessing note

All ESPI images are loaded as grayscale and resized to `256 x 256`.
The random baseline uses a one-channel ResNet-18 `conv1` adaptation with mean/std `0.5/0.5`.
The ImageNet baseline repeats grayscale to RGB and applies ImageNet mean/std normalization.

## Quality summary

| Variant | Preprocessing strategy | Samples | Dim | NaN | Inf | Duplicate paths | Nonzero variance | Internal comparison | Final publication tables | Output NPZ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

## Output schema

Each NPZ uses schema `{SCHEMA_VERSION}` and contains:

`embedding, label, label_name, path, board, material, frequency_hz, split, split_group, distribution_group, checkpoint_path, checkpoint_sha256, embedding_point, model_name, source_npz, schema_version`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract random/ImageNet ResNet-18 baseline embeddings.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--variant", choices=["random", "imagenet"], action="append", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_deterministic()
    variants = args.variant or ["random", "imagenet"]
    manifest = load_manifest(args.manifest)
    indices = selected_indices(len(manifest["paths"]), args.max_samples, args.full)
    print(f"[info] variants={variants} samples={len(indices)} full={args.full}")

    summaries = []
    for variant in variants:
        summaries.append(extract_variant(variant, args, manifest, indices))

    report_path = args.report_dir / "RESNET18_BASELINE_EXTRACTION.md"
    json_path = args.report_dir / "resnet18_baseline_extraction.json"
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "manifest": str(args.manifest),
        "image_root": str(args.image_root),
        "out_dir": str(args.out_dir),
        "full": bool(args.full),
        "max_samples": int(args.max_samples),
        "variants": summaries,
    }
    write_report(report_path, args, summaries)
    write_json(json_path, payload)

    print(f"[done] report={report_path}")
    print(f"[done] key_numbers={json_path}")
    for summary in summaries:
        print(
            "[done] {variant} output={output} shape={shape} internal={internal} final={final}".format(
                variant=summary["variant"],
                output=summary["output_npz"],
                shape=summary["quality_checks"]["embedding_shape"],
                internal=summary["acceptable_for_internal_comparison"],
                final=summary["acceptable_for_final_publication_tables"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
