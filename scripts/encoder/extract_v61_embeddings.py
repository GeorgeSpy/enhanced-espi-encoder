#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract frozen v6.1 embeddings in the normalized encoder feature schema.

This script performs deterministic eval-mode feature extraction only. It does
not train, fine-tune, evaluate, implement LeFFT, or run acoustic-response tasks.
"""

from __future__ import annotations

import argparse
import hashlib
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
from torchvision.models import resnet18


DEFAULT_OUT_DIR = Path("outputs/encoder_features_normalized_v001")
DEFAULT_REPORT_DIR = Path("reports/encoder_baselines")
DEFAULT_OUTPUT_NAME = "features_v61.normalized.npz"
SCHEMA_VERSION = "encoder_feature_schema_v001"

LABEL_NAMES = {
    0: "1_1H",
    1: "1_1T",
    2: "1_2",
    3: "2_1",
    4: "higher",
}


def set_deterministic(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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


class ResNet18HeadOnly(nn.Module):
    """Documented v6.1 architecture reconstructed without training dependencies."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint_with_audit(model: nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    raw = safe_torch_load(checkpoint_path)
    state_dict = raw.get("state_dict", raw.get("model_state_dict", raw)) if isinstance(raw, dict) else raw
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(raw)}")

    model_state = model.state_dict()
    loadable = {}
    shape_mismatches = []
    unexpected = []
    for key, value in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": list(value.shape),
                    "model_shape": list(model_state[key].shape),
                }
            )
            continue
        loadable[key] = value

    missing = [key for key in model_state.keys() if key not in loadable]
    load_result = model.load_state_dict(loadable, strict=False)
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_keys": int(len(state_dict)),
        "model_keys": int(len(model_state)),
        "matched_keys": int(len(loadable)),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": unexpected + list(load_result.unexpected_keys),
        "shape_mismatches": shape_mismatches,
        "status": "clean" if not missing and not unexpected and not shape_mismatches else "non_strict",
    }


class ESPIDataset(Dataset):
    def __init__(self, manifest: dict[str, Any], indices: np.ndarray, image_root: Path) -> None:
        self.manifest = manifest
        self.indices = indices.astype(np.int64)
        self.image_root = image_root

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        source_idx = int(self.indices[item])
        path_text = str(self.manifest["paths"][source_idx])
        image_path = resolve_image_path(path_text, self.image_root)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path} (source path: {path_text})")
        image = Image.open(image_path).convert("L").resize((256, 256))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - float(array.mean())) / (float(array.std()) + 1e-6)
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor, source_idx


def build_metadata(manifest: dict[str, Any], indices: np.ndarray) -> dict[str, np.ndarray]:
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


def extract_embeddings(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(args.manifest)
    indices = selected_indices(len(manifest["paths"]), args.max_samples, args.full)
    device = torch.device(args.device)

    model = ResNet18HeadOnly(num_classes=5)
    load_audit = load_checkpoint_with_audit(model, args.checkpoint)
    model.eval()
    model.to(device)

    dataset = ESPIDataset(manifest, indices, args.image_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    embeddings: list[np.ndarray] = []
    seen_indices: list[np.ndarray] = []

    with torch.inference_mode():
        for batch_idx, (images, source_indices) in enumerate(loader, start=1):
            images = images.to(device)
            features = model.forward_features(images).detach().cpu().numpy().astype(np.float32)
            embeddings.append(features)
            seen_indices.append(source_indices.numpy().astype(np.int64))
            if batch_idx == 1 or batch_idx % 10 == 0:
                print(f"[v6.1] batch={batch_idx} samples={sum(len(item) for item in seen_indices)}")

    embedding = np.concatenate(embeddings, axis=0)
    output_indices = np.concatenate(seen_indices, axis=0)
    metadata = build_metadata(manifest, output_indices)
    quality = quality_checks(embedding, metadata)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / DEFAULT_OUTPUT_NAME
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {out_path}")
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
            checkpoint_path=np.array(str(args.checkpoint), dtype=object),
            checkpoint_sha256=np.array(load_audit["checkpoint_sha256"], dtype=object),
            embedding_point=np.array("avgpool_pre_classifier", dtype=object),
            model_name=np.array("v6.1 ResNet18_HeadOnly", dtype=object),
            source_npz=np.array("generated_from_manifest_images", dtype=object),
            schema_version=np.array(SCHEMA_VERSION, dtype=object),
        )
    tmp_path.replace(out_path)

    acceptable = quality_passed(quality) and load_audit["status"] == "clean"
    return {
        "output_npz": str(out_path),
        "manifest": str(args.manifest),
        "image_root": str(args.image_root),
        "full": bool(args.full),
        "max_samples": int(args.max_samples),
        "batch_size": int(args.batch_size),
        "device": str(args.device),
        "preprocessing_strategy": "v6.1 documented preprocessing: grayscale 1-channel, resize 256x256, per-image z-score normalization",
        "model_name": "v6.1 ResNet18_HeadOnly",
        "embedding_point": "avgpool_pre_classifier",
        "embedding_dim": int(embedding.shape[1]),
        "quality_checks": quality,
        "load_audit": load_audit,
        "acceptable_for_internal_comparison": bool(acceptable),
        "acceptable_for_final_publication_tables": bool(acceptable),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    audit = summary["load_audit"]
    quality = summary["quality_checks"]
    text = f"""# v6.1 Frozen Embedding Extraction

## Scope

This report documents deterministic eval-mode extraction of v6.1 frozen embeddings. No training, fine-tuning, unified evaluation, LeFFT, or acoustic-response prediction was performed.

## Inputs

- Manifest: `{summary['manifest']}`
- Image root: `{summary['image_root']}`
- Checkpoint: `{audit['checkpoint_path']}`
- Checkpoint SHA256: `{audit['checkpoint_sha256']}`
- Device: `{summary['device']}`
- Batch size: `{summary['batch_size']}`
- Full extraction: `{summary['full']}`
- Max samples when not full: `{summary['max_samples']}`

## Model

- Model: `{summary['model_name']}`
- Architecture source: reconstructed from `scripts/v6_1/train_v6_1_head_only.py`
- Backbone: ResNet-18 style, 1-channel ESPI input
- Classifier head: Linear(512, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 5)
- Embedding point: `{summary['embedding_point']}`
- Embedding dimension: `{summary['embedding_dim']}`
- Preprocessing: `{summary['preprocessing_strategy']}`

## Load audit

- Status: `{audit['status']}`
- Checkpoint keys: `{audit['checkpoint_keys']}`
- Model keys: `{audit['model_keys']}`
- Matched keys: `{audit['matched_keys']}`
- Missing keys: `{len(audit['missing_keys'])}`
- Unexpected keys: `{len(audit['unexpected_keys'])}`
- Shape mismatches: `{len(audit['shape_mismatches'])}`

## Quality checks

- Sample count: `{quality['sample_count']}`
- Embedding shape: `{quality['embedding_shape']}`
- NaN count: `{quality['nan_count']}`
- Inf count: `{quality['inf_count']}`
- Duplicate paths: `{quality['duplicate_paths']}`
- Nonzero variance: `{quality['nonzero_variance']}`
- Metadata lengths match embeddings: `{quality['metadata_lengths_match']}`

## Output

- NPZ: `{summary['output_npz']}`
- Internal comparison acceptable: `{summary['acceptable_for_internal_comparison']}`
- Final publication tables acceptable: `{summary['acceptable_for_final_publication_tables']}`
- Schema version: `{SCHEMA_VERSION}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract v6.1 frozen embeddings.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_deterministic()
    summary = extract_embeddings(args)
    report_path = args.report_dir / "V61_EMBEDDING_EXTRACTION.md"
    json_path = args.report_dir / "v61_embedding_extraction.json"
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        **summary,
    }
    write_report(report_path, summary)
    write_json(json_path, payload)
    print(f"[done] report={report_path}")
    print(f"[done] key_numbers={json_path}")
    print(
        "[done] output={output} shape={shape} load_status={load_status} internal={internal} final={final}".format(
            output=summary["output_npz"],
            shape=summary["quality_checks"]["embedding_shape"],
            load_status=summary["load_audit"]["status"],
            internal=summary["acceptable_for_internal_comparison"],
            final=summary["acceptable_for_final_publication_tables"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
