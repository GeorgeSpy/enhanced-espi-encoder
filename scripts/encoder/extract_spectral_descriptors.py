#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract deterministic spectral descriptors for ESPI images.

The output is a lightweight spectral / LeFFT-inspired descriptor baseline. It
is not a trained LeFFT model and does not train, fine-tune, evaluate encoders,
or run acoustic-response prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_OUT_DIR = Path("outputs/encoder_features_normalized_v001")
DEFAULT_REPORT_DIR = Path("reports/publication_assets/methodological_hardening_v001/lefft_descriptor_ablation")
DEFAULT_OUTPUT_NAME = "features_spectral_descriptors.normalized.npz"
SCHEMA_VERSION = "encoder_feature_schema_v001"

LABEL_NAMES = {
    0: "1_1H",
    1: "1_1T",
    2: "1_2",
    3: "2_1",
    4: "higher",
}


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
    return np.arange(max(1, min(max_samples, n_samples)), dtype=np.int64)


def robust_normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        mean = float(np.mean(finite))
        std = float(np.std(finite))
        if std <= 1e-8:
            return np.zeros_like(image, dtype=np.float32)
        return np.clip((image - mean) / (3.0 * std), -1.0, 1.0).astype(np.float32)
    normalized = np.clip((image - low) / (high - low), 0.0, 1.0)
    return normalized.astype(np.float32)


def load_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32)


def load_roi_mask(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"ROI mask not found: {path}")
    with Image.open(path) as image:
        mask = image.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
        return (np.asarray(mask, dtype=np.float32) > 127.0).astype(np.float32)


def spectral_geometry(size: int = 256) -> dict[str, np.ndarray]:
    y, x = np.indices((size, size), dtype=np.float32)
    center = (size - 1) / 2.0
    x = x - center
    y = y - center
    radius = np.sqrt(x * x + y * y)
    radius_norm = radius / (np.max(radius) + 1e-12)
    angle = (np.arctan2(y, x) + math.pi) / (2.0 * math.pi)
    return {"radius": radius_norm.astype(np.float32), "angle": angle.astype(np.float32), "x": x, "y": y}


def descriptor_names() -> list[str]:
    names: list[str] = []
    names.extend([f"radial_energy_bin_{idx:02d}" for idx in range(8)])
    names.extend([f"angular_energy_bin_{idx:02d}" for idx in range(12)])
    names.extend(["low_frequency_energy_ratio", "mid_frequency_energy_ratio", "high_frequency_energy_ratio"])
    names.extend(["spectral_centroid_radius", "spectral_bandwidth", "spectral_entropy"])
    names.extend(["peak_concentration_top1", "peak_concentration_top5", "peak_concentration_top10"])
    names.extend(
        [
            "orientation_anisotropy_max_over_mean",
            "horizontal_energy_ratio",
            "vertical_energy_ratio",
            "diagonal_energy_ratio",
        ]
    )
    names.extend(
        [
            "patch_low_energy_mean",
            "patch_mid_energy_mean",
            "patch_high_energy_mean",
            "patch_low_energy_std",
            "patch_mid_energy_std",
            "patch_high_energy_std",
        ]
    )
    return names


def normalized_sum(values: np.ndarray, total: float) -> float:
    if total <= 1e-12:
        return 0.0
    return float(np.sum(values) / total)


def spectrum_from_image(image: np.ndarray, roi_mask: np.ndarray | None, hann: np.ndarray) -> np.ndarray:
    normalized = robust_normalize(image)
    if roi_mask is not None:
        normalized = normalized * roi_mask
    centered = normalized - float(np.mean(normalized))
    windowed = centered * hann
    fft = np.fft.fftshift(np.fft.fft2(windowed))
    magnitude = np.log1p(np.abs(fft)).astype(np.float64)
    center = magnitude.shape[0] // 2
    magnitude[center - 1 : center + 2, center - 1 : center + 2] = 0.0
    return magnitude


def band_ratios(spectrum: np.ndarray, radius: np.ndarray) -> tuple[float, float, float]:
    total = float(np.sum(spectrum)) + 1e-12
    low = normalized_sum(spectrum[radius < 0.15], total)
    mid = normalized_sum(spectrum[(radius >= 0.15) & (radius < 0.45)], total)
    high = normalized_sum(spectrum[radius >= 0.45], total)
    return low, mid, high


def extract_one_descriptor(image: np.ndarray, roi_mask: np.ndarray | None, geometry: dict[str, np.ndarray], hann: np.ndarray) -> np.ndarray:
    spectrum = spectrum_from_image(image, roi_mask, hann)
    radius = geometry["radius"]
    angle = geometry["angle"]
    total = float(np.sum(spectrum)) + 1e-12

    features: list[float] = []

    radial_edges = np.linspace(0.0, 1.0, 9)
    for start, end in zip(radial_edges[:-1], radial_edges[1:]):
        mask = (radius >= start) & (radius < end)
        features.append(normalized_sum(spectrum[mask], total))

    angular_edges = np.linspace(0.0, 1.0, 13)
    angular_values = []
    for start, end in zip(angular_edges[:-1], angular_edges[1:]):
        mask = (angle >= start) & (angle < end)
        value = normalized_sum(spectrum[mask], total)
        angular_values.append(value)
        features.append(value)

    low, mid, high = band_ratios(spectrum, radius)
    features.extend([low, mid, high])

    probability = spectrum / total
    centroid = float(np.sum(probability * radius))
    bandwidth = float(np.sqrt(np.sum(probability * (radius - centroid) ** 2)))
    entropy = float(-np.sum(probability[probability > 0] * np.log2(probability[probability > 0] + 1e-12)))
    entropy /= float(np.log2(probability.size))
    features.extend([centroid, bandwidth, entropy])

    flat = np.sort(spectrum.reshape(-1))[::-1]
    features.extend([float(np.sum(flat[:k]) / total) for k in (1, 5, 10)])

    angular_array = np.asarray(angular_values, dtype=np.float64)
    features.append(float(np.max(angular_array) / (np.mean(angular_array) + 1e-12)))

    angle_radians = angle * 2.0 * math.pi - math.pi
    horizontal = (np.abs(np.sin(angle_radians)) < np.sin(np.deg2rad(15.0))) | (
        np.abs(np.sin(angle_radians)) > np.sin(np.deg2rad(165.0))
    )
    vertical = np.abs(np.cos(angle_radians)) < np.cos(np.deg2rad(75.0))
    diagonal = (np.abs(np.abs(np.rad2deg(angle_radians)) - 45.0) < 15.0) | (
        np.abs(np.abs(np.rad2deg(angle_radians)) - 135.0) < 15.0
    )
    features.extend(
        [
            normalized_sum(spectrum[horizontal], total),
            normalized_sum(spectrum[vertical], total),
            normalized_sum(spectrum[diagonal], total),
        ]
    )

    patch_low: list[float] = []
    patch_mid: list[float] = []
    patch_high: list[float] = []
    patch_size = image.shape[0] // 4
    patch_hann_1d = np.hanning(patch_size).astype(np.float32)
    patch_hann = np.outer(patch_hann_1d, patch_hann_1d).astype(np.float32)
    patch_geometry = spectral_geometry(patch_size)
    for row in range(4):
        for col in range(4):
            patch = image[row * patch_size : (row + 1) * patch_size, col * patch_size : (col + 1) * patch_size]
            patch_mask = None
            if roi_mask is not None:
                patch_mask = roi_mask[row * patch_size : (row + 1) * patch_size, col * patch_size : (col + 1) * patch_size]
            patch_spectrum = spectrum_from_image(patch, patch_mask, patch_geometry["radius"] * 0.0 + patch_hann)
            patch_total = float(np.sum(patch_spectrum)) + 1e-12
            patch_radius = patch_geometry["radius"]
            p_low, p_mid, p_high = band_ratios(patch_spectrum, patch_radius)
            if patch_total > 1e-12:
                patch_low.append(p_low)
                patch_mid.append(p_mid)
                patch_high.append(p_high)
    for values in (patch_low, patch_mid, patch_high):
        array = np.asarray(values, dtype=np.float64)
        features.append(float(np.mean(array)) if array.size else 0.0)
    for values in (patch_low, patch_mid, patch_high):
        array = np.asarray(values, dtype=np.float64)
        features.append(float(np.std(array)) if array.size else 0.0)

    return np.asarray(features, dtype=np.float32)


def build_metadata(manifest: dict[str, Any], indices: np.ndarray) -> dict[str, np.ndarray]:
    paths = manifest["paths"][indices].astype(str)
    labels = manifest["labels"][indices].astype(np.int64)
    distribution_group = manifest["distribution_group"][indices].astype(str)
    frequency_hz = manifest["frequency_hz"][indices].astype(np.float32)
    split = manifest["split"][indices].astype(object)

    boards = []
    materials = []
    split_groups = []
    names = []
    for label, path_text, group in zip(labels, paths, distribution_group):
        board = extract_board(group, path_text)
        material = derive_material(board, group, path_text)
        boards.append(board)
        materials.append(material)
        split_groups.append(board)
        names.append(LABEL_NAMES.get(int(label), str(int(label))))

    return {
        "label": labels,
        "label_name": np.asarray(names, dtype=object),
        "path": paths.astype(object),
        "board": np.asarray(boards, dtype=object),
        "material": np.asarray(materials, dtype=object),
        "frequency_hz": frequency_hz,
        "split": split,
        "split_group": np.asarray(split_groups, dtype=object),
        "distribution_group": distribution_group.astype(object),
    }


def quality_checks(embedding: np.ndarray, metadata: dict[str, np.ndarray], names: list[str]) -> dict[str, Any]:
    paths = metadata["path"].astype(str)
    metadata_lengths_match = all(len(value) == len(embedding) for value in metadata.values())
    return {
        "sample_count": int(len(embedding)),
        "descriptor_dimension": int(embedding.shape[1]) if embedding.ndim == 2 else 0,
        "embedding_shape": [int(dim) for dim in embedding.shape],
        "nan_count": int(np.isnan(embedding).sum()),
        "inf_count": int(np.isinf(embedding).sum()),
        "duplicate_paths": int(len(paths) - len(set(paths.tolist()))),
        "nonzero_variance": bool(np.any(np.nanstd(embedding, axis=0) > 1e-12)) if len(embedding) else False,
        "metadata_lengths_match": bool(metadata_lengths_match),
        "descriptor_names_count": int(len(names)),
        "descriptor_names_match_dimension": bool(embedding.ndim == 2 and len(names) == embedding.shape[1]),
    }


def quality_passed(quality: dict[str, Any]) -> bool:
    return (
        quality["sample_count"] > 0
        and quality["descriptor_dimension"] > 0
        and quality["nan_count"] == 0
        and quality["inf_count"] == 0
        and quality["duplicate_paths"] == 0
        and quality["nonzero_variance"]
        and quality["metadata_lengths_match"]
        and quality["descriptor_names_match_dimension"]
    )


def save_npz_atomic(path: Path, overwrite: bool, embedding: np.ndarray, metadata: dict[str, np.ndarray], names: list[str]) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
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
            checkpoint_path=np.array("not_applicable:deterministic_spectral_descriptors", dtype=object),
            checkpoint_sha256=np.array("not_applicable", dtype=object),
            embedding_point=np.array("deterministic_fft_spectral_descriptors", dtype=object),
            model_name=np.array("spectral_descriptors_lefft_inspired", dtype=object),
            source_npz=np.array("generated_from_manifest_images", dtype=object),
            schema_version=np.array(SCHEMA_VERSION, dtype=object),
            descriptor_names=np.asarray(names, dtype=object),
        )
    tmp_path.replace(path)


def write_report(path: Path, args: argparse.Namespace, output_path: Path, quality: dict[str, Any], names: list[str]) -> None:
    families = [
        "8 normalized radial FFT energy bands",
        "12 normalized angular FFT energy bins",
        "low/mid/high spatial-frequency energy ratios",
        "spectral centroid radius, bandwidth, and entropy",
        "top-1/top-5/top-10 peak concentration",
        "orientation anisotropy and horizontal/vertical/diagonal energy ratios",
        "4x4 local patch low/mid/high energy mean and standard deviation",
    ]
    family_lines = "\n".join(f"- {item}" for item in families)
    quality_rows = "\n".join(f"| {key} | {value} |" for key, value in quality.items())
    text = f"""# Spectral Descriptor Extraction Report

## Scope

This report documents deterministic spatial-frequency descriptor extraction for ESPI images. The descriptors are spectral / LeFFT-inspired descriptors, not a trained LeFFT model. No deep model training, encoder fine-tuning, evaluation, LeFFT superiority claim, or acoustic-response prediction was performed.

## Inputs

- Manifest: `{args.manifest}`
- Image root: `{args.image_root}`
- ROI mask: `{args.roi_mask if args.roi_mask else 'none'}`
- Device argument: `{args.device}` (descriptors are computed deterministically on CPU with NumPy)
- Full extraction: `{args.full}`
- Max samples when not full: `{args.max_samples}`

## Output

- Output NPZ: `{output_path}`
- Model name: `spectral_descriptors_lefft_inspired`
- Embedding point: `deterministic_fft_spectral_descriptors`
- Schema version: `{SCHEMA_VERSION}`

## Preprocessing strategy

Each ESPI image is loaded as grayscale, resized to `256 x 256`, robustly normalized per image using percentile clipping, optionally masked with the supplied ROI mask, Hann-windowed, transformed with a 2D FFT, converted to log-magnitude spectrum, and processed after suppressing the DC component.

## Descriptor families

{family_lines}

## Quality checks

| Check | Value |
|---|---:|
{quality_rows}

## Descriptor names

`{', '.join(names)}`

## Claim boundary

These outputs are deterministic spectral descriptors and may be used as a lightweight spectral / LeFFT-inspired baseline. They are not a trained LeFFT model and do not support any claim of LeFFT superiority.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def update_manuscript_map(map_path: Path, report_path: Path) -> None:
    if not map_path.exists():
        return
    marker = "Spectral / LeFFT-inspired descriptor extraction"
    text = map_path.read_text(encoding="utf-8")
    if marker in text:
        return
    line = (
        f"| {marker} | `scripts/encoder/extract_spectral_descriptors.py` | manifest + image root | "
        f"`{report_path.as_posix()}` | done-external |"
    )
    section = (
        "\n## Spectral Descriptor Ablation\n\n"
        "| Artifact | Source script | Input artifact | Output/report | Status |\n"
        "|---|---|---|---|---|\n"
        f"{line}\n"
    )
    release_gate = "\n## Release Gate\n"
    if release_gate in text:
        text = text.replace(release_gate, section + release_gate, 1)
    else:
        text = text.rstrip() + "\n" + section
    map_path.write_text(text, encoding="utf-8")


def extract_descriptors(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(args.manifest)
    indices = selected_indices(len(manifest["paths"]), args.max_samples, args.full)
    names = descriptor_names()
    geometry = spectral_geometry(256)
    hann_1d = np.hanning(256).astype(np.float32)
    hann = np.outer(hann_1d, hann_1d).astype(np.float32)
    roi_mask = load_roi_mask(args.roi_mask)

    descriptors: list[np.ndarray] = []
    successful_indices: list[int] = []
    for count, source_idx in enumerate(indices, start=1):
        path_text = str(manifest["paths"][int(source_idx)])
        image_path = resolve_image_path(path_text, args.image_root)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path} (source path: {path_text})")
        image = load_grayscale(image_path)
        descriptors.append(extract_one_descriptor(image, roi_mask, geometry, hann))
        successful_indices.append(int(source_idx))
        if count == 1 or count % 250 == 0 or count == len(indices):
            print(f"[progress] extracted {count}/{len(indices)} descriptors")

    embedding = np.vstack(descriptors).astype(np.float32)
    output_indices = np.asarray(successful_indices, dtype=np.int64)
    metadata = build_metadata(manifest, output_indices)
    quality = quality_checks(embedding, metadata, names)

    out_path = args.out_dir / DEFAULT_OUTPUT_NAME
    save_npz_atomic(out_path, args.overwrite, embedding, metadata, names)

    report_path = DEFAULT_REPORT_DIR / "SPECTRAL_DESCRIPTOR_EXTRACTION_REPORT.md"
    json_path = DEFAULT_REPORT_DIR / "spectral_descriptor_extraction.json"
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "image_root": str(args.image_root),
        "out_dir": str(args.out_dir),
        "output_npz": str(out_path),
        "full": bool(args.full),
        "max_samples": int(args.max_samples),
        "roi_mask": str(args.roi_mask) if args.roi_mask else None,
        "schema_version": SCHEMA_VERSION,
        "model_name": "spectral_descriptors_lefft_inspired",
        "embedding_point": "deterministic_fft_spectral_descriptors",
        "descriptor_families": [
            "radial_fft_energy_bands",
            "angular_fft_energy_bins",
            "low_mid_high_spatial_frequency_energy_ratios",
            "spectral_centroid_bandwidth_entropy",
            "peak_concentration",
            "orientation_anisotropy",
            "local_patch_spectral_statistics",
        ],
        "descriptor_names": names,
        "quality_checks": quality,
        "status": "pass" if quality_passed(quality) else "warning",
        "claim_boundary": "Deterministic spectral / LeFFT-inspired descriptors only; not a trained LeFFT model; no LeFFT superiority claim.",
    }
    write_report(report_path, args, out_path, quality, names)
    write_json(json_path, payload)
    update_manuscript_map(Path("docs") / "MANUSCRIPT_MAP.md", report_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract deterministic spectral / LeFFT-inspired descriptors.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--roi-mask", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if str(args.device).lower() != "cpu":
        print(f"[warn] device={args.device} was requested, but descriptors are deterministic NumPy CPU computations.")
    payload = extract_descriptors(args)
    print(f"[done] output={payload['output_npz']}")
    print(f"[done] report={DEFAULT_REPORT_DIR / 'SPECTRAL_DESCRIPTOR_EXTRACTION_REPORT.md'}")
    print(f"[done] key_numbers={DEFAULT_REPORT_DIR / 'spectral_descriptor_extraction.json'}")
    print(f"[done] status={payload['status']} shape={payload['quality_checks']['embedding_shape']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
