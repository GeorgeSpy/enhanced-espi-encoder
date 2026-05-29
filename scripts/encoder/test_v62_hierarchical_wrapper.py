#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Smoke test for the frozen hierarchical v6.2 encoder wrapper."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
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


DEFAULT_MODEL_FILE = REPO_ROOT / "scripts" / "v6_2" / "espi_v6_2_fixed.py"
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "hierarchical_encoder"
DEFAULT_MANIFEST = REPO_ROOT / "artifacts" / "manifests" / "manifest_v1_5class.npz"
DEFAULT_COMPAT = REPO_ROOT / "external" / "compat"
REPORT_NAME = "WRAPPER_SMOKE_TEST.md"
JSON_NAME = "wrapper_smoke_test.json"


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


def load_manifest(path: Path) -> dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    return {
        "paths": data["paths"].astype(str),
        "labels": data["labels"].astype(np.int64),
    }


def choose_sample_indices(labels: np.ndarray, max_samples: int) -> list[int]:
    selected: list[int] = []
    for label in sorted(set(int(value) for value in labels.tolist())):
        matches = np.where(labels == label)[0]
        if len(matches):
            selected.append(int(matches[0]))
        if len(selected) >= max_samples:
            return selected
    if len(selected) < max_samples:
        for idx in range(len(labels)):
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= max_samples:
                break
    return selected


def image_to_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = float(array.mean())
    std = float(array.std()) + 1e-6
    array = (array - mean) / std
    return torch.from_numpy(array).unsqueeze(0)


def tensor_stats(tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None:
        return {"available": False}
    detached = tensor.detach().cpu()
    finite = torch.isfinite(detached)
    return {
        "available": True,
        "shape": list(detached.shape),
        "nan_count": int(torch.isnan(detached).sum().item()),
        "inf_count": int(torch.isinf(detached).sum().item()),
        "finite": bool(finite.all().item()),
        "mean": float(detached.float().mean().item()) if detached.numel() else math.nan,
        "std": float(detached.float().std(unbiased=False).item()) if detached.numel() else math.nan,
    }


def outputs_are_close(
    first: dict[str, torch.Tensor | None],
    second: dict[str, torch.Tensor | None],
    atol: float,
    rtol: float,
) -> tuple[bool, dict[str, float | None]]:
    max_abs_diff: dict[str, float | None] = {}
    ok = True
    for key, first_value in first.items():
        second_value = second.get(key)
        if first_value is None and second_value is None:
            max_abs_diff[key] = None
            continue
        if first_value is None or second_value is None:
            ok = False
            max_abs_diff[key] = math.inf
            continue
        first_cpu = first_value.detach().cpu()
        second_cpu = second_value.detach().cpu()
        diff = torch.max(torch.abs(first_cpu - second_cpu)).item()
        max_abs_diff[key] = float(diff)
        if not torch.allclose(first_cpu, second_cpu, atol=atol, rtol=rtol):
            ok = False
    return ok, max_abs_diff


def write_reports(out_dir: Path, report: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / JSON_NAME
    md_path = out_dir / REPORT_NAME
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    checks = report["checks"]
    lines = [
        "# v6.2 Hierarchical Encoder Wrapper Smoke Test",
        "",
        "## Scope",
        "",
        "This smoke test validates the frozen hierarchical v6.2 encoder wrapper on a tiny manifest subset. It does not train and does not run full embedding extraction.",
        "",
        "## Inputs",
        "",
        f"- Model file: `{report['model_file']}`",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Manifest: `{report['manifest']}`",
        f"- Device: `{report['device']}`",
        f"- Samples tested: `{report['n_samples']}`",
        f"- Image size: `{report['image_size']}`",
        f"- Created UTC: `{report['created_utc']}`",
        "",
        "## Result",
        "",
        f"- Overall status: **{report['status']}**",
        f"- Required embeddings available: `{checks['required_embeddings_available']}`",
        f"- No NaN: `{checks['no_nan']}`",
        f"- No Inf: `{checks['no_inf']}`",
        f"- Deterministic repeated forward: `{checks['deterministic_repeated_forward']}`",
        f"- Repeated forward max absolute difference: `{report['max_abs_diff']}`",
        "",
        "## Output Tensor Shapes",
        "",
        "| Output | Available | Shape | NaN | Inf |",
        "|---|---:|---|---:|---:|",
    ]
    for key, stats in report["output_stats"].items():
        lines.append(
            f"| `{key}` | {stats.get('available', False)} | `{stats.get('shape', '')}` | "
            f"{stats.get('nan_count', '')} | {stats.get('inf_count', '')} |"
        )
    lines.extend(
        [
            "",
            "## Sample Paths",
            "",
            *[f"- `{item}`" for item in report["sample_paths"]],
            "",
            "## Notes",
            "",
            "- `z_expert_prelogit` and `z_arcface_input` intentionally reference `outputs[\"embeddings\"]`.",
            "- `z_physics_fusion` is intentionally not exposed in this wrapper.",
            "- Placeholder physics modules are refused by the wrapper import path.",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the frozen hierarchical v6.2 encoder wrapper.")
    parser.add_argument("--model-file", type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dependency-path", type=Path, action="append", default=[])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()

    max_samples = max(1, min(5, int(args.max_samples)))
    model_file = resolve_path(args.model_file)
    checkpoint = resolve_path(args.checkpoint)
    manifest_path = resolve_path(args.manifest)
    out_dir = resolve_path(args.out_dir)
    dependency_paths = [DEFAULT_COMPAT, *[resolve_path(path) for path in args.dependency_path]]

    manifest = load_manifest(manifest_path)
    sample_indices = choose_sample_indices(manifest["labels"], max_samples)
    sample_paths = [Path(manifest["paths"][idx]) for idx in sample_indices]
    missing_paths = [str(path) for path in sample_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Smoke-test image paths do not exist: {missing_paths[:3]}")

    batch = torch.stack([image_to_tensor(path, args.image_size) for path in sample_paths], dim=0)
    encoder = ESPIv62HierarchicalEncoder(
        checkpoint=checkpoint,
        model_file=model_file,
        device=args.device,
        dependency_paths=dependency_paths,
        strict=True,
    )
    first = encoder.encode(batch)
    second = encoder.encode(batch)
    deterministic, max_abs_diff = outputs_are_close(first, second, atol=args.atol, rtol=args.rtol)
    output_stats = {key: tensor_stats(value) for key, value in first.items()}

    required_available = (
        output_stats["z_expert_prelogit"].get("available") is True
        and output_stats["z_arcface_input"].get("available") is True
    )
    no_nan = all(stats.get("nan_count", 0) == 0 for stats in output_stats.values() if stats.get("available"))
    no_inf = all(stats.get("inf_count", 0) == 0 for stats in output_stats.values() if stats.get("available"))
    expected_batch_shapes = all(
        stats.get("shape", [None])[0] == len(sample_paths)
        for stats in output_stats.values()
        if stats.get("available")
    )
    same_embedding_object = first["z_expert_prelogit"] is first["z_arcface_input"]

    checks = {
        "required_embeddings_available": bool(required_available),
        "no_nan": bool(no_nan),
        "no_inf": bool(no_inf),
        "deterministic_repeated_forward": bool(deterministic),
        "expected_batch_shapes": bool(expected_batch_shapes),
        "z_alias_consistency": bool(same_embedding_object),
        "load_missing_keys_zero": len(encoder.load_missing_keys) == 0,
        "load_unexpected_keys_zero": len(encoder.load_unexpected_keys) == 0,
    }
    status = "pass" if all(checks.values()) else "fail"
    report = {
        "created_utc": utc_now(),
        "status": status,
        "model_file": str(model_file),
        "checkpoint": str(checkpoint),
        "manifest": str(manifest_path),
        "dependency_paths": [str(path) for path in dependency_paths],
        "device": str(encoder.device),
        "n_samples": len(sample_paths),
        "sample_indices": sample_indices,
        "sample_paths": [str(path) for path in sample_paths],
        "image_size": args.image_size,
        "output_stats": output_stats,
        "max_abs_diff": max_abs_diff,
        "checks": checks,
        "load_missing_keys": encoder.load_missing_keys,
        "load_unexpected_keys": encoder.load_unexpected_keys,
        "import_log": encoder.import_log,
        "notes": [
            "z_expert_prelogit and z_arcface_input are aliases for outputs['embeddings'].",
            "z_physics_fusion is intentionally not exposed.",
        ],
    }
    write_reports(out_dir, report)
    print(f"[wrapper-smoke] wrote {out_dir / REPORT_NAME}")
    print(f"[wrapper-smoke] wrote {out_dir / JSON_NAME}")
    print(f"[wrapper-smoke] status: {status}")
    print(f"[wrapper-smoke] z shape: {output_stats['z_expert_prelogit'].get('shape')}")
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
