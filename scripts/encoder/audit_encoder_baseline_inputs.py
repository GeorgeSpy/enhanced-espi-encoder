#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit input availability for a unified encoder baseline comparison.

This script is read-only with respect to experiment artifacts. It does not train,
does not extract embeddings, and does not run evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PLAYGROUND = ROOT.parent
PRIVATE_ESPI_ROOT = Path(os.environ["ESPI_PRIVATE_ROOT"]) if os.environ.get("ESPI_PRIVATE_ROOT") else None

DEFAULT_OUT_DIR = Path("reports/encoder_baselines")

EXPECTED_METADATA = ["label", "path", "board", "material", "frequency_hz", "split", "split_group"]
EXPECTED_FEATURE_FIELDS = [
    "embedding",
    "label",
    "path",
    "board",
    "material",
    "frequency_hz",
    "split_group",
    "checkpoint_sha256",
    "embedding_point",
]

FIELD_ALIASES = {
    "embedding": ["embedding", "embeddings"],
    "label": ["label", "labels", "y", "targets"],
    "path": ["path", "paths", "image_path", "image_paths"],
    "board": ["board", "boards"],
    "material": ["material", "materials"],
    "frequency_hz": ["frequency_hz", "frequency", "freq_hz", "freq"],
    "split": ["split", "splits", "train_idx"],
    "split_group": ["split_group", "split_groups", "group", "groups"],
    "checkpoint_sha256": ["checkpoint_sha256", "checkpoint_path"],
    "embedding_point": ["embedding_point", "embedding_layer"],
}


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    output = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        output.append(path)
    return output


def search_files(roots: list[Path], patterns: list[str], max_results: int = 12) -> list[Path]:
    results: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            try:
                for match in root.rglob(pattern):
                    if match.is_file():
                        results.append(match)
                        if len(results) >= max_results:
                            return dedupe_paths(results)
            except Exception:
                continue
    return dedupe_paths(results)


def matrix_row(
    encoder: str,
    artifact: str,
    path: Path | None,
    status: str,
    blocking_issue: str = "",
) -> dict[str, Any]:
    return {
        "Encoder": encoder,
        "Required artifact": artifact,
        "Found?": "yes" if path and path.exists() else "no",
        "Path": rel_or_abs(path) if path else "",
        "Status": status,
        "Blocking issue": blocking_issue,
    }


def scalar_to_str(value: Any) -> str:
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    if array.size == 1:
        return str(array.reshape(-1)[0])
    return str(array)


def unknown_fraction(values: np.ndarray) -> float:
    if len(values) == 0:
        return 1.0
    unknown = {"", "unknown", "none", "nan", "null", "na"}
    return float(np.mean([str(value).strip().lower() in unknown for value in values.astype(str)]))


def inspect_npz(path: Path | None, expected_fields: list[str]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "exists": False,
            "path": str(path) if path else "",
            "keys": [],
            "missing_fields": expected_fields,
            "n_samples": None,
            "embedding_dim": None,
            "metadata_issues": [],
        }

    metadata_issues: list[str] = []
    with np.load(str(path), allow_pickle=True) as data:
        keys = list(data.files)
        lower_keys = {key.lower(): key for key in keys}
        field_map = {}
        missing_fields = []
        for field in expected_fields:
            aliases = FIELD_ALIASES.get(field, [field])
            found = next((lower_keys[alias.lower()] for alias in aliases if alias.lower() in lower_keys), None)
            if found:
                field_map[field] = found
            else:
                missing_fields.append(field)
        n_samples = None
        embedding_dim = None
        embedding_key = field_map.get("embedding")
        if embedding_key:
            embedding = data[embedding_key]
            n_samples = int(embedding.shape[0]) if embedding.ndim >= 1 else None
            embedding_dim = int(embedding.shape[1]) if embedding.ndim == 2 else None

        if field_map.get("board") and unknown_fraction(data[field_map["board"]]) > 0.95:
            metadata_issues.append("board is stored as unknown for nearly all samples")
        if field_map.get("split_group") and unknown_fraction(data[field_map["split_group"]]) > 0.95:
            metadata_issues.append("split_group is stored as unknown for nearly all samples")
        if "checkpoint_sha256" in keys:
            checkpoint_sha256 = scalar_to_str(data["checkpoint_sha256"])
        else:
            checkpoint_sha256 = ""
            if "checkpoint_path" in keys:
                metadata_issues.append("checkpoint SHA256 is missing; checkpoint_path is available instead")
        if "embedding_point" in keys:
            embedding_point = scalar_to_str(data["embedding_point"])
        else:
            embedding_point = ""
            if "embedding_layer" in keys:
                embedding_point = scalar_to_str(data["embedding_layer"])
                metadata_issues.append("embedding_point is missing; embedding_layer is available instead")

    return {
        "exists": True,
        "path": str(path),
        "keys": keys,
        "field_map": field_map,
        "missing_fields": missing_fields,
        "n_samples": n_samples,
        "embedding_dim": embedding_dim,
        "metadata_issues": metadata_issues,
        "checkpoint_sha256": checkpoint_sha256,
        "embedding_point": embedding_point,
    }


def inspect_manifest(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "exists": False,
            "path": str(path) if path else "",
            "keys": [],
            "missing_metadata": EXPECTED_METADATA,
            "n_samples": None,
            "path_key": None,
            "existing_image_paths_checked": 0,
            "existing_image_paths_found": 0,
        }

    with np.load(str(path), allow_pickle=True) as data:
        keys = list(data.files)
        lower_map = {key.lower(): key for key in keys}
        field_map = {}
        missing = []
        for field in EXPECTED_METADATA:
            aliases = FIELD_ALIASES.get(field, [field])
            if field == "split":
                aliases = aliases + ["val_idx"]
            found = next((lower_map[alias.lower()] for alias in aliases if alias.lower() in lower_map), None)
            if found:
                field_map[field] = found
            else:
                missing.append(field)
        path_key = field_map.get("path")
        label_key = field_map.get("label")
        n_samples = None
        if label_key:
            n_samples = int(len(data[label_key]))
        elif path_key:
            n_samples = int(len(data[path_key]))

        existing_checked = 0
        existing_found = 0
        if path_key:
            paths = data[path_key].astype(str)
            for value in paths[: min(25, len(paths))]:
                existing_checked += 1
                if Path(value).exists():
                    existing_found += 1

    return {
        "exists": True,
        "path": str(path),
        "keys": keys,
        "field_map": field_map,
        "missing_metadata": missing,
        "n_samples": n_samples,
        "path_key": path_key,
        "existing_image_paths_checked": existing_checked,
        "existing_image_paths_found": existing_found,
    }


def candidate_paths() -> dict[str, list[Path]]:
    candidates = {
        "manifest": [
            ROOT / "artifacts/manifests/manifest_v1_5class.npz",
            ROOT / "manifest/manifest_v1_5class.npz",
            PLAYGROUND / "ESPI_v62_v62A_Encoder_package_20260430_224510/core_v62_files/manifest/manifest_v1_5class.npz",
        ],
        "v62a_features": [
            ROOT / "outputs/features_v62a_epoch25.npz",
            PLAYGROUND / "v62_encoder_evidence/features_v62a_epoch25.npz",
            PLAYGROUND / "v62_encoder_evidence/outputs/features_v62a_epoch25.npz",
            PLAYGROUND / "ESPI_v62_v62A_Encoder_package_20260430_224510/encoder_evidence/features_v62a_epoch25.npz",
        ],
        "hier_features": [
            ROOT / "outputs/hierarchical_embeddings_v001/features_hier_z_expert_prelogit.npz",
        ],
        "v62a_checkpoint": [
            ROOT / "artifacts/checkpoints/checkpoint_epoch25_20260211_035150.pt",
            PLAYGROUND / "ESPI_v62_v62A_Encoder_package_20260430_224510/core_v62_files/checkpoints/checkpoint_epoch25_20260211_035150.pt",
        ],
        "hier_checkpoint": [
            ROOT / "artifacts/checkpoints/ckpt_phase2_expert.pt",
        ],
        "image_roots": [
            PLAYGROUND / "ESPI_v62_v62A_Encoder_package_20260430_224510/core_v62_files/data",
            ROOT / "data",
        ],
    }
    if PRIVATE_ESPI_ROOT is not None:
        candidates["manifest"].extend([
            PRIVATE_ESPI_ROOT / "manifest/manifest_v1_5class.npz",
            PRIVATE_ESPI_ROOT / "manifest/manifest_v1.npz",
        ])
        candidates["v62a_checkpoint"].extend([
            PRIVATE_ESPI_ROOT / "checkpoints/checkpoint_epoch25_20260211_035150.pt",
            PRIVATE_ESPI_ROOT / "logs/checkpoint_epoch25_20260211_035150.pt",
        ])
        candidates["hier_checkpoint"].append(PRIVATE_ESPI_ROOT / "logs/train_v6.2_opt/ckpt_phase2_expert.pt")
        candidates["image_roots"].append(PRIVATE_ESPI_ROOT / "data")
    return candidates


def derive_status(found: bool, missing: list[str] | None = None, metadata_issue: bool = False) -> tuple[str, str]:
    if not found:
        return "missing", "Required artifact not found."
    if missing:
        return "incomplete", "Missing fields: " + ", ".join(missing)
    if metadata_issue:
        return "usable-with-caveat", "Metadata issue requires regeneration before final publication tables."
    return "available", ""


def build_audit(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = candidate_paths()
    if args.manifest:
        candidates["manifest"].insert(0, args.manifest)
    if args.v62a_features:
        candidates["v62a_features"].insert(0, args.v62a_features)
    if args.hier_features:
        candidates["hier_features"].insert(0, args.hier_features)
    if args.v61_checkpoint:
        candidates["v61_checkpoint"] = [args.v61_checkpoint]
    else:
        search_roots = [ROOT, PLAYGROUND]
        if PRIVATE_ESPI_ROOT is not None:
            search_roots.append(PRIVATE_ESPI_ROOT)
        candidates["v61_checkpoint"] = search_files(
            search_roots,
            ["*v6*1*.pt", "*v61*.pt", "*v6*1*.pth", "*v61*.pth", "*v6*1*.ckpt", "*v61*.ckpt"],
            max_results=8,
        )

    manifest_path = first_existing(candidates["manifest"])
    v62a_features_path = first_existing(candidates["v62a_features"])
    hier_features_path = first_existing(candidates["hier_features"])
    v62a_checkpoint_path = first_existing(candidates["v62a_checkpoint"])
    hier_checkpoint_path = first_existing(candidates["hier_checkpoint"])
    image_root = first_existing(candidates["image_roots"])
    v61_checkpoint_path = first_existing(candidates["v61_checkpoint"])

    manifest = inspect_manifest(manifest_path)
    v62a_features = inspect_npz(v62a_features_path, EXPECTED_FEATURE_FIELDS)
    hier_features = inspect_npz(hier_features_path, EXPECTED_FEATURE_FIELDS)

    required_scripts = {
        "random_imagenet_extraction": ROOT / "scripts/encoder/extract_v62_embeddings.py",
        "v62a_audit": ROOT / "scripts/encoder/audit_v62_embeddings.py",
        "hierarchical_wrapper": ROOT / "src/enhanced_espi_encoder/v62_hierarchical_encoder.py",
        "hierarchical_extraction": ROOT / "scripts/encoder/extract_v62_hierarchical_embeddings.py",
        "grouped_evaluation": ROOT / "scripts/encoder/evaluate_encoder_grouped_generalization.py",
        "v61_training_script": ROOT / "scripts/v6_1/train_v6_1_head_only.py",
    }

    rows: list[dict[str, Any]] = []
    manifest_status, manifest_block = derive_status(manifest["exists"], manifest.get("missing_metadata"))
    rows.append(matrix_row("all baselines", "manifest file with expected metadata", manifest_path, manifest_status, manifest_block))

    image_status = "available" if image_root else "missing"
    image_block = "" if image_root else "Image root not found; extraction baselines cannot be run."
    if manifest["exists"] and manifest["existing_image_paths_checked"] and manifest["existing_image_paths_found"] == 0:
        image_status = "incomplete"
        image_block = "Manifest sample paths were checked but none existed on this machine."
    rows.append(matrix_row("random ResNet-18 embeddings", "image paths / root", image_root, image_status, image_block))
    rows.append(matrix_row("ImageNet-pretrained ResNet-18 embeddings", "image paths / root", image_root, image_status, image_block))

    script_path = required_scripts["random_imagenet_extraction"]
    status, block = derive_status(script_path.exists())
    rows.append(matrix_row("random ResNet-18 embeddings", "baseline extraction script / loader", script_path, status, block))
    rows.append(matrix_row("ImageNet-pretrained ResNet-18 embeddings", "baseline extraction script / loader", script_path, status, block))

    status, block = derive_status(v61_checkpoint_path is not None)
    rows.append(matrix_row("v6.1 embeddings", "v6.1 model/checkpoint", v61_checkpoint_path, status, block))
    v61_script = required_scripts["v61_training_script"]
    status, block = derive_status(v61_script.exists())
    rows.append(matrix_row("v6.1 embeddings", "v6.1 model/training script", v61_script, status, block))

    status, block = derive_status(v62a_features["exists"], v62a_features["missing_fields"], bool(v62a_features["metadata_issues"]))
    if v62a_features["exists"]:
        rows.append(matrix_row("v6.2-A frozen embeddings", "existing v6.2-A feature dump", v62a_features_path, status, block))
    else:
        status, block = derive_status(v62a_checkpoint_path is not None)
        rows.append(matrix_row("v6.2-A frozen embeddings", "v6.2-A feature dump or checkpoint", v62a_checkpoint_path, status, block))

    status, block = derive_status(hier_features["exists"], hier_features["missing_fields"], bool(hier_features["metadata_issues"]))
    if hier_features["metadata_issues"]:
        block = (
            "Hierarchical NPZ stored board/split_group as unknown; acceptable for internal comparison "
            "after recovery from distribution_group/path, but must be regenerated before final publication tables."
        )
    rows.append(matrix_row("hierarchical v6.2 phase2 expert embeddings", "existing hierarchical feature dump", hier_features_path, status, block))

    status, block = derive_status(hier_checkpoint_path is not None)
    rows.append(matrix_row("hierarchical v6.2 phase2 expert embeddings", "hierarchical phase2 checkpoint", hier_checkpoint_path, status, block))

    for name, path in required_scripts.items():
        status, block = derive_status(path.exists())
        rows.append(matrix_row("required scripts/loaders", name, path, status, block))

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "manifest": manifest,
        "image_root": str(image_root) if image_root else "",
        "v61_checkpoint_candidates": [str(path) for path in candidates["v61_checkpoint"]],
        "v61_checkpoint": str(v61_checkpoint_path) if v61_checkpoint_path else "",
        "v62a_features": v62a_features,
        "v62a_checkpoint": str(v62a_checkpoint_path) if v62a_checkpoint_path else "",
        "hierarchical_features": hier_features,
        "hierarchical_checkpoint": str(hier_checkpoint_path) if hier_checkpoint_path else "",
        "required_scripts": {name: {"path": str(path), "exists": path.exists()} for name, path in required_scripts.items()},
        "availability_matrix": rows,
        "known_issues": [
            "Hierarchical NPZ stored board/split_group as unknown and required recovery from distribution_group/path.",
            "This is acceptable for internal comparison but requires regeneration before final publication tables.",
        ],
        "blocked_for_unified_extraction": any(row["Status"] == "missing" for row in rows if row["Encoder"] != "v6.1 embeddings"),
        "v61_available": v61_checkpoint_path is not None,
    }
    return rows, summary


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = ["Encoder", "Required artifact", "Found?", "Path", "Status", "Blocking issue"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|---|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]).replace("\n", " ") for column in columns) + " |")
    return "\n".join(lines)


def write_report(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    missing_rows = [row for row in rows if row["Status"] == "missing"]
    caveat_rows = [row for row in rows if row["Status"] in {"usable-with-caveat", "incomplete"}]
    missing_lines = "\n".join(f"- {row['Encoder']}: {row['Required artifact']} ({row['Blocking issue']})" for row in missing_rows) or "- None"
    caveat_lines = "\n".join(f"- {row['Encoder']}: {row['Required artifact']} ({row['Blocking issue']})" for row in caveat_rows) or "- None"

    text = f"""# Encoder Baseline Input Audit

## Scope

This report audits whether the inputs required for a unified encoder baseline comparison are available. It does not train models, extract embeddings, run evaluations, implement LeFFT scripts, or implement acoustic-response prediction.

## Target encoders

1. Random ResNet-18 embeddings.
2. ImageNet-pretrained ResNet-18 embeddings.
3. v6.1 embeddings, if checkpoint/model is available.
4. Existing v6.2-A frozen embeddings.
5. Hierarchical v6.2 phase2 expert embeddings.

## Availability matrix

{markdown_table(rows)}

## Feature dump checks

### v6.2-A

- Feature dump exists: `{summary['v62a_features']['exists']}`
- Path: `{summary['v62a_features']['path']}`
- Samples: `{summary['v62a_features']['n_samples']}`
- Embedding dimension: `{summary['v62a_features']['embedding_dim']}`
- Missing expected fields: `{', '.join(summary['v62a_features']['missing_fields']) or 'none'}`
- Metadata issues: `{', '.join(summary['v62a_features']['metadata_issues']) or 'none'}`

### Hierarchical v6.2 phase2 expert

- Feature dump exists: `{summary['hierarchical_features']['exists']}`
- Path: `{summary['hierarchical_features']['path']}`
- Samples: `{summary['hierarchical_features']['n_samples']}`
- Embedding dimension: `{summary['hierarchical_features']['embedding_dim']}`
- Missing expected fields: `{', '.join(summary['hierarchical_features']['missing_fields']) or 'none'}`
- Metadata issues: `{', '.join(summary['hierarchical_features']['metadata_issues']) or 'none'}`

## Known metadata caveat

The hierarchical NPZ stored `board` and `split_group` as `unknown`. Previous audits recovered those fields from `distribution_group` and `path`. This is acceptable for internal comparison, but the hierarchical feature dump must be regenerated with correct `board` and `split_group` metadata before final publication tables.

## Blocking issues

{missing_lines}

## Non-blocking caveats

{caveat_lines}

## Decision

- Existing v6.2-A and hierarchical reports can be used for report-level comparison.
- No blocking missing artifact was found for internal baseline-comparison preparation.
- Unified baseline extraction should still wait for an explicit implementation step.
- Publication-grade tables require metadata cleanup/regeneration for the caveats listed above.
- v6.1 should be included only if a compatible model/checkpoint is available.
- Random and ImageNet ResNet-18 baselines require image-root and manifest availability before extraction.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit inputs for unified encoder baseline comparison.")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--v62a-features", type=Path, default=None)
    parser.add_argument("--hier-features", type=Path, default=None)
    parser.add_argument("--v61-checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, summary = build_audit(args)
    out_dir = args.out_dir
    report_path = out_dir / "ENCODER_BASELINE_INPUT_AUDIT.md"
    json_path = out_dir / "encoder_baseline_input_audit.json"
    write_report(report_path, rows, summary)
    write_json(json_path, summary)

    print(f"[done] report={report_path}")
    print(f"[done] key_numbers={json_path}")
    for row in rows:
        print(f"[{row['Status']}] {row['Encoder']} | {row['Required artifact']} | found={row['Found?']} | {row['Path']}")
        if row["Blocking issue"]:
            print(f"  issue: {row['Blocking issue']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
