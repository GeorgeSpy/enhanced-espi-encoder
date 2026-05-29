#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Safe load audit for the hierarchical / physics-aware v6.2 branch.

This script does not train, does not load raw images, and only performs a
single optional dummy forward pass after checkpoint loading. It refuses to use
placeholder physics modules silently.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_FILE = REPO_ROOT / "scripts" / "v6_2" / "espi_v6_2_fixed.py"
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "hierarchical_encoder"
REPORT_NAME = "LOAD_AUDIT_V62_HIERARCHICAL.md"
JSON_NAME = "load_audit_v62_hierarchical.json"


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def resolve_path(path_text: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path.resolve()
    return (base / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_parameters(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def is_tensor_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "detach")


def looks_like_state_dict(candidate: Any) -> bool:
    return isinstance(candidate, Mapping) and any(is_tensor_like(value) for value in candidate.values())


def unwrap_state_dict(checkpoint_obj: Any) -> tuple[dict[str, Any], str]:
    if looks_like_state_dict(checkpoint_obj):
        return dict(checkpoint_obj), "checkpoint root"

    if not isinstance(checkpoint_obj, Mapping):
        raise TypeError(f"Checkpoint object is not a mapping: {type(checkpoint_obj).__name__}")

    for key in (
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "module",
        "ema_state_dict",
        "model_ema",
    ):
        value = checkpoint_obj.get(key)
        if looks_like_state_dict(value):
            return dict(value), f"checkpoint['{key}']"

    raise ValueError("Could not find a tensor state_dict in the checkpoint.")


def strip_prefix(state_dict: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    stripped: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key[len(prefix) :] if key.startswith(prefix) else key
        stripped[new_key] = value
    return stripped


def state_dict_variants(state_dict: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    variants: list[tuple[str, dict[str, Any]]] = [("raw", dict(state_dict))]
    for prefix in ("module.", "model.", "net.", "model.module.", "module.model."):
        if any(key.startswith(prefix) for key in state_dict.keys()):
            variants.append((f"strip:{prefix}", strip_prefix(state_dict, prefix)))
    return variants


def score_state_dict(candidate: Mapping[str, Any], model_state: Mapping[str, Any]) -> dict[str, Any]:
    matched_keys: list[str] = []
    unexpected_keys: list[str] = []
    shape_mismatches: list[dict[str, str]] = []

    for key, value in candidate.items():
        if not is_tensor_like(value):
            unexpected_keys.append(key)
            continue
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        checkpoint_shape = tuple(value.shape)
        model_shape = tuple(model_state[key].shape)
        if checkpoint_shape == model_shape:
            matched_keys.append(key)
        else:
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": str(checkpoint_shape),
                    "model_shape": str(model_shape),
                }
            )

    matched_set = set(matched_keys)
    missing_keys = [key for key in model_state.keys() if key not in matched_set]
    return {
        "matched_keys": matched_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatches": shape_mismatches,
    }


def choose_best_state_dict(
    variants: list[tuple[str, dict[str, Any]]],
    model_state: Mapping[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    best_name = ""
    best_state: dict[str, Any] = {}
    best_score: dict[str, Any] | None = None

    for name, candidate in variants:
        score = score_state_dict(candidate, model_state)
        if best_score is None:
            best_name, best_state, best_score = name, candidate, score
            continue

        current_rank = (
            len(score["matched_keys"]),
            -len(score["shape_mismatches"]),
            -len(score["unexpected_keys"]),
        )
        best_rank = (
            len(best_score["matched_keys"]),
            -len(best_score["shape_mismatches"]),
            -len(best_score["unexpected_keys"]),
        )
        if current_rank > best_rank:
            best_name, best_state, best_score = name, candidate, score

    if best_score is None:
        raise ValueError("No state_dict variants were available for scoring.")
    return best_name, best_state, best_score


def import_model_module(model_file: Path) -> tuple[Any, str]:
    for candidate in (
        model_file.parent,
        REPO_ROOT / "scripts" / "pipeline",
        REPO_ROOT / "scripts" / "v6_2",
        REPO_ROOT,
        Path.cwd(),
    ):
        candidate_text = str(candidate)
        if candidate.exists() and candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)

    spec = importlib.util.spec_from_file_location("v62_hierarchical_model_audit", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {model_file}")

    module = importlib.util.module_from_spec(spec)
    import_log = io.StringIO()
    with contextlib.redirect_stdout(import_log), contextlib.redirect_stderr(import_log):
        spec.loader.exec_module(module)
    return module, import_log.getvalue().strip()


def run_dummy_forward(model: Any, device: Any, dummy_size: int, torch_module: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "attempted": True,
        "ok": False,
        "dummy_shape": [1, 1, dummy_size, dummy_size],
        "output_keys": [],
        "error": None,
        "availability": {
            "embeddings": False,
            "denoised": False,
            "wrapped_phase": False,
            "unwrapped_phase": False,
            "amplitude": False,
            "physics_feats": False,
            "spectral_filter": False,
        },
    }
    try:
        model.eval()
        dummy = torch_module.zeros((1, 1, dummy_size, dummy_size), device=device)
        with torch_module.no_grad():
            outputs = model(dummy)
        if isinstance(outputs, Mapping):
            keys = sorted(str(key) for key in outputs.keys())
            result["output_keys"] = keys
            for key in result["availability"].keys():
                result["availability"][key] = key in outputs
            result["ok"] = True
        else:
            result["error"] = f"Model returned {type(outputs).__name__}, not a mapping."
    except Exception as exc:  # pragma: no cover - depends on external model components
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def classify_status(audit: dict[str, Any]) -> tuple[str, str]:
    if audit["errors"]:
        return "fatal", audit["errors"][0]

    load = audit.get("load_audit", {})
    matched = int(load.get("matched_key_count", 0))
    missing = int(load.get("missing_key_count", 0))
    unexpected = int(load.get("unexpected_key_count", 0))
    shape_mismatch = int(load.get("shape_mismatch_count", 0))

    if matched == 0:
        return "fatal", "No checkpoint tensors matched the hierarchical model state_dict."
    if missing == 0 and unexpected == 0 and shape_mismatch == 0:
        return "clean", "Checkpoint keys match the hierarchical model state_dict cleanly."

    return (
        "acceptable with explanation",
        "Partial compatible-key load succeeded for audit purposes, but missing, unexpected, "
        "or shape-mismatched keys must be resolved before making representation claims.",
    )


def truncated_list(items: list[Any], limit: int = 40) -> str:
    if not items:
        return "- none\n"
    lines = [f"- `{item}`" for item in items[:limit]]
    if len(items) > limit:
        lines.append(f"- ... truncated {len(items) - limit} additional entries")
    return "\n".join(lines) + "\n"


def write_markdown_report(path: Path, audit: dict[str, Any]) -> None:
    load = audit.get("load_audit", {})
    dummy = audit.get("dummy_forward", {})
    availability = dummy.get("availability", {})

    text = f"""# v6.2 Hierarchical Checkpoint Load Audit

## Scope

This is a safe, read-only checkpoint load audit for the hierarchical / physics-aware v6.2 branch. It does not train a model, does not load raw images, and does not run full dataset inference.

## Inputs

- Model file: `{audit.get("model_file")}`
- Checkpoint path: `{audit.get("checkpoint_path")}`
- Checkpoint SHA256: `{audit.get("checkpoint_sha256") or "not available"}`
- Device: `{audit.get("device")}`
- Created UTC: `{audit.get("created_utc")}`

## Model Import

- Model class: `{audit.get("model_class") or "not available"}`
- Base physics modules imported: `{audit.get("base_imported")}`
- Placeholder physics modules refused: `{audit.get("placeholder_refused")}`
- Total parameters: `{audit.get("total_parameters") if audit.get("total_parameters") is not None else "not available"}`

## Load Status

- Load mismatch status: **{audit.get("load_mismatch_status")}**
- Explanation: {audit.get("load_mismatch_explanation")}
- Checkpoint state source: `{load.get("state_source") or "not available"}`
- State-dict variant: `{load.get("state_dict_variant") or "not available"}`
- Matched keys: `{load.get("matched_key_count", 0)}`
- Missing keys: `{load.get("missing_key_count", 0)}`
- Unexpected keys: `{load.get("unexpected_key_count", 0)}`
- Shape mismatches: `{load.get("shape_mismatch_count", 0)}`

## Dummy Forward Output Keys

- Dummy forward attempted: `{dummy.get("attempted", False)}`
- Dummy forward OK: `{dummy.get("ok", False)}`
- Dummy shape: `{dummy.get("dummy_shape")}`
- Output keys: `{dummy.get("output_keys", [])}`
- Dummy forward error: `{dummy.get("error") or "none"}`

## Output Availability

| Output | Available |
|---|---:|
| `embeddings` | {availability.get("embeddings", False)} |
| `denoised` | {availability.get("denoised", False)} |
| `wrapped_phase` | {availability.get("wrapped_phase", False)} |
| `unwrapped_phase` | {availability.get("unwrapped_phase", False)} |
| `amplitude` | {availability.get("amplitude", False)} |
| `physics_feats` | {availability.get("physics_feats", False)} |
| `spectral_filter` | {availability.get("spectral_filter", False)} |

`outputs["embeddings"]` is the first practical embedding point for `z_expert_prelogit` / `z_arcface_input`. `physics_feats` and `spectral_filter` are expected to be unavailable in the current forward output unless the model is later extended or instrumented with hooks.

## Missing Keys

{truncated_list(load.get("missing_keys", []))}
## Unexpected Keys

{truncated_list(load.get("unexpected_keys", []))}
## Shape Mismatches

{truncated_list([item.get("key", item) for item in load.get("shape_mismatches", [])])}
## Warnings

{truncated_list(audit.get("warnings", []))}
## Errors

{truncated_list(audit.get("errors", []))}
"""
    path.write_text(text, encoding="utf-8")


def save_reports(out_dir: Path, audit: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    status, explanation = classify_status(audit)
    audit["load_mismatch_status"] = status
    audit["load_mismatch_explanation"] = explanation

    json_path = out_dir / JSON_NAME
    md_path = out_dir / REPORT_NAME
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(md_path, audit)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely audit hierarchical v6.2 checkpoint compatibility without training."
    )
    parser.add_argument("--model-file", type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dummy-size", type=int, default=64)
    parser.add_argument("--skip-dummy-forward", action="store_true")
    args = parser.parse_args()

    model_file = resolve_path(args.model_file)
    checkpoint_path = resolve_path(args.checkpoint)
    out_dir = resolve_path(args.out_dir)

    audit: dict[str, Any] = {
        "created_utc": utc_now(),
        "model_file": str(model_file),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": None,
        "device": args.device,
        "model_class": None,
        "base_imported": None,
        "placeholder_refused": False,
        "total_parameters": None,
        "load_audit": {
            "state_source": None,
            "state_dict_variant": None,
            "matched_key_count": 0,
            "missing_key_count": 0,
            "unexpected_key_count": 0,
            "shape_mismatch_count": 0,
            "missing_keys": [],
            "unexpected_keys": [],
            "shape_mismatches": [],
        },
        "dummy_forward": {
            "attempted": False,
            "ok": False,
            "dummy_shape": None,
            "output_keys": [],
            "error": None,
            "availability": {
                "embeddings": False,
                "denoised": False,
                "wrapped_phase": False,
                "unwrapped_phase": False,
                "amplitude": False,
                "physics_feats": False,
                "spectral_filter": False,
            },
        },
        "warnings": [],
        "errors": [],
        "import_log": "",
    }

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        audit["errors"].append(f"Could not import torch: {type(exc).__name__}: {exc}")
        save_reports(out_dir, audit)
        print(f"[audit] wrote {out_dir / REPORT_NAME}")
        print(f"[audit] status: {audit['load_mismatch_status']}")
        return 2

    if not model_file.exists():
        audit["errors"].append(f"Model file does not exist: {model_file}")
        save_reports(out_dir, audit)
        print(f"[audit] wrote {out_dir / REPORT_NAME}")
        print(f"[audit] status: {audit['load_mismatch_status']}")
        return 2

    if not checkpoint_path.exists():
        audit["errors"].append(f"Checkpoint does not exist: {checkpoint_path}")
        save_reports(out_dir, audit)
        print(f"[audit] wrote {out_dir / REPORT_NAME}")
        print(f"[audit] status: {audit['load_mismatch_status']}")
        return 2

    try:
        audit["checkpoint_sha256"] = sha256_file(checkpoint_path)
    except Exception as exc:
        audit["errors"].append(f"Could not hash checkpoint: {type(exc).__name__}: {exc}")

    try:
        module, import_log = import_model_module(model_file)
        audit["import_log"] = import_log
        audit["base_imported"] = getattr(module, "BASE_IMPORTED", None)
        if audit["base_imported"] is False:
            audit["placeholder_refused"] = True
            audit["errors"].append(
                "Model file fell back to placeholder physics modules (BASE_IMPORTED=False); "
                "refusing to instantiate for a scientific load audit."
            )
            save_reports(out_dir, audit)
            print(f"[audit] wrote {out_dir / REPORT_NAME}")
            print(f"[audit] status: {audit['load_mismatch_status']}")
            return 2

        model_cls = getattr(module, "EnhancedHybridPhysicsESPI_V6_2", None)
        if model_cls is None:
            audit["errors"].append("EnhancedHybridPhysicsESPI_V6_2 was not found in the model file.")
            save_reports(out_dir, audit)
            print(f"[audit] wrote {out_dir / REPORT_NAME}")
            print(f"[audit] status: {audit['load_mismatch_status']}")
            return 2

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            audit["warnings"].append("CUDA requested but unavailable; falling back to CPU.")
            device = torch.device("cpu")
            audit["device"] = "cpu"

        model = model_cls().to(device)
        audit["model_class"] = f"{model_cls.__module__}.{model_cls.__name__}"
        audit["total_parameters"] = count_parameters(model)

        checkpoint_obj = torch.load(checkpoint_path, map_location=device)
        checkpoint_state, state_source = unwrap_state_dict(checkpoint_obj)
        model_state = model.state_dict()
        variant_name, chosen_state, score = choose_best_state_dict(
            state_dict_variants(checkpoint_state), model_state
        )
        compatible_state = {key: chosen_state[key] for key in score["matched_keys"]}
        load_result = model.load_state_dict(compatible_state, strict=False)

        missing_keys = list(load_result.missing_keys)
        unexpected_keys = list(score["unexpected_keys"])
        shape_mismatches = list(score["shape_mismatches"])

        audit["load_audit"] = {
            "state_source": state_source,
            "state_dict_variant": variant_name,
            "matched_key_count": len(score["matched_keys"]),
            "missing_key_count": len(missing_keys),
            "unexpected_key_count": len(unexpected_keys),
            "shape_mismatch_count": len(shape_mismatches),
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "shape_mismatches": shape_mismatches,
        }

        if not args.skip_dummy_forward:
            audit["dummy_forward"] = run_dummy_forward(model, device, args.dummy_size, torch)
        else:
            audit["dummy_forward"]["attempted"] = False
            audit["dummy_forward"]["error"] = "Skipped by --skip-dummy-forward."

    except Exception as exc:  # pragma: no cover - external artifact dependent
        audit["errors"].append(f"{type(exc).__name__}: {exc}")
        audit["traceback"] = traceback.format_exc()

    save_reports(out_dir, audit)
    print(f"[audit] wrote {out_dir / REPORT_NAME}")
    print(f"[audit] wrote {out_dir / JSON_NAME}")
    print(f"[audit] status: {audit['load_mismatch_status']}")
    print(f"[audit] matched keys: {audit['load_audit']['matched_key_count']}")
    print(f"[audit] missing keys: {audit['load_audit']['missing_key_count']}")
    print(f"[audit] unexpected keys: {audit['load_audit']['unexpected_key_count']}")
    return 0 if audit["load_mismatch_status"] in {"clean", "acceptable with explanation"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
