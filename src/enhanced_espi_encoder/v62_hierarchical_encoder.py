"""Frozen wrapper for the hierarchical / physics-aware v6.2 ESPI encoder."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_FILE = REPO_ROOT / "scripts" / "v6_2" / "espi_v6_2_fixed.py"


def _is_tensor_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "detach")


def _looks_like_state_dict(value: Any) -> bool:
    return isinstance(value, Mapping) and any(_is_tensor_like(item) for item in value.values())


def _unwrap_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if _looks_like_state_dict(checkpoint):
        return checkpoint
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint is not a mapping: {type(checkpoint).__name__}")
    for key in ("state_dict", "model_state_dict", "model", "net", "module"):
        value = checkpoint.get(key)
        if _looks_like_state_dict(value):
            return value
    raise ValueError("Could not find a tensor state_dict in checkpoint.")


def _add_import_paths(paths: Sequence[Path | str]) -> None:
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            continue
        text = str(resolved)
        if text not in sys.path:
            sys.path.insert(0, text)


def _import_hierarchical_module(model_file: Path, dependency_paths: Sequence[Path | str]) -> tuple[Any, str]:
    _add_import_paths(
        [
            *dependency_paths,
            model_file.parent,
            REPO_ROOT / "external" / "compat",
            REPO_ROOT / "scripts" / "v6_2",
            REPO_ROOT,
        ]
    )
    spec = importlib.util.spec_from_file_location("v62_hierarchical_encoder_model", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {model_file}")
    module = importlib.util.module_from_spec(spec)
    import_log = io.StringIO()
    with contextlib.redirect_stdout(import_log), contextlib.redirect_stderr(import_log):
        spec.loader.exec_module(module)
    if getattr(module, "BASE_IMPORTED", None) is not True:
        raise RuntimeError(
            "Refusing to instantiate hierarchical v6.2 model because BASE_IMPORTED is not True. "
            "Placeholder physics modules must not be used."
        )
    if not hasattr(module, "EnhancedHybridPhysicsESPI_V6_2"):
        raise AttributeError("EnhancedHybridPhysicsESPI_V6_2 was not found in the model file.")
    return module, import_log.getvalue().strip()


class ESPIv62HierarchicalEncoder:
    """Read-only frozen encoder wrapper for hierarchical v6.2 checkpoints."""

    def __init__(
        self,
        checkpoint: Path | str,
        model_file: Path | str = DEFAULT_MODEL_FILE,
        device: str | torch.device = "cpu",
        dependency_paths: Sequence[Path | str] = (),
        strict: bool = True,
    ) -> None:
        self.model_file = Path(model_file).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        if not self.model_file.exists():
            raise FileNotFoundError(f"Model file does not exist: {self.model_file}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint}")

        module, import_log = _import_hierarchical_module(self.model_file, dependency_paths)
        self.import_log = import_log
        self.module = module
        self.model = module.EnhancedHybridPhysicsESPI_V6_2().to(self.device)

        checkpoint_obj = torch.load(self.checkpoint, map_location=self.device)
        state_dict = _unwrap_state_dict(checkpoint_obj)
        load_result = self.model.load_state_dict(state_dict, strict=strict)
        self.load_missing_keys = list(load_result.missing_keys)
        self.load_unexpected_keys = list(load_result.unexpected_keys)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def output_names(self) -> list[str]:
        return [
            "z_expert_prelogit",
            "z_arcface_input",
            "gatekeeper_logits",
            "denoised",
            "wrapped_phase",
            "unwrapped_phase",
            "amplitude",
        ]

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> dict[str, torch.Tensor | None]:
        if images.ndim != 4:
            raise ValueError(f"Expected BCHW tensor, got shape {tuple(images.shape)}")
        outputs = self.model(images.to(self.device))
        if not isinstance(outputs, Mapping):
            raise TypeError(f"Expected model outputs to be a mapping, got {type(outputs).__name__}")
        if "embeddings" not in outputs:
            raise KeyError('Hierarchical model output does not contain required key "embeddings".')

        embeddings = outputs["embeddings"]
        return {
            "z_expert_prelogit": embeddings,
            "z_arcface_input": embeddings,
            "gatekeeper_logits": outputs.get("gate_logits"),
            "denoised": outputs.get("denoised"),
            "wrapped_phase": outputs.get("wrapped_phase"),
            "unwrapped_phase": outputs.get("unwrapped_phase"),
            "amplitude": outputs.get("amplitude"),
        }

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> dict[str, torch.Tensor | None]:
        return self.encode(images)

