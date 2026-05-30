"""Compatibility shim for legacy ESPI checkpointing helper imports.

This module exists only so physics/model components from the legacy
``enhanced_espi_pipeline.py`` can be imported during safe checkpoint load audits.
It is not a replacement for the original checkpointing implementation.
"""


def save_checkpoint(*args, **kwargs):
    """Fail closed if training code attempts to save through the shim."""
    raise RuntimeError(
        "checkpointing.save_checkpoint compatibility shim is load-audit-only. "
        "Restore the original checkpointing.py before training or checkpointing."
    )

