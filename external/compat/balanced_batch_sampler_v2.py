"""Compatibility shim for legacy ESPI training-only sampler imports.

The hierarchical load-audit path imports legacy physics components from
``enhanced_espi_pipeline.py``. That module imports ``BalancedBatchSampler`` at
module import time, although the sampler is only needed for training/data-loader
construction and is not used when instantiating the physics model or running a
dummy forward pass.

This shim is intentionally fail-closed: it lets model components import for
checkpoint load-auditing, but raises immediately if training code tries to use
the sampler.
"""

__version__ = "compat-shim-load-audit-only"


class BalancedBatchSampler:
    """Fail-closed placeholder for load-audit-only imports."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "BalancedBatchSampler compatibility shim is load-audit-only. "
            "Restore the original balanced_batch_sampler_v2.py before training "
            "or constructing data loaders."
        )

