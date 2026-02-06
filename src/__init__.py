"""
BlockFFN + Mamba Hybrid Research Prototype.

This package provides tools for integrating Mamba2 sidecars with BlockFFN-3B-SFT
to test whether routing signals can gate between cheap (Mamba) and expensive
(attention) sequence mixing.
"""

from .routing_extractor import RoutingExtractor, RoutingHook
from .mamba_sidecar import MambaSidecar, GatedHybridLayer, attach_mamba_sidecar
from .eval_utils import (
    compute_perplexity,
    generate_text,
    MemoryTracker,
    benchmark_throughput,
)

__all__ = [
    # Routing extraction
    "RoutingExtractor",
    "RoutingHook",
    # Mamba integration
    "MambaSidecar",
    "GatedHybridLayer",
    "attach_mamba_sidecar",
    # Evaluation utilities
    "compute_perplexity",
    "generate_text",
    "MemoryTracker",
    "benchmark_throughput",
]

__version__ = "0.1.0"
