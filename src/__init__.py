"""
BlockFFN Adaptive Inference Framework

A single-model inference acceleration framework that exploits BlockFFN's
intrinsic properties (TLS, CLS, routing sparsity) to dynamically allocate
compute per token without sacrificing correctness.

Key components:
- Adaptive Attention: Variable window sizes based on routing sparsity
- CLS FFN: Chunk-level expert batching and reuse
- Verification: Confidence-based escalation decisions
- Engine: Unified orchestrator for all optimizations
"""

from src.adaptive.config import AdaptiveInferenceConfig, PRESETS
from src.engine.adaptive_engine import (
    AdaptiveInferenceEngine,
    GenerationStats,
    create_adaptive_engine,
    print_stats_summary,
)

__version__ = "0.2.0"
__all__ = [
    # Config
    "AdaptiveInferenceConfig",
    "PRESETS",
    # Engine
    "AdaptiveInferenceEngine",
    "GenerationStats",
    "create_adaptive_engine",
    "print_stats_summary",
]
