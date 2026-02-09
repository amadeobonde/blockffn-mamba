"""CLS-aware FFN optimization components."""

from src.ffn.chunk_analyzer import (
    ChunkExpertAnalyzer,
    ChunkAnalysis,
    SequenceAnalysis,
    create_expert_activation_matrix,
    compute_chunk_overlap_matrix,
)
from src.ffn.expert_batcher import (
    ChunkExpertBatcher,
    StreamingExpertBatcher,
    BatchedExpertResult,
    ExpertResultCache,
)
from src.ffn.cls_ffn import (
    CLSAwareFFN,
    CLSStatistics,
    attach_cls_ffn,
    detach_cls_ffn,
    get_cls_layers,
    get_all_cls_statistics,
    reset_all_cls_statistics,
)

__all__ = [
    # Analyzer
    "ChunkExpertAnalyzer",
    "ChunkAnalysis",
    "SequenceAnalysis",
    "create_expert_activation_matrix",
    "compute_chunk_overlap_matrix",
    # Batcher
    "ChunkExpertBatcher",
    "StreamingExpertBatcher",
    "BatchedExpertResult",
    "ExpertResultCache",
    # CLS FFN
    "CLSAwareFFN",
    "CLSStatistics",
    "attach_cls_ffn",
    "detach_cls_ffn",
    "get_cls_layers",
    "get_all_cls_statistics",
    "reset_all_cls_statistics",
]
