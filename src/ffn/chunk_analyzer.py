"""
Chunk Expert Analyzer

Analyzes expert activation patterns across chunks of tokens to identify
opportunities for CLS (Chunk-Level Sparsity) optimization.

Key insight: BlockFFN has ~70% expert overlap across 8-token chunks.
This means we can compute the union of experts once and scatter results.
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import torch
from torch import Tensor


@dataclass
class ChunkAnalysis:
    """Analysis results for a single chunk of tokens.

    Attributes:
        chunk_idx: Index of this chunk in the sequence
        start_pos: Starting position in sequence
        end_pos: Ending position (exclusive)
        expert_sets: Per-token set of active expert indices
        union_experts: Union of all active experts in chunk
        overlap_ratio: Pairwise overlap ratio (0-1)
        can_batch: Whether this chunk benefits from batched execution
    """
    chunk_idx: int
    start_pos: int
    end_pos: int
    expert_sets: List[Set[int]]
    union_experts: Set[int]
    overlap_ratio: float
    can_batch: bool

    @property
    def chunk_size(self) -> int:
        return self.end_pos - self.start_pos

    @property
    def num_union_experts(self) -> int:
        return len(self.union_experts)

    @property
    def avg_experts_per_token(self) -> float:
        if not self.expert_sets:
            return 0.0
        return sum(len(s) for s in self.expert_sets) / len(self.expert_sets)

    @property
    def efficiency_gain(self) -> float:
        """Estimate efficiency gain from batching.

        Returns ratio of (batched cost) / (per-token cost).
        Lower is better (more savings).
        """
        if not self.expert_sets:
            return 1.0
        total_per_token = sum(len(s) for s in self.expert_sets)
        if total_per_token == 0:
            return 1.0
        return len(self.union_experts) / total_per_token


@dataclass
class SequenceAnalysis:
    """Analysis results for an entire sequence.

    Attributes:
        chunks: List of per-chunk analysis
        total_tokens: Total tokens analyzed
        mean_overlap: Mean overlap ratio across chunks
        mean_efficiency_gain: Mean efficiency gain from batching
    """
    chunks: List[ChunkAnalysis]
    total_tokens: int
    mean_overlap: float
    mean_efficiency_gain: float

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def batchable_chunks(self) -> int:
        return sum(1 for c in self.chunks if c.can_batch)


class ChunkExpertAnalyzer:
    """
    Analyzes expert activation patterns to identify CLS opportunities.

    Given routing signals from BlockFFN's router, this class:
    1. Identifies which experts are active per token
    2. Groups tokens into chunks
    3. Computes expert overlap within chunks
    4. Determines which chunks benefit from batched execution

    Attributes:
        chunk_size: Number of tokens per chunk (default 8)
        overlap_threshold: Min overlap to enable batching (default 0.5)
        min_active_experts: Minimum active experts per token
    """

    def __init__(
        self,
        chunk_size: int = 8,
        overlap_threshold: float = 0.5,
        min_active_experts: int = 1,
    ):
        """
        Initialize the analyzer.

        Args:
            chunk_size: Tokens per chunk (default 8, as in paper)
            overlap_threshold: Min overlap ratio to enable batching
            min_active_experts: Minimum experts that must be active
        """
        assert chunk_size > 0
        assert 0.0 <= overlap_threshold <= 1.0

        self.chunk_size = chunk_size
        self.overlap_threshold = overlap_threshold
        self.min_active_experts = min_active_experts

    def get_active_experts(
        self,
        routing_signals: Tensor,
        threshold: float = 0.0,
    ) -> List[Set[int]]:
        """
        Extract active expert sets from routing signals.

        Args:
            routing_signals: Post-ReLU routing signals [seq, num_experts]
                            or [batch, seq, num_experts]
            threshold: Activation threshold (0.0 = any positive activation)

        Returns:
            List of sets, one per token, containing active expert indices
        """
        # Handle batch dimension
        if routing_signals.dim() == 3:
            # Take first batch element for now
            routing_signals = routing_signals[0]

        seq_len, num_experts = routing_signals.shape
        expert_sets = []

        for t in range(seq_len):
            # Find experts above threshold
            active_mask = routing_signals[t] > threshold
            active_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)

            if active_indices.dim() == 0:
                # Single expert case
                active_set = {active_indices.item()} if active_indices.numel() > 0 else set()
            else:
                active_set = set(active_indices.tolist())

            expert_sets.append(active_set)

        return expert_sets

    def compute_pairwise_overlap(
        self,
        expert_sets: List[Set[int]],
    ) -> float:
        """
        Compute mean pairwise overlap ratio for a list of expert sets.

        Overlap(A, B) = |A ∩ B| / |A ∪ B|  (Jaccard similarity)

        Args:
            expert_sets: List of expert sets

        Returns:
            Mean pairwise Jaccard similarity
        """
        if len(expert_sets) < 2:
            return 1.0  # Single token = perfect overlap

        total_overlap = 0.0
        num_pairs = 0

        for i in range(len(expert_sets)):
            for j in range(i + 1, len(expert_sets)):
                set_a = expert_sets[i]
                set_b = expert_sets[j]

                if not set_a and not set_b:
                    # Both empty - consider as full overlap
                    overlap = 1.0
                elif not set_a or not set_b:
                    # One empty - no overlap
                    overlap = 0.0
                else:
                    intersection = len(set_a & set_b)
                    union = len(set_a | set_b)
                    overlap = intersection / union if union > 0 else 0.0

                total_overlap += overlap
                num_pairs += 1

        return total_overlap / num_pairs if num_pairs > 0 else 0.0

    def compute_consecutive_overlap(
        self,
        expert_sets: List[Set[int]],
    ) -> float:
        """
        Compute mean overlap between consecutive tokens.

        This is more relevant for CLS since we typically process
        tokens sequentially and can reuse recent expert results.

        Args:
            expert_sets: List of expert sets

        Returns:
            Mean consecutive overlap ratio
        """
        if len(expert_sets) < 2:
            return 1.0

        total_overlap = 0.0

        for i in range(len(expert_sets) - 1):
            set_a = expert_sets[i]
            set_b = expert_sets[i + 1]

            if not set_a and not set_b:
                overlap = 1.0
            elif not set_a or not set_b:
                overlap = 0.0
            else:
                intersection = len(set_a & set_b)
                union = len(set_a | set_b)
                overlap = intersection / union if union > 0 else 0.0

            total_overlap += overlap

        return total_overlap / (len(expert_sets) - 1)

    def analyze_chunk(
        self,
        expert_sets: List[Set[int]],
        chunk_idx: int,
        start_pos: int,
    ) -> ChunkAnalysis:
        """
        Analyze a single chunk of tokens.

        Args:
            expert_sets: Expert sets for tokens in this chunk
            chunk_idx: Index of this chunk
            start_pos: Starting position in sequence

        Returns:
            ChunkAnalysis with overlap and batching information
        """
        # Compute union of all experts
        union_experts = set()
        for s in expert_sets:
            union_experts |= s

        # Compute overlap ratio
        overlap = self.compute_pairwise_overlap(expert_sets)

        # Determine if batching is beneficial
        # Batching helps when overlap is high (reuse experts)
        # but not when union is too large (memory pressure)
        can_batch = (
            overlap >= self.overlap_threshold and
            len(union_experts) >= self.min_active_experts
        )

        return ChunkAnalysis(
            chunk_idx=chunk_idx,
            start_pos=start_pos,
            end_pos=start_pos + len(expert_sets),
            expert_sets=expert_sets,
            union_experts=union_experts,
            overlap_ratio=overlap,
            can_batch=can_batch,
        )

    def analyze_sequence(
        self,
        routing_signals: Tensor,
        threshold: float = 0.0,
    ) -> SequenceAnalysis:
        """
        Analyze an entire sequence for CLS opportunities.

        Args:
            routing_signals: [seq, num_experts] or [batch, seq, num_experts]
            threshold: Activation threshold

        Returns:
            SequenceAnalysis with per-chunk and aggregate statistics
        """
        # Get active experts per token
        expert_sets = self.get_active_experts(routing_signals, threshold)
        seq_len = len(expert_sets)

        # Analyze chunks
        chunks = []
        for chunk_idx, start_pos in enumerate(range(0, seq_len, self.chunk_size)):
            end_pos = min(start_pos + self.chunk_size, seq_len)
            chunk_expert_sets = expert_sets[start_pos:end_pos]

            analysis = self.analyze_chunk(chunk_expert_sets, chunk_idx, start_pos)
            chunks.append(analysis)

        # Compute aggregate statistics
        if chunks:
            mean_overlap = sum(c.overlap_ratio for c in chunks) / len(chunks)
            mean_efficiency = sum(c.efficiency_gain for c in chunks) / len(chunks)
        else:
            mean_overlap = 0.0
            mean_efficiency = 1.0

        return SequenceAnalysis(
            chunks=chunks,
            total_tokens=seq_len,
            mean_overlap=mean_overlap,
            mean_efficiency_gain=mean_efficiency,
        )

    def get_expert_usage_stats(
        self,
        routing_signals: Tensor,
        threshold: float = 0.0,
    ) -> Dict[str, float]:
        """
        Get statistics about expert usage patterns.

        Args:
            routing_signals: [seq, num_experts] or [batch, seq, num_experts]
            threshold: Activation threshold

        Returns:
            Dict with usage statistics
        """
        expert_sets = self.get_active_experts(routing_signals, threshold)

        if not expert_sets:
            return {
                "mean_experts_per_token": 0.0,
                "total_unique_experts": 0,
                "sparsity": 1.0,
                "consecutive_overlap": 1.0,
            }

        # Count experts
        num_experts = routing_signals.shape[-1]
        all_experts = set()
        total_active = 0

        for s in expert_sets:
            all_experts |= s
            total_active += len(s)

        mean_per_token = total_active / len(expert_sets)
        sparsity = 1.0 - (mean_per_token / num_experts)

        return {
            "mean_experts_per_token": mean_per_token,
            "total_unique_experts": len(all_experts),
            "sparsity": sparsity,
            "consecutive_overlap": self.compute_consecutive_overlap(expert_sets),
        }


def create_expert_activation_matrix(
    routing_signals: Tensor,
    threshold: float = 0.0,
) -> Tensor:
    """
    Create a binary expert activation matrix.

    Args:
        routing_signals: [batch, seq, num_experts]
        threshold: Activation threshold

    Returns:
        Binary tensor [batch, seq, num_experts] where 1 = active
    """
    return (routing_signals > threshold).float()


def compute_chunk_overlap_matrix(
    activation_matrix: Tensor,
    chunk_size: int = 8,
) -> Tensor:
    """
    Compute overlap matrix between consecutive tokens within chunks.

    Args:
        activation_matrix: Binary [batch, seq, num_experts]
        chunk_size: Tokens per chunk

    Returns:
        Overlap values [batch, num_chunks, chunk_size-1]
    """
    batch_size, seq_len, num_experts = activation_matrix.shape
    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    overlaps = []

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = activation_matrix[:, start:end, :]  # [batch, chunk_len, experts]

        if chunk.shape[1] < 2:
            continue

        # Compute pairwise overlaps within chunk
        chunk_overlaps = []
        for t in range(chunk.shape[1] - 1):
            a = chunk[:, t, :]      # [batch, experts]
            b = chunk[:, t + 1, :]  # [batch, experts]

            intersection = (a * b).sum(dim=-1)  # [batch]
            union = ((a + b) > 0).float().sum(dim=-1)  # [batch]

            overlap = intersection / (union + 1e-8)  # [batch]
            chunk_overlaps.append(overlap)

        if chunk_overlaps:
            overlaps.append(torch.stack(chunk_overlaps, dim=1))  # [batch, chunk_len-1]

    if not overlaps:
        return torch.tensor([])

    return torch.stack(overlaps, dim=1)  # [batch, num_chunks, chunk_len-1]
