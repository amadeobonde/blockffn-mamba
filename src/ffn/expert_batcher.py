"""
Chunk Expert Batcher

Batches expert computations across token chunks for efficiency.
Key optimization: compute union-of-experts once per chunk, scatter to tokens.

This exploits BlockFFN's ~70% CLS (Chunk-Level Sparsity) by:
1. Computing which experts are needed for the entire chunk
2. Loading those expert weights once
3. Running them on all relevant tokens in parallel
4. Scattering results back to individual token positions
"""

from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.ffn.chunk_analyzer import ChunkAnalysis, SequenceAnalysis


@dataclass
class BatchedExpertResult:
    """Result from batched expert computation.

    Attributes:
        output: Computed FFN output [batch, seq, hidden]
        experts_computed: Set of expert indices that were computed
        tokens_per_expert: Count of tokens using each expert
        cache_hits: Number of expert results reused from cache
    """
    output: Tensor
    experts_computed: Set[int]
    tokens_per_expert: Dict[int, int]
    cache_hits: int = 0


class ExpertResultCache:
    """
    LRU cache for expert computation results.

    Caches expert outputs to avoid redundant computation when
    the same expert is needed for nearby tokens.
    """

    def __init__(self, max_size_mb: int = 512):
        """
        Initialize cache.

        Args:
            max_size_mb: Maximum cache size in megabytes
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        self.cache: Dict[Tuple[int, int], Tensor] = {}  # (expert_idx, token_hash) -> result
        self.access_order: List[Tuple[int, int]] = []

    def _compute_tensor_size(self, t: Tensor) -> int:
        """Compute tensor size in bytes."""
        return t.numel() * t.element_size()

    def _evict_if_needed(self, required_bytes: int) -> None:
        """Evict oldest entries to make room."""
        while (
            self.access_order and
            self.current_size_bytes + required_bytes > self.max_size_bytes
        ):
            key = self.access_order.pop(0)
            if key in self.cache:
                tensor = self.cache.pop(key)
                self.current_size_bytes -= self._compute_tensor_size(tensor)

    def get(self, expert_idx: int, token_hash: int) -> Optional[Tensor]:
        """Get cached result if available."""
        key = (expert_idx, token_hash)
        if key in self.cache:
            # Move to end (most recent)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, expert_idx: int, token_hash: int, result: Tensor) -> None:
        """Cache a result."""
        key = (expert_idx, token_hash)
        size = self._compute_tensor_size(result)

        # Evict if needed
        self._evict_if_needed(size)

        # Store
        if key in self.cache:
            old_size = self._compute_tensor_size(self.cache[key])
            self.current_size_bytes -= old_size
            self.access_order.remove(key)

        self.cache[key] = result.detach()
        self.current_size_bytes += size
        self.access_order.append(key)

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()
        self.current_size_bytes = 0


class ChunkExpertBatcher:
    """
    Batches expert computations for CLS optimization.

    Given a chunk of tokens and their required experts, this class:
    1. Computes the union of required experts
    2. Loads those expert weights (once per chunk)
    3. Runs each expert on all tokens that need it
    4. Accumulates results per token

    Key optimizations:
    - Union-of-experts: Load each expert weight once per chunk
    - Token batching: Run expert on multiple tokens at once
    - Result caching: Reuse results for identical inputs
    """

    def __init__(
        self,
        cache_size_mb: int = 512,
        enable_caching: bool = True,
        min_tokens_for_batching: int = 2,
    ):
        """
        Initialize the batcher.

        Args:
            cache_size_mb: Expert result cache size
            enable_caching: Whether to cache expert results
            min_tokens_for_batching: Minimum tokens to benefit from batching
        """
        self.cache = ExpertResultCache(cache_size_mb) if enable_caching else None
        self.min_tokens_for_batching = min_tokens_for_batching

        # Statistics
        self.stats = {
            "chunks_processed": 0,
            "experts_computed": 0,
            "cache_hits": 0,
            "tokens_batched": 0,
        }

    def batch_experts_for_chunk(
        self,
        hidden_states: Tensor,
        expert_sets: List[Set[int]],
        expert_weights: Dict[int, Tuple[Tensor, Tensor]],
        activation_fn: nn.Module,
    ) -> BatchedExpertResult:
        """
        Execute experts in batched fashion for a chunk.

        Args:
            hidden_states: Input hidden states [chunk_size, hidden_dim]
            expert_sets: Per-token sets of active expert indices
            expert_weights: Dict mapping expert_idx to (up_proj, down_proj) weights
            activation_fn: Activation function between up and down projections

        Returns:
            BatchedExpertResult with outputs and statistics
        """
        chunk_size, hidden_dim = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Compute union of experts
        union_experts = set()
        for s in expert_sets:
            union_experts |= s

        # Track which tokens need each expert
        expert_to_tokens: Dict[int, List[int]] = {e: [] for e in union_experts}
        for t_idx, expert_set in enumerate(expert_sets):
            for e in expert_set:
                expert_to_tokens[e].append(t_idx)

        # Initialize output accumulator
        # Each token accumulates contributions from its active experts
        output = torch.zeros(chunk_size, hidden_dim, device=device, dtype=dtype)
        expert_counts = torch.zeros(chunk_size, device=device, dtype=dtype)

        cache_hits = 0
        tokens_per_expert = {}

        # Process each expert in the union
        for expert_idx in union_experts:
            token_indices = expert_to_tokens[expert_idx]
            if not token_indices:
                continue

            tokens_per_expert[expert_idx] = len(token_indices)

            # Get expert weights
            if expert_idx not in expert_weights:
                continue
            up_proj, down_proj = expert_weights[expert_idx]

            # Gather tokens for this expert
            token_indices_tensor = torch.tensor(token_indices, device=device)
            expert_inputs = hidden_states[token_indices_tensor]  # [num_tokens, hidden]

            # Check cache for each token
            cached_results = []
            uncached_indices = []
            uncached_inputs = []

            if self.cache is not None:
                for i, t_idx in enumerate(token_indices):
                    # Simple hash based on input values
                    token_hash = hash(tuple(hidden_states[t_idx, :8].tolist()))
                    cached = self.cache.get(expert_idx, token_hash)
                    if cached is not None:
                        cached_results.append((i, cached))
                        cache_hits += 1
                    else:
                        uncached_indices.append(i)
                        uncached_inputs.append(expert_inputs[i])
            else:
                uncached_indices = list(range(len(token_indices)))
                uncached_inputs = [expert_inputs[i] for i in uncached_indices]

            # Compute for uncached tokens
            if uncached_inputs:
                uncached_batch = torch.stack(uncached_inputs, dim=0)  # [num_uncached, hidden]

                # Expert forward: up_proj -> activation -> down_proj
                # up_proj: [hidden, intermediate]
                # down_proj: [intermediate, hidden]
                intermediate = F.linear(uncached_batch, up_proj)
                intermediate = activation_fn(intermediate)
                expert_output = F.linear(intermediate, down_proj)

                # Cache results
                if self.cache is not None:
                    for i, local_idx in enumerate(uncached_indices):
                        t_idx = token_indices[local_idx]
                        token_hash = hash(tuple(hidden_states[t_idx, :8].tolist()))
                        self.cache.put(expert_idx, token_hash, expert_output[i])

                # Scatter results back
                for i, local_idx in enumerate(uncached_indices):
                    t_idx = token_indices[local_idx]
                    output[t_idx] += expert_output[i]
                    expert_counts[t_idx] += 1

            # Add cached results
            for local_idx, cached_result in cached_results:
                t_idx = token_indices[local_idx]
                output[t_idx] += cached_result.to(device)
                expert_counts[t_idx] += 1

        # Average by number of experts per token (or leave as sum if that's the design)
        # BlockFFN typically sums, not averages
        # output = output / expert_counts.unsqueeze(-1).clamp(min=1)

        # Update stats
        self.stats["chunks_processed"] += 1
        self.stats["experts_computed"] += len(union_experts)
        self.stats["cache_hits"] += cache_hits
        self.stats["tokens_batched"] += chunk_size

        return BatchedExpertResult(
            output=output,
            experts_computed=union_experts,
            tokens_per_expert=tokens_per_expert,
            cache_hits=cache_hits,
        )

    def batch_experts_indexed(
        self,
        hidden_states: Tensor,
        routing_weights: Tensor,
        top_k_indices: Tensor,
        expert_weights: List[Tuple[Tensor, Tensor]],
        activation_fn: nn.Module,
    ) -> Tensor:
        """
        Execute experts using top-k routing indices.

        This is an alternative interface that works with standard
        MoE routing outputs (top-k indices and weights).

        Args:
            hidden_states: [batch, seq, hidden]
            routing_weights: [batch, seq, k] weights for top-k experts
            top_k_indices: [batch, seq, k] indices of top-k experts
            expert_weights: List of (up_proj, down_proj) for each expert
            activation_fn: Activation function

        Returns:
            output: [batch, seq, hidden]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        k = top_k_indices.shape[-1]
        device = hidden_states.device
        dtype = hidden_states.dtype

        output = torch.zeros_like(hidden_states)

        # Process in chunks for memory efficiency
        chunk_size = 8
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk_len = end - start

            # Get chunk data
            chunk_hidden = hidden_states[:, start:end, :]  # [batch, chunk, hidden]
            chunk_weights = routing_weights[:, start:end, :]  # [batch, chunk, k]
            chunk_indices = top_k_indices[:, start:end, :]  # [batch, chunk, k]

            # Find unique experts in this chunk
            unique_experts = torch.unique(chunk_indices).tolist()

            # Process each expert
            for expert_idx in unique_experts:
                if expert_idx >= len(expert_weights):
                    continue

                up_proj, down_proj = expert_weights[expert_idx]

                # Find all (batch, token, k_position) that use this expert
                mask = chunk_indices == expert_idx  # [batch, chunk, k]

                if not mask.any():
                    continue

                # Gather inputs and weights for this expert
                # This is complex due to 3D indexing; simplify with loop
                for b in range(batch_size):
                    for t in range(chunk_len):
                        for ki in range(k):
                            if mask[b, t, ki]:
                                # This position uses this expert
                                inp = chunk_hidden[b, t, :]  # [hidden]
                                weight = chunk_weights[b, t, ki]

                                # Compute expert output
                                intermediate = F.linear(inp.unsqueeze(0), up_proj)
                                intermediate = activation_fn(intermediate)
                                expert_out = F.linear(intermediate, down_proj)

                                # Accumulate weighted output
                                output[b, start + t, :] += weight * expert_out.squeeze(0)

        return output

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = {
            "chunks_processed": 0,
            "experts_computed": 0,
            "cache_hits": 0,
            "tokens_batched": 0,
        }

    def clear_cache(self) -> None:
        """Clear the result cache."""
        if self.cache is not None:
            self.cache.clear()


class StreamingExpertBatcher:
    """
    Expert batcher optimized for streaming/autoregressive generation.

    Maintains state across tokens for efficient incremental processing.
    Key optimization: reuses expert results when consecutive tokens
    share experts (common in BlockFFN due to high CLS).
    """

    def __init__(
        self,
        window_size: int = 8,
        overlap_threshold: float = 0.7,
    ):
        """
        Initialize streaming batcher.

        Args:
            window_size: Number of recent tokens to consider for reuse
            overlap_threshold: Min overlap to attempt result reuse
        """
        self.window_size = window_size
        self.overlap_threshold = overlap_threshold

        # Recent expert results: (expert_idx, input_hash) -> result
        self.recent_results: Dict[Tuple[int, int], Tensor] = {}
        self.recent_expert_sets: List[Set[int]] = []

        # Statistics
        self.reuse_count = 0
        self.compute_count = 0

    def process_token(
        self,
        hidden_state: Tensor,
        active_experts: Set[int],
        expert_weights: Dict[int, Tuple[Tensor, Tensor]],
        activation_fn: nn.Module,
    ) -> Tensor:
        """
        Process a single token, reusing recent results where possible.

        Args:
            hidden_state: [hidden_dim] input for this token
            active_experts: Set of active expert indices
            expert_weights: Expert weight dictionary
            activation_fn: Activation function

        Returns:
            output: [hidden_dim] FFN output for this token
        """
        hidden_dim = hidden_state.shape[0]
        device = hidden_state.device
        dtype = hidden_state.dtype

        output = torch.zeros(hidden_dim, device=device, dtype=dtype)

        # Compute hash for this input
        input_hash = hash(tuple(hidden_state[:8].tolist()))

        for expert_idx in active_experts:
            key = (expert_idx, input_hash)

            # Check if we can reuse a recent result
            if key in self.recent_results:
                output += self.recent_results[key]
                self.reuse_count += 1
                continue

            # Need to compute
            if expert_idx not in expert_weights:
                continue

            up_proj, down_proj = expert_weights[expert_idx]

            intermediate = F.linear(hidden_state.unsqueeze(0), up_proj)
            intermediate = activation_fn(intermediate)
            expert_out = F.linear(intermediate, down_proj).squeeze(0)

            output += expert_out
            self.compute_count += 1

            # Cache this result
            self.recent_results[key] = expert_out.detach()

        # Update recent expert sets
        self.recent_expert_sets.append(active_experts)
        if len(self.recent_expert_sets) > self.window_size:
            self.recent_expert_sets.pop(0)

        # Prune old cache entries
        self._prune_cache()

        return output

    def _prune_cache(self) -> None:
        """Remove cache entries for experts no longer in recent window."""
        if not self.recent_expert_sets:
            self.recent_results.clear()
            return

        # Find all experts in recent window
        active_in_window = set()
        for expert_set in self.recent_expert_sets:
            active_in_window |= expert_set

        # Remove entries for experts not in window
        keys_to_remove = [
            key for key in self.recent_results
            if key[0] not in active_in_window
        ]
        for key in keys_to_remove:
            del self.recent_results[key]

    def reset(self) -> None:
        """Reset state for new sequence."""
        self.recent_results.clear()
        self.recent_expert_sets.clear()
        self.reuse_count = 0
        self.compute_count = 0

    @property
    def reuse_rate(self) -> float:
        """Fraction of expert computations that were reused."""
        total = self.reuse_count + self.compute_count
        return self.reuse_count / total if total > 0 else 0.0
