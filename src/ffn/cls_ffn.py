"""
CLS-Aware FFN Layer

Wrapper for BlockFFN's MLP layers that exploits Chunk-Level Sparsity (CLS).
Intercepts the forward pass to:
1. Analyze routing patterns across token chunks
2. Batch expert computations for efficiency
3. Reuse expert results for overlapping activations

This is the main integration point for FFN optimization.
"""

from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.adaptive.config import AdaptiveInferenceConfig
from src.ffn.chunk_analyzer import ChunkExpertAnalyzer, SequenceAnalysis
from src.ffn.expert_batcher import (
    ChunkExpertBatcher,
    StreamingExpertBatcher,
    BatchedExpertResult,
)


@dataclass
class CLSStatistics:
    """Statistics from CLS-aware FFN processing."""
    total_tokens: int = 0
    total_experts_computed: int = 0
    total_experts_reused: int = 0
    chunks_processed: int = 0
    mean_overlap: float = 0.0
    mean_efficiency_gain: float = 0.0
    expert_load_count: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def reuse_rate(self) -> float:
        total = self.total_experts_computed + self.total_experts_reused
        return self.total_experts_reused / total if total > 0 else 0.0

    @property
    def compute_savings(self) -> float:
        """Estimated compute savings from CLS optimization."""
        total = self.total_experts_computed + self.total_experts_reused
        if total == 0:
            return 0.0
        return self.total_experts_reused / total


class CLSAwareFFN(nn.Module):
    """
    CLS-aware wrapper for BlockFFN's MLP layers.

    Wraps an existing MLP layer and intercepts forward pass to apply
    chunk-level sparsity optimization. The original layer's weights
    are not modified.

    Architecture of BlockFFN MLP:
        hidden -> router_proj -> router_act_fn -> router_norm -> routing_weights
        hidden -> gate_proj -> activation -> gate_output
        hidden -> up_proj -> up_output
        gate_output * up_output -> intermediate
        intermediate -> down_proj -> output

    With BlockFFN, the routing determines which experts (blocks) to compute.
    We exploit that neighboring tokens often use similar experts (CLS).

    Attributes:
        original_mlp: The wrapped MLP layer
        layer_idx: Layer index in the model
        chunk_analyzer: Analyzer for routing patterns
        chunk_batcher: Batcher for expert computations
        collect_statistics: Whether to collect runtime stats
    """

    def __init__(
        self,
        original_mlp: nn.Module,
        layer_idx: int,
        config: AdaptiveInferenceConfig,
    ):
        """
        Initialize CLS-aware FFN.

        Args:
            original_mlp: Original MLP layer to wrap
            layer_idx: Index of this layer
            config: Adaptive inference configuration
        """
        super().__init__()

        self.original_mlp = original_mlp
        self.layer_idx = layer_idx
        self.config = config
        self.collect_statistics = config.collect_statistics

        # Initialize components
        self.chunk_analyzer = ChunkExpertAnalyzer(
            chunk_size=config.chunk_size,
            overlap_threshold=config.overlap_threshold,
        )

        self.chunk_batcher = ChunkExpertBatcher(
            cache_size_mb=config.expert_cache_mb,
            enable_caching=True,
        )

        self.streaming_batcher = StreamingExpertBatcher(
            window_size=config.chunk_size,
            overlap_threshold=config.overlap_threshold,
        )

        # Statistics
        self._stats = CLSStatistics() if self.collect_statistics else None

        # Cache for extracted expert weights
        self._expert_weights: Optional[Dict[int, Tuple[Tensor, Tensor]]] = None
        self._num_experts: Optional[int] = None

        # Extract expert structure
        self._extract_expert_structure()

    def _extract_expert_structure(self) -> None:
        """Extract expert weight structure from the original MLP."""
        mlp = self.original_mlp

        # BlockFFN has a specific structure with router and block weights
        # The exact structure depends on the implementation

        # Try to find number of experts
        if hasattr(mlp, "num_experts"):
            self._num_experts = mlp.num_experts
        elif hasattr(mlp, "router_proj"):
            # Number of experts = output dimension of router projection
            self._num_experts = mlp.router_proj.out_features
        else:
            # Fallback: try to infer from other attributes
            self._num_experts = None

    def _get_expert_weights(self) -> Dict[int, Tuple[Tensor, Tensor]]:
        """
        Extract or cache expert weights from the MLP.

        Returns:
            Dict mapping expert_idx to (up_proj, down_proj) weight tensors
        """
        if self._expert_weights is not None:
            return self._expert_weights

        mlp = self.original_mlp
        expert_weights = {}

        # BlockFFN uses a different structure than standard MoE
        # It has gate_proj, up_proj, down_proj with routing determining
        # which blocks (contiguous slices) to compute

        if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
            # Standard structure - weights are shared, routing selects blocks
            # For now, treat the entire FFN as a single "expert"
            # More sophisticated block extraction would depend on BlockFFN internals
            expert_weights[0] = (
                mlp.up_proj.weight,  # up projection
                mlp.down_proj.weight,  # down projection
            )

        self._expert_weights = expert_weights
        return expert_weights

    def forward(
        self,
        hidden_states: Tensor,
        routing_weights: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Forward pass with CLS optimization.

        Args:
            hidden_states: Input [batch, seq, hidden]
            routing_weights: Optional pre-computed routing weights
            **kwargs: Additional arguments passed to original MLP

        Returns:
            output: FFN output [batch, seq, hidden]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # For short sequences, use original forward
        if seq_len < self.config.chunk_size:
            return self.original_mlp(hidden_states, **kwargs)

        # Step 1: Get routing signals
        routing_signals = self._compute_routing(hidden_states)

        if routing_signals is None:
            # Routing not available, fall back to original
            return self.original_mlp(hidden_states, **kwargs)

        # Step 2: Analyze routing patterns
        analysis = self.chunk_analyzer.analyze_sequence(routing_signals)

        # Step 3: Decide whether CLS optimization is beneficial
        if analysis.mean_overlap < self.config.overlap_threshold:
            # Low overlap, CLS won't help much
            if self._stats:
                self._stats.total_tokens += seq_len
            return self.original_mlp(hidden_states, **kwargs)

        # Step 4: Execute with CLS optimization
        output = self._execute_with_cls(
            hidden_states, routing_signals, analysis
        )

        return output

    def _compute_routing(self, hidden_states: Tensor) -> Optional[Tensor]:
        """
        Compute routing signals from hidden states.

        Args:
            hidden_states: [batch, seq, hidden]

        Returns:
            routing_signals: [batch, seq, num_experts] or None
        """
        mlp = self.original_mlp

        if not hasattr(mlp, "router_proj"):
            return None

        with torch.no_grad():
            # Router forward: proj -> activation -> norm
            router_logits = mlp.router_proj(hidden_states)

            if hasattr(mlp, "router_act_fn"):
                routing_signals = mlp.router_act_fn(router_logits)
            else:
                routing_signals = F.relu(router_logits)

            # Note: We don't apply router_norm here since we just need
            # the sparsity pattern, not the exact weights

        return routing_signals

    def _execute_with_cls(
        self,
        hidden_states: Tensor,
        routing_signals: Tensor,
        analysis: SequenceAnalysis,
    ) -> Tensor:
        """
        Execute FFN with CLS optimization.

        Args:
            hidden_states: [batch, seq, hidden]
            routing_signals: [batch, seq, num_experts]
            analysis: Pre-computed sequence analysis

        Returns:
            output: [batch, seq, hidden]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # For batch size > 1, process each sequence separately
        # (CLS patterns differ across sequences)
        if batch_size > 1:
            outputs = []
            for b in range(batch_size):
                out = self._execute_single_sequence(
                    hidden_states[b],
                    routing_signals[b],
                )
                outputs.append(out)
            return torch.stack(outputs, dim=0)

        return self._execute_single_sequence(
            hidden_states[0],
            routing_signals[0],
        ).unsqueeze(0)

    def _execute_single_sequence(
        self,
        hidden_states: Tensor,
        routing_signals: Tensor,
    ) -> Tensor:
        """
        Execute FFN for a single sequence with CLS optimization.

        Args:
            hidden_states: [seq, hidden]
            routing_signals: [seq, num_experts]

        Returns:
            output: [seq, hidden]
        """
        seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Get expert activation patterns
        expert_sets = self.chunk_analyzer.get_active_experts(routing_signals)

        # Initialize output
        output = torch.zeros(seq_len, hidden_dim, device=device, dtype=dtype)

        # Process in chunks
        chunk_size = self.config.chunk_size
        expert_weights = self._get_expert_weights()

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk_hidden = hidden_states[start:end]
            chunk_expert_sets = expert_sets[start:end]

            # Analyze this chunk
            chunk_analysis = self.chunk_analyzer.analyze_chunk(
                chunk_expert_sets, start // chunk_size, start
            )

            if chunk_analysis.can_batch and expert_weights:
                # Use batched execution
                activation_fn = self._get_activation_fn()
                result = self.chunk_batcher.batch_experts_for_chunk(
                    chunk_hidden,
                    chunk_expert_sets,
                    expert_weights,
                    activation_fn,
                )
                output[start:end] = result.output

                # Update statistics
                if self._stats:
                    self._stats.chunks_processed += 1
                    self._stats.total_experts_reused += result.cache_hits
                    self._stats.total_experts_computed += len(result.experts_computed)
            else:
                # Fall back to original MLP for this chunk
                # We need to handle the batch dimension
                chunk_output = self.original_mlp(chunk_hidden.unsqueeze(0))
                output[start:end] = chunk_output.squeeze(0)

        if self._stats:
            self._stats.total_tokens += seq_len

        return output

    def _get_activation_fn(self) -> nn.Module:
        """Get activation function from original MLP."""
        mlp = self.original_mlp

        if hasattr(mlp, "act_fn"):
            return mlp.act_fn
        elif hasattr(mlp, "activation_fn"):
            return mlp.activation_fn
        else:
            # Default to SiLU (common in modern LLMs)
            return nn.SiLU()

    def forward_streaming(
        self,
        hidden_state: Tensor,
        routing_signal: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Streaming forward for autoregressive generation.

        Optimized for single-token processing with result reuse.

        Args:
            hidden_state: [hidden] or [1, hidden] single token input
            routing_signal: Optional pre-computed routing

        Returns:
            output: [hidden] or [1, hidden] single token output
        """
        # Handle shape
        squeeze_output = hidden_state.dim() == 1
        if squeeze_output:
            hidden_state = hidden_state.unsqueeze(0)

        # Compute routing if not provided
        if routing_signal is None:
            routing_signal = self._compute_routing(hidden_state.unsqueeze(0))
            if routing_signal is not None:
                routing_signal = routing_signal.squeeze(0)

        if routing_signal is None:
            # No routing available, use original
            output = self.original_mlp(hidden_state.unsqueeze(0))
            output = output.squeeze(0)
            if squeeze_output:
                output = output.squeeze(0)
            return output

        # Get active experts for this token
        expert_sets = self.chunk_analyzer.get_active_experts(routing_signal)
        if not expert_sets:
            active_experts = set()
        else:
            active_experts = expert_sets[0]

        # Use streaming batcher
        expert_weights = self._get_expert_weights()
        activation_fn = self._get_activation_fn()

        output = self.streaming_batcher.process_token(
            hidden_state.squeeze(0),
            active_experts,
            expert_weights,
            activation_fn,
        )

        if not squeeze_output:
            output = output.unsqueeze(0)

        return output

    def get_statistics(self) -> Dict[str, Any]:
        """Get collected statistics."""
        if self._stats is None:
            return {}

        return {
            "total_tokens": self._stats.total_tokens,
            "total_experts_computed": self._stats.total_experts_computed,
            "total_experts_reused": self._stats.total_experts_reused,
            "chunks_processed": self._stats.chunks_processed,
            "reuse_rate": self._stats.reuse_rate,
            "compute_savings": self._stats.compute_savings,
            "batcher_stats": self.chunk_batcher.stats,
            "streaming_reuse_rate": self.streaming_batcher.reuse_rate,
        }

    def reset_statistics(self) -> None:
        """Reset statistics."""
        if self._stats:
            self._stats = CLSStatistics()
        self.chunk_batcher.reset_stats()
        self.chunk_batcher.clear_cache()
        self.streaming_batcher.reset()


# ==========================================================================
# Model Patching Utilities
# ==========================================================================


def attach_cls_ffn(
    model: nn.Module,
    config: AdaptiveInferenceConfig,
    layer_indices: Optional[List[int]] = None,
) -> nn.Module:
    """
    Attach CLS-aware FFN wrappers to a BlockFFN model.

    Args:
        model: BlockFFN model
        config: Adaptive inference configuration
        layer_indices: Which layers to modify (None = use config)

    Returns:
        Modified model (mutated in place)
    """
    if not config.enable_cls_optimization:
        print("CLS optimization disabled in config")
        return model

    num_layers = len(model.model.layers)

    if layer_indices is None:
        layer_indices = config.get_target_layer_indices(num_layers)

    print(f"Attaching CLS-aware FFN to layers: {layer_indices}")

    for idx in layer_indices:
        if idx < 0 or idx >= num_layers:
            print(f"  Skipping invalid layer index: {idx}")
            continue

        layer = model.model.layers[idx]

        if not hasattr(layer, "mlp"):
            print(f"  Layer {idx}: No MLP found, skipping")
            continue

        original_mlp = layer.mlp

        # Check if already wrapped
        if isinstance(original_mlp, CLSAwareFFN):
            print(f"  Layer {idx}: Already wrapped, skipping")
            continue

        cls_ffn = CLSAwareFFN(
            original_mlp=original_mlp,
            layer_idx=idx,
            config=config,
        )

        layer.mlp = cls_ffn
        print(f"  Layer {idx}: Wrapped with CLSAwareFFN")

    return model


def detach_cls_ffn(model: nn.Module) -> nn.Module:
    """
    Remove CLS-aware FFN wrappers and restore original MLPs.

    Args:
        model: Model with CLS-aware FFN layers

    Returns:
        Model with original MLPs restored
    """
    for idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and isinstance(layer.mlp, CLSAwareFFN):
            layer.mlp = layer.mlp.original_mlp
            print(f"  Layer {idx}: Restored original MLP")

    return model


def get_cls_layers(
    model: nn.Module,
) -> List[Tuple[int, CLSAwareFFN]]:
    """
    Get all CLSAwareFFN layers in the model.

    Args:
        model: Model to inspect

    Returns:
        List of (layer_idx, cls_ffn) tuples
    """
    cls_layers = []
    for idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and isinstance(layer.mlp, CLSAwareFFN):
            cls_layers.append((idx, layer.mlp))
    return cls_layers


def get_all_cls_statistics(model: nn.Module) -> Dict[int, Dict[str, Any]]:
    """
    Get statistics from all CLS-aware FFN layers.

    Args:
        model: Model with CLS layers

    Returns:
        Dict mapping layer_idx to statistics dict
    """
    stats = {}
    for idx, cls_ffn in get_cls_layers(model):
        stats[idx] = cls_ffn.get_statistics()
    return stats


def reset_all_cls_statistics(model: nn.Module) -> None:
    """Reset statistics on all CLS-aware FFN layers."""
    for _, cls_ffn in get_cls_layers(model):
        cls_ffn.reset_statistics()
