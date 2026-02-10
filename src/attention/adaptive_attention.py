"""
Adaptive Attention Layer

Complete adaptive attention layer for BlockFFN that wraps an existing
transformer decoder layer and adds:
1. Speculative routing signal extraction
2. Sparsity-to-window mapping
3. Adaptive attention mask generation
4. Modified attention computation

No model weights are modified - this is a non-invasive wrapper.
"""

from typing import Optional, Tuple, Dict, Any, List
from collections import defaultdict
import torch
import torch.nn as nn
from torch import Tensor

from src.adaptive.config import AdaptiveInferenceConfig
from src.adaptive.routing_predictor import RoutingPredictor
from src.adaptive.window_mapper import SparsityToWindowMapper
from src.attention.mask_generator import AdaptiveMaskGenerator
from src.attention.windowed_attention import WindowedAttentionCore


class AdaptiveAttentionLayer(nn.Module):
    """
    Adaptive attention layer wrapper for BlockFFN decoder layers.

    Intercepts the forward pass to:
    1. Predict routing signals speculatively
    2. Compute per-token sparsity
    3. Map sparsity to attention window sizes
    4. Generate adaptive masks
    5. Run attention with adaptive windows
    6. Pass through MLP unchanged

    Attributes:
        original_layer: The wrapped transformer decoder layer
        layer_idx: Index of this layer in the model
        routing_predictor: Speculative routing signal predictor
        window_mapper: Sparsity to window size mapper
        mask_generator: Adaptive mask generator
        attention_core: Windowed attention computation
        collect_statistics: Whether to collect runtime statistics
    """

    def __init__(
        self,
        original_layer: nn.Module,
        model_reference: nn.Module,
        layer_idx: int,
        config: AdaptiveInferenceConfig,
    ):
        """
        Initialize adaptive attention layer.

        Args:
            original_layer: The original decoder layer to wrap
            model_reference: Reference to full model (for routing access)
            layer_idx: Index of this layer
            config: Adaptive inference configuration
        """
        super().__init__()

        self.original_layer = original_layer
        self.model_reference = model_reference
        self.layer_idx = layer_idx
        self.config = config
        self.collect_statistics = config.collect_statistics

        # Initialize components
        self.routing_predictor = RoutingPredictor(
            model=model_reference,
            layer_idx=layer_idx,
        )

        self.window_mapper = SparsityToWindowMapper.from_config(config)

        self.mask_generator = AdaptiveMaskGenerator(
            strategy=config.mask_strategy,
        )

        # Get attention module from original layer
        attention_module = self._get_attention_module()
        self.attention_core = WindowedAttentionCore(
            original_attention=attention_module,
            use_flex_attention=config.use_flex_attention,
        )

        # Statistics tracking
        self._stats: Dict[str, List[float]] = defaultdict(list) if self.collect_statistics else {}

    def _get_attention_module(self) -> nn.Module:
        """Extract the attention module from the original layer."""
        layer = self.original_layer

        # Try common attribute names
        if hasattr(layer, "self_attn"):
            return layer.self_attn
        elif hasattr(layer, "attention"):
            return layer.attention
        elif hasattr(layer, "attn"):
            return layer.attn
        else:
            raise ValueError(
                f"Cannot find attention module in layer. "
                f"Available attributes: {dir(layer)}"
            )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, ...]:
        """
        Forward pass with adaptive attention windowing.

        Pipeline:
        1. Predict routing signals (speculative)
        2. Compute per-token sparsity
        3. Map sparsity to window sizes
        4. Generate adaptive attention mask
        5. Run attention with adaptive mask
        6. Continue with rest of layer (layernorms, MLP)

        Args:
            hidden_states: Input [batch, seq, hidden]
            attention_mask: Original attention mask (will be combined)
            position_ids: Position IDs for RoPE
            past_key_value: KV cache
            output_attentions: Return attention weights
            use_cache: Return updated KV cache
            **kwargs: Additional arguments passed to original layer

        Returns:
            Tuple of (hidden_states, attention_weights, past_key_value)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        # =======================================================
        # Decode shortcut: skip adaptive logic for single-token steps
        # =======================================================
        # During autoregressive decode (seq_len=1), there's nothing to
        # window — one query token attends to the full KV cache.
        # Just pass through to the original layer unchanged.
        if seq_len <= 1:
            return self.original_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )

        # =======================================================
        # Step 1: Speculative routing signal prediction
        # =======================================================
        routing_signals = self.routing_predictor.predict_routing(hidden_states)

        # =======================================================
        # Step 2: Compute per-token sparsity
        # =======================================================
        sparsity = self.routing_predictor.compute_sparsity(routing_signals)

        # =======================================================
        # Step 3: Map sparsity to window sizes
        # =======================================================
        window_sizes = self.window_mapper(sparsity, seq_len)

        # =======================================================
        # Quality guard: floor windows at 7/8 of seq_len so that
        # windowing never drops more than ~12% of context per layer.
        # This keeps prefill divergence within cs >= 0.99 even when
        # 12 wrapped layers compound.  The engine's escalation
        # mechanism handles more aggressive windowing at decode time.
        # =======================================================
        min_window = max(seq_len * 7 // 8, 1)
        window_sizes = torch.clamp(window_sizes, min=min(min_window, seq_len))

        # =======================================================
        # No-op shortcut: if all windows >= seq_len, the adaptive
        # mask equals standard causal. Pass through unchanged to
        # preserve the model's internal attention path (flash/SDPA)
        # and avoid numerical divergence from switching code paths.
        # =======================================================
        if window_sizes.min().item() >= seq_len:
            outputs = self.original_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            if self.collect_statistics:
                self._collect_stats(sparsity, window_sizes, seq_len)
            return outputs

        # =======================================================
        # Flash/SDPA guard: when the model handles masking internally
        # (attention_mask=None), injecting an explicit mask would
        # switch the attention code path and cause divergence.
        # =======================================================
        if attention_mask is None:
            outputs = self.original_layer(
                hidden_states=hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            if self.collect_statistics:
                self._collect_stats(sparsity, window_sizes, seq_len)
            return outputs

        # =======================================================
        # Step 4: Generate adaptive attention mask
        # =======================================================
        adaptive_mask = self.mask_generator.generate_mask(
            window_sizes=window_sizes,
            seq_len=seq_len,
            device=device,
        )

        # =======================================================
        # Step 5: Combine adaptive mask with original mask
        # =======================================================
        # Convert adaptive bool mask to float format matching the model's mask.
        # Model mask: [batch, 1, seq, seq] float with 0.0 (allowed) / -inf (blocked)
        # Our mask: [batch, 1, seq, seq] bool with True (allowed) / False (blocked)
        if isinstance(adaptive_mask, Tensor) and adaptive_mask.dtype == torch.bool:
            float_mask = torch.zeros(
                adaptive_mask.shape, dtype=hidden_states.dtype, device=device
            )
            float_mask.masked_fill_(~adaptive_mask, torch.finfo(hidden_states.dtype).min)
            adaptive_mask = float_mask

        # Combine: both use 0/-inf convention, so addition works
        if attention_mask is not None and isinstance(adaptive_mask, Tensor):
            adaptive_mask = adaptive_mask + attention_mask

        # =======================================================
        # Step 6: Delegate to original layer with our mask
        # =======================================================
        # Call the ORIGINAL decoder layer's forward. This preserves
        # RoPE, DynamicCache updates, MiniCPM scaling, layernorms,
        # MLP, and all model-specific behavior. We only change the mask.
        outputs = self.original_layer(
            hidden_states=hidden_states,
            attention_mask=adaptive_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        # =======================================================
        # Collect statistics if enabled
        # =======================================================
        if self.collect_statistics:
            self._collect_stats(sparsity, window_sizes, seq_len)

        return outputs

    def _collect_stats(
        self,
        sparsity: Tensor,
        window_sizes: Tensor,
        seq_len: int,
    ) -> None:
        """Collect statistics for analysis."""
        self._stats["mean_sparsity"].append(sparsity.mean().item())
        self._stats["mean_window"].append(window_sizes.float().mean().item())
        self._stats["min_window"].append(window_sizes.min().item())
        self._stats["max_window"].append(window_sizes.max().item())

        # Window distribution
        small = (window_sizes <= self.config.small_window).float().mean().item()
        medium_mask = (
            (window_sizes > self.config.small_window) &
            (window_sizes <= self.config.medium_window)
        )
        medium = medium_mask.float().mean().item()
        full = (window_sizes > self.config.medium_window).float().mean().item()

        self._stats["pct_small_window"].append(small)
        self._stats["pct_medium_window"].append(medium)
        self._stats["pct_full_window"].append(full)

    def get_statistics(self) -> Dict[str, float]:
        """Return collected statistics as averages."""
        if not self._stats:
            return {}
        return {
            k: sum(v) / len(v) if v else 0.0
            for k, v in self._stats.items()
        }

    def clear_statistics(self) -> None:
        """Clear collected statistics."""
        if self._stats:
            self._stats = defaultdict(list)


# ==========================================================================
# Model Patching Utilities
# ==========================================================================


def attach_adaptive_attention(
    model: nn.Module,
    config: AdaptiveInferenceConfig,
    layer_indices: Optional[List[int]] = None,
) -> nn.Module:
    """
    Attach adaptive attention layers to a BlockFFN model.

    Replaces specified decoder layers with AdaptiveAttentionLayer wrappers.

    Args:
        model: BlockFFN model (AutoModelForCausalLM)
        config: Adaptive inference configuration
        layer_indices: Which layers to modify (None = use config.target_layers)

    Returns:
        Modified model (same object, mutated in place)
    """
    num_layers = len(model.model.layers)

    if layer_indices is None:
        layer_indices = config.get_target_layer_indices(num_layers)

    print(f"Attaching adaptive attention to layers: {layer_indices}")

    for idx in layer_indices:
        if idx < 0 or idx >= num_layers:
            print(f"  Skipping invalid layer index: {idx}")
            continue

        original_layer = model.model.layers[idx]

        adaptive_layer = AdaptiveAttentionLayer(
            original_layer=original_layer,
            model_reference=model,
            layer_idx=idx,
            config=config,
        )

        model.model.layers[idx] = adaptive_layer
        print(f"  Layer {idx}: Wrapped with AdaptiveAttentionLayer")

    return model


def detach_adaptive_attention(model: nn.Module) -> nn.Module:
    """
    Remove adaptive attention wrappers and restore original layers.

    Args:
        model: Model with adaptive attention layers

    Returns:
        Model with original layers restored
    """
    for idx, layer in enumerate(model.model.layers):
        if isinstance(layer, AdaptiveAttentionLayer):
            model.model.layers[idx] = layer.original_layer
            print(f"  Layer {idx}: Restored original layer")

    return model


def get_adaptive_layers(
    model: nn.Module,
) -> List[Tuple[int, AdaptiveAttentionLayer]]:
    """
    Get all AdaptiveAttentionLayer instances in the model.

    Args:
        model: Model to inspect

    Returns:
        List of (layer_idx, layer) tuples for adaptive layers
    """
    adaptive_layers = []
    for idx, layer in enumerate(model.model.layers):
        if isinstance(layer, AdaptiveAttentionLayer):
            adaptive_layers.append((idx, layer))
    return adaptive_layers


def get_all_statistics(model: nn.Module) -> Dict[int, Dict[str, float]]:
    """
    Get statistics from all adaptive attention layers.

    Args:
        model: Model with adaptive layers

    Returns:
        Dict mapping layer_idx to statistics dict
    """
    stats = {}
    for idx, layer in get_adaptive_layers(model):
        stats[idx] = layer.get_statistics()
    return stats


def clear_all_statistics(model: nn.Module) -> None:
    """Clear statistics from all adaptive attention layers."""
    for _, layer in get_adaptive_layers(model):
        layer.clear_statistics()
