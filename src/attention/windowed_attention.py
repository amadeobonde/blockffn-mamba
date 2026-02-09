"""
Windowed Attention Core

Core attention computation with support for variable-length windows.
Integrates with existing BlockFFN attention while adding adaptive windowing.

Provides two backends:
1. Standard PyTorch attention with dense masks (CPU/MPS compatible)
2. FlexAttention for efficient sparse computation (CUDA, PyTorch 2.5+)
"""

from typing import Optional, Tuple, Union, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class WindowedAttentionCore(nn.Module):
    """
    Attention computation with variable-length window support.

    Wraps existing attention modules and adds adaptive window masking.
    Can operate in two modes:
    1. Wrapper mode: Uses original attention module's projections
    2. Standalone mode: Has its own Q, K, V, O projections

    Attributes:
        original_attention: Reference to original attention module (optional)
        hidden_size: Model hidden size
        num_heads: Number of Q attention heads
        num_kv_heads: Number of KV heads (for GQA; equals num_heads for MHA)
        head_dim: Dimension per head
        use_flex_attention: Whether to use FlexAttention backend
    """

    def __init__(
        self,
        original_attention: Optional[nn.Module] = None,
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        use_flex_attention: bool = False,
    ):
        """
        Initialize windowed attention.

        Args:
            original_attention: Original attention module to wrap
            hidden_size: Hidden size (required if no original_attention)
            num_heads: Number of Q heads (required if no original_attention)
            num_kv_heads: Number of KV heads for GQA (defaults to num_heads)
            use_flex_attention: Use FlexAttention if available
        """
        super().__init__()

        self.original_attention = original_attention
        self.use_flex_attention = use_flex_attention

        # Extract dimensions from original attention or use provided
        if original_attention is not None:
            self._extract_dimensions()
        else:
            assert hidden_size is not None and num_heads is not None
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads or num_heads
            self.head_dim = hidden_size // num_heads

        # Number of times to repeat KV heads to match Q heads (GQA)
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # Check FlexAttention availability
        self._flex_available = self._check_flex_attention()

    def _extract_dimensions(self) -> None:
        """Extract dimensions from original attention module (supports GQA)."""
        attn = self.original_attention

        # Try common attribute names for hidden size
        if hasattr(attn, "hidden_size"):
            self.hidden_size = attn.hidden_size
        elif hasattr(attn, "embed_dim"):
            self.hidden_size = attn.embed_dim
        elif hasattr(attn, "q_proj"):
            self.hidden_size = attn.q_proj.in_features
        else:
            raise ValueError("Cannot determine hidden_size from attention module")

        # Number of Q heads
        if hasattr(attn, "num_heads"):
            self.num_heads = attn.num_heads
        elif hasattr(attn, "num_attention_heads"):
            self.num_heads = attn.num_attention_heads
        else:
            raise ValueError("Cannot determine num_heads from attention module")

        # Number of KV heads (for Grouped Query Attention)
        # If not present, assume MHA (num_kv_heads == num_heads)
        if hasattr(attn, "num_key_value_heads"):
            self.num_kv_heads = attn.num_key_value_heads
        elif hasattr(attn, "num_kv_heads"):
            self.num_kv_heads = attn.num_kv_heads
        elif hasattr(attn, "k_proj"):
            # Infer from k_proj output features: out_features = num_kv_heads * head_dim
            head_dim = self.hidden_size // self.num_heads
            self.num_kv_heads = attn.k_proj.out_features // head_dim
        else:
            self.num_kv_heads = self.num_heads

        self.head_dim = self.hidden_size // self.num_heads

    def _check_flex_attention(self) -> bool:
        """Check if FlexAttention is available."""
        if not self.use_flex_attention:
            return False
        try:
            from torch.nn.attention.flex_attention import flex_attention
            return True
        except ImportError:
            return False

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Union[Tensor, Any],
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, ...]:
        """
        Forward pass with adaptive attention masking.

        Args:
            hidden_states: Input [batch, seq, hidden]
            attention_mask: Adaptive mask from MaskGenerator
            position_ids: Position IDs for RoPE (optional)
            past_key_value: Cached K, V for incremental decoding
            output_attentions: Return attention weights
            use_cache: Return updated KV cache

        Returns:
            Tuple of (output, attention_weights, past_key_value)
            Components are None if not requested
        """
        if self._flex_available and self._is_block_mask(attention_mask):
            return self._flex_attention_forward(
                hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache, **kwargs
            )
        else:
            return self._standard_attention_forward(
                hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache, **kwargs
            )

    def _is_block_mask(self, mask: Any) -> bool:
        """Check if mask is a FlexAttention BlockMask."""
        try:
            from torch.nn.attention.flex_attention import BlockMask
            return isinstance(mask, BlockMask)
        except ImportError:
            return False

    def _standard_attention_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_ids: Optional[Tensor],
        past_key_value: Optional[Tuple[Tensor, Tensor]],
        output_attentions: bool,
        use_cache: bool,
        **kwargs,
    ) -> Tuple[Tensor, ...]:
        """
        Standard attention with dense mask.

        Uses the original attention module's projections if available,
        otherwise performs manual computation.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Get Q, K, V from original attention or compute manually
        if self.original_attention is not None:
            query_states = self.original_attention.q_proj(hidden_states)
            key_states = self.original_attention.k_proj(hidden_states)
            value_states = self.original_attention.v_proj(hidden_states)
        else:
            raise NotImplementedError(
                "Standalone mode requires Q, K, V projections"
            )

        # Reshape for multi-head attention (supports GQA: num_kv_heads <= num_heads)
        # Q: [batch, seq, hidden] -> [batch, num_heads, seq, head_dim]
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # K/V: [batch, seq, num_kv_heads * head_dim] -> [batch, num_kv_heads, seq, head_dim]
        key_states = key_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # Handle KV cache
        # In transformers 4.x+, past_key_value is typically a DynamicCache object.
        # We MUST call cache.update() for our layer so all layers have entries —
        # otherwise non-wrapped layers after us crash with IndexError.
        if past_key_value is not None:
            if hasattr(past_key_value, 'update'):
                # DynamicCache: update it with our K/V so the cache stays in sync.
                # update() both stores and returns the concatenated K/V.
                layer_idx = getattr(self.original_attention, 'layer_idx', None)
                if layer_idx is not None:
                    cache_kwargs = {"cache_position": kwargs.get("cache_position")}
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, layer_idx, cache_kwargs
                    )
            elif isinstance(past_key_value, (list, tuple)) and len(past_key_value) >= 2:
                past_key, past_value = past_key_value[0], past_key_value[1]
                key_states = torch.cat([past_key, key_states], dim=2)
                value_states = torch.cat([past_value, value_states], dim=2)

        # Return the cache object itself (DynamicCache) or a tuple
        if use_cache:
            if hasattr(past_key_value, 'update'):
                new_past_key_value = past_key_value  # DynamicCache manages itself
            else:
                new_past_key_value = (key_states, value_states)
        else:
            new_past_key_value = None

        # Expand KV heads to match Q heads for GQA
        if self.num_kv_groups > 1:
            key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)

        # Compute attention scores
        # [batch, heads, seq_q, head_dim] @ [batch, heads, head_dim, seq_kv]
        # -> [batch, heads, seq_q, seq_kv]
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale

        # Apply attention mask
        if attention_mask is not None:
            # Ensure mask is correct shape
            if attention_mask.dtype == torch.bool:
                # Convert bool to float: True->0, False->-inf
                attn_mask_float = torch.zeros_like(
                    attn_weights, dtype=attn_weights.dtype
                )
                # Expand mask from [batch, 1, seq, seq] to [batch, heads, seq, seq]
                expanded_mask = attention_mask.expand(-1, self.num_heads, -1, -1)
                attn_mask_float.masked_fill_(~expanded_mask, float("-inf"))
                attn_weights = attn_weights + attn_mask_float
            else:
                # Already float mask with -inf for blocked positions
                attn_weights = attn_weights + attention_mask

        # Softmax and apply to values
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Optional: dropout (not included for inference)

        # Compute output
        # [batch, heads, seq_q, seq_kv] @ [batch, heads, seq_kv, head_dim]
        # -> [batch, heads, seq_q, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back
        # [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, hidden_size)

        # Output projection
        if self.original_attention is not None:
            attn_output = self.original_attention.o_proj(attn_output)

        # Prepare outputs
        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        else:
            outputs += (None,)
        if use_cache:
            outputs += (new_past_key_value,)
        else:
            outputs += (None,)

        return outputs

    def _flex_attention_forward(
        self,
        hidden_states: Tensor,
        block_mask: Any,
        position_ids: Optional[Tensor],
        past_key_value: Optional[Tuple[Tensor, Tensor]],
        output_attentions: bool,
        use_cache: bool,
        **kwargs,
    ) -> Tuple[Tensor, ...]:
        """
        FlexAttention forward pass with block-sparse mask.

        Uses PyTorch 2.5+ FlexAttention for efficient sparse computation.
        """
        from torch.nn.attention.flex_attention import flex_attention

        batch_size, seq_len, hidden_size = hidden_states.shape

        # Get Q, K, V
        if self.original_attention is not None:
            query_states = self.original_attention.q_proj(hidden_states)
            key_states = self.original_attention.k_proj(hidden_states)
            value_states = self.original_attention.v_proj(hidden_states)
        else:
            raise NotImplementedError("Standalone mode not supported for FlexAttention")

        # Reshape for multi-head (supports GQA)
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # Handle KV cache (same DynamicCache logic as standard path)
        if past_key_value is not None:
            if hasattr(past_key_value, 'update'):
                layer_idx = getattr(self.original_attention, 'layer_idx', None)
                if layer_idx is not None:
                    cache_kwargs = {"cache_position": kwargs.get("cache_position")}
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, layer_idx, cache_kwargs
                    )
            elif isinstance(past_key_value, (list, tuple)) and len(past_key_value) >= 2:
                past_key, past_value = past_key_value[0], past_key_value[1]
                key_states = torch.cat([past_key, key_states], dim=2)
                value_states = torch.cat([past_value, value_states], dim=2)

        if use_cache:
            if hasattr(past_key_value, 'update'):
                new_past_key_value = past_key_value
            else:
                new_past_key_value = (key_states, value_states)
        else:
            new_past_key_value = None

        # Expand KV heads to match Q heads for GQA
        if self.num_kv_groups > 1:
            key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)

        # FlexAttention
        attn_output = flex_attention(
            query_states,
            key_states,
            value_states,
            block_mask=block_mask,
        )

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, hidden_size)

        # Output projection
        if self.original_attention is not None:
            attn_output = self.original_attention.o_proj(attn_output)

        # Note: FlexAttention doesn't return attention weights
        outputs = (attn_output, None, new_past_key_value)

        return outputs


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
) -> Tensor:
    """
    Scaled dot-product attention with optional mask.

    This is a simple implementation for reference/fallback.

    Args:
        query: [batch, heads, seq_q, head_dim]
        key: [batch, heads, seq_kv, head_dim]
        value: [batch, heads, seq_kv, head_dim]
        attn_mask: [batch, heads, seq_q, seq_kv] or broadcastable
        dropout_p: Dropout probability
        scale: Attention scale (default: 1/sqrt(head_dim))

    Returns:
        output: [batch, heads, seq_q, head_dim]
    """
    head_dim = query.size(-1)
    scale = scale or (1.0 / math.sqrt(head_dim))

    # Compute attention scores
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Apply mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weights.masked_fill_(~attn_mask, float("-inf"))
        else:
            attn_weights = attn_weights + attn_mask

    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1)

    # Dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Apply to values
    return torch.matmul(attn_weights, value)
