"""
Adaptive Attention Mask Generator

Generates variable-length causal attention masks where each query position
can attend to a different number of previous tokens based on its window size.

Mask formula: attend(q, kv) = (kv <= q) AND (q - kv < window_size[q])

Supports multiple strategies:
1. Dense: Full [B, 1, S, S] tensor (CPU/MPS compatible)
2. Block-sparse: FlexAttention BlockMask (CUDA, PyTorch 2.5+)
3. Grouped: Group tokens by window size for efficient batching
"""

from typing import Union, Dict, List, Optional, Any
import torch
import torch.nn as nn
from torch import Tensor


class AdaptiveMaskGenerator:
    """
    Generates attention masks with per-token variable window sizes.

    The key challenge is efficient batch processing when each token
    may have a different window size. This class provides multiple
    strategies to handle this efficiently.

    Attributes:
        strategy: Mask generation strategy ("dense", "block_sparse", "grouped")
        block_size: Block size for block_sparse strategy
        num_window_groups: Number of groups for grouped strategy
    """

    STRATEGY_DENSE = "dense"
    STRATEGY_BLOCK_SPARSE = "block_sparse"
    STRATEGY_GROUPED = "grouped"

    def __init__(
        self,
        strategy: str = "dense",
        block_size: int = 64,
        num_window_groups: int = 3,
    ):
        """
        Initialize the mask generator.

        Args:
            strategy: Generation strategy
            block_size: Block size for block_sparse (FlexAttention)
            num_window_groups: Number of groups for grouped execution
        """
        assert strategy in (
            self.STRATEGY_DENSE,
            self.STRATEGY_BLOCK_SPARSE,
            self.STRATEGY_GROUPED,
        )
        self.strategy = strategy
        self.block_size = block_size
        self.num_window_groups = num_window_groups

        # Check FlexAttention availability
        self._flex_available = self._check_flex_attention()

    def _check_flex_attention(self) -> bool:
        """Check if PyTorch FlexAttention is available."""
        try:
            from torch.nn.attention.flex_attention import flex_attention
            return True
        except ImportError:
            return False

    def generate_mask(
        self,
        window_sizes: Tensor,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.bool,
    ) -> Union[Tensor, Dict, Any]:
        """
        Generate attention mask based on per-token window sizes.

        Args:
            window_sizes: Per-token window sizes [batch, seq]
            seq_len: Sequence length
            device: Target device
            dtype: Mask data type (torch.bool or torch.float)

        Returns:
            Mask representation depends on strategy:
            - dense: Tensor [batch, 1, seq, seq]
            - block_sparse: FlexAttention BlockMask (or fallback to dense)
            - grouped: Dict with grouped token indices and masks
        """
        if self.strategy == self.STRATEGY_DENSE:
            return self._generate_dense_mask(window_sizes, seq_len, device, dtype)
        elif self.strategy == self.STRATEGY_BLOCK_SPARSE:
            return self._generate_block_sparse_mask(window_sizes, seq_len, device)
        elif self.strategy == self.STRATEGY_GROUPED:
            return self._generate_grouped_mask(window_sizes, seq_len, device)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _generate_dense_mask(
        self,
        window_sizes: Tensor,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        Generate dense [batch, 1, seq, seq] mask tensor.

        Each query position q can attend to positions in [q - window + 1, q].
        Combined with causal masking: only attend to positions <= q.

        Args:
            window_sizes: [batch, seq] window sizes per token
            seq_len: Sequence length
            device: Target device
            dtype: Output dtype

        Returns:
            mask: [batch, 1, seq, seq] attention mask
                  True = attend, False = don't attend (for bool)
                  0.0 = attend, -inf = don't attend (for float)
        """
        batch_size = window_sizes.shape[0]

        # Create position indices
        # q_pos: [1, seq, 1], kv_pos: [1, 1, seq]
        positions = torch.arange(seq_len, device=device)
        q_pos = positions.view(1, -1, 1)   # Query positions
        kv_pos = positions.view(1, 1, -1)  # Key/Value positions

        # Compute distance: q_pos - kv_pos
        # Positive distance means kv is before q (causal)
        distance = q_pos - kv_pos  # [1, seq, seq]

        # Expand window_sizes: [batch, seq] -> [batch, seq, 1]
        windows = window_sizes.unsqueeze(-1)  # [batch, seq, 1]

        # Causal mask: kv_pos <= q_pos (distance >= 0)
        causal_mask = distance >= 0  # [1, seq, seq]

        # Window mask: distance < window_size (kv is within window)
        window_mask = distance < windows  # [batch, seq, seq]

        # Combine: must satisfy both causal AND window constraints
        mask = causal_mask & window_mask  # [batch, seq, seq]

        # Add head dimension: [batch, seq, seq] -> [batch, 1, seq, seq]
        mask = mask.unsqueeze(1)

        # Convert to requested dtype
        if dtype == torch.bool:
            return mask
        else:
            # Convert to float mask: True->0.0, False->-inf
            float_mask = torch.zeros_like(mask, dtype=dtype)
            float_mask.masked_fill_(~mask, float("-inf"))
            return float_mask

    def _generate_block_sparse_mask(
        self,
        window_sizes: Tensor,
        seq_len: int,
        device: torch.device,
    ):
        """
        Generate FlexAttention-compatible BlockMask.

        Requires PyTorch 2.5+ with torch.nn.attention.flex_attention.
        Falls back to dense mask if FlexAttention is not available.

        Args:
            window_sizes: [batch, seq] window sizes
            seq_len: Sequence length
            device: Target device

        Returns:
            BlockMask for FlexAttention, or dense mask as fallback
        """
        if not self._flex_available:
            # Fallback to dense mask
            return self._generate_dense_mask(
                window_sizes, seq_len, device, torch.bool
            )

        try:
            from torch.nn.attention.flex_attention import create_block_mask

            batch_size = window_sizes.shape[0]

            # Store window_sizes for closure capture
            _window_sizes = window_sizes

            def adaptive_causal_mask(b, h, q_idx, kv_idx):
                """
                FlexAttention mask_mod for adaptive windowed causal attention.

                Returns True if position (q_idx, kv_idx) should be computed.
                """
                # Causal constraint: can only attend to earlier positions
                causal = q_idx >= kv_idx

                # Window constraint: must be within window
                window = _window_sizes[b, q_idx]
                in_window = (q_idx - kv_idx) < window

                return causal & in_window

            # Create BlockMask
            block_mask = create_block_mask(
                adaptive_causal_mask,
                B=batch_size,
                H=None,  # Broadcast over all heads
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                _compile=True,
            )
            return block_mask

        except Exception:
            # Fallback to dense mask on any error
            return self._generate_dense_mask(
                window_sizes, seq_len, device, torch.bool
            )

    def _generate_grouped_mask(
        self,
        window_sizes: Tensor,
        seq_len: int,
        device: torch.device,
    ) -> Dict:
        """
        Group tokens by similar window sizes for efficient batch processing.

        Instead of per-token masks, groups tokens into buckets and
        processes each group with uniform window size.

        Args:
            window_sizes: [batch, seq] window sizes
            seq_len: Sequence length
            device: Target device

        Returns:
            Dict with group information:
            {
                group_id: {
                    "indices": token indices in this group,
                    "window_size": uniform window for this group,
                    "mask": dense mask for this group,
                }
            }
        """
        window_min = window_sizes.min().item()
        window_max = window_sizes.max().item()

        if window_min == window_max:
            # All same window size - single group
            return {
                0: {
                    "indices": torch.arange(seq_len, device=device),
                    "window_size": int(window_min),
                    "mask": self._generate_dense_mask(
                        window_sizes, seq_len, device, torch.bool
                    ),
                }
            }

        # Compute bucket boundaries
        step = (window_max - window_min) / self.num_window_groups
        boundaries = [
            window_min + i * step
            for i in range(self.num_window_groups + 1)
        ]

        groups = {}
        for g in range(self.num_window_groups):
            low = boundaries[g]
            high = boundaries[g + 1] if g < self.num_window_groups - 1 else float("inf")

            # Find tokens in this bucket
            # Use first batch dimension for grouping
            mask = (window_sizes[0] >= low) & (window_sizes[0] < high)
            indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)

            if len(indices) > 0:
                # Use max window in group (conservative)
                group_window = int(window_sizes[0, indices].max().item())

                # Create uniform window tensor for this group
                uniform_windows = torch.full(
                    (1, len(indices)),
                    group_window,
                    device=device,
                    dtype=torch.long,
                )

                groups[g] = {
                    "indices": indices,
                    "window_size": group_window,
                    "count": len(indices),
                }

        return groups


def generate_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.bool,
) -> Tensor:
    """
    Generate a standard causal attention mask.

    Args:
        seq_len: Sequence length
        device: Target device
        dtype: Mask dtype

    Returns:
        mask: [1, 1, seq, seq] causal mask
    """
    mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=dtype, device=device)
    )
    return mask.unsqueeze(0).unsqueeze(0)


def generate_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.bool,
) -> Tensor:
    """
    Generate a uniform sliding window causal mask.

    All positions use the same window size.

    Args:
        seq_len: Sequence length
        window_size: Uniform window size
        device: Target device
        dtype: Mask dtype

    Returns:
        mask: [1, 1, seq, seq] sliding window causal mask
    """
    # Create position indices
    positions = torch.arange(seq_len, device=device)
    q_pos = positions.view(-1, 1)
    kv_pos = positions.view(1, -1)

    distance = q_pos - kv_pos

    # Causal and window constraints
    mask = (distance >= 0) & (distance < window_size)

    if dtype != torch.bool:
        float_mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
        float_mask.masked_fill_(~mask, float("-inf"))
        mask = float_mask

    return mask.unsqueeze(0).unsqueeze(0)
