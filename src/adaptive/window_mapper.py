"""
Sparsity to Window Size Mapper

Maps routing sparsity values to attention window sizes.
The core idea is that high-sparsity tokens (few active experts)
are "easy" and only need local context, while low-sparsity tokens
(many active experts) are "hard" and need full attention.

Window sizing policy:
- Very high sparsity (>0.9) → small local window (16 tokens)
- Medium sparsity (0.7-0.9) → medium window (64 tokens)
- Low sparsity (<0.7) → full attention (all previous tokens)
"""

from typing import Tuple, Optional
import torch
import torch.nn.functional as F
from torch import Tensor

from src.adaptive.config import AdaptiveInferenceConfig


class SparsityToWindowMapper:
    """
    Maps routing sparsity values to attention window sizes.

    Supports two modes:
    1. Discrete: Bucket sparsity into fixed window sizes
    2. Continuous: Smooth interpolation between window sizes

    Attributes:
        high_threshold: Sparsity threshold for small window
        medium_threshold: Sparsity threshold for medium window
        small_window: Window size for high sparsity tokens
        medium_window: Window size for medium sparsity tokens
        mode: "discrete" or "continuous"
        temperature: Temperature for continuous mode sigmoid
    """

    def __init__(
        self,
        high_sparsity_threshold: float = 0.9,
        medium_sparsity_threshold: float = 0.7,
        small_window: int = 16,
        medium_window: int = 64,
        mode: str = "discrete",
        temperature: float = 1.0,
    ):
        """
        Initialize the window mapper.

        Args:
            high_sparsity_threshold: Threshold for small window (default 0.9)
            medium_sparsity_threshold: Threshold for medium window (default 0.7)
            small_window: Window size for high sparsity (default 16)
            medium_window: Window size for medium sparsity (default 64)
            mode: "discrete" or "continuous"
            temperature: Sigmoid temperature for continuous mode
        """
        assert 0.0 <= medium_sparsity_threshold < high_sparsity_threshold <= 1.0
        assert 0 < small_window <= medium_window

        self.high_threshold = high_sparsity_threshold
        self.medium_threshold = medium_sparsity_threshold
        self.small_window = small_window
        self.medium_window = medium_window
        self.mode = mode
        self.temperature = temperature

    @classmethod
    def from_config(cls, config: AdaptiveInferenceConfig) -> "SparsityToWindowMapper":
        """Create mapper from config."""
        return cls(
            high_sparsity_threshold=config.high_sparsity_threshold,
            medium_sparsity_threshold=config.medium_sparsity_threshold,
            small_window=config.small_window,
            medium_window=config.medium_window,
            mode=config.window_mode,
            temperature=config.window_temperature,
        )

    def __call__(
        self,
        sparsity: Tensor,
        seq_len: int,
    ) -> Tensor:
        """
        Map sparsity values to window sizes.

        Args:
            sparsity: Per-token sparsity values [batch, seq] in range [0, 1]
            seq_len: Sequence length (used for full window size)

        Returns:
            window_sizes: Per-token attention window sizes [batch, seq]
        """
        if self.mode == "discrete":
            return self._discrete_mapping(sparsity, seq_len)
        elif self.mode == "continuous":
            return self._continuous_mapping(sparsity, seq_len)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _discrete_mapping(
        self,
        sparsity: Tensor,
        seq_len: int,
    ) -> Tensor:
        """
        Bucket sparsity into discrete window sizes.

        Policy:
        - sparsity >= high_threshold: small_window
        - sparsity >= medium_threshold: medium_window
        - sparsity < medium_threshold: full attention (seq_len)
        """
        # Initialize with full attention
        window_sizes = torch.full_like(sparsity, seq_len, dtype=torch.long)

        # Medium sparsity → medium window
        medium_mask = sparsity >= self.medium_threshold
        window_sizes = torch.where(
            medium_mask,
            torch.tensor(self.medium_window, device=sparsity.device),
            window_sizes,
        )

        # High sparsity → small window
        high_mask = sparsity >= self.high_threshold
        window_sizes = torch.where(
            high_mask,
            torch.tensor(self.small_window, device=sparsity.device),
            window_sizes,
        )

        return window_sizes

    def _continuous_mapping(
        self,
        sparsity: Tensor,
        seq_len: int,
    ) -> Tensor:
        """
        Continuous interpolation between window sizes.

        Uses sigmoid blending for smooth transitions:
        - Low sparsity → full attention
        - High sparsity → small window
        - Intermediate values blend smoothly
        """
        # Compute blend weight using scaled sigmoid
        # Higher sparsity → higher weight → smaller window
        blend = torch.sigmoid(
            (sparsity - self.medium_threshold) * self.temperature * 10
        )

        # Interpolate between full attention and small window
        window_sizes = (
            (1 - blend) * seq_len +
            blend * self.small_window
        )

        # Ensure minimum window size
        window_sizes = torch.clamp(window_sizes, min=self.small_window)

        return window_sizes.long()

    def get_window_distribution(
        self,
        sparsity: Tensor,
        seq_len: int,
    ) -> dict:
        """
        Get statistics about window size distribution.

        Useful for monitoring and debugging.

        Args:
            sparsity: Per-token sparsity values [batch, seq]
            seq_len: Sequence length

        Returns:
            Dict with window distribution statistics
        """
        window_sizes = self(sparsity, seq_len).float()

        # Count tokens in each category
        small_count = (window_sizes <= self.small_window).sum().item()
        medium_count = (
            (window_sizes > self.small_window) &
            (window_sizes <= self.medium_window)
        ).sum().item()
        full_count = (window_sizes > self.medium_window).sum().item()
        total = window_sizes.numel()

        return {
            "small_window_pct": small_count / total if total > 0 else 0,
            "medium_window_pct": medium_count / total if total > 0 else 0,
            "full_window_pct": full_count / total if total > 0 else 0,
            "mean_window": window_sizes.mean().item(),
            "min_window": window_sizes.min().item(),
            "max_window": window_sizes.max().item(),
            "mean_sparsity": sparsity.mean().item(),
        }


def compute_effective_attention_cost(
    window_sizes: Tensor,
    seq_len: int,
) -> float:
    """
    Compute the effective attention cost ratio.

    Compares the sum of window sizes to full attention cost.

    Args:
        window_sizes: Per-token window sizes [batch, seq]
        seq_len: Full sequence length

    Returns:
        Cost ratio (1.0 = full attention, lower = savings)
    """
    # Full attention cost: sum of 1..seq_len = seq_len * (seq_len + 1) / 2
    full_cost = seq_len * (seq_len + 1) / 2

    # Adaptive cost: sum of actual window sizes
    # Each token at position i attends to min(window_size[i], i+1) tokens
    positions = torch.arange(seq_len, device=window_sizes.device).unsqueeze(0)
    actual_windows = torch.minimum(window_sizes, positions + 1)
    adaptive_cost = actual_windows.sum().item()

    return adaptive_cost / full_cost if full_cost > 0 else 1.0


def estimate_speedup(
    sparsity: Tensor,
    seq_len: int,
    mapper: SparsityToWindowMapper,
) -> dict:
    """
    Estimate speedup from adaptive attention.

    Args:
        sparsity: Per-token sparsity [batch, seq]
        seq_len: Sequence length
        mapper: Window mapper to use

    Returns:
        Dict with speedup estimates
    """
    window_sizes = mapper(sparsity, seq_len)
    cost_ratio = compute_effective_attention_cost(window_sizes, seq_len)

    return {
        "cost_ratio": cost_ratio,
        "theoretical_speedup": 1.0 / cost_ratio if cost_ratio > 0 else 1.0,
        "attention_flops_saved_pct": (1.0 - cost_ratio) * 100,
    }
