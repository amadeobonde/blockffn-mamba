"""
Adaptive Inference Configuration

Centralized configuration for the BlockFFN adaptive inference framework.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import copy


@dataclass
class AdaptiveInferenceConfig:
    """Configuration for adaptive inference with BlockFFN.

    This configuration controls:
    1. Window mapping: How routing sparsity maps to attention window sizes
    2. Verification: Confidence thresholds for accepting reduced compute
    3. Escalation: When and how to fall back to full attention
    4. CLS settings: Chunk-level sparsity optimization parameters
    5. Device: Target device and optimization flags
    """

    # ==========================================================================
    # Window Mapping Configuration
    # ==========================================================================

    # Sparsity thresholds for window size buckets
    high_sparsity_threshold: float = 0.9   # >90% sparse → small window
    medium_sparsity_threshold: float = 0.7  # 70-90% sparse → medium window
    # <70% sparse → full attention

    # Attention window sizes (tokens)
    small_window: int = 16    # For high sparsity tokens
    medium_window: int = 64   # For medium sparsity tokens
    # full_window = seq_len   # For low sparsity tokens (computed dynamically)

    # Window mapping mode
    window_mode: str = "discrete"  # "discrete" or "continuous"
    window_temperature: float = 1.0  # For continuous mode sigmoid blending

    # ==========================================================================
    # Verification Thresholds
    # ==========================================================================

    # Confidence thresholds (higher = stricter)
    tau_green: float = 0.85   # High confidence - accept reduced compute
    tau_yellow: float = 0.65  # Moderate confidence - monitor for streak
    tau_red: float = 0.45     # Low confidence - immediate escalation

    # Streak detection
    yellow_streak_limit: int = 3  # Escalate after N consecutive yellow tokens

    # Confidence metric weights
    entropy_weight: float = 0.5
    margin_weight: float = 0.5
    stability_weight: float = 0.0  # Set >0 to enable stability checking

    # Top-k for margin computation
    top_k_margin: int = 5

    # ==========================================================================
    # Verification Scheduling Configuration
    # ==========================================================================

    # Strategy 1: Token-stride scheduling
    # Verify every k-th token (1 = every token, 4 = every 4th token)
    verification_stride: int = 1
    stride_default_accept: bool = True  # True = accept skipped, False = inherit previous

    # Strategy 2: Layer-gated scheduling
    # Only verify in the final N% of layers
    verification_layer_ratio: float = 1.0  # 1.0 = all layers, 0.25 = last 25%
    verification_layer_mode: str = "ratio"  # "ratio" | "count" | "all"
    verification_layer_count: int = 0  # If mode="count", verify in last N layers

    # Strategy 3: Cheap-gate scheduling (margin-first)
    # Compute margin first; only compute entropy if margin is uncertain
    enable_cheap_gate: bool = False
    cheap_gate_high_threshold: float = 0.95  # Margin > this = accept (skip entropy)
    cheap_gate_low_threshold: float = 0.40   # Margin < this = escalate (skip entropy)
    # Margin in between = compute full confidence (margin + entropy)

    # ==========================================================================
    # Escalation Configuration
    # ==========================================================================

    # Enable/disable escalation paths
    enable_layer_escalation: bool = True
    enable_token_escalation: bool = True
    enable_suffix_escalation: bool = True

    # Escalation budget (max fraction of tokens that can escalate)
    max_escalation_rate: float = 0.20  # 20% max

    # Graceful degradation
    cascade_threshold: int = 3  # Disable adaptive after N suffix escalations

    # Bounded suffix escalation
    # 0 = recompute to end, >0 = fixed length recomputation
    suffix_escalation_length: int = 0

    # ==========================================================================
    # CLS Settings (Phase 4)
    # ==========================================================================

    chunk_size: int = 8  # Tokens per chunk for CLS analysis
    overlap_threshold: float = 0.9  # Expert overlap for reuse
    expert_cache_mb: int = 512  # Expert weight cache size
    enable_cls_optimization: bool = False  # Disabled until Phase 4

    # ==========================================================================
    # Device Configuration
    # ==========================================================================

    device: str = "cpu"  # "cpu", "mps", "cuda"
    dtype: str = "float32"  # "float32", "float16", "bfloat16"

    # Attention backend
    use_flex_attention: bool = False  # True for CUDA + PyTorch 2.5+
    mask_strategy: str = "dense"  # "dense", "block_sparse", "grouped"

    # ==========================================================================
    # Target Layers
    # ==========================================================================

    # Which layers to apply adaptive attention to
    # "all" = all layers, "middle" = middle third, list = specific indices
    target_layers: Any = "middle"

    # ==========================================================================
    # Statistics & Debugging
    # ==========================================================================

    collect_statistics: bool = True
    verbose: bool = False

    def __post_init__(self):
        """Validate configuration."""
        assert 0.0 <= self.high_sparsity_threshold <= 1.0
        assert 0.0 <= self.medium_sparsity_threshold <= 1.0
        assert self.medium_sparsity_threshold < self.high_sparsity_threshold
        assert self.small_window > 0
        assert self.medium_window >= self.small_window
        assert 0.0 <= self.tau_green <= 1.0
        assert 0.0 <= self.tau_yellow <= self.tau_green
        assert 0.0 <= self.tau_red <= self.tau_yellow
        assert self.yellow_streak_limit > 0
        assert self.chunk_size > 0
        assert self.device in ("cpu", "mps", "cuda")
        assert self.window_mode in ("discrete", "continuous")
        assert self.mask_strategy in ("dense", "block_sparse", "grouped")
        # Verification scheduling validations
        assert self.verification_stride >= 1, "verification_stride must be >= 1"
        assert 0.0 <= self.verification_layer_ratio <= 1.0
        assert self.verification_layer_mode in ("ratio", "count", "all")
        assert 0.0 <= self.cheap_gate_low_threshold <= self.cheap_gate_high_threshold <= 1.0
        assert self.suffix_escalation_length >= 0, "suffix_escalation_length must be >= 0"

    def for_device(self, device: str) -> "AdaptiveInferenceConfig":
        """Return optimized config for specific device."""
        config = copy.copy(self)
        config.device = device

        if device == "cpu":
            # CPU: Prioritize memory efficiency, no FlexAttention
            config.use_flex_attention = False
            config.mask_strategy = "dense"
            config.dtype = "float32"
            config.expert_cache_mb = 256

        elif device == "mps":
            # MPS: Similar to CPU but can use float16
            config.use_flex_attention = False
            config.mask_strategy = "dense"
            config.dtype = "float16"
            config.expert_cache_mb = 512

        elif device == "cuda":
            # CUDA: Can use FlexAttention if available
            config.use_flex_attention = True
            config.mask_strategy = "block_sparse"
            config.dtype = "float16"
            config.expert_cache_mb = 1024

        return config

    def get_target_layer_indices(self, num_layers: int) -> list:
        """Get list of layer indices to apply adaptive attention."""
        if self.target_layers == "all":
            return list(range(num_layers))
        elif self.target_layers == "middle":
            start = num_layers // 3
            end = 2 * num_layers // 3
            return list(range(start, end))
        elif isinstance(self.target_layers, (list, tuple)):
            return [i for i in self.target_layers if 0 <= i < num_layers]
        else:
            raise ValueError(f"Invalid target_layers: {self.target_layers}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "high_sparsity_threshold": self.high_sparsity_threshold,
            "medium_sparsity_threshold": self.medium_sparsity_threshold,
            "small_window": self.small_window,
            "medium_window": self.medium_window,
            "window_mode": self.window_mode,
            "window_temperature": self.window_temperature,
            "tau_green": self.tau_green,
            "tau_yellow": self.tau_yellow,
            "tau_red": self.tau_red,
            "yellow_streak_limit": self.yellow_streak_limit,
            "entropy_weight": self.entropy_weight,
            "margin_weight": self.margin_weight,
            "stability_weight": self.stability_weight,
            "top_k_margin": self.top_k_margin,
            # Verification scheduling
            "verification_stride": self.verification_stride,
            "stride_default_accept": self.stride_default_accept,
            "verification_layer_ratio": self.verification_layer_ratio,
            "verification_layer_mode": self.verification_layer_mode,
            "verification_layer_count": self.verification_layer_count,
            "enable_cheap_gate": self.enable_cheap_gate,
            "cheap_gate_high_threshold": self.cheap_gate_high_threshold,
            "cheap_gate_low_threshold": self.cheap_gate_low_threshold,
            # Escalation
            "enable_layer_escalation": self.enable_layer_escalation,
            "enable_token_escalation": self.enable_token_escalation,
            "enable_suffix_escalation": self.enable_suffix_escalation,
            "max_escalation_rate": self.max_escalation_rate,
            "cascade_threshold": self.cascade_threshold,
            "suffix_escalation_length": self.suffix_escalation_length,
            "chunk_size": self.chunk_size,
            "overlap_threshold": self.overlap_threshold,
            "expert_cache_mb": self.expert_cache_mb,
            "enable_cls_optimization": self.enable_cls_optimization,
            "device": self.device,
            "dtype": self.dtype,
            "use_flex_attention": self.use_flex_attention,
            "mask_strategy": self.mask_strategy,
            "target_layers": self.target_layers,
            "collect_statistics": self.collect_statistics,
            "verbose": self.verbose,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AdaptiveInferenceConfig":
        """Create from dictionary."""
        return cls(**d)

    @classmethod
    def from_preset(cls, preset_name: str) -> "AdaptiveInferenceConfig":
        """Create from a named preset."""
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
        return copy.copy(PRESETS[preset_name])


# ==========================================================================
# Configuration Presets
# ==========================================================================

PRESETS: Dict[str, AdaptiveInferenceConfig] = {
    # Conservative: Minimal changes, focus on very high sparsity tokens
    # Use when quality is paramount
    "conservative": AdaptiveInferenceConfig(
        high_sparsity_threshold=0.95,
        medium_sparsity_threshold=0.85,
        small_window=32,
        medium_window=128,
        tau_green=0.90,
        tau_yellow=0.75,
        tau_red=0.50,
        yellow_streak_limit=2,
    ),

    # Balanced: Default settings, good quality/speed tradeoff
    "balanced": AdaptiveInferenceConfig(),  # Uses all defaults

    # Aggressive: Maximum savings, may affect quality slightly
    # Use when speed is paramount
    "aggressive": AdaptiveInferenceConfig(
        high_sparsity_threshold=0.80,
        medium_sparsity_threshold=0.60,
        small_window=8,
        medium_window=32,
        tau_green=0.75,
        tau_yellow=0.55,
        tau_red=0.35,
        yellow_streak_limit=5,
        max_escalation_rate=0.30,
    ),

    # CPU-optimized: Best settings for CPU inference
    "cpu": AdaptiveInferenceConfig(
        device="cpu",
        dtype="float32",
        use_flex_attention=False,
        mask_strategy="dense",
        expert_cache_mb=256,
        # Slightly more conservative to avoid overhead
        high_sparsity_threshold=0.88,
        medium_sparsity_threshold=0.70,
    ),

    # MPS-optimized: Best settings for Apple Silicon
    "mps": AdaptiveInferenceConfig(
        device="mps",
        dtype="float16",
        use_flex_attention=False,
        mask_strategy="dense",
        expert_cache_mb=512,
    ),

    # CUDA-optimized: Best settings for NVIDIA GPUs
    "cuda": AdaptiveInferenceConfig(
        device="cuda",
        dtype="float16",
        use_flex_attention=True,
        mask_strategy="block_sparse",
        expert_cache_mb=1024,
    ),

    # Debug: Maximum logging and statistics
    "debug": AdaptiveInferenceConfig(
        collect_statistics=True,
        verbose=True,
        # Conservative to see clear patterns
        high_sparsity_threshold=0.90,
        tau_green=0.85,
    ),

    # Fast: Verification scheduling for reduced overhead
    # Use when latency is critical but quality must be maintained
    "fast": AdaptiveInferenceConfig(
        # Aggressive attention settings
        high_sparsity_threshold=0.80,
        medium_sparsity_threshold=0.60,
        small_window=8,
        medium_window=32,
        # Relaxed thresholds
        tau_green=0.75,
        tau_yellow=0.55,
        # Verification scheduling
        verification_stride=4,  # Verify every 4th token
        stride_default_accept=True,
        enable_cheap_gate=True,
        cheap_gate_high_threshold=0.90,
        cheap_gate_low_threshold=0.35,
    ),

    # Turbo: Maximum speed with aggressive scheduling
    # Use for latency-critical applications with quality tradeoffs acceptable
    "turbo": AdaptiveInferenceConfig(
        # Very aggressive attention
        high_sparsity_threshold=0.75,
        medium_sparsity_threshold=0.55,
        small_window=4,
        medium_window=16,
        # Relaxed thresholds
        tau_green=0.70,
        tau_yellow=0.50,
        tau_red=0.30,
        yellow_streak_limit=6,
        max_escalation_rate=0.35,
        # Aggressive verification scheduling
        verification_stride=8,  # Verify every 8th token
        stride_default_accept=True,
        verification_layer_ratio=0.25,  # Only verify in last 25% of layers
        enable_cheap_gate=True,
        cheap_gate_high_threshold=0.85,
        cheap_gate_low_threshold=0.30,
    ),
}
