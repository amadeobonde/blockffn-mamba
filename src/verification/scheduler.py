"""
Verification Scheduling

Reduces verification overhead by running confidence checks less frequently.

Strategies:
1. Token-stride: Only verify every k-th token
2. Layer-gate: Only verify in final N layers
3. Cheap-gate: Use margin-first gating before expensive entropy computation
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Set, List
import time
import torch
from torch import Tensor

from src.adaptive.config import AdaptiveInferenceConfig
from src.verification.confidence import compute_margin, compute_entropy


@dataclass
class VerificationResult:
    """Result from scheduled verification."""
    should_verify: bool           # Whether verification was performed
    confidence: Optional[float]   # Confidence value if computed
    used_cheap_gate: bool         # True if only margin was computed
    skipped_reason: str           # "token_stride" | "layer_gate" | "cheap_gate" | "none"
    gate_result: str              # "accept" | "reject" | "uncertain" | "full" | "skipped"


@dataclass
class SchedulingStats:
    """Statistics for verification scheduling."""
    total_tokens: int = 0
    verified_tokens: int = 0
    skipped_by_stride: int = 0
    skipped_by_layer: int = 0

    # Cheap gate breakdown
    cheap_gate_checks: int = 0
    cheap_gate_fast_accepts: int = 0
    cheap_gate_fast_rejects: int = 0
    cheap_gate_full_compute: int = 0

    # Timing instrumentation (milliseconds)
    total_verification_time_ms: float = 0.0
    total_margin_time_ms: float = 0.0
    total_entropy_time_ms: float = 0.0

    @property
    def verification_rate(self) -> float:
        """Fraction of tokens that were verified."""
        return self.verified_tokens / self.total_tokens if self.total_tokens > 0 else 0.0

    @property
    def entropy_skip_rate(self) -> float:
        """Fraction of cheap gate checks that skipped entropy."""
        if self.cheap_gate_checks == 0:
            return 0.0
        return (self.cheap_gate_fast_accepts + self.cheap_gate_fast_rejects) / self.cheap_gate_checks

    @property
    def avg_verification_time_ms(self) -> float:
        """Average time per verified token in milliseconds."""
        if self.verified_tokens == 0:
            return 0.0
        return self.total_verification_time_ms / self.verified_tokens

    @property
    def avg_margin_time_ms(self) -> float:
        """Average margin computation time per check."""
        checks = self.cheap_gate_checks if self.cheap_gate_checks > 0 else self.verified_tokens
        if checks == 0:
            return 0.0
        return self.total_margin_time_ms / checks

    @property
    def avg_entropy_time_ms(self) -> float:
        """Average entropy computation time per check."""
        if self.cheap_gate_full_compute == 0:
            return 0.0
        return self.total_entropy_time_ms / self.cheap_gate_full_compute

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tokens": self.total_tokens,
            "verified_tokens": self.verified_tokens,
            "verification_rate": self.verification_rate,
            "skipped_by_stride": self.skipped_by_stride,
            "skipped_by_layer": self.skipped_by_layer,
            "cheap_gate_checks": self.cheap_gate_checks,
            "cheap_gate_fast_accepts": self.cheap_gate_fast_accepts,
            "cheap_gate_fast_rejects": self.cheap_gate_fast_rejects,
            "cheap_gate_full_compute": self.cheap_gate_full_compute,
            "entropy_skip_rate": self.entropy_skip_rate,
            # Timing stats
            "total_verification_time_ms": self.total_verification_time_ms,
            "total_margin_time_ms": self.total_margin_time_ms,
            "total_entropy_time_ms": self.total_entropy_time_ms,
            "avg_verification_time_ms": self.avg_verification_time_ms,
            "avg_margin_time_ms": self.avg_margin_time_ms,
            "avg_entropy_time_ms": self.avg_entropy_time_ms,
        }


class TokenStrideScheduler:
    """
    Skip verification for non-stride tokens.

    When stride=4, only tokens at positions 0, 4, 8, ... are verified.
    Skipped tokens either accept (default_accept=True) or inherit
    the previous decision (default_accept=False).
    """

    def __init__(self, stride: int, default_accept: bool):
        self.stride = stride
        self.default_accept = default_accept
        self._last_confidence: Optional[float] = None
        self._last_should_escalate: bool = False

    def should_verify(self, token_position: int) -> bool:
        """Check if this token position should be verified."""
        # Always verify first token
        if token_position == 0:
            return True
        return token_position % self.stride == 0

    def get_skipped_confidence(self) -> Tuple[float, str]:
        """
        Get confidence value to use when verification is skipped.

        Returns:
            (confidence, reason)
        """
        if self.default_accept:
            # Accept with high confidence
            return 1.0, f"stride_accept (stride={self.stride})"
        elif self._last_confidence is not None:
            # Inherit previous confidence
            return self._last_confidence, f"stride_inherit (stride={self.stride})"
        else:
            # First token with no previous - accept with neutral confidence
            return 0.75, "stride_first_token"

    def record_result(self, confidence: float, should_escalate: bool) -> None:
        """Record a verified decision for future inheritance."""
        self._last_confidence = confidence
        self._last_should_escalate = should_escalate

    def reset(self) -> None:
        """Reset state for new generation."""
        self._last_confidence = None
        self._last_should_escalate = False


class LayerGateScheduler:
    """
    Only verify in final layers.

    This scheduler determines which layers should perform verification.
    For engine-level verification (after the final layer), this controls
    whether verification runs based on which layer's output we're checking.
    """

    def __init__(
        self,
        mode: str,
        layer_ratio: float,
        layer_count: int,
    ):
        self.mode = mode
        self.layer_ratio = layer_ratio
        self.layer_count = layer_count
        self._verification_layers: Optional[Set[int]] = None
        self._total_layers: int = 0

    def configure_for_model(self, total_layers: int) -> None:
        """Compute which layers should verify."""
        self._total_layers = total_layers

        if self.mode == "all" or self.layer_ratio >= 1.0:
            self._verification_layers = set(range(total_layers))
        elif self.mode == "ratio":
            # Top N% of layers
            start_layer = int(total_layers * (1 - self.layer_ratio))
            self._verification_layers = set(range(start_layer, total_layers))
        elif self.mode == "count":
            # Last N layers
            start_layer = max(0, total_layers - self.layer_count)
            self._verification_layers = set(range(start_layer, total_layers))
        else:
            self._verification_layers = set(range(total_layers))

    def should_verify_at_layer(self, layer_idx: int) -> bool:
        """Check if this layer should run verification."""
        if self._verification_layers is None:
            return True  # Not configured, verify all
        return layer_idx in self._verification_layers

    def should_verify_final(self) -> bool:
        """
        Check if verification should run at the final layer.

        For engine-level verification after the model forward pass,
        this is typically True unless layer_ratio or layer_count
        excludes the final layer (which would be unusual).
        """
        if self._verification_layers is None or self._total_layers == 0:
            return True
        # Final layer index is total_layers - 1
        return (self._total_layers - 1) in self._verification_layers

    def get_verification_layers(self) -> Set[int]:
        """Return set of layer indices that verify."""
        return self._verification_layers or set()


class CheapGateScheduler:
    """
    Margin-first gating before expensive entropy computation.

    The key insight is that margin (top-k extraction) is much cheaper
    than entropy (full softmax + log + sum). If margin is clearly
    high or low, we can skip entropy computation entirely.
    """

    def __init__(
        self,
        high_threshold: float,
        low_threshold: float,
        top_k: int = 5,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.top_k = top_k

        # Statistics
        self.total_checks = 0
        self.entropy_skipped = 0
        self.fast_accepts = 0
        self.fast_rejects = 0

    def compute_gated_confidence(
        self,
        logits: Tensor,
        margin_weight: float,
        entropy_weight: float,
    ) -> Tuple[float, bool, str, float, float]:
        """
        Compute confidence with cheap-gate optimization.

        Args:
            logits: Model output logits [..., vocab_size]
            margin_weight: Weight for margin in composite score
            entropy_weight: Weight for entropy in composite score

        Returns:
            (confidence, used_cheap_gate, gate_result, margin_time_ms, entropy_time_ms)
            gate_result: "accept" | "reject" | "uncertain"
        """
        self.total_checks += 1

        # Step 1: Compute margin (cheap - just top-k extraction)
        margin_start = time.perf_counter()
        margin = compute_margin(logits, k=self.top_k, normalized=True)
        margin_val = margin.mean().item() if margin.numel() > 1 else margin.item()
        margin_time_ms = (time.perf_counter() - margin_start) * 1000

        # Step 2: Check if we can short-circuit
        if margin_val >= self.high_threshold:
            # High margin = very confident, skip entropy
            self.entropy_skipped += 1
            self.fast_accepts += 1
            # Use margin as confidence (conservative but reasonable)
            return margin_val, True, "accept", margin_time_ms, 0.0

        if margin_val <= self.low_threshold:
            # Low margin = definitely uncertain, skip entropy
            self.entropy_skipped += 1
            self.fast_rejects += 1
            return margin_val, True, "reject", margin_time_ms, 0.0

        # Step 3: Uncertain zone - compute full confidence
        entropy_start = time.perf_counter()
        entropy = compute_entropy(logits, normalized=True)
        entropy_val = entropy.mean().item() if entropy.numel() > 1 else entropy.item()
        entropy_time_ms = (time.perf_counter() - entropy_start) * 1000

        # Weighted combination (entropy inverted: 1 - entropy)
        total_weight = margin_weight + entropy_weight
        if total_weight > 0:
            confidence = (
                (margin_weight / total_weight) * margin_val +
                (entropy_weight / total_weight) * (1.0 - entropy_val)
            )
        else:
            confidence = margin_val

        return confidence, False, "uncertain", margin_time_ms, entropy_time_ms

    def get_stats(self) -> Dict[str, Any]:
        """Return gating statistics."""
        return {
            "total_checks": self.total_checks,
            "entropy_skipped": self.entropy_skipped,
            "entropy_skip_rate": (
                self.entropy_skipped / self.total_checks
                if self.total_checks > 0 else 0.0
            ),
            "fast_accepts": self.fast_accepts,
            "fast_rejects": self.fast_rejects,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.total_checks = 0
        self.entropy_skipped = 0
        self.fast_accepts = 0
        self.fast_rejects = 0


class VerificationScheduler:
    """
    Coordinates all verification scheduling strategies.

    Reduces verification overhead by:
    1. Token stride: Only verify every k-th token
    2. Layer gate: Only verify in final N layers
    3. Cheap gate: Skip entropy when margin is conclusive

    Usage:
        scheduler = VerificationScheduler(config)
        scheduler.configure_for_model(num_layers)

        for position in range(num_tokens):
            result = scheduler.compute_confidence(logits, position, confidence_computer)
            if result.should_verify:
                decision = decision_maker.decide(result.confidence, position)
                scheduler.record_result(result.confidence, decision.should_escalate)
            else:
                decision = get_default_decision()
    """

    def __init__(self, config: AdaptiveInferenceConfig):
        self.config = config

        # Initialize sub-schedulers
        self.token_scheduler = TokenStrideScheduler(
            stride=config.verification_stride,
            default_accept=config.stride_default_accept,
        )

        self.layer_scheduler = LayerGateScheduler(
            mode=config.verification_layer_mode,
            layer_ratio=config.verification_layer_ratio,
            layer_count=config.verification_layer_count,
        )

        self.cheap_gate: Optional[CheapGateScheduler] = None
        if config.enable_cheap_gate:
            self.cheap_gate = CheapGateScheduler(
                high_threshold=config.cheap_gate_high_threshold,
                low_threshold=config.cheap_gate_low_threshold,
                top_k=config.top_k_margin,
            )

        self._configured = False
        self._stats = SchedulingStats()

    def configure_for_model(self, total_layers: int) -> None:
        """Configure layer-dependent settings."""
        self.layer_scheduler.configure_for_model(total_layers)
        self._configured = True

    def should_verify_token(self, token_position: int) -> Tuple[bool, str]:
        """Check token-stride gate."""
        if self.config.verification_stride <= 1:
            return True, "none"

        if self.token_scheduler.should_verify(token_position):
            return True, "none"
        else:
            return False, "token_stride"

    def should_verify_layer(self, layer_idx: int) -> Tuple[bool, str]:
        """Check layer gate (for layer-level verification)."""
        if self.config.verification_layer_ratio >= 1.0:
            return True, "none"

        if not self._configured:
            return True, "none"  # Not configured, verify all

        if self.layer_scheduler.should_verify_at_layer(layer_idx):
            return True, "none"
        else:
            return False, "layer_gate"

    def compute_confidence(
        self,
        logits: Tensor,
        token_position: int,
        confidence_computer: Optional[Any] = None,
    ) -> VerificationResult:
        """
        Compute confidence with all scheduling optimizations.

        Args:
            logits: Model output logits [batch, vocab] or [batch, seq, vocab]
            token_position: Current token position
            confidence_computer: Standard confidence computer (optional if using cheap gate)

        Returns:
            VerificationResult with confidence and metadata
        """
        self._stats.total_tokens += 1

        # Check token stride
        should_verify_tok, tok_reason = self.should_verify_token(token_position)
        if not should_verify_tok:
            self._stats.skipped_by_stride += 1
            confidence, reason = self.token_scheduler.get_skipped_confidence()
            return VerificationResult(
                should_verify=False,
                confidence=confidence,
                used_cheap_gate=False,
                skipped_reason=tok_reason,
                gate_result="skipped",
            )

        self._stats.verified_tokens += 1
        verify_start = time.perf_counter()

        # Check cheap gate (if enabled)
        if self.cheap_gate is not None:
            self._stats.cheap_gate_checks += 1
            confidence, used_gate, gate_result, margin_time_ms, entropy_time_ms = (
                self.cheap_gate.compute_gated_confidence(
                    logits,
                    margin_weight=self.config.margin_weight,
                    entropy_weight=self.config.entropy_weight,
                )
            )

            # Track timing
            self._stats.total_margin_time_ms += margin_time_ms
            self._stats.total_entropy_time_ms += entropy_time_ms
            self._stats.total_verification_time_ms += (time.perf_counter() - verify_start) * 1000

            if used_gate:
                if gate_result == "accept":
                    self._stats.cheap_gate_fast_accepts += 1
                else:
                    self._stats.cheap_gate_fast_rejects += 1
            else:
                self._stats.cheap_gate_full_compute += 1

            return VerificationResult(
                should_verify=True,
                confidence=confidence,
                used_cheap_gate=used_gate,
                skipped_reason="cheap_gate" if used_gate else "none",
                gate_result=gate_result,
            )

        # Full confidence computation (using provided confidence_computer)
        if confidence_computer is not None:
            confidence_tensor = confidence_computer.compute_confidence(logits)
            confidence_val = (
                confidence_tensor.mean().item()
                if confidence_tensor.numel() > 1
                else confidence_tensor.item()
            )
        else:
            # Fallback: compute basic confidence from margin and entropy
            margin_start = time.perf_counter()
            margin = compute_margin(logits, k=self.config.top_k_margin, normalized=True)
            margin_val = margin.mean().item() if margin.numel() > 1 else margin.item()
            self._stats.total_margin_time_ms += (time.perf_counter() - margin_start) * 1000

            entropy_start = time.perf_counter()
            entropy = compute_entropy(logits, normalized=True)
            entropy_val = entropy.mean().item() if entropy.numel() > 1 else entropy.item()
            self._stats.total_entropy_time_ms += (time.perf_counter() - entropy_start) * 1000

            total_weight = self.config.margin_weight + self.config.entropy_weight
            if total_weight > 0:
                confidence_val = (
                    (self.config.margin_weight / total_weight) * margin_val +
                    (self.config.entropy_weight / total_weight) * (1.0 - entropy_val)
                )
            else:
                confidence_val = margin_val

        self._stats.total_verification_time_ms += (time.perf_counter() - verify_start) * 1000

        return VerificationResult(
            should_verify=True,
            confidence=confidence_val,
            used_cheap_gate=False,
            skipped_reason="none",
            gate_result="full",
        )

    def record_result(self, confidence: float, should_escalate: bool) -> None:
        """Record verification result for stride inheritance."""
        self.token_scheduler.record_result(confidence, should_escalate)

    def reset(self) -> None:
        """Reset state for new generation."""
        self.token_scheduler.reset()
        if self.cheap_gate:
            self.cheap_gate.reset_stats()
        self._stats = SchedulingStats()

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduling statistics."""
        stats = {
            "token_stride": self.config.verification_stride,
            "layer_ratio": self.config.verification_layer_ratio,
            "cheap_gate_enabled": self.cheap_gate is not None,
            "scheduling_stats": self._stats.to_dict(),
        }
        if self.cheap_gate:
            stats["cheap_gate_stats"] = self.cheap_gate.get_stats()
        return stats

    @property
    def is_scheduling_enabled(self) -> bool:
        """Check if any scheduling optimization is enabled."""
        return (
            self.config.verification_stride > 1 or
            self.config.verification_layer_ratio < 1.0 or
            self.config.enable_cheap_gate
        )
