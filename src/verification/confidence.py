"""
Confidence Metrics

Cheap confidence checks for verifying adaptive inference outputs.
All metrics run in O(vocab) time or less.

Key metrics:
1. Entropy: Uncertainty in output distribution
2. Margin: Gap between top predictions
3. Stability: Agreement between reduced and full compute (optional)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch
import torch.nn.functional as F
from torch import Tensor

from src.adaptive.config import AdaptiveInferenceConfig


def compute_entropy(
    logits: Tensor,
    normalized: bool = True,
) -> Tensor:
    """
    Compute entropy of output distribution.

    Lower entropy indicates higher confidence (one dominant prediction).
    Higher entropy indicates uncertainty (multiple candidates).

    Args:
        logits: Output logits [..., vocab_size]
        normalized: If True, normalize to [0, 1] range

    Returns:
        entropy: Entropy values [...] (same shape without last dim)
    """
    # Compute probabilities
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)

    # Entropy: -sum(p * log(p))
    entropy = -torch.sum(probs * log_probs, dim=-1)

    if normalized:
        # Normalize by max entropy (uniform distribution)
        vocab_size = logits.size(-1)
        max_entropy = math.log(vocab_size)
        entropy = entropy / max_entropy

    return entropy


def compute_margin(
    logits: Tensor,
    k: int = 5,
    normalized: bool = True,
    temperature: float = 1.0,
) -> Tensor:
    """
    Compute margin between top-1 and top-k predictions.

    Larger margin indicates higher confidence (clear winner).
    Smaller margin indicates uncertainty (multiple close candidates).

    Args:
        logits: Output logits [..., vocab_size]
        k: Compare top-1 with top-k
        normalized: If True, apply sigmoid normalization
        temperature: Temperature for sigmoid (if normalized)

    Returns:
        margin: Margin values [...] (same shape without last dim)
    """
    # Get top-k logits
    topk_values, _ = torch.topk(logits, k=k, dim=-1)

    # Margin = top1 - topk
    margin = topk_values[..., 0] - topk_values[..., -1]

    if normalized:
        # Apply sigmoid to normalize to [0, 1]
        margin = torch.sigmoid(margin / temperature)

    return margin


def compute_stability(
    logits_reduced: Tensor,
    logits_full: Tensor,
    method: str = "cosine",
) -> Tensor:
    """
    Compute stability between reduced and full compute outputs.

    Higher stability indicates agreement (reduced compute is reliable).
    Lower stability indicates disagreement (should escalate).

    Args:
        logits_reduced: Logits from reduced attention [..., vocab]
        logits_full: Logits from full attention [..., vocab]
        method: Comparison method ("cosine", "kl", "topk_agreement")

    Returns:
        stability: Stability scores [...] in [-1, 1] for cosine, [0, 1] for others
    """
    if method == "cosine":
        # Cosine similarity between logit vectors
        reduced_norm = F.normalize(logits_reduced, dim=-1)
        full_norm = F.normalize(logits_full, dim=-1)
        stability = torch.sum(reduced_norm * full_norm, dim=-1)

    elif method == "kl":
        # Negative KL divergence (higher = more similar)
        probs_reduced = F.softmax(logits_reduced, dim=-1)
        log_probs_full = F.log_softmax(logits_full, dim=-1)
        kl = F.kl_div(log_probs_full, probs_reduced, reduction="none").sum(dim=-1)
        # Convert to stability: exp(-kl) gives [0, 1]
        stability = torch.exp(-kl)

    elif method == "topk_agreement":
        # Check if top-k predictions agree
        k = 5
        topk_reduced = torch.topk(logits_reduced, k, dim=-1).indices
        topk_full = torch.topk(logits_full, k, dim=-1).indices

        # Count agreement
        agreement = torch.zeros_like(logits_reduced[..., 0])
        for i in range(k):
            for j in range(k):
                agreement += (topk_reduced[..., i] == topk_full[..., j]).float()
        stability = agreement / k

    else:
        raise ValueError(f"Unknown method: {method}")

    return stability


@dataclass
class ConfidenceMetrics:
    """
    Container for confidence metrics.

    Attributes:
        entropy: Normalized entropy [0, 1] (0 = confident, 1 = uncertain)
        margin: Normalized margin [0, 1] (0 = uncertain, 1 = confident)
        stability: Optional stability score (computed on demand)
    """
    entropy: Tensor  # [batch, seq] or [batch]
    margin: Tensor   # [batch, seq] or [batch]
    stability: Optional[Tensor] = None  # [batch, seq] or [batch]

    def composite_score(
        self,
        w_entropy: float = 0.5,
        w_margin: float = 0.5,
        w_stability: float = 0.0,
        use_minimum: bool = False,
    ) -> Tensor:
        """
        Compute composite confidence score.

        Args:
            w_entropy: Weight for entropy (inverted: 1 - entropy)
            w_margin: Weight for margin
            w_stability: Weight for stability (if available)
            use_minimum: If True, use min of components instead of weighted sum

        Returns:
            confidence: Composite confidence score [0, 1]
        """
        # Convert entropy to confidence (invert)
        entropy_conf = 1.0 - self.entropy

        if use_minimum:
            # Conservative: take minimum of all metrics
            confidence = torch.minimum(entropy_conf, self.margin)
            if self.stability is not None:
                confidence = torch.minimum(confidence, self.stability)
            return confidence

        # Weighted sum
        if self.stability is not None and w_stability > 0:
            total_weight = w_entropy + w_margin + w_stability
            confidence = (
                (w_entropy / total_weight) * entropy_conf +
                (w_margin / total_weight) * self.margin +
                (w_stability / total_weight) * self.stability
            )
        else:
            total_weight = w_entropy + w_margin
            confidence = (
                (w_entropy / total_weight) * entropy_conf +
                (w_margin / total_weight) * self.margin
            )

        return confidence


class ConfidenceComputer:
    """
    Computes confidence metrics for verification.

    Combines entropy, margin, and optional stability checks
    into a composite confidence score.
    """

    def __init__(self, config: AdaptiveInferenceConfig):
        """
        Initialize confidence computer.

        Args:
            config: Adaptive inference configuration
        """
        self.config = config
        self.entropy_weight = config.entropy_weight
        self.margin_weight = config.margin_weight
        self.stability_weight = config.stability_weight
        self.top_k = config.top_k_margin

    def compute(
        self,
        logits: Tensor,
        logits_full: Optional[Tensor] = None,
    ) -> ConfidenceMetrics:
        """
        Compute all confidence metrics.

        Args:
            logits: Output logits [batch, seq, vocab] or [batch, vocab]
            logits_full: Optional full-attention logits for stability check

        Returns:
            ConfidenceMetrics containing all computed metrics
        """
        entropy = compute_entropy(logits, normalized=True)
        margin = compute_margin(logits, k=self.top_k, normalized=True)

        stability = None
        if logits_full is not None and self.stability_weight > 0:
            stability = compute_stability(logits, logits_full, method="cosine")
            # Convert cosine [-1, 1] to [0, 1]
            stability = (stability + 1.0) / 2.0

        return ConfidenceMetrics(
            entropy=entropy,
            margin=margin,
            stability=stability,
        )

    def compute_confidence(
        self,
        logits: Tensor,
        logits_full: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute composite confidence score directly.

        Convenience method that returns just the confidence score.

        Args:
            logits: Output logits
            logits_full: Optional full-attention logits

        Returns:
            confidence: Composite confidence [0, 1]
        """
        metrics = self.compute(logits, logits_full)
        return metrics.composite_score(
            w_entropy=self.entropy_weight,
            w_margin=self.margin_weight,
            w_stability=self.stability_weight,
        )


def classify_confidence(
    confidence: Tensor,
    tau_green: float = 0.85,
    tau_yellow: float = 0.65,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Classify tokens into green/yellow/red confidence zones.

    Args:
        confidence: Confidence scores [batch, seq] or [batch]
        tau_green: Threshold for green zone (accept)
        tau_yellow: Threshold for yellow zone (monitor)

    Returns:
        green_mask: High confidence (accept reduced compute)
        yellow_mask: Moderate confidence (monitor for streak)
        red_mask: Low confidence (escalate immediately)
    """
    green_mask = confidence >= tau_green
    yellow_mask = (confidence >= tau_yellow) & ~green_mask
    red_mask = confidence < tau_yellow

    return green_mask, yellow_mask, red_mask


def get_confidence_stats(confidence: Tensor) -> dict:
    """
    Get statistics about confidence distribution.

    Args:
        confidence: Confidence scores

    Returns:
        Dict with confidence statistics
    """
    return {
        "mean": confidence.mean().item(),
        "std": confidence.std().item() if confidence.numel() > 1 else 0.0,
        "min": confidence.min().item(),
        "max": confidence.max().item(),
        "pct_above_0.9": (confidence >= 0.9).float().mean().item(),
        "pct_above_0.8": (confidence >= 0.8).float().mean().item(),
        "pct_below_0.5": (confidence < 0.5).float().mean().item(),
    }
