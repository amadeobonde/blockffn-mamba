"""
Escalation Decision Logic

Determines when and how to escalate from reduced to full attention
based on confidence metrics and streak detection.

Decision hierarchy:
- Green zone (confidence >= 0.85): Accept reduced compute
- Yellow zone (0.65-0.85): Monitor, escalate if streak
- Red zone (< 0.65): Immediate escalation
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import torch
from torch import Tensor

from src.adaptive.config import AdaptiveInferenceConfig
from src.verification.confidence import ConfidenceComputer, classify_confidence


class EscalationLevel(Enum):
    """
    Escalation levels for adaptive inference.

    NONE: Accept reduced compute, no escalation needed
    LAYER: Re-run current layer with full attention
    TOKEN: Re-run current token with full attention (all layers)
    SUFFIX: Re-run from current token to end with full attention
    FULL_CONTEXT: Re-compute entire context (last resort)
    """
    NONE = auto()
    LAYER = auto()
    TOKEN = auto()
    SUFFIX = auto()
    FULL_CONTEXT = auto()


@dataclass
class EscalationDecision:
    """
    Decision about whether and how to escalate.

    Attributes:
        level: The escalation level
        reason: Why this decision was made
        confidence: The confidence score that triggered the decision
        position: Token position (if applicable)
    """
    level: EscalationLevel
    reason: str
    confidence: float
    position: Optional[int] = None

    @property
    def should_escalate(self) -> bool:
        """Whether any escalation is needed."""
        return self.level != EscalationLevel.NONE


class EscalationDecisionMaker:
    """
    Makes escalation decisions based on confidence and history.

    Tracks yellow streak (consecutive moderate-confidence tokens)
    and decides when to escalate to prevent error accumulation.

    Attributes:
        config: Adaptive inference configuration
        yellow_streak: Current count of consecutive yellow tokens
        total_decisions: Count of decisions made
        escalation_counts: Counts by escalation level
    """

    def __init__(self, config: AdaptiveInferenceConfig):
        """
        Initialize decision maker.

        Args:
            config: Adaptive inference configuration
        """
        self.config = config
        self.tau_green = config.tau_green
        self.tau_yellow = config.tau_yellow
        self.tau_red = config.tau_red
        self.streak_limit = config.yellow_streak_limit

        # State tracking
        self.yellow_streak = 0
        self.total_decisions = 0
        self.escalation_counts = {level: 0 for level in EscalationLevel}

        # Cascade detection
        self.suffix_escalation_count = 0
        self.cascade_threshold = config.cascade_threshold

    def decide(
        self,
        confidence: float,
        position: Optional[int] = None,
    ) -> EscalationDecision:
        """
        Make escalation decision for a single token.

        Args:
            confidence: Token's confidence score [0, 1]
            position: Optional token position

        Returns:
            EscalationDecision with level and reason
        """
        self.total_decisions += 1

        # Check for cascade (too many suffix escalations)
        if self.suffix_escalation_count >= self.cascade_threshold:
            # Disable adaptive compute entirely
            decision = EscalationDecision(
                level=EscalationLevel.TOKEN,
                reason=f"Cascade detected ({self.suffix_escalation_count} suffix escalations)",
                confidence=confidence,
                position=position,
            )
            self.escalation_counts[EscalationLevel.TOKEN] += 1
            return decision

        # Green zone: high confidence, accept
        if confidence >= self.tau_green:
            self.yellow_streak = 0  # Reset streak
            decision = EscalationDecision(
                level=EscalationLevel.NONE,
                reason=f"High confidence ({confidence:.3f} >= {self.tau_green})",
                confidence=confidence,
                position=position,
            )
            self.escalation_counts[EscalationLevel.NONE] += 1
            return decision

        # Yellow zone: moderate confidence, check streak
        if confidence >= self.tau_yellow:
            self.yellow_streak += 1

            if self.yellow_streak >= self.streak_limit:
                # Streak exceeded, escalate suffix
                self.suffix_escalation_count += 1
                self.yellow_streak = 0  # Reset after escalation
                decision = EscalationDecision(
                    level=EscalationLevel.SUFFIX,
                    reason=f"Yellow streak ({self.streak_limit} tokens)",
                    confidence=confidence,
                    position=position,
                )
                self.escalation_counts[EscalationLevel.SUFFIX] += 1
                return decision
            else:
                # Accept but flag
                decision = EscalationDecision(
                    level=EscalationLevel.NONE,
                    reason=f"Yellow zone ({confidence:.3f}), streak={self.yellow_streak}",
                    confidence=confidence,
                    position=position,
                )
                self.escalation_counts[EscalationLevel.NONE] += 1
                return decision

        # Red zone: low confidence, immediate escalation
        if confidence >= self.tau_red:
            self.yellow_streak = 0
            decision = EscalationDecision(
                level=EscalationLevel.TOKEN,
                reason=f"Low confidence ({confidence:.3f} < {self.tau_yellow})",
                confidence=confidence,
                position=position,
            )
            self.escalation_counts[EscalationLevel.TOKEN] += 1
            return decision

        # Very low confidence: escalate with more context
        self.yellow_streak = 0
        decision = EscalationDecision(
            level=EscalationLevel.SUFFIX,
            reason=f"Very low confidence ({confidence:.3f} < {self.tau_red})",
            confidence=confidence,
            position=position,
        )
        self.suffix_escalation_count += 1
        self.escalation_counts[EscalationLevel.SUFFIX] += 1
        return decision

    def decide_batch(
        self,
        confidence: Tensor,
    ) -> list:
        """
        Make decisions for a batch of tokens.

        Note: Batch mode doesn't track streak across batch
        (each sequence in batch is independent).

        Args:
            confidence: Confidence scores [batch, seq]

        Returns:
            List of lists of EscalationDecision
        """
        batch_size, seq_len = confidence.shape
        decisions = []

        for b in range(batch_size):
            seq_decisions = []
            self.reset_streak()  # Reset for each sequence

            for t in range(seq_len):
                conf = confidence[b, t].item()
                decision = self.decide(conf, position=t)
                seq_decisions.append(decision)

            decisions.append(seq_decisions)

        return decisions

    def reset_streak(self) -> None:
        """Reset yellow streak counter."""
        self.yellow_streak = 0

    def reset_cascade(self) -> None:
        """Reset cascade detection counter."""
        self.suffix_escalation_count = 0

    def reset_all(self) -> None:
        """Reset all state."""
        self.yellow_streak = 0
        self.suffix_escalation_count = 0
        self.total_decisions = 0
        self.escalation_counts = {level: 0 for level in EscalationLevel}

    @property
    def escalation_rate(self) -> float:
        """Fraction of tokens that required escalation."""
        if self.total_decisions == 0:
            return 0.0
        escalated = sum(
            count for level, count in self.escalation_counts.items()
            if level != EscalationLevel.NONE
        )
        return escalated / self.total_decisions

    def get_stats(self) -> dict:
        """Get decision statistics."""
        return {
            "total_decisions": self.total_decisions,
            "escalation_rate": self.escalation_rate,
            "yellow_streak": self.yellow_streak,
            "suffix_escalations": self.suffix_escalation_count,
            "counts_by_level": {
                level.name: count
                for level, count in self.escalation_counts.items()
            },
        }


class BatchEscalationTracker:
    """
    Tracks escalation decisions across a batch for efficient processing.

    Identifies which tokens need escalation and groups them for
    efficient batch recomputation.
    """

    def __init__(self, config: AdaptiveInferenceConfig):
        self.config = config
        self.tau_green = config.tau_green
        self.tau_yellow = config.tau_yellow

    def identify_escalations(
        self,
        confidence: Tensor,
    ) -> dict:
        """
        Identify tokens needing escalation in a batch.

        Args:
            confidence: Confidence scores [batch, seq]

        Returns:
            Dict with:
            - 'token_escalation': Positions needing token escalation
            - 'suffix_escalation': Positions needing suffix escalation
            - 'escalation_mask': Boolean mask of escalation positions
        """
        batch_size, seq_len = confidence.shape

        # Classify all tokens
        green, yellow, red = classify_confidence(
            confidence,
            tau_green=self.tau_green,
            tau_yellow=self.tau_yellow,
        )

        # Red tokens need immediate escalation
        token_escalation_mask = red

        # Find yellow streaks (simplified: mark end of streaks)
        # More sophisticated streak detection would track per-sequence
        suffix_escalation_mask = torch.zeros_like(confidence, dtype=torch.bool)

        for b in range(batch_size):
            streak = 0
            for t in range(seq_len):
                if yellow[b, t]:
                    streak += 1
                    if streak >= self.config.yellow_streak_limit:
                        suffix_escalation_mask[b, t] = True
                        streak = 0
                else:
                    streak = 0

        # Combine escalation masks
        escalation_mask = token_escalation_mask | suffix_escalation_mask

        # Get positions
        token_positions = torch.nonzero(token_escalation_mask)
        suffix_positions = torch.nonzero(suffix_escalation_mask)

        return {
            "token_escalation": token_positions,
            "suffix_escalation": suffix_positions,
            "escalation_mask": escalation_mask,
            "pct_escalated": escalation_mask.float().mean().item(),
        }

    def group_escalations(
        self,
        escalation_info: dict,
    ) -> list:
        """
        Group escalation positions for efficient batch processing.

        Finds contiguous regions that need escalation and groups them.

        Args:
            escalation_info: Output from identify_escalations

        Returns:
            List of (batch_idx, start_pos, end_pos) tuples
        """
        mask = escalation_info["escalation_mask"]
        batch_size, seq_len = mask.shape

        groups = []

        for b in range(batch_size):
            # Find contiguous runs of True values
            seq_mask = mask[b]
            positions = torch.nonzero(seq_mask).squeeze(-1).tolist()

            if not positions:
                continue

            # Group contiguous positions
            start = positions[0]
            end = positions[0]

            for pos in positions[1:]:
                if pos == end + 1:
                    end = pos
                else:
                    groups.append((b, start, end + 1))
                    start = pos
                    end = pos

            groups.append((b, start, end + 1))

        return groups
