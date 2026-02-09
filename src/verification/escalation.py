"""
Escalation Module

Handles escalation when verification fails. Provides mechanisms to
recompute tokens with full attention when reduced compute is unreliable.

Escalation paths:
1. Layer escalation: Re-run current layer with full attention
2. Token escalation: Re-run current token with full attention (all layers)
3. Suffix escalation: Re-run from position N to end with full attention
"""

from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
from torch import Tensor

from src.adaptive.config import AdaptiveInferenceConfig
from src.attention.adaptive_attention import (
    AdaptiveAttentionLayer,
    get_adaptive_layers,
)
from src.verification.decision import EscalationLevel


class EscalationModule:
    """
    Handles escalation when verification fails.

    Provides methods to recompute tokens with full attention
    when reduced compute produces unreliable outputs.

    Attributes:
        model: The BlockFFN model
        config: Adaptive inference configuration
        adaptive_layers: List of adaptive attention layers
    """

    def __init__(
        self,
        model: nn.Module,
        config: AdaptiveInferenceConfig,
    ):
        """
        Initialize escalation module.

        Args:
            model: BlockFFN model with adaptive attention layers
            config: Adaptive inference configuration
        """
        self.model = model
        self.config = config

        # Get adaptive layers
        self.adaptive_layers = get_adaptive_layers(model)

        # Cache original layer references
        self._original_layers: Dict[int, nn.Module] = {}
        for idx, layer in self.adaptive_layers:
            self._original_layers[idx] = layer.original_layer

    def escalate(
        self,
        input_ids: Tensor,
        level: EscalationLevel,
        position: Optional[int] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        layer_idx: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        """
        Perform escalation at the specified level.

        Args:
            input_ids: Input token IDs [batch, seq]
            level: Escalation level
            position: Token position (for TOKEN and SUFFIX)
            attention_mask: Attention mask
            past_key_values: KV cache
            layer_idx: Layer index (for LAYER escalation)
            **kwargs: Additional arguments for model forward

        Returns:
            logits: Recomputed logits
        """
        if level == EscalationLevel.NONE:
            raise ValueError("Cannot escalate with level NONE")

        elif level == EscalationLevel.LAYER:
            if layer_idx is None:
                raise ValueError("layer_idx required for LAYER escalation")
            return self.escalate_layer(
                input_ids, layer_idx, attention_mask, **kwargs
            )

        elif level == EscalationLevel.TOKEN:
            if position is None:
                position = input_ids.shape[1] - 1  # Default to last token
            return self.escalate_token(
                input_ids, position, attention_mask, past_key_values, **kwargs
            )

        elif level == EscalationLevel.SUFFIX:
            if position is None:
                position = input_ids.shape[1] - 1
            # Check if bounded suffix is configured
            if self.config.suffix_escalation_length > 0:
                return self.escalate_suffix_bounded(
                    input_ids, position, self.config.suffix_escalation_length,
                    attention_mask, **kwargs
                )
            else:
                return self.escalate_suffix(
                    input_ids, position, attention_mask, **kwargs
                )

        elif level == EscalationLevel.FULL_CONTEXT:
            return self.escalate_full_context(
                input_ids, attention_mask, **kwargs
            )

        else:
            raise ValueError(f"Unknown escalation level: {level}")

    def escalate_layer(
        self,
        input_ids: Tensor,
        layer_idx: int,
        attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Re-run a single layer with full attention.

        Temporarily bypasses adaptive attention for the specified layer.

        Args:
            input_ids: Input token IDs [batch, seq]
            layer_idx: Layer to escalate
            attention_mask: Attention mask
            **kwargs: Additional arguments

        Returns:
            logits: Model output logits
        """
        # Check if this layer is adaptive
        adaptive_layer = None
        for idx, layer in self.adaptive_layers:
            if idx == layer_idx:
                adaptive_layer = layer
                break

        if adaptive_layer is None:
            # Layer is not adaptive, just run normally
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    **kwargs,
                )
            return outputs.logits

        # Temporarily replace adaptive layer with original
        original_layer = adaptive_layer.original_layer

        try:
            self.model.model.layers[layer_idx] = original_layer

            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    **kwargs,
                )

            return outputs.logits

        finally:
            # Restore adaptive layer
            self.model.model.layers[layer_idx] = adaptive_layer

    def escalate_token(
        self,
        input_ids: Tensor,
        position: int,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        **kwargs,
    ) -> Tensor:
        """
        Recompute a single token with full attention across all layers.

        Args:
            input_ids: Input token IDs [batch, seq]
            position: Token position to recompute
            attention_mask: Attention mask
            past_key_values: KV cache (for incremental decoding)
            **kwargs: Additional arguments

        Returns:
            logits: Logits for the specified position [batch, vocab]
        """
        # Temporarily replace all adaptive layers with originals
        try:
            self._set_full_attention()

            if past_key_values is not None and position > 0:
                # Use KV cache for efficiency
                token_input = input_ids[:, position:position + 1]
                token_mask = attention_mask[:, :position + 1] if attention_mask is not None else None

                with torch.no_grad():
                    outputs = self.model(
                        token_input,
                        attention_mask=token_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        **kwargs,
                    )

                return outputs.logits[:, -1, :]
            else:
                # Recompute from scratch up to and including position
                with torch.no_grad():
                    outputs = self.model(
                        input_ids[:, :position + 1],
                        attention_mask=attention_mask[:, :position + 1] if attention_mask is not None else None,
                        **kwargs,
                    )

                return outputs.logits[:, -1, :]

        finally:
            self._restore_adaptive_layers()

    def escalate_suffix(
        self,
        input_ids: Tensor,
        start_position: int,
        attention_mask: Optional[Tensor] = None,
        past_key_values_prefix: Optional[Tuple] = None,
        **kwargs,
    ) -> Tensor:
        """
        Recompute from start_position to end with full attention.

        Used when we detect that errors may have propagated from
        a misclassified token.

        Args:
            input_ids: Input token IDs [batch, seq]
            start_position: Position to start recomputation from
            attention_mask: Attention mask
            past_key_values_prefix: KV cache for prefix (before start_position)
            **kwargs: Additional arguments

        Returns:
            logits: Logits for all positions from start_position [batch, seq-start, vocab]
        """
        try:
            self._set_full_attention()

            if past_key_values_prefix is not None and start_position > 0:
                # Use cached prefix
                suffix_input = input_ids[:, start_position:]
                suffix_mask = attention_mask[:, :] if attention_mask is not None else None

                with torch.no_grad():
                    outputs = self.model(
                        suffix_input,
                        attention_mask=suffix_mask,
                        past_key_values=past_key_values_prefix,
                        use_cache=True,
                        **kwargs,
                    )

                return outputs.logits
            else:
                # Recompute entire sequence from start
                with torch.no_grad():
                    outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        **kwargs,
                    )

                # Return logits from start_position onwards
                return outputs.logits[:, start_position:, :]

        finally:
            self._restore_adaptive_layers()

    def escalate_suffix_bounded(
        self,
        input_ids: Tensor,
        start_position: int,
        suffix_length: int,
        attention_mask: Optional[Tensor] = None,
        past_key_values_prefix: Optional[Tuple] = None,
        **kwargs,
    ) -> Tensor:
        """
        Recompute a bounded suffix with full attention.

        Unlike escalate_suffix which recomputes to the end of sequence,
        this recomputes only a fixed number of tokens from start_position.

        Args:
            input_ids: Input token IDs [batch, seq]
            start_position: Position to start recomputation from
            suffix_length: Number of tokens to recompute (L=2,4,8,16)
            attention_mask: Attention mask
            past_key_values_prefix: KV cache for prefix (before start_position)
            **kwargs: Additional arguments

        Returns:
            logits: Logits for recomputed positions [batch, suffix_length, vocab]
        """
        seq_len = input_ids.shape[1]
        end_position = min(start_position + suffix_length, seq_len)
        actual_length = end_position - start_position

        if actual_length <= 0:
            # Nothing to recompute
            return torch.empty(
                input_ids.shape[0], 0, self.model.config.vocab_size,
                device=input_ids.device, dtype=torch.float32
            )

        try:
            self._set_full_attention()

            if past_key_values_prefix is not None and start_position > 0:
                # Use cached prefix - recompute only the suffix portion
                suffix_input = input_ids[:, start_position:end_position]
                suffix_mask = attention_mask[:, :end_position] if attention_mask is not None else None

                with torch.no_grad():
                    outputs = self.model(
                        suffix_input,
                        attention_mask=suffix_mask,
                        past_key_values=past_key_values_prefix,
                        use_cache=True,
                        **kwargs,
                    )

                return outputs.logits
            else:
                # Recompute from start of sequence to end_position
                with torch.no_grad():
                    outputs = self.model(
                        input_ids[:, :end_position],
                        attention_mask=attention_mask[:, :end_position] if attention_mask is not None else None,
                        **kwargs,
                    )

                # Return logits from start_position to end_position
                return outputs.logits[:, start_position:end_position, :]

        finally:
            self._restore_adaptive_layers()

    def escalate_full_context(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Recompute entire context with full attention.

        Last resort when cascade is detected.

        Args:
            input_ids: Input token IDs [batch, seq]
            attention_mask: Attention mask
            **kwargs: Additional arguments

        Returns:
            logits: Full logits [batch, seq, vocab]
        """
        try:
            self._set_full_attention()

            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    **kwargs,
                )

            return outputs.logits

        finally:
            self._restore_adaptive_layers()

    def _set_full_attention(self) -> None:
        """Replace all adaptive layers with original layers."""
        for idx, layer in self.adaptive_layers:
            self.model.model.layers[idx] = layer.original_layer

    def _restore_adaptive_layers(self) -> None:
        """Restore adaptive layers."""
        for idx, layer in self.adaptive_layers:
            self.model.model.layers[idx] = layer


class BatchEscalationExecutor:
    """
    Executes escalations efficiently across a batch.

    Groups escalation positions and processes them together
    to minimize redundant computation.
    """

    def __init__(
        self,
        model: nn.Module,
        config: AdaptiveInferenceConfig,
    ):
        self.model = model
        self.config = config
        self.escalation_module = EscalationModule(model, config)

    def execute_escalations(
        self,
        input_ids: Tensor,
        original_logits: Tensor,
        escalation_groups: List[Tuple[int, int, int]],
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Execute escalations for grouped positions.

        Args:
            input_ids: Input token IDs [batch, seq]
            original_logits: Original logits from adaptive forward [batch, seq, vocab]
            escalation_groups: List of (batch_idx, start_pos, end_pos) tuples
            attention_mask: Attention mask

        Returns:
            logits: Updated logits with escalated positions replaced
        """
        output_logits = original_logits.clone()

        for batch_idx, start_pos, end_pos in escalation_groups:
            # Recompute this segment with full attention
            recomputed = self.escalation_module.escalate_suffix(
                input_ids[batch_idx:batch_idx + 1],
                start_position=start_pos,
                attention_mask=attention_mask[batch_idx:batch_idx + 1] if attention_mask is not None else None,
            )

            # Replace logits for recomputed positions
            segment_len = end_pos - start_pos
            output_logits[batch_idx, start_pos:end_pos] = recomputed[:, :segment_len]

        return output_logits

    def execute_selective_escalation(
        self,
        input_ids: Tensor,
        original_logits: Tensor,
        escalation_mask: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Execute escalation for all marked positions.

        Simpler interface that takes a boolean mask instead of groups.

        Args:
            input_ids: Input token IDs [batch, seq]
            original_logits: Original logits [batch, seq, vocab]
            escalation_mask: Boolean mask [batch, seq] of positions to escalate
            attention_mask: Attention mask

        Returns:
            logits: Updated logits
        """
        if not escalation_mask.any():
            return original_logits

        # Find groups from mask
        batch_size, seq_len = input_ids.shape
        groups = []

        for b in range(batch_size):
            positions = escalation_mask[b].nonzero().squeeze(-1).tolist()
            if not positions:
                continue

            # Group contiguous positions
            if isinstance(positions, int):
                positions = [positions]

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

        return self.execute_escalations(
            input_ids, original_logits, groups, attention_mask
        )
