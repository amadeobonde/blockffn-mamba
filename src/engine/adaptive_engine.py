"""
Adaptive Inference Engine

Main orchestrator for adaptive inference with BlockFFN.
Combines all components:
1. Adaptive attention with variable windows
2. Verification with confidence metrics
3. Escalation when verification fails

Provides both generation and batch processing interfaces.
"""

from typing import Optional, Dict, Any, Union, Tuple, List
from dataclasses import dataclass, field
from collections import defaultdict
import time
import torch
import torch.nn as nn
from torch import Tensor

from src.adaptive.config import AdaptiveInferenceConfig, PRESETS
from src.adaptive.routing_predictor import LayerRoutingPredictors
from src.adaptive.window_mapper import SparsityToWindowMapper, estimate_speedup
from src.attention.adaptive_attention import (
    attach_adaptive_attention,
    detach_adaptive_attention,
    get_adaptive_layers,
    get_all_statistics,
    clear_all_statistics,
)
from src.verification.confidence import ConfidenceComputer, get_confidence_stats
from src.verification.decision import EscalationDecisionMaker, EscalationLevel
from src.verification.escalation import EscalationModule, BatchEscalationExecutor
from src.verification.scheduler import VerificationScheduler
from src.ffn.cls_ffn import (
    attach_cls_ffn,
    detach_cls_ffn,
    get_cls_layers,
    get_all_cls_statistics,
    reset_all_cls_statistics,
)


@dataclass
class GenerationStats:
    """Statistics from a generation run."""
    total_tokens: int = 0
    reduced_compute_tokens: int = 0
    escalated_tokens: int = 0
    escalation_by_level: Dict[EscalationLevel, int] = field(default_factory=lambda: defaultdict(int))
    mean_confidence: float = 0.0
    mean_sparsity: float = 0.0
    wall_time_seconds: float = 0.0
    # CLS statistics
    cls_expert_reuse_rate: float = 0.0
    cls_compute_savings: float = 0.0
    cls_chunks_processed: int = 0

    @property
    def escalation_rate(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.escalated_tokens / self.total_tokens

    @property
    def reduced_compute_rate(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.reduced_compute_tokens / self.total_tokens

    @property
    def tokens_per_second(self) -> float:
        if self.wall_time_seconds == 0:
            return 0.0
        return self.total_tokens / self.wall_time_seconds

    @property
    def combined_savings_estimate(self) -> float:
        """Estimate total compute savings from attention + FFN optimizations."""
        # Attention savings from reduced window tokens
        attn_savings = self.reduced_compute_rate * 0.5  # Conservative estimate
        # FFN savings from CLS
        ffn_savings = self.cls_compute_savings * 0.5  # Weight by FFN portion
        return attn_savings + ffn_savings


class AdaptiveInferenceEngine:
    """
    Main engine for adaptive inference with BlockFFN.

    Orchestrates:
    - Model patching with adaptive attention layers
    - Verification with confidence metrics
    - Escalation when confidence is low

    Usage:
        engine = AdaptiveInferenceEngine(model, tokenizer, config)
        output = engine.generate("Hello, world!", max_tokens=50)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[AdaptiveInferenceConfig] = None,
    ):
        """
        Initialize adaptive inference engine.

        Args:
            model: BlockFFN model (AutoModelForCausalLM)
            tokenizer: Tokenizer for the model
            config: Configuration (uses "balanced" preset if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or PRESETS["balanced"]

        # Determine device
        self.device = next(model.parameters()).device

        # Initialize components
        self.confidence_computer = ConfidenceComputer(self.config)
        self.decision_maker = EscalationDecisionMaker(self.config)
        self.verification_scheduler = VerificationScheduler(self.config)
        self.escalation_module: Optional[EscalationModule] = None

        # Attach adaptive attention layers
        self._attach_adaptive_layers()

        # Configure verification scheduler for model layer count
        num_layers = len(model.model.layers)
        self.verification_scheduler.configure_for_model(num_layers)

        # Initialize escalation module after layers are attached
        self.escalation_module = EscalationModule(model, self.config)
        self.batch_escalator = BatchEscalationExecutor(model, self.config)

        # Statistics tracking
        self.stats = GenerationStats()
        self._confidence_history: List[float] = []
        self._sparsity_history: List[float] = []

    def _attach_adaptive_layers(self) -> None:
        """Attach adaptive attention and CLS FFN layers to the model."""
        # Attach adaptive attention
        attach_adaptive_attention(self.model, self.config)

        # Attach CLS-aware FFN if enabled
        if self.config.enable_cls_optimization:
            attach_cls_ffn(self.model, self.config)

    def detach_adaptive_layers(self) -> None:
        """Remove adaptive attention and CLS FFN layers, restore original model."""
        detach_adaptive_attention(self.model)
        detach_cls_ffn(self.model)
        self.escalation_module = None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = False,
        return_stats: bool = False,
    ) -> Union[str, Tuple[str, GenerationStats]]:
        """
        Generate text with adaptive inference.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample (vs greedy)
            return_stats: Whether to return generation statistics

        Returns:
            Generated text, optionally with statistics
        """
        # Reset statistics
        self.stats = GenerationStats()
        self._confidence_history = []
        self._sparsity_history = []
        self.decision_maker.reset_all()
        clear_all_statistics(self.model)

        start_time = time.time()

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Prefill: process the full prompt once
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            # Use prefill logits for the first generated token
            logits = outputs.logits[:, -1, :]

        generated_tokens = []

        # Generate tokens
        for step in range(max_new_tokens):
            # Scheduled verification
            step_position = len(generated_tokens)
            verification_result = self.verification_scheduler.compute_confidence(
                logits.unsqueeze(1),
                token_position=step_position,
                confidence_computer=self.confidence_computer,
            )

            if verification_result.should_verify and verification_result.confidence is not None:
                confidence_val = verification_result.confidence
                self._confidence_history.append(confidence_val)

                # Make escalation decision
                decision = self.decision_maker.decide(
                    confidence_val,
                    position=step_position,
                )

                # Record for stride inheritance
                self.verification_scheduler.record_result(
                    confidence_val, decision.should_escalate
                )
            else:
                # Use skipped confidence value (from stride inheritance or default)
                confidence_val = verification_result.confidence if verification_result.confidence is not None else 1.0
                self._confidence_history.append(confidence_val)

                # Default decision: no escalation for skipped tokens
                from src.verification.decision import EscalationDecision
                decision = EscalationDecision(
                    level=EscalationLevel.NONE,
                    reason=f"Skipped ({verification_result.skipped_reason})",
                    confidence=confidence_val,
                    position=step_position,
                )

            # Handle escalation
            if decision.should_escalate:
                self.stats.escalated_tokens += 1
                self.stats.escalation_by_level[decision.level] += 1

                # Recompute with full attention
                full_input = torch.cat([
                    input_ids,
                    torch.tensor([generated_tokens], device=self.device) if generated_tokens else torch.empty(1, 0, dtype=torch.long, device=self.device)
                ], dim=1)

                logits = self.escalation_module.escalate(
                    full_input,
                    level=decision.level,
                    position=full_input.shape[1] - 1,
                    attention_mask=attention_mask,
                )
            else:
                self.stats.reduced_compute_tokens += 1

            self.stats.total_tokens += 1

            # Sample next token
            if do_sample and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = logits.argmax(dim=-1)

            next_token_id = next_token.item()
            generated_tokens.append(next_token_id)

            # Check for EOS
            if next_token_id == self.tokenizer.eos_token_id:
                break

            # Extend attention mask BEFORE next forward pass
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(1, 1, device=self.device, dtype=attention_mask.dtype)
                ], dim=1)

            # Forward pass: process the newly generated token
            current_input = torch.tensor(
                [[next_token_id]], device=self.device,
            )
            with torch.no_grad():
                outputs = self.model(
                    current_input,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

        # Finalize statistics
        self.stats.wall_time_seconds = time.time() - start_time
        if self._confidence_history:
            self.stats.mean_confidence = sum(self._confidence_history) / len(self._confidence_history)

        # Collect layer statistics
        layer_stats = get_all_statistics(self.model)
        if layer_stats:
            sparsities = [s.get("mean_sparsity", 0) for s in layer_stats.values()]
            if sparsities:
                self.stats.mean_sparsity = sum(sparsities) / len(sparsities)

        # Collect CLS statistics
        if self.config.enable_cls_optimization:
            cls_stats = get_all_cls_statistics(self.model)
            if cls_stats:
                reuse_rates = [s.get("reuse_rate", 0) for s in cls_stats.values()]
                savings = [s.get("compute_savings", 0) for s in cls_stats.values()]
                chunks = [s.get("chunks_processed", 0) for s in cls_stats.values()]
                if reuse_rates:
                    self.stats.cls_expert_reuse_rate = sum(reuse_rates) / len(reuse_rates)
                if savings:
                    self.stats.cls_compute_savings = sum(savings) / len(savings)
                if chunks:
                    self.stats.cls_chunks_processed = sum(chunks)

        # Decode output
        full_tokens = input_ids[0].tolist() + generated_tokens
        output_text = self.tokenizer.decode(full_tokens, skip_special_tokens=True)

        if return_stats:
            return output_text, self.stats
        return output_text

    def process_batch(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        verify: bool = True,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Process a batch with adaptive inference.

        For prefill/prompt processing (not generation).

        Args:
            input_ids: Input token IDs [batch, seq]
            attention_mask: Attention mask [batch, seq]
            verify: Whether to run verification and escalation

        Returns:
            logits: Output logits [batch, seq, vocab]
            info: Dict with processing information
        """
        start_time = time.time()

        # Forward pass with adaptive attention
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
            )

        logits = outputs.logits

        if not verify:
            return logits, {"verified": False}

        # Compute confidence for all tokens
        confidence = self.confidence_computer.compute_confidence(logits)

        # Identify escalations
        from src.verification.decision import BatchEscalationTracker
        tracker = BatchEscalationTracker(self.config)
        escalation_info = tracker.identify_escalations(confidence)

        # Execute escalations if needed
        if escalation_info["pct_escalated"] > 0:
            groups = tracker.group_escalations(escalation_info)
            logits = self.batch_escalator.execute_escalations(
                input_ids, logits, groups, attention_mask
            )

        processing_time = time.time() - start_time

        return logits, {
            "verified": True,
            "pct_escalated": escalation_info["pct_escalated"],
            "escalation_mask": escalation_info["escalation_mask"],
            "confidence_stats": get_confidence_stats(confidence),
            "processing_time": processing_time,
        }

    def get_layer_statistics(self) -> Dict[int, Dict[str, float]]:
        """Get statistics from all adaptive attention layers."""
        return get_all_statistics(self.model)

    def get_cls_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics from all CLS-aware FFN layers."""
        return get_all_cls_statistics(self.model)

    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get statistics from the decision maker."""
        return self.decision_maker.get_stats()

    def get_all_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components."""
        return {
            "generation": {
                "total_tokens": self.stats.total_tokens,
                "reduced_compute_rate": self.stats.reduced_compute_rate,
                "escalation_rate": self.stats.escalation_rate,
                "tokens_per_second": self.stats.tokens_per_second,
                "mean_confidence": self.stats.mean_confidence,
                "mean_sparsity": self.stats.mean_sparsity,
            },
            "attention_layers": self.get_layer_statistics(),
            "cls_layers": self.get_cls_statistics(),
            "decisions": self.get_decision_statistics(),
            "cls_summary": {
                "expert_reuse_rate": self.stats.cls_expert_reuse_rate,
                "compute_savings": self.stats.cls_compute_savings,
                "chunks_processed": self.stats.cls_chunks_processed,
            },
            "verification_scheduling": self.verification_scheduler.get_stats(),
            "combined_savings_estimate": self.stats.combined_savings_estimate,
        }

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.stats = GenerationStats()
        self._confidence_history = []
        self._sparsity_history = []
        self.decision_maker.reset_all()
        self.verification_scheduler.reset()
        clear_all_statistics(self.model)
        if self.config.enable_cls_optimization:
            reset_all_cls_statistics(self.model)


def create_adaptive_engine(
    model_name: str = "SparseLLM/BlockFFN-3B-SFT",
    preset: str = "balanced",
    device: Optional[str] = None,
    **config_overrides,
) -> AdaptiveInferenceEngine:
    """
    Convenience function to create an adaptive inference engine.

    Args:
        model_name: HuggingFace model name or path
        preset: Configuration preset name
        device: Target device (auto-detect if None)
        **config_overrides: Override specific config values

    Returns:
        Configured AdaptiveInferenceEngine
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Load model and tokenizer
    print(f"Loading model {model_name} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Get config from preset
    config = PRESETS[preset].for_device(device)

    # Apply overrides
    for key, value in config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Create engine
    engine = AdaptiveInferenceEngine(model, tokenizer, config)

    print(f"Created adaptive engine with preset '{preset}'")
    print(f"  Target layers: {config.get_target_layer_indices(len(model.model.layers))}")
    print(f"  Window sizes: small={config.small_window}, medium={config.medium_window}")
    print(f"  Thresholds: green={config.tau_green}, yellow={config.tau_yellow}")
    print(f"  CLS optimization: {'enabled' if config.enable_cls_optimization else 'disabled'}")
    if config.enable_cls_optimization:
        print(f"    Chunk size: {config.chunk_size}, overlap threshold: {config.overlap_threshold}")
    # Show verification scheduling info
    if config.verification_stride > 1:
        print(f"  Verification stride: every {config.verification_stride} tokens")
    if config.verification_layer_ratio < 1.0:
        print(f"  Verification layers: last {config.verification_layer_ratio*100:.0f}%")
    if config.enable_cheap_gate:
        print(f"  Cheap gate: enabled (high={config.cheap_gate_high_threshold}, low={config.cheap_gate_low_threshold})")

    return engine


def print_stats_summary(stats: GenerationStats) -> None:
    """Print a human-readable summary of generation statistics."""
    print("\n" + "=" * 60)
    print("ADAPTIVE INFERENCE STATISTICS")
    print("=" * 60)
    print(f"Tokens generated: {stats.total_tokens}")
    print(f"Wall time: {stats.wall_time_seconds:.2f}s ({stats.tokens_per_second:.1f} tok/s)")
    print()
    print("ATTENTION OPTIMIZATION:")
    print(f"  Reduced compute tokens: {stats.reduced_compute_tokens} ({stats.reduced_compute_rate*100:.1f}%)")
    print(f"  Escalated tokens: {stats.escalated_tokens} ({stats.escalation_rate*100:.1f}%)")
    print(f"  Mean confidence: {stats.mean_confidence:.3f}")
    print(f"  Mean sparsity: {stats.mean_sparsity:.3f}")
    print()
    print("FFN/CLS OPTIMIZATION:")
    print(f"  Expert reuse rate: {stats.cls_expert_reuse_rate*100:.1f}%")
    print(f"  Compute savings: {stats.cls_compute_savings*100:.1f}%")
    print(f"  Chunks processed: {stats.cls_chunks_processed}")
    print()
    print(f"COMBINED SAVINGS ESTIMATE: {stats.combined_savings_estimate*100:.1f}%")
    print("=" * 60)
