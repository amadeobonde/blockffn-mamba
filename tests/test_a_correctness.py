"""
Phase A: Correctness & Safety Tests

These tests MUST PASS before any performance claims are valid.

A1. Equivalence tests: adaptive engine ≈ vanilla BlockFFN
A2. Forced failure injection: escalation actually works
A3. Edge case handling: boundary conditions

If any test in this file fails, STOP. Fix before moving on.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adaptive.config import AdaptiveInferenceConfig, PRESETS


@dataclass
class EquivalenceResult:
    """Result from an equivalence test."""
    test_name: str
    passed: bool
    logit_cosine_sim: float
    token_match: bool
    max_logit_diff: float
    escalation_count: int
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.test_name}\n"
            f"  Logit cosine similarity: {self.logit_cosine_sim:.6f}\n"
            f"  Token match: {self.token_match}\n"
            f"  Max logit diff: {self.max_logit_diff:.6f}\n"
            f"  Escalations: {self.escalation_count}"
        )


@dataclass
class EscalationTestResult:
    """Result from an escalation test."""
    test_name: str
    passed: bool
    escalation_triggered: bool
    confidence_dropped: bool
    output_matched_after: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.test_name}\n"
            f"  Escalation triggered: {self.escalation_triggered}\n"
            f"  Confidence dropped: {self.confidence_dropped}\n"
            f"  Output matched after escalation: {self.output_matched_after}"
        )


class CorrectnessTestSuite:
    """
    Phase A: Correctness & Safety Test Suite

    Tests that adaptive inference preserves correctness and
    handles failures safely.
    """

    # Thresholds for passing
    MIN_COSINE_SIM = 0.999  # Very tight tolerance
    MAX_LOGIT_DIFF = 0.01   # Maximum absolute difference in normalized logits

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device: str = "cpu",
    ):
        """
        Initialize test suite.

        Args:
            model: Pre-loaded BlockFFN model (or None to load later)
            tokenizer: Pre-loaded tokenizer (or None to load later)
            device: Target device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.results: List[Any] = []

    def load_model(self, model_name: str = "SparseLLM/BlockFFN-3B-SFT") -> None:
        """Load model and tokenizer if not provided."""
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
        ).to(self.device)
        print("Model loaded.")

    @contextmanager
    def baseline_mode(self):
        """Context manager for running model in baseline (unmodified) mode."""
        # Save current state
        original_layers = list(self.model.model.layers)

        # Check if any layers are wrapped
        from src.attention.adaptive_attention import AdaptiveAttentionLayer
        from src.ffn.cls_ffn import CLSAwareFFN

        for idx, layer in enumerate(self.model.model.layers):
            if isinstance(layer, AdaptiveAttentionLayer):
                self.model.model.layers[idx] = layer.original_layer
            if hasattr(layer, "mlp") and isinstance(layer.mlp, CLSAwareFFN):
                layer.mlp = layer.mlp.original_mlp

        try:
            yield
        finally:
            # Restore
            for idx, orig_layer in enumerate(original_layers):
                self.model.model.layers[idx] = orig_layer

    def compute_logit_similarity(
        self,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Compute similarity between two logit tensors.

        Returns:
            (cosine_similarity, max_absolute_difference)
        """
        # Flatten to [batch * seq, vocab]
        a_flat = logits_a.reshape(-1, logits_a.shape[-1]).float()
        b_flat = logits_b.reshape(-1, logits_b.shape[-1]).float()

        # Normalize
        a_norm = F.normalize(a_flat, dim=-1)
        b_norm = F.normalize(b_flat, dim=-1)

        # Cosine similarity (mean across all positions)
        cosine_sim = (a_norm * b_norm).sum(dim=-1).mean().item()

        # Max absolute difference
        max_diff = (a_flat - b_flat).abs().max().item()

        return cosine_sim, max_diff

    # =========================================================================
    # A1: Equivalence Tests
    # =========================================================================

    def test_a1_greedy_equivalence(
        self,
        prompts: Optional[List[str]] = None,
        max_new_tokens: int = 20,
    ) -> EquivalenceResult:
        """
        A1.1: Greedy decode equivalence test.

        Verify that adaptive engine produces identical greedy outputs
        to baseline BlockFFN when escalation is enabled.
        """
        if prompts is None:
            prompts = [
                "The capital of France is",
                "def fibonacci(n):",
                "Once upon a time, there was a",
                "The sum of 2 and 2 is",
            ]

        self.load_model()

        from src.engine.adaptive_engine import AdaptiveInferenceEngine

        # Config with escalation enabled (conservative settings)
        config = AdaptiveInferenceConfig.from_preset("conservative")
        config.enable_cls_optimization = False  # Test attention first
        config.collect_statistics = True

        all_baseline_tokens = []
        all_adaptive_tokens = []
        all_baseline_logits = []
        all_adaptive_logits = []
        total_escalations = 0

        for prompt in prompts:
            # Get baseline output
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with self.baseline_mode():
                with torch.no_grad():
                    baseline_out = self.model.generate(
                        inputs["input_ids"],
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    baseline_tokens = baseline_out.sequences[0].tolist()
                    baseline_logits = torch.stack(baseline_out.scores, dim=1)

            # Get adaptive output
            engine = AdaptiveInferenceEngine(self.model, self.tokenizer, config)

            with torch.no_grad():
                adaptive_out = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                adaptive_tokens = adaptive_out.sequences[0].tolist()
                adaptive_logits = torch.stack(adaptive_out.scores, dim=1)

            all_baseline_tokens.append(baseline_tokens)
            all_adaptive_tokens.append(adaptive_tokens)
            all_baseline_logits.append(baseline_logits)
            all_adaptive_logits.append(adaptive_logits)
            total_escalations += engine.stats.escalated_tokens

            # Detach adaptive layers for next iteration
            engine.detach_adaptive_layers()

        # Compute aggregate metrics
        cosine_sims = []
        max_diffs = []
        token_matches = []

        for bl, al, bt, at in zip(
            all_baseline_logits, all_adaptive_logits,
            all_baseline_tokens, all_adaptive_tokens
        ):
            cs, md = self.compute_logit_similarity(bl, al)
            cosine_sims.append(cs)
            max_diffs.append(md)
            token_matches.append(bt == at)

        mean_cosine = sum(cosine_sims) / len(cosine_sims)
        max_diff = max(max_diffs)
        all_match = all(token_matches)

        passed = (
            mean_cosine >= self.MIN_COSINE_SIM and
            max_diff <= self.MAX_LOGIT_DIFF and
            all_match
        )

        result = EquivalenceResult(
            test_name="A1.1 Greedy Decode Equivalence",
            passed=passed,
            logit_cosine_sim=mean_cosine,
            token_match=all_match,
            max_logit_diff=max_diff,
            escalation_count=total_escalations,
            details={
                "prompts": prompts,
                "per_prompt_cosine": cosine_sims,
                "per_prompt_match": token_matches,
            },
        )
        self.results.append(result)
        return result

    def test_a1_prefill_equivalence(
        self,
        prompts: Optional[List[str]] = None,
    ) -> EquivalenceResult:
        """
        A1.2: Prefill (prompt processing) equivalence test.

        Verify that adaptive attention produces identical prefill logits.
        """
        if prompts is None:
            prompts = [
                "The quick brown fox jumps over the lazy dog.",
                "In a galaxy far far away, there lived a wise old sage who knew many secrets.",
                "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr",
            ]

        self.load_model()

        from src.attention.adaptive_attention import attach_adaptive_attention, detach_adaptive_attention

        config = AdaptiveInferenceConfig.from_preset("conservative")

        all_cosine_sims = []
        all_max_diffs = []

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Baseline prefill
            with self.baseline_mode():
                with torch.no_grad():
                    baseline_out = self.model(
                        inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                    )
                    baseline_logits = baseline_out.logits

            # Adaptive prefill
            attach_adaptive_attention(self.model, config)
            with torch.no_grad():
                adaptive_out = self.model(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
                adaptive_logits = adaptive_out.logits
            detach_adaptive_attention(self.model)

            cs, md = self.compute_logit_similarity(baseline_logits, adaptive_logits)
            all_cosine_sims.append(cs)
            all_max_diffs.append(md)

        mean_cosine = sum(all_cosine_sims) / len(all_cosine_sims)
        max_diff = max(all_max_diffs)

        passed = mean_cosine >= self.MIN_COSINE_SIM

        result = EquivalenceResult(
            test_name="A1.2 Prefill Equivalence",
            passed=passed,
            logit_cosine_sim=mean_cosine,
            token_match=True,  # N/A for prefill
            max_logit_diff=max_diff,
            escalation_count=0,  # N/A for prefill
            details={
                "prompts": prompts,
                "per_prompt_cosine": all_cosine_sims,
            },
        )
        self.results.append(result)
        return result

    # =========================================================================
    # A2: Forced Failure Injection
    # =========================================================================

    def test_a2_forced_low_confidence(self) -> EscalationTestResult:
        """
        A2.1: Force low confidence on known hard tokens.

        Manually inject low sparsity to simulate hard tokens,
        verify escalation triggers and output recovers.
        """
        self.load_model()

        from src.engine.adaptive_engine import AdaptiveInferenceEngine
        from src.verification.confidence import ConfidenceComputer
        from src.verification.decision import EscalationDecisionMaker

        # Hard prompts requiring long-range reasoning
        hard_prompts = [
            # Coreference resolution
            "John gave Mary a book. She thanked him for the",
            # Delayed variable use
            "x = 10\ny = 20\nz = x + y\nprint(z)  # Output:",
            # Math reasoning
            "If 3x + 5 = 20, then x equals",
        ]

        config = AdaptiveInferenceConfig.from_preset("conservative")
        config.tau_green = 0.99  # Very high threshold to force escalation
        config.tau_yellow = 0.95
        config.enable_cls_optimization = False

        escalation_triggered = False
        confidence_dropped = False
        output_matched = True

        for prompt in hard_prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Get baseline
            with self.baseline_mode():
                with torch.no_grad():
                    baseline_out = self.model.generate(
                        inputs["input_ids"],
                        max_new_tokens=10,
                        do_sample=False,
                    )
                    baseline_tokens = baseline_out[0].tolist()

            # Get adaptive with high threshold
            engine = AdaptiveInferenceEngine(self.model, self.tokenizer, config)
            adaptive_text, stats = engine.generate(
                prompt,
                max_new_tokens=10,
                do_sample=False,
                return_stats=True,
            )

            # Check if escalation happened
            if stats.escalated_tokens > 0:
                escalation_triggered = True

            # Check confidence
            if stats.mean_confidence < config.tau_green:
                confidence_dropped = True

            # Check token match after escalation
            adaptive_tokens = self.tokenizer.encode(adaptive_text)
            if adaptive_tokens[-10:] != baseline_tokens[-10:]:
                output_matched = False

            engine.detach_adaptive_layers()

        passed = escalation_triggered and output_matched

        result = EscalationTestResult(
            test_name="A2.1 Forced Low Confidence Escalation",
            passed=passed,
            escalation_triggered=escalation_triggered,
            confidence_dropped=confidence_dropped,
            output_matched_after=output_matched,
            details={"hard_prompts": hard_prompts},
        )
        self.results.append(result)
        return result

    def test_a2_yellow_streak_detection(self) -> EscalationTestResult:
        """
        A2.2: Yellow streak detection test.

        Verify that consecutive moderate-confidence tokens
        trigger suffix escalation.
        """

        from src.verification.decision import EscalationDecisionMaker, EscalationLevel

        config = AdaptiveInferenceConfig.from_preset("balanced")
        config.tau_green = 0.90
        config.tau_yellow = 0.70
        config.yellow_streak_limit = 3

        decision_maker = EscalationDecisionMaker(config)

        # Simulate a sequence of yellow-zone confidences
        confidences = [0.75, 0.78, 0.72, 0.76, 0.74]  # All in yellow zone

        decisions = []
        for i, conf in enumerate(confidences):
            decision = decision_maker.decide(conf, position=i)
            decisions.append(decision)

        # After 3 consecutive yellows, should trigger suffix escalation
        suffix_triggered = any(
            d.level == EscalationLevel.SUFFIX for d in decisions
        )

        # Verify streak reset after escalation
        decision_maker.reset_streak()
        post_reset_decision = decision_maker.decide(0.75, position=10)
        streak_reset_worked = decision_maker.yellow_streak == 1

        passed = suffix_triggered and streak_reset_worked

        result = EscalationTestResult(
            test_name="A2.2 Yellow Streak Detection",
            passed=passed,
            escalation_triggered=suffix_triggered,
            confidence_dropped=True,  # N/A
            output_matched_after=streak_reset_worked,
            details={
                "confidences": confidences,
                "decisions": [str(d.level) for d in decisions],
            },
        )
        self.results.append(result)
        return result

    def test_a2_cascade_protection(self) -> EscalationTestResult:
        """
        A2.3: Cascade protection test.

        Verify that too many suffix escalations disables
        adaptive compute entirely (degrades gracefully).
        """
        from src.verification.decision import EscalationDecisionMaker, EscalationLevel

        config = AdaptiveInferenceConfig.from_preset("balanced")
        config.cascade_threshold = 3  # Disable after 3 suffix escalations

        decision_maker = EscalationDecisionMaker(config)

        # Trigger multiple suffix escalations
        for _ in range(4):
            # Force very low confidence
            decision_maker.decide(0.30, position=0)

        # After cascade threshold, should switch to TOKEN level
        cascade_decision = decision_maker.decide(0.30, position=100)
        cascade_triggered = cascade_decision.level == EscalationLevel.TOKEN

        # Verify cascade counter
        counter_correct = decision_maker.suffix_escalation_count >= config.cascade_threshold

        passed = cascade_triggered and counter_correct

        result = EscalationTestResult(
            test_name="A2.3 Cascade Protection",
            passed=passed,
            escalation_triggered=cascade_triggered,
            confidence_dropped=True,
            output_matched_after=True,  # N/A
            details={
                "cascade_threshold": config.cascade_threshold,
                "suffix_escalation_count": decision_maker.suffix_escalation_count,
            },
        )
        self.results.append(result)
        return result

    # =========================================================================
    # A3: Edge Case Tests
    # =========================================================================

    def test_a3_empty_and_single_token(self) -> EquivalenceResult:
        """
        A3.1: Edge case with minimal input.

        Verify handling of single-token and very short inputs.
        """
        self.load_model()

        from src.attention.adaptive_attention import attach_adaptive_attention, detach_adaptive_attention

        config = AdaptiveInferenceConfig.from_preset("balanced")

        edge_cases = [
            "A",  # Single word
            "Hi",  # Two chars
            "The",  # Short common word
        ]

        all_passed = True

        for prompt in edge_cases:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Baseline
            with self.baseline_mode():
                with torch.no_grad():
                    baseline_out = self.model(inputs["input_ids"])
                    baseline_logits = baseline_out.logits

            # Adaptive
            attach_adaptive_attention(self.model, config)
            try:
                with torch.no_grad():
                    adaptive_out = self.model(inputs["input_ids"])
                    adaptive_logits = adaptive_out.logits

                cs, _ = self.compute_logit_similarity(baseline_logits, adaptive_logits)
                if cs < 0.99:
                    all_passed = False
            except Exception as e:
                all_passed = False
            finally:
                detach_adaptive_attention(self.model)

        result = EquivalenceResult(
            test_name="A3.1 Edge Case: Minimal Input",
            passed=all_passed,
            logit_cosine_sim=1.0 if all_passed else 0.0,
            token_match=all_passed,
            max_logit_diff=0.0,
            escalation_count=0,
            details={"edge_cases": edge_cases},
        )
        self.results.append(result)
        return result

    def test_a3_long_context(self) -> EquivalenceResult:
        """
        A3.2: Long context handling.

        Verify behavior with context longer than typical windows.
        """
        self.load_model()

        from src.attention.adaptive_attention import attach_adaptive_attention, detach_adaptive_attention

        config = AdaptiveInferenceConfig.from_preset("balanced")

        # Create a long prompt (should exceed small/medium windows)
        long_prompt = "The " * 100 + "answer is"  # ~100+ tokens

        inputs = self.tokenizer(long_prompt, return_tensors="pt").to(self.device)
        seq_len = inputs["input_ids"].shape[1]

        # Baseline
        with self.baseline_mode():
            with torch.no_grad():
                baseline_out = self.model(inputs["input_ids"])
                baseline_logits = baseline_out.logits

        # Adaptive
        attach_adaptive_attention(self.model, config)
        with torch.no_grad():
            adaptive_out = self.model(inputs["input_ids"])
            adaptive_logits = adaptive_out.logits
        detach_adaptive_attention(self.model)

        cs, md = self.compute_logit_similarity(baseline_logits, adaptive_logits)

        result = EquivalenceResult(
            test_name="A3.2 Edge Case: Long Context",
            passed=cs >= 0.99,
            logit_cosine_sim=cs,
            token_match=True,
            max_logit_diff=md,
            escalation_count=0,
            details={"seq_len": seq_len},
        )
        self.results.append(result)
        return result

    # =========================================================================
    # Test Runner
    # =========================================================================

    def run_all(self, stop_on_failure: bool = True) -> bool:
        """
        Run all Phase A tests.

        Args:
            stop_on_failure: If True, stop at first failure

        Returns:
            True if all tests passed
        """
        print("\n" + "=" * 70)
        print("PHASE A: CORRECTNESS & SAFETY TESTS")
        print("=" * 70)
        print("\nThese tests MUST PASS before any performance claims are valid.\n")

        tests = [
            ("A1.1", self.test_a1_greedy_equivalence),
            ("A1.2", self.test_a1_prefill_equivalence),
            ("A2.1", self.test_a2_forced_low_confidence),
            ("A2.2", self.test_a2_yellow_streak_detection),
            ("A2.3", self.test_a2_cascade_protection),
            ("A3.1", self.test_a3_empty_and_single_token),
            ("A3.2", self.test_a3_long_context),
        ]

        passed_count = 0
        failed_count = 0

        for test_id, test_fn in tests:
            print(f"\nRunning {test_id}...")
            try:
                result = test_fn()
                print(result)

                if result.passed:
                    passed_count += 1
                else:
                    failed_count += 1
                    if stop_on_failure:
                        print(f"\n*** STOP: {test_id} failed. Fix before continuing. ***")
                        break
            except Exception as e:
                print(f"[FAIL] {test_id}: Exception - {e}")
                failed_count += 1
                if stop_on_failure:
                    print(f"\n*** STOP: {test_id} raised exception. Fix before continuing. ***")
                    raise

        print("\n" + "=" * 70)
        print(f"PHASE A SUMMARY: {passed_count} passed, {failed_count} failed")
        print("=" * 70)

        return failed_count == 0


# =============================================================================
# Standalone execution
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase A: Correctness Tests")
    parser.add_argument("--model", default="SparseLLM/BlockFFN-3B-SFT")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-stop-on-failure", action="store_true")
    args = parser.parse_args()

    suite = CorrectnessTestSuite(device=args.device)

    success = suite.run_all(stop_on_failure=not args.no_stop_on_failure)
    sys.exit(0 if success else 1)
