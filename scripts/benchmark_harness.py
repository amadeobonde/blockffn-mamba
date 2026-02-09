#!/usr/bin/env python3
"""
BlockFFN Adaptive Inference Benchmarking Harness

Comprehensive benchmarking CLI that sweeps configurations and outputs
detailed performance metrics in CSV format.

Usage:
    python scripts/benchmark_harness.py --output results.csv
    python scripts/benchmark_harness.py --preset balanced --batch-sizes 1,2,4
    python scripts/benchmark_harness.py --resume checkpoint.json
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator
from itertools import product

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adaptive.config import AdaptiveInferenceConfig, PRESETS
from src.engine.adaptive_engine import AdaptiveInferenceEngine
from src.attention.adaptive_attention import get_all_statistics
from src.ffn.cls_ffn import get_all_cls_statistics
from src.verification.decision import EscalationLevel


# =============================================================================
# Data Classes for Metrics
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Single benchmark configuration point."""
    batch_size: int
    context_length: int
    preset: str
    num_tokens: int = 50
    warmup_runs: int = 2
    benchmark_runs: int = 5


@dataclass
class LatencyMetrics:
    """Latency measurements from a benchmark run."""
    ttft_ms: float                    # Time to first token
    p50_token_latency_ms: float       # Median per-token latency
    p95_token_latency_ms: float       # 95th percentile latency
    p99_token_latency_ms: float       # 99th percentile latency
    total_time_ms: float              # Total generation time
    tokens_per_second: float          # Overall throughput


@dataclass
class ComputeProxyMetrics:
    """Proxy metrics for FLOPs/compute savings."""
    # Attention metrics
    attention_flops_ratio: float      # Adaptive / Full attention cost
    pct_small_window: float           # % tokens using small window
    pct_medium_window: float          # % tokens using medium window
    pct_full_window: float            # % tokens using full attention
    mean_window_size: float           # Average window size used
    theoretical_attention_savings: float  # 1 - cost_ratio

    # FFN/CLS metrics
    cls_expert_reuse_rate: float      # Expert computation reuse
    cls_compute_savings: float        # Estimated FFN savings
    cls_chunks_processed: int         # Total chunks analyzed


@dataclass
class EscalationMetrics:
    """Escalation statistics."""
    overall_escalation_rate: float
    layer_escalation_count: int
    token_escalation_count: int
    suffix_escalation_count: int
    full_context_escalation_count: int
    mean_confidence: float


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""
    peak_memory_mb: float
    allocated_memory_mb: float
    reserved_memory_mb: float


@dataclass
class BenchmarkResult:
    """Complete result from a single benchmark run."""
    # Configuration
    batch_size: int
    context_length: int
    preset: str
    device: str
    timestamp: str

    # Metrics
    latency: LatencyMetrics
    compute_proxy: ComputeProxyMetrics
    escalation: EscalationMetrics
    memory: MemoryMetrics

    # Metadata
    num_tokens_generated: int
    seed: int
    model_name: str


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """
    Main benchmark runner class.

    Handles model loading, warmup, timing, and metric collection.
    """

    def __init__(
        self,
        model_name: str = "SparseLLM/BlockFFN-3B-SFT",
        device: str = "auto",
        seed: int = 42,
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.seed = seed
        self.verbose = verbose

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """Load model and tokenizer."""
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.verbose:
            print(f"Loading {self.model_name} on {self.device}...")

        dtype = torch.float32 if self.device == "cpu" else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)

        if self.verbose:
            print("Model loaded.")

    def _generate_test_prompt(self, target_length: int) -> str:
        """Generate a prompt targeting a specific token length."""
        # Base prompts of varying complexity
        base_prompts = [
            "The quick brown fox jumps over the lazy dog. ",
            "In machine learning, neural networks are computational systems. ",
            "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2) ",
        ]

        # Build up to target length
        prompt = ""
        idx = 0
        while True:
            test_prompt = prompt + base_prompts[idx % len(base_prompts)]
            test_tokens = self.tokenizer.encode(test_prompt)
            if len(test_tokens) >= target_length:
                break
            prompt = test_prompt
            idx += 1

        return prompt

    def _measure_ttft(
        self,
        engine: AdaptiveInferenceEngine,
        prompt: str,
    ) -> Tuple[float, int]:
        """
        Measure Time To First Token (TTFT).

        TTFT = time from starting generation to first token emitted.
        This is essentially the prefill time.

        Returns:
            (ttft_ms, first_token_id)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Synchronize before timing
        if self.device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        # Prefill pass
        with torch.no_grad():
            outputs = engine.model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        # Get first token
        logits = outputs.logits[:, -1, :]
        first_token = logits.argmax(dim=-1).item()

        if self.device == "cuda":
            torch.cuda.synchronize()

        ttft_ms = (time.perf_counter() - start_time) * 1000

        return ttft_ms, first_token

    def _measure_token_latencies(
        self,
        engine: AdaptiveInferenceEngine,
        prompt: str,
        num_tokens: int,
    ) -> List[float]:
        """
        Measure per-token generation latencies.

        Returns list of latencies in milliseconds for each token after the first.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Prefill
        with torch.no_grad():
            outputs = engine.model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

        latencies = []
        generated_tokens = []

        # Generate tokens and measure each
        for step in range(num_tokens):
            if generated_tokens:
                current_input = torch.tensor(
                    [[generated_tokens[-1]]],
                    device=self.device,
                )
            else:
                current_input = input_ids[:, -1:]

            if self.device == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            with torch.no_grad():
                outputs = engine.model(
                    current_input,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            logits = outputs.logits[:, -1, :]
            next_token = logits.argmax(dim=-1).item()
            past_key_values = outputs.past_key_values

            if self.device == "cuda":
                torch.cuda.synchronize()

            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)
            generated_tokens.append(next_token)

            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(1, 1, device=self.device, dtype=attention_mask.dtype)
                ], dim=1)

            # Check EOS
            if next_token == self.tokenizer.eos_token_id:
                break

        return latencies

    def _compute_attention_flops_proxy(
        self,
        engine: AdaptiveInferenceEngine,
        context_length: int,
    ) -> Dict[str, float]:
        """
        Compute attention FLOPs proxy from window statistics.

        The proxy is based on:
        - Window size distribution across tokens
        - Effective attention cost ratio = sum(window_sizes) / full_attention_cost
        """
        layer_stats = engine.get_layer_statistics()

        if not layer_stats:
            return {
                "attention_flops_ratio": 1.0,
                "pct_small_window": 0.0,
                "pct_medium_window": 0.0,
                "pct_full_window": 1.0,
                "mean_window_size": float(context_length),
                "theoretical_attention_savings": 0.0,
            }

        # Aggregate across layers
        pct_small = []
        pct_medium = []
        pct_full = []
        mean_windows = []

        config = engine.config

        for layer_idx, stats in layer_stats.items():
            pct_small.append(stats.get("pct_small_window", 0))
            pct_medium.append(stats.get("pct_medium_window", 0))
            pct_full.append(stats.get("pct_full_window", 1.0))
            mean_windows.append(stats.get("mean_window", context_length))

        avg_pct_small = sum(pct_small) / len(pct_small) if pct_small else 0
        avg_pct_medium = sum(pct_medium) / len(pct_medium) if pct_medium else 0
        avg_pct_full = sum(pct_full) / len(pct_full) if pct_full else 1.0
        avg_mean_window = sum(mean_windows) / len(mean_windows) if mean_windows else context_length

        # Compute effective attention cost ratio
        small_cost = config.small_window / context_length
        medium_cost = config.medium_window / context_length

        effective_cost = (
            avg_pct_small * small_cost +
            avg_pct_medium * medium_cost +
            avg_pct_full * 1.0
        )

        return {
            "attention_flops_ratio": effective_cost,
            "pct_small_window": avg_pct_small,
            "pct_medium_window": avg_pct_medium,
            "pct_full_window": avg_pct_full,
            "mean_window_size": avg_mean_window,
            "theoretical_attention_savings": 1.0 - effective_cost,
        }

    def _compute_cls_metrics(
        self,
        engine: AdaptiveInferenceEngine,
    ) -> Dict[str, Any]:
        """Collect FFN/CLS statistics."""
        cls_stats = engine.get_cls_statistics()

        if not cls_stats:
            return {
                "cls_expert_reuse_rate": 0.0,
                "cls_compute_savings": 0.0,
                "cls_chunks_processed": 0,
            }

        reuse_rates = []
        savings = []
        chunks = 0

        for layer_idx, stats in cls_stats.items():
            reuse_rates.append(stats.get("reuse_rate", 0))
            savings.append(stats.get("compute_savings", 0))
            chunks += stats.get("chunks_processed", 0)

        return {
            "cls_expert_reuse_rate": sum(reuse_rates) / len(reuse_rates) if reuse_rates else 0,
            "cls_compute_savings": sum(savings) / len(savings) if savings else 0,
            "cls_chunks_processed": chunks,
        }

    def _get_escalation_metrics(
        self,
        engine: AdaptiveInferenceEngine,
    ) -> Dict[str, Any]:
        """Extract escalation statistics."""
        stats = engine.stats
        decision_stats = engine.get_decision_statistics()

        counts_by_level = decision_stats.get("counts_by_level", {})

        return {
            "overall_escalation_rate": stats.escalation_rate,
            "layer_escalation_count": counts_by_level.get("LAYER", 0),
            "token_escalation_count": counts_by_level.get("TOKEN", 0),
            "suffix_escalation_count": counts_by_level.get("SUFFIX", 0),
            "full_context_escalation_count": counts_by_level.get("FULL_CONTEXT", 0),
            "mean_confidence": stats.mean_confidence,
        }

    def _get_memory_metrics(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        if self.device == "cuda":
            return {
                "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
                "allocated_memory_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "reserved_memory_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            }
        else:
            try:
                import psutil
                process = psutil.Process()
                return {
                    "peak_memory_mb": process.memory_info().rss / 1024 / 1024,
                    "allocated_memory_mb": 0.0,
                    "reserved_memory_mb": 0.0,
                }
            except ImportError:
                return {
                    "peak_memory_mb": 0.0,
                    "allocated_memory_mb": 0.0,
                    "reserved_memory_mb": 0.0,
                }

    def run_single_benchmark(
        self,
        config: BenchmarkConfig,
    ) -> BenchmarkResult:
        """
        Run a single benchmark configuration.

        Args:
            config: Benchmark configuration

        Returns:
            BenchmarkResult with all metrics
        """
        self.load_model()

        # Set seed for reproducibility
        torch.manual_seed(self.seed)
        if self.device == "cuda":
            torch.cuda.manual_seed(self.seed)
            torch.cuda.reset_peak_memory_stats()

        # Create engine with specified preset
        adaptive_config = AdaptiveInferenceConfig.from_preset(config.preset)
        adaptive_config = adaptive_config.for_device(self.device)
        adaptive_config.enable_cls_optimization = True
        adaptive_config.collect_statistics = True

        engine = AdaptiveInferenceEngine(
            self.model,
            self.tokenizer,
            adaptive_config,
        )

        # Generate test prompt
        prompt = self._generate_test_prompt(config.context_length)

        # Warmup runs
        for _ in range(config.warmup_runs):
            engine.reset_statistics()
            _ = engine.generate(
                prompt,
                max_new_tokens=config.num_tokens,
                do_sample=False,
            )

        # Benchmark runs
        ttft_samples = []
        all_latencies = []
        total_times = []

        for run in range(config.benchmark_runs):
            engine.reset_statistics()

            # Measure TTFT
            ttft, _ = self._measure_ttft(engine, prompt)
            ttft_samples.append(ttft)

            # Measure token latencies
            engine.reset_statistics()
            latencies = self._measure_token_latencies(
                engine, prompt, config.num_tokens
            )
            all_latencies.extend(latencies)
            total_times.append(ttft + sum(latencies))

        # Final run to collect statistics
        engine.reset_statistics()
        _, stats = engine.generate(
            prompt,
            max_new_tokens=config.num_tokens,
            do_sample=False,
            return_stats=True,
        )

        # Compute metrics
        sorted_latencies = sorted(all_latencies)
        n = len(sorted_latencies)

        latency_metrics = LatencyMetrics(
            ttft_ms=sum(ttft_samples) / len(ttft_samples),
            p50_token_latency_ms=sorted_latencies[n // 2] if n > 0 else 0,
            p95_token_latency_ms=sorted_latencies[int(n * 0.95)] if n > 0 else 0,
            p99_token_latency_ms=sorted_latencies[int(n * 0.99)] if n > 0 else 0,
            total_time_ms=sum(total_times) / len(total_times),
            tokens_per_second=config.num_tokens / (sum(total_times) / len(total_times) / 1000) if total_times else 0,
        )

        attention_metrics = self._compute_attention_flops_proxy(
            engine, config.context_length
        )
        cls_metrics = self._compute_cls_metrics(engine)
        escalation_metrics = self._get_escalation_metrics(engine)
        memory_metrics = self._get_memory_metrics()

        # Cleanup
        engine.detach_adaptive_layers()
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return BenchmarkResult(
            batch_size=config.batch_size,
            context_length=config.context_length,
            preset=config.preset,
            device=self.device,
            timestamp=datetime.now().isoformat(),
            latency=latency_metrics,
            compute_proxy=ComputeProxyMetrics(
                attention_flops_ratio=attention_metrics["attention_flops_ratio"],
                pct_small_window=attention_metrics["pct_small_window"],
                pct_medium_window=attention_metrics["pct_medium_window"],
                pct_full_window=attention_metrics["pct_full_window"],
                mean_window_size=attention_metrics["mean_window_size"],
                theoretical_attention_savings=attention_metrics["theoretical_attention_savings"],
                cls_expert_reuse_rate=cls_metrics["cls_expert_reuse_rate"],
                cls_compute_savings=cls_metrics["cls_compute_savings"],
                cls_chunks_processed=cls_metrics["cls_chunks_processed"],
            ),
            escalation=EscalationMetrics(
                overall_escalation_rate=escalation_metrics["overall_escalation_rate"],
                layer_escalation_count=escalation_metrics["layer_escalation_count"],
                token_escalation_count=escalation_metrics["token_escalation_count"],
                suffix_escalation_count=escalation_metrics["suffix_escalation_count"],
                full_context_escalation_count=escalation_metrics["full_context_escalation_count"],
                mean_confidence=escalation_metrics["mean_confidence"],
            ),
            memory=MemoryMetrics(**memory_metrics),
            num_tokens_generated=stats.total_tokens,
            seed=self.seed,
            model_name=self.model_name,
        )


# =============================================================================
# Sweep Orchestrator
# =============================================================================

class BenchmarkSweep:
    """
    Orchestrates sweeping across multiple configurations.

    Supports checkpointing for partial runs.
    """

    DEFAULT_BATCH_SIZES = [1, 2, 4, 8]
    DEFAULT_CONTEXT_LENGTHS = [128, 256, 512, 1024, 2048]
    DEFAULT_PRESETS = ["conservative", "balanced", "aggressive"]

    def __init__(
        self,
        runner: BenchmarkRunner,
        batch_sizes: Optional[List[int]] = None,
        context_lengths: Optional[List[int]] = None,
        presets: Optional[List[str]] = None,
        checkpoint_file: Optional[str] = None,
    ):
        self.runner = runner
        self.batch_sizes = batch_sizes or self.DEFAULT_BATCH_SIZES
        self.context_lengths = context_lengths or self.DEFAULT_CONTEXT_LENGTHS
        self.presets = presets or self.DEFAULT_PRESETS
        self.checkpoint_file = checkpoint_file

        self.completed_configs: set = set()
        self.results: List[BenchmarkResult] = []

        if checkpoint_file and os.path.exists(checkpoint_file):
            self._load_checkpoint()

    def _config_key(self, config: BenchmarkConfig) -> str:
        """Create unique key for a configuration."""
        return f"{config.batch_size}_{config.context_length}_{config.preset}"

    def _load_checkpoint(self) -> None:
        """Load progress from checkpoint file."""
        with open(self.checkpoint_file, "r") as f:
            data = json.load(f)
            self.completed_configs = set(data.get("completed", []))
            print(f"Resumed from checkpoint: {len(self.completed_configs)} configs completed")

    def _save_checkpoint(self) -> None:
        """Save progress to checkpoint file."""
        if not self.checkpoint_file:
            return
        with open(self.checkpoint_file, "w") as f:
            json.dump({
                "completed": list(self.completed_configs),
                "timestamp": datetime.now().isoformat(),
            }, f)

    def generate_configs(self) -> Iterator[BenchmarkConfig]:
        """Generate all configuration combinations."""
        for batch_size, ctx_len, preset in product(
            self.batch_sizes, self.context_lengths, self.presets
        ):
            config = BenchmarkConfig(
                batch_size=batch_size,
                context_length=ctx_len,
                preset=preset,
            )

            # Skip if already completed
            if self._config_key(config) in self.completed_configs:
                continue

            yield config

    def run_sweep(
        self,
        output_file: str,
        progress_callback: Optional[callable] = None,
    ) -> List[BenchmarkResult]:
        """
        Run the full benchmark sweep.

        Args:
            output_file: Path to output CSV file
            progress_callback: Optional callback(current, total, result)

        Returns:
            List of all benchmark results
        """
        configs = list(self.generate_configs())
        total = len(configs) + len(self.completed_configs)
        current = len(self.completed_configs)

        print(f"\nBenchmark sweep: {len(configs)} configurations remaining")
        print(f"  Batch sizes: {self.batch_sizes}")
        print(f"  Context lengths: {self.context_lengths}")
        print(f"  Presets: {self.presets}")
        print(f"  Output: {output_file}\n")

        # Open CSV file (append mode if resuming)
        file_exists = os.path.exists(output_file) and self.completed_configs

        with open(output_file, "a" if file_exists else "w", newline="") as f:
            writer = None

            for config in configs:
                current += 1
                print(f"[{current}/{total}] batch={config.batch_size}, "
                      f"ctx={config.context_length}, preset={config.preset}")

                try:
                    result = self.runner.run_single_benchmark(config)
                    self.results.append(result)

                    # Write to CSV
                    row = self._result_to_row(result)
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=row.keys())
                        if not file_exists:
                            writer.writeheader()
                    writer.writerow(row)
                    f.flush()

                    # Mark completed
                    self.completed_configs.add(self._config_key(config))
                    self._save_checkpoint()

                    if progress_callback:
                        progress_callback(current, total, result)

                    print(f"  -> TTFT={result.latency.ttft_ms:.1f}ms, "
                          f"p50={result.latency.p50_token_latency_ms:.1f}ms, "
                          f"tok/s={result.latency.tokens_per_second:.1f}")

                except Exception as e:
                    print(f"  -> ERROR: {e}")
                    continue

        return self.results

    def _result_to_row(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Flatten BenchmarkResult to CSV row."""
        return {
            # Configuration
            "batch_size": result.batch_size,
            "context_length": result.context_length,
            "preset": result.preset,
            "device": result.device,
            "model_name": result.model_name,
            "seed": result.seed,
            "timestamp": result.timestamp,

            # Latency
            "ttft_ms": result.latency.ttft_ms,
            "p50_token_latency_ms": result.latency.p50_token_latency_ms,
            "p95_token_latency_ms": result.latency.p95_token_latency_ms,
            "p99_token_latency_ms": result.latency.p99_token_latency_ms,
            "total_time_ms": result.latency.total_time_ms,
            "tokens_per_second": result.latency.tokens_per_second,

            # Attention FLOPs proxy
            "attention_flops_ratio": result.compute_proxy.attention_flops_ratio,
            "pct_small_window": result.compute_proxy.pct_small_window,
            "pct_medium_window": result.compute_proxy.pct_medium_window,
            "pct_full_window": result.compute_proxy.pct_full_window,
            "mean_window_size": result.compute_proxy.mean_window_size,
            "theoretical_attention_savings": result.compute_proxy.theoretical_attention_savings,

            # FFN/CLS proxy
            "cls_expert_reuse_rate": result.compute_proxy.cls_expert_reuse_rate,
            "cls_compute_savings": result.compute_proxy.cls_compute_savings,
            "cls_chunks_processed": result.compute_proxy.cls_chunks_processed,

            # Escalation
            "overall_escalation_rate": result.escalation.overall_escalation_rate,
            "layer_escalation_count": result.escalation.layer_escalation_count,
            "token_escalation_count": result.escalation.token_escalation_count,
            "suffix_escalation_count": result.escalation.suffix_escalation_count,
            "full_context_escalation_count": result.escalation.full_context_escalation_count,
            "mean_confidence": result.escalation.mean_confidence,

            # Memory
            "peak_memory_mb": result.memory.peak_memory_mb,
            "allocated_memory_mb": result.memory.allocated_memory_mb,

            # Tokens
            "num_tokens_generated": result.num_tokens_generated,
        }


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="BlockFFN Adaptive Inference Benchmarking Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full sweep with default settings
  python scripts/benchmark_harness.py --output results.csv

  # Custom configuration
  python scripts/benchmark_harness.py \\
      --batch-sizes 1,2,4 \\
      --context-lengths 256,512,1024 \\
      --presets balanced,aggressive \\
      --output custom_results.csv

  # Resume from checkpoint
  python scripts/benchmark_harness.py \\
      --output results.csv \\
      --resume checkpoint.json

  # Single configuration
  python scripts/benchmark_harness.py \\
      --batch-sizes 1 \\
      --context-lengths 512 \\
      --presets balanced \\
      --output single_test.csv
        """,
    )

    # Required
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output CSV file path",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        default="SparseLLM/BlockFFN-3B-SFT",
        help="Model name or path (default: SparseLLM/BlockFFN-3B-SFT)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run on (default: auto)",
    )

    # Sweep configuration
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8",
        help="Comma-separated batch sizes (default: 1,2,4,8)",
    )
    parser.add_argument(
        "--context-lengths",
        type=str,
        default="128,256,512,1024,2048",
        help="Comma-separated context lengths (default: 128,256,512,1024,2048)",
    )
    parser.add_argument(
        "--presets",
        type=str,
        default="conservative,balanced,aggressive",
        help="Comma-separated presets (default: conservative,balanced,aggressive)",
    )

    # Benchmark parameters
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=50,
        help="Number of tokens to generate per run (default: 50)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Number of warmup runs (default: 2)",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=5,
        help="Number of benchmark runs (default: 5)",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode (may be slower)",
    )

    # Checkpointing
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file for resuming partial runs",
    )

    # Output options
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Parse configuration lists
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    context_lengths = [int(x.strip()) for x in args.context_lengths.split(",")]
    presets = [x.strip() for x in args.presets.split(",")]

    # Validate presets
    valid_presets = list(PRESETS.keys())
    for preset in presets:
        if preset not in valid_presets:
            print(f"Error: Unknown preset '{preset}'. Valid: {valid_presets}")
            sys.exit(1)

    # Set up deterministic mode
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)

    # Create runner and sweep
    runner = BenchmarkRunner(
        model_name=args.model,
        device=args.device,
        seed=args.seed,
        verbose=not args.quiet,
    )

    sweep = BenchmarkSweep(
        runner=runner,
        batch_sizes=batch_sizes,
        context_lengths=context_lengths,
        presets=presets,
        checkpoint_file=args.resume,
    )

    # Run sweep
    print("\n" + "=" * 70)
    print("BlockFFN Adaptive Inference Benchmark")
    print("=" * 70)

    start_time = time.time()
    results = sweep.run_sweep(args.output)
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"COMPLETE: {len(results)} configurations benchmarked")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Output: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
