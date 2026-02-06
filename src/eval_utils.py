"""
Evaluation Utilities for BlockFFN + Mamba Hybrid.

This module provides:
- Perplexity computation on WikiText and other datasets
- Text generation helpers
- Memory tracking for GPU experiments
- Throughput benchmarking
"""

from typing import Optional, List, Dict, Any, Union
from contextlib import contextmanager
import time
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm


class MemoryTracker:
    """
    Track GPU memory usage during experiments.

    Usage:
        tracker = MemoryTracker()
        tracker.snapshot("before_load")
        model = load_model()
        tracker.snapshot("after_load")
        print(tracker.report())
    """

    def __init__(self):
        self.snapshots: List[Dict[str, Any]] = []
        self.cuda_available = torch.cuda.is_available()

    def snapshot(self, label: str):
        """
        Take a memory snapshot with the given label.

        Args:
            label: Descriptive name for this snapshot
        """
        if self.cuda_available:
            torch.cuda.synchronize()
            self.snapshots.append({
                "label": label,
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                "timestamp": time.time(),
            })
        else:
            self.snapshots.append({
                "label": label,
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "max_allocated_gb": 0.0,
                "timestamp": time.time(),
            })

    def reset_peak(self):
        """Reset peak memory statistics."""
        if self.cuda_available:
            torch.cuda.reset_peak_memory_stats()

    def clear(self):
        """Clear all snapshots."""
        self.snapshots.clear()

    def report(self) -> str:
        """Generate a formatted memory report."""
        if not self.snapshots:
            return "No snapshots recorded."

        lines = ["Memory Usage Report:", "-" * 60]
        for snap in self.snapshots:
            lines.append(
                f"{snap['label']:30s} | "
                f"Alloc: {snap['allocated_gb']:.2f} GB | "
                f"Reserved: {snap['reserved_gb']:.2f} GB | "
                f"Peak: {snap['max_allocated_gb']:.2f} GB"
            )
        lines.append("-" * 60)
        return "\n".join(lines)

    def get_delta(self, label1: str, label2: str) -> float:
        """Get memory delta between two snapshots in GB."""
        snap1 = next((s for s in self.snapshots if s["label"] == label1), None)
        snap2 = next((s for s in self.snapshots if s["label"] == label2), None)
        if snap1 and snap2:
            return snap2["allocated_gb"] - snap1["allocated_gb"]
        return 0.0


@contextmanager
def memory_efficient_inference(model: nn.Module):
    """
    Context manager for memory-efficient inference.

    Disables gradients and uses autocast for reduced memory.
    """
    original_training = model.training
    model.set_train_mode = lambda x: None  # Avoid issues
    was_training = original_training
    model.train(False)

    try:
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    yield
            else:
                yield
    finally:
        model.train(was_training)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compute_perplexity(
    model: nn.Module,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_samples: int = 100,
    max_length: int = 512,
    batch_size: int = 1,
    stride: int = 256,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Compute perplexity on a dataset.

    Uses a sliding window approach to handle long sequences.

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration
        split: Dataset split to use
        max_samples: Maximum samples to process
        max_length: Maximum sequence length
        batch_size: Batch size for processing
        stride: Stride for sliding window
        show_progress: Show progress bar

    Returns:
        Dict with "perplexity" and "loss" keys
    """
    from datasets import load_dataset

    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    # Concatenate all text
    all_text = "\n\n".join(dataset["text"][:max_samples])

    # Tokenize
    encodings = tokenizer(all_text, return_tensors="pt")
    input_ids = encodings["input_ids"]

    # Get device
    device = next(model.parameters()).device

    # Sliding window perplexity
    seq_len = input_ids.size(1)
    nlls = []

    iterator = range(0, seq_len, stride)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing perplexity")

    model.train(False)
    with torch.no_grad():
        for begin_idx in iterator:
            end_idx = min(begin_idx + max_length, seq_len)
            target_len = end_idx - begin_idx

            # Get input chunk
            input_chunk = input_ids[:, begin_idx:end_idx].to(device)

            # Create labels (shifted by 1)
            target_chunk = input_chunk.clone()

            # Forward pass
            outputs = model(input_chunk, labels=target_chunk)
            neg_log_likelihood = outputs.loss * target_len

            nlls.append(neg_log_likelihood.item())

            if end_idx == seq_len:
                break

    # Compute perplexity
    total_loss = sum(nlls) / (seq_len - 1)
    perplexity = torch.exp(torch.tensor(total_loss)).item()

    return {
        "perplexity": perplexity,
        "loss": total_loss,
        "num_tokens": seq_len,
    }


def generate_text(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
) -> str:
    """
    Generate text continuation from a prompt.

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        prompt: Starting text
        max_new_tokens: Maximum tokens to generate
        do_sample: Use sampling (vs greedy)
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling

    Returns:
        Generated text (including prompt)
    """
    device = next(model.parameters()).device

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    # Generate
    model.train(False)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def benchmark_throughput(
    model: nn.Module,
    tokenizer,
    batch_size: int = 1,
    seq_lengths: List[int] = [128, 256, 512],
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark model throughput at various sequence lengths.

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        batch_size: Batch size for benchmarking
        seq_lengths: Sequence lengths to test
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of timed iterations
        show_progress: Show progress bar

    Returns:
        Dict with throughput metrics per sequence length
    """
    device = next(model.parameters()).device
    results = {}

    model.train(False)

    for seq_len in seq_lengths:
        # Create dummy input
        input_ids = torch.randint(
            100, 10000,
            (batch_size, seq_len),
            device=device,
        )

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_ids)

        # Synchronize before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        times = []
        iterator = range(benchmark_runs)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Benchmarking seq_len={seq_len}")

        with torch.no_grad():
            for _ in iterator:
                start = time.perf_counter()
                _ = model(input_ids)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)

        # Compute metrics
        avg_time = sum(times) / len(times)
        total_tokens = batch_size * seq_len
        tokens_per_second = total_tokens / avg_time

        results[seq_len] = {
            "avg_time_ms": avg_time * 1000,
            "tokens_per_second": tokens_per_second,
            "batch_size": batch_size,
            "total_tokens": total_tokens,
        }

    return results


def compare_outputs(
    model1: nn.Module,
    model2: nn.Module,
    tokenizer,
    prompts: List[str],
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> Dict[str, Any]:
    """
    Compare outputs of two models for the same inputs.

    Useful for verifying that modifications (like adding Mamba sidecar with
    alpha=0) don't change model behavior.

    Args:
        model1: First model
        model2: Second model
        tokenizer: Shared tokenizer
        prompts: List of prompts to test
        atol: Absolute tolerance for torch.allclose
        rtol: Relative tolerance for torch.allclose

    Returns:
        Dict with comparison results per prompt
    """
    device1 = next(model1.parameters()).device
    device2 = next(model2.parameters()).device

    results = {
        "prompts": [],
        "all_close": True,
        "max_diff": 0.0,
    }

    model1.train(False)
    model2.train(False)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids1 = inputs["input_ids"].to(device1)
        input_ids2 = inputs["input_ids"].to(device2)

        with torch.no_grad():
            out1 = model1(input_ids1)
            out2 = model2(input_ids2)

        # Compare logits
        logits1 = out1.logits.cpu()
        logits2 = out2.logits.cpu()

        is_close = torch.allclose(logits1, logits2, atol=atol, rtol=rtol)
        max_diff = (logits1 - logits2).abs().max().item()

        results["prompts"].append({
            "prompt": prompt,
            "is_close": is_close,
            "max_diff": max_diff,
        })

        if not is_close:
            results["all_close"] = False
        results["max_diff"] = max(results["max_diff"], max_diff)

    return results


def print_generation_comparison(
    generations: Dict[str, str],
    prompt: str,
):
    """
    Print a formatted comparison of generations from different models/configs.

    Args:
        generations: Dict mapping config name to generated text
        prompt: The original prompt
    """
    print("=" * 80)
    print(f"Prompt: {prompt!r}")
    print("=" * 80)

    for config_name, text in generations.items():
        # Extract only the generated part
        if text.startswith(prompt):
            generated = text[len(prompt):]
        else:
            generated = text

        print(f"\n[{config_name}]")
        print(f"  {generated!r}")

    print("=" * 80)


def save_results_csv(
    results: List[Dict[str, Any]],
    filepath: str,
    fieldnames: Optional[List[str]] = None,
):
    """
    Save experiment results to a CSV file.

    Args:
        results: List of result dicts
        filepath: Output file path
        fieldnames: Column names (auto-detected if None)
    """
    import csv

    if not results:
        print(f"No results to save to {filepath}")
        return

    if fieldnames is None:
        fieldnames = list(results[0].keys())

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved results to {filepath}")


def load_model_safe(
    model_name: str,
    cache_dir: Optional[str] = None,
    device: str = "auto",
    dtype: str = "float16",
    use_8bit: bool = False,
) -> tuple:
    """
    Load a model with error handling and memory optimizations.

    Args:
        model_name: HuggingFace model name
        cache_dir: Local cache directory
        device: Device specification ("auto", "cuda", "cpu")
        dtype: Data type ("float16", "float32", "bfloat16")
        use_8bit: Use 8-bit quantization

    Returns:
        Tuple of (model, tokenizer, status_dict)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    status = {
        "model_loaded": False,
        "dtype": dtype,
        "device": device,
        "quantized": use_8bit,
        "warnings": [],
    }

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    # Load model
    load_kwargs = {
        "cache_dir": cache_dir,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if use_8bit:
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto"
        status["warnings"].append("Using 8-bit quantization")
    else:
        load_kwargs["torch_dtype"] = torch_dtype
        if device == "auto":
            load_kwargs["device_map"] = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        status["model_loaded"] = True

        # Move to device if not using device_map
        if device not in ["auto"] and "device_map" not in load_kwargs:
            model = model.to(device)

        status["device"] = str(next(model.parameters()).device)

    except RuntimeError as e:
        if "out of memory" in str(e).lower() and not use_8bit:
            status["warnings"].append("OOM, retrying with 8-bit quantization")
            return load_model_safe(
                model_name, cache_dir, device, dtype, use_8bit=True
            )
        raise

    return model, tokenizer, status
