#!/usr/bin/env python3
"""
Script 02: Inspect BlockFFN Model Architecture

This is the most important exploration script. It prints the full layer
structure to identify:
- Where the attention module is
- Where the MLP/FFN is
- Where the router is located
- Key weight matrix shapes

This information is needed before we can attach Mamba sidecars or extract
routing signals.

Usage:
    python scripts/02_inspect_model.py
    python scripts/02_inspect_model.py --layer 0  # Inspect specific layer
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODEL_NAME, MODEL_CACHE_DIR, DEVICE


def print_module_tree(module, prefix="", max_depth=3, current_depth=0):
    """Recursively print module structure."""
    if current_depth >= max_depth:
        return

    for name, child in module.named_children():
        child_type = type(child).__name__
        num_params = sum(p.numel() for p in child.parameters(recurse=False))

        # Highlight router-related modules
        highlight = ""
        if any(kw in name.lower() for kw in ["router", "gate", "expert"]):
            highlight = " <-- ROUTER/GATE"
        elif "mlp" in name.lower():
            highlight = " <-- MLP"
        elif "attn" in name.lower() or "attention" in name.lower():
            highlight = " <-- ATTENTION"

        print(f"{prefix}{name}: {child_type} (params: {num_params:,}){highlight}")
        print_module_tree(child, prefix + "  ", max_depth, current_depth + 1)


def inspect_layer_detailed(layer, layer_idx):
    """Print detailed information about a single layer."""
    print(f"\n{'=' * 60}")
    print(f"DETAILED INSPECTION: Layer {layer_idx}")
    print("=" * 60)

    print(f"\nLayer type: {type(layer).__name__}")

    # List all attributes
    print("\nDirect attributes:")
    for name in dir(layer):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(layer, name)
            if callable(attr) and not isinstance(attr, type):
                continue
            if hasattr(attr, "shape"):
                print(f"  {name}: Tensor{list(attr.shape)}")
            elif hasattr(attr, "weight"):
                print(f"  {name}: {type(attr).__name__} (has weight)")
        except Exception:
            pass

    # Print full module structure
    print("\nFull module tree:")
    print_module_tree(layer, prefix="  ", max_depth=5)

    # Look for router specifically
    print("\nSearching for router components:")
    for name, module in layer.named_modules():
        name_lower = name.lower()
        if any(kw in name_lower for kw in ["router", "gate", "expert", "block"]):
            print(f"  Found: {name} ({type(module).__name__})")
            if hasattr(module, "weight"):
                print(f"    weight shape: {module.weight.shape}")

    # Examine MLP if present
    if hasattr(layer, "mlp"):
        mlp = layer.mlp
        print(f"\nMLP module: {type(mlp).__name__}")
        print("MLP children:")
        for name, child in mlp.named_children():
            print(f"  {name}: {type(child).__name__}")
            if hasattr(child, "weight"):
                print(f"    weight: {child.weight.shape}")

        # Look for router_act_fn (the key hook point)
        if hasattr(mlp, "router_act_fn"):
            print(f"\n  router_act_fn: {mlp.router_act_fn}")
            print("  ^ This is where we capture post-ReLU routing signals")
        if hasattr(mlp, "router_proj"):
            print(f"  router_proj: {type(mlp.router_proj).__name__}")
            if hasattr(mlp.router_proj, "weight"):
                print(f"    weight: {mlp.router_proj.weight.shape}")
                print(f"    -> num_experts likely: {mlp.router_proj.weight.shape[0]}")
        if hasattr(mlp, "router_norm"):
            print(f"  router_norm: {type(mlp.router_norm).__name__}")


def main():
    parser = argparse.ArgumentParser(description="Inspect BlockFFN model architecture")
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Which layer to inspect in detail (default: 0)",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Show summary for all layers",
    )
    parser.add_argument(
        "--weights",
        action="store_true",
        help="Show weight shapes for all named parameters",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BlockFFN Model Architecture Inspection")
    print("=" * 60)

    # Import and load model
    print("\nLoading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=str(MODEL_CACHE_DIR),
        trust_remote_code=True,
    )

    # Load on CPU to save memory during inspection
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=str(MODEL_CACHE_DIR),
        trust_remote_code=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )

    print(f"\nModel loaded: {type(model).__name__}")

    # Top-level structure
    print("\n" + "=" * 60)
    print("TOP-LEVEL STRUCTURE")
    print("=" * 60)
    print_module_tree(model, max_depth=2)

    # Model config
    print("\n" + "=" * 60)
    print("KEY CONFIGURATION VALUES")
    print("=" * 60)
    config = model.config
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    if hasattr(config, "num_key_value_heads"):
        print(f"  num_key_value_heads: {config.num_key_value_heads}")
    if hasattr(config, "intermediate_size"):
        print(f"  intermediate_size: {config.intermediate_size}")

    # Check for MoE-specific config
    moe_attrs = ["num_experts", "num_experts_per_tok", "moe_intermediate_size"]
    print("\nMoE configuration (if present):")
    for attr in moe_attrs:
        if hasattr(config, attr):
            print(f"  {attr}: {getattr(config, attr)}")

    # Layer structure
    num_layers = len(model.model.layers)
    print(f"\n" + "=" * 60)
    print(f"LAYER STRUCTURE ({num_layers} layers)")
    print("=" * 60)

    if args.all_layers:
        for i in range(num_layers):
            layer = model.model.layers[i]
            layer_type = type(layer).__name__
            mlp_type = type(layer.mlp).__name__ if hasattr(layer, "mlp") else "N/A"
            print(f"  Layer {i:2d}: {layer_type} | MLP: {mlp_type}")
    else:
        print(f"  (Use --all-layers to show all {num_layers} layers)")

    # Detailed inspection of one layer
    if 0 <= args.layer < num_layers:
        inspect_layer_detailed(model.model.layers[args.layer], args.layer)
    else:
        print(f"\nInvalid layer index: {args.layer} (max: {num_layers - 1})")

    # Weight shapes
    if args.weights:
        print("\n" + "=" * 60)
        print("ALL WEIGHT SHAPES")
        print("=" * 60)
        for name, param in model.named_parameters():
            print(f"  {name}: {list(param.shape)}")

    # Summary and next steps
    print("\n" + "=" * 60)
    print("SUMMARY FOR MAMBA INTEGRATION")
    print("=" * 60)

    # Try to auto-detect router location
    sample_layer = model.model.layers[0]
    router_found = False
    if hasattr(sample_layer, "mlp"):
        mlp = sample_layer.mlp
        if hasattr(mlp, "router_act_fn"):
            print("\nRouter location detected:")
            print("  model.model.layers[i].mlp.router_act_fn")
            print("  -> Wrap this to capture post-ReLU routing signals")
            router_found = True
        if hasattr(mlp, "router_proj"):
            proj = mlp.router_proj
            if hasattr(proj, "weight"):
                num_experts = proj.weight.shape[0]
                print(f"\n  Number of experts: {num_experts}")
                print(f"  Routing signal shape will be: [batch*seq, {num_experts}]")

    if not router_found:
        print("\nWARNING: Could not auto-detect router location.")
        print("You may need to manually inspect the model to find the router.")

    print(f"\nHidden size: {config.hidden_size}")
    print("  -> This is the d_model for Mamba sidecar")

    print("\n" + "=" * 60)
    print("Next step: Run 03_extract_routing.py to capture routing signals")
    print("=" * 60)


if __name__ == "__main__":
    main()
