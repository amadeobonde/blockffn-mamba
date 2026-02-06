"""
Routing Signal Extraction for BlockFFN.

This module captures post-ReLU, pre-RMSNorm routing activations from BlockFFN's
MLP layers. These signals indicate which experts are active for each token and
can be used to gate between Mamba and attention.

BlockFFN Router Architecture:
    1. router_proj: Linear projection [hidden_size] -> [num_experts]
    2. router_act_fn: ReLU activation (produces sparse, binary-ish signals)
    3. router_norm: RMSNorm (scales magnitudes)

We capture after step 2 (post-ReLU) to get the raw sparsity pattern before
normalization affects the signal.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Callable, Any
import torch
from torch import Tensor
import warnings


class RoutingHook:
    """
    Manages forward hooks for capturing routing signals.

    This class handles the registration, storage, and cleanup of PyTorch
    forward hooks used to intercept routing activations.
    """

    def __init__(self):
        self.activations: Dict[str, List[Tensor]] = defaultdict(list)
        self.handles: List[Any] = []  # RemovableHandle objects
        self._enabled = True

    def make_hook(self, layer_name: str) -> Callable:
        """
        Create a hook function that captures activations.

        Args:
            layer_name: Identifier for this layer's activations

        Returns:
            Hook function compatible with register_forward_hook
        """
        def hook(module, input, output):
            if self._enabled:
                # Detach and move to CPU to avoid GPU memory accumulation
                self.activations[layer_name].append(output.detach().cpu())
        return hook

    def enable(self):
        """Enable activation capture."""
        self._enabled = True

    def disable(self):
        """Disable activation capture (hooks remain registered)."""
        self._enabled = False

    def clear(self):
        """Clear all stored activations."""
        self.activations.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def get_latest(self, layer_name: str) -> Optional[Tensor]:
        """Get the most recent activation for a layer."""
        if layer_name in self.activations and self.activations[layer_name]:
            return self.activations[layer_name][-1]
        return None

    def get_all(self) -> Dict[str, List[Tensor]]:
        """Get all stored activations."""
        return dict(self.activations)


class RoutingExtractor:
    """
    High-level interface for extracting routing signals from BlockFFN.

    This class wraps the router activation function in each MLP layer to
    capture the post-ReLU routing scores before RMSNorm is applied.

    Usage:
        model = load_blockffn_model()
        extractor = RoutingExtractor(model)

        # Run forward pass
        with torch.no_grad():
            outputs = model(input_ids)

        # Get routing signals
        signals = extractor.get_routing_signals()
        stats = extractor.compute_sparsity_stats(signals)
    """

    def __init__(
        self,
        model,
        target_layers: Optional[List[int]] = None,
        store_history: bool = False,
    ):
        """
        Initialize the routing extractor.

        Args:
            model: BlockFFN model with model.model.layers structure
            target_layers: Specific layer indices to capture (None = all)
            store_history: If True, keep all activations; if False, keep only latest
        """
        self.model = model
        self.store_history = store_history
        self.hook_manager = RoutingHook()
        self._captured_values: Dict[int, Dict[str, Tensor]] = {}
        self._original_act_fns: Dict[int, Callable] = {}

        # Determine target layers
        if target_layers is None:
            self.target_layers = list(range(len(model.model.layers)))
        else:
            self.target_layers = target_layers

        # Set up capturing wrappers
        self._setup_captures()

    def _setup_captures(self):
        """
        Inject capturing wrappers around router_act_fn in target layers.

        BlockFFN's MLP uses router_act_fn (ReLU) after router_proj.
        We wrap this function to capture its output before router_norm.
        """
        for layer_idx in self.target_layers:
            try:
                mlp = self.model.model.layers[layer_idx].mlp

                # Check if this MLP has the expected router structure
                if not hasattr(mlp, 'router_act_fn'):
                    warnings.warn(
                        f"Layer {layer_idx} MLP has no router_act_fn. "
                        "Model architecture may differ from expected BlockMLP."
                    )
                    continue

                # Store original activation function
                original_act = mlp.router_act_fn
                self._original_act_fns[layer_idx] = original_act

                # Create storage for this layer
                self._captured_values[layer_idx] = {"value": None}
                captured = self._captured_values[layer_idx]

                # Create capturing wrapper
                def make_wrapper(orig_fn, cap_dict, idx):
                    def wrapper(x):
                        result = orig_fn(x)
                        # Store the post-ReLU values
                        cap_dict["value"] = result.detach()
                        return result
                    return wrapper

                # Replace the activation function with our wrapper
                mlp.router_act_fn = make_wrapper(original_act, captured, layer_idx)

            except AttributeError as e:
                warnings.warn(
                    f"Could not set up capture for layer {layer_idx}: {e}. "
                    "The model structure may differ from expected."
                )

    def get_routing_signals(self) -> Dict[int, Tensor]:
        """
        Get all captured routing signals after a forward pass.

        Returns:
            Dict mapping layer indices to routing tensors.
            Shape of each tensor: [batch*seq, num_experts] or [batch, seq, num_experts]
        """
        signals = {}
        for layer_idx, captured in self._captured_values.items():
            if captured["value"] is not None:
                signals[layer_idx] = captured["value"]
        return signals

    def get_routing_signals_reshaped(
        self,
        batch_size: int,
        seq_len: int,
    ) -> Dict[int, Tensor]:
        """
        Get routing signals reshaped to [batch, seq, num_experts].

        Args:
            batch_size: Batch size of the input
            seq_len: Sequence length of the input

        Returns:
            Dict mapping layer indices to tensors of shape [batch, seq, num_experts]
        """
        signals = self.get_routing_signals()
        reshaped = {}
        for layer_idx, tensor in signals.items():
            if tensor.dim() == 2:
                # [batch*seq, num_experts] -> [batch, seq, num_experts]
                num_experts = tensor.shape[-1]
                reshaped[layer_idx] = tensor.view(batch_size, seq_len, num_experts)
            else:
                reshaped[layer_idx] = tensor
        return reshaped

    def compute_sparsity_stats(
        self,
        signals: Optional[Dict[int, Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Compute sparsity statistics for routing signals.

        Args:
            signals: Routing signals (uses latest captured if None)

        Returns:
            Dict with sparsity metrics:
                - per_layer: Dict[layer_idx, float] - sparsity per layer
                - overall: float - average sparsity across all layers
                - per_token: Tensor - sparsity per token position
        """
        if signals is None:
            signals = self.get_routing_signals()

        if not signals:
            return {"per_layer": {}, "overall": 0.0, "per_token": None}

        per_layer = {}
        all_sparsities = []

        for layer_idx, tensor in signals.items():
            # Sparsity = fraction of zeros
            sparsity = (tensor == 0).float().mean().item()
            per_layer[layer_idx] = sparsity
            all_sparsities.append(sparsity)

        # Per-token sparsity (using first layer as reference)
        first_layer_idx = min(signals.keys())
        first_tensor = signals[first_layer_idx]
        per_token = (first_tensor == 0).float().mean(dim=-1)

        return {
            "per_layer": per_layer,
            "overall": sum(all_sparsities) / len(all_sparsities) if all_sparsities else 0.0,
            "per_token": per_token,
        }

    def compute_importance_scores(
        self,
        signals: Optional[Dict[int, Tensor]] = None,
    ) -> Dict[int, Tensor]:
        """
        Convert routing signals to importance scores.

        "Important" tokens activate more experts (lower sparsity).
        Score is 1 - sparsity, so higher = more important.

        Args:
            signals: Routing signals (uses latest captured if None)

        Returns:
            Dict mapping layer indices to importance tensors [batch*seq] or [batch, seq]
        """
        if signals is None:
            signals = self.get_routing_signals()

        importance = {}
        for layer_idx, tensor in signals.items():
            # Importance = 1 - sparsity = fraction of non-zero experts
            # Average across expert dimension
            sparsity = (tensor == 0).float().mean(dim=-1)
            importance[layer_idx] = 1.0 - sparsity

        return importance

    def clear(self):
        """Clear all captured routing signals."""
        for captured in self._captured_values.values():
            captured["value"] = None

    def restore_original(self):
        """Restore original activation functions (remove capturing wrappers)."""
        for layer_idx, original_act in self._original_act_fns.items():
            try:
                mlp = self.model.model.layers[layer_idx].mlp
                mlp.router_act_fn = original_act
            except (AttributeError, IndexError):
                pass
        self._original_act_fns.clear()
        self._captured_values.clear()

    def __del__(self):
        """Clean up on deletion."""
        self.restore_original()


def extract_routing_from_prompt(
    model,
    tokenizer,
    prompt: str,
    target_layers: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to extract routing signals for a single prompt.

    Args:
        model: BlockFFN model
        tokenizer: Tokenizer for the model
        prompt: Text prompt to analyze
        target_layers: Specific layers to capture

    Returns:
        Dict containing:
            - signals: Dict[layer_idx, Tensor]
            - tokens: List[str] - tokenized prompt
            - sparsity_stats: Dict with sparsity metrics
            - importance: Dict[layer_idx, Tensor]
    """
    extractor = RoutingExtractor(model, target_layers)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    batch_size, seq_len = input_ids.shape

    # Forward pass
    with torch.no_grad():
        _ = model(input_ids)

    # Collect results
    signals = extractor.get_routing_signals_reshaped(batch_size, seq_len)
    sparsity_stats = extractor.compute_sparsity_stats()
    importance = extractor.compute_importance_scores()

    # Get token strings for visualization
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # Clean up
    extractor.restore_original()

    return {
        "signals": signals,
        "tokens": tokens,
        "sparsity_stats": sparsity_stats,
        "importance": importance,
    }
