"""
Routing Signal Predictor

Speculatively extracts routing signals from BlockFFN layers BEFORE attention
computation. This enables adaptive window sizing based on routing sparsity.

The key insight is that we can run just the router projection (one matmul)
to predict which tokens are "easy" (high sparsity) vs "hard" (low sparsity)
before deciding how much attention compute to allocate.
"""

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch import Tensor


class RoutingPredictor:
    """
    Predicts routing signals for adaptive attention window sizing.

    BlockFFN's router architecture:
        router_proj (Linear) -> router_act_fn (ReLU) -> router_norm (RMSNorm)

    We speculatively run router_proj + ReLU to get routing signals,
    which indicate expert activation patterns. High sparsity (many zeros)
    indicates "easy" tokens that can use smaller attention windows.

    Attributes:
        model: Reference to the BlockFFN model
        layer_idx: Which layer this predictor is for
        router_proj: Cached reference to the router projection layer
        router_act_fn: Cached reference to the router activation function
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
    ):
        """
        Initialize the routing predictor.

        Args:
            model: BlockFFN model (expects model.model.layers[i].mlp structure)
            layer_idx: Index of the layer to predict routing for
        """
        self.model = model
        self.layer_idx = layer_idx

        # Cache router components
        self.router_proj: Optional[nn.Linear] = None
        self.router_act_fn: Optional[nn.Module] = None
        self.has_router = False

        self._cache_router_components()

    def _cache_router_components(self) -> None:
        """Extract router_proj and router_act_fn from the model."""
        try:
            # Navigate to the MLP layer
            layer = self.model.model.layers[self.layer_idx]
            mlp = layer.mlp

            # Check for router components
            if hasattr(mlp, "router_proj") and hasattr(mlp, "router_act_fn"):
                self.router_proj = mlp.router_proj
                self.router_act_fn = mlp.router_act_fn
                self.has_router = True
            else:
                # Try alternative naming conventions
                for proj_name in ["router_proj", "gate_proj", "router"]:
                    if hasattr(mlp, proj_name):
                        self.router_proj = getattr(mlp, proj_name)
                        break

                # Default activation to ReLU if not found
                if self.router_proj is not None:
                    self.router_act_fn = getattr(mlp, "router_act_fn", nn.ReLU())
                    self.has_router = True

        except (AttributeError, IndexError) as e:
            self.has_router = False

    def predict_routing(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        """
        Predict routing signals from hidden states.

        This is the speculative routing computation - we run just the
        router projection and activation to get routing signals before
        the full attention computation.

        Args:
            hidden_states: Input hidden states [batch, seq, hidden_dim]

        Returns:
            routing_signals: Post-activation routing signals [batch, seq, num_experts]
                            Values are >= 0 (post-ReLU), sparse (many zeros)
        """
        if not self.has_router:
            # Fallback: return uniform "all experts active" signal
            batch, seq, hidden = hidden_states.shape
            return torch.ones(batch, seq, 1, device=hidden_states.device)

        with torch.no_grad():
            # Router projection: [batch, seq, hidden] -> [batch, seq, num_experts]
            router_logits = self.router_proj(hidden_states)

            # Apply activation (ReLU) to induce sparsity
            routing_signals = self.router_act_fn(router_logits)

        return routing_signals

    def compute_sparsity(
        self,
        routing_signals: Tensor,
    ) -> Tensor:
        """
        Compute per-token sparsity from routing signals.

        Sparsity = fraction of inactive (zero) experts per token.
        Higher sparsity indicates "easier" tokens that need fewer experts.

        Args:
            routing_signals: Post-activation routing signals [batch, seq, num_experts]

        Returns:
            sparsity: Per-token sparsity values [batch, seq] in range [0, 1]
        """
        # Count zeros along expert dimension
        is_zero = routing_signals == 0  # [batch, seq, num_experts]
        sparsity = is_zero.float().mean(dim=-1)  # [batch, seq]
        return sparsity

    def predict_and_compute_sparsity(
        self,
        hidden_states: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Combined prediction and sparsity computation.

        Convenience method that returns both routing signals and sparsity.

        Args:
            hidden_states: Input hidden states [batch, seq, hidden_dim]

        Returns:
            routing_signals: [batch, seq, num_experts]
            sparsity: [batch, seq]
        """
        routing_signals = self.predict_routing(hidden_states)
        sparsity = self.compute_sparsity(routing_signals)
        return routing_signals, sparsity

    def get_num_experts(self) -> int:
        """Get the number of experts in the router."""
        if self.router_proj is not None:
            return self.router_proj.out_features
        return 1


class LayerRoutingPredictors:
    """
    Collection of routing predictors for multiple layers.

    Manages RoutingPredictor instances for each target layer,
    enabling efficient batch prediction across layers.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_indices: Optional[list] = None,
    ):
        """
        Initialize routing predictors for specified layers.

        Args:
            model: BlockFFN model
            target_layer_indices: List of layer indices, or None for middle third
        """
        self.model = model
        self.predictors: Dict[int, RoutingPredictor] = {}

        # Determine target layers
        num_layers = len(model.model.layers)
        if target_layer_indices is None:
            # Default: middle third
            start = num_layers // 3
            end = 2 * num_layers // 3
            target_layer_indices = list(range(start, end))

        # Create predictors
        for layer_idx in target_layer_indices:
            if 0 <= layer_idx < num_layers:
                self.predictors[layer_idx] = RoutingPredictor(model, layer_idx)

    def predict_all(
        self,
        hidden_states: Tensor,
    ) -> Dict[int, Tensor]:
        """
        Predict routing signals for all target layers.

        Args:
            hidden_states: Input hidden states [batch, seq, hidden_dim]

        Returns:
            Dict mapping layer_idx to routing_signals [batch, seq, num_experts]
        """
        return {
            layer_idx: predictor.predict_routing(hidden_states)
            for layer_idx, predictor in self.predictors.items()
        }

    def compute_sparsity_all(
        self,
        hidden_states: Tensor,
    ) -> Dict[int, Tensor]:
        """
        Compute sparsity for all target layers.

        Args:
            hidden_states: Input hidden states [batch, seq, hidden_dim]

        Returns:
            Dict mapping layer_idx to sparsity [batch, seq]
        """
        return {
            layer_idx: predictor.compute_sparsity(
                predictor.predict_routing(hidden_states)
            )
            for layer_idx, predictor in self.predictors.items()
        }

    def compute_average_sparsity(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        """
        Compute average sparsity across all target layers.

        Useful when making a single window-sizing decision for
        tokens across multiple layers.

        Args:
            hidden_states: Input hidden states [batch, seq, hidden_dim]

        Returns:
            Average sparsity [batch, seq]
        """
        sparsities = self.compute_sparsity_all(hidden_states)
        if not sparsities:
            return torch.ones(
                hidden_states.shape[0],
                hidden_states.shape[1],
                device=hidden_states.device,
            )

        stacked = torch.stack(list(sparsities.values()), dim=0)
        return stacked.mean(dim=0)

    def __getitem__(self, layer_idx: int) -> RoutingPredictor:
        """Get predictor for a specific layer."""
        return self.predictors[layer_idx]

    def __contains__(self, layer_idx: int) -> bool:
        """Check if layer has a predictor."""
        return layer_idx in self.predictors

    def __len__(self) -> int:
        """Number of predictors."""
        return len(self.predictors)
