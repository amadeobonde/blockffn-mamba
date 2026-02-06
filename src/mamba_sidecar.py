"""
Mamba Sidecar Integration for BlockFFN.

This module provides the MambaSidecar class that wraps Mamba2 for use as a
parallel path alongside transformer attention, and the GatedHybridLayer that
combines both paths using routing-based gating.

Architecture:
    hidden_states -> [Attention] -> attn_output
                 \-> [Mamba]     -> mamba_output

    gate = f(routing_signals)  # 0=Mamba, 1=Attention
    output = gate * attn_output + (1-gate) * mamba_output
"""

from typing import Optional, Tuple, Dict, List, Any, Union
import warnings
import torch
import torch.nn as nn
from torch import Tensor


class MambaSidecar(nn.Module):
    """
    Mamba2 block configured as a sidecar to transformer layers.

    Handles automatic fallback to a linear layer when mamba_ssm is not
    available (e.g., on Mac for development).

    Args:
        d_model: Model dimension (must match transformer hidden_size)
        d_state: SSM state dimension (default 64, reduced for memory)
        d_conv: Local convolution width (default 4)
        expand: Block expansion factor (default 2)
        headdim: Head dimension (default 64)
        device: Device to place the module on
        fallback_mode: What to use when Mamba unavailable
                      "linear" (default), "gru", or "identity"
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        fallback_mode: str = "linear",
    ):
        super().__init__()
        self.d_model = d_model
        self.fallback_mode = fallback_mode
        self.using_real_mamba = False

        # Try to import and initialize Mamba2
        try:
            from mamba_ssm import Mamba2

            self.mamba = Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
            )
            if device:
                self.mamba = self.mamba.to(device)
            if dtype:
                self.mamba = self.mamba.to(dtype)

            self.using_real_mamba = True

        except ImportError as e:
            warnings.warn(
                f"mamba_ssm not available ({e}). "
                f"Using {fallback_mode} fallback for development. "
                "This won't reflect real Mamba performance."
            )
            self._init_fallback(d_model, fallback_mode, device, dtype)

    def _init_fallback(
        self,
        d_model: int,
        mode: str,
        device: Optional[str],
        dtype: Optional[torch.dtype],
    ):
        """Initialize fallback module when Mamba is unavailable."""
        if mode == "linear":
            self.mamba = nn.Linear(d_model, d_model)
        elif mode == "gru":
            # GRU as a simple SSM-like fallback
            self.mamba = nn.GRU(
                d_model,
                d_model,
                batch_first=True,
                bidirectional=False,
            )
        elif mode == "identity":
            self.mamba = nn.Identity()
        else:
            raise ValueError(f"Unknown fallback_mode: {mode}")

        if device:
            self.mamba = self.mamba.to(device)
        if dtype and mode != "identity":
            self.mamba = self.mamba.to(dtype)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Process hidden states through Mamba (or fallback).

        Args:
            hidden_states: [batch_size, seq_len, d_model]

        Returns:
            Tensor of same shape [batch_size, seq_len, d_model]
        """
        if self.using_real_mamba or self.fallback_mode != "gru":
            return self.mamba(hidden_states)
        else:
            # GRU returns (output, hidden)
            output, _ = self.mamba(hidden_states)
            return output

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"using_real_mamba={self.using_real_mamba}, "
            f"fallback_mode={self.fallback_mode}"
        )


class GatedHybridLayer(nn.Module):
    """
    Hybrid layer that gates between transformer attention and Mamba.

    Uses BlockFFN's routing sparsity as the gating signal:
    - High sparsity (many inactive experts) -> token is "easy" -> use Mamba
    - Low sparsity (many active experts) -> token is "hard" -> use Attention

    Args:
        original_layer: The original transformer decoder layer
        mamba_sidecar: MambaSidecar instance
        alpha: Default gate value (0=Mamba, 1=Attention) for fixed mode
        gate_mode: "fixed", "routing", or "learned"
        sparsity_threshold: Threshold for hard gating (routing mode)
        hard_gating: Use binary (hard) vs soft gating
        temperature: Temperature for gate computation
    """

    def __init__(
        self,
        original_layer: nn.Module,
        mamba_sidecar: MambaSidecar,
        alpha: float = 0.5,
        gate_mode: str = "routing",
        sparsity_threshold: float = 0.7,
        hard_gating: bool = False,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.mamba_sidecar = mamba_sidecar

        # Gate configuration
        self.register_buffer("alpha", torch.tensor(alpha))
        self.gate_mode = gate_mode
        self.sparsity_threshold = sparsity_threshold
        self.hard_gating = hard_gating
        self.temperature = temperature

        # External routing signal (set before forward pass)
        self._routing_signal: Optional[Tensor] = None

        # Learned gate projection (for "learned" mode)
        if gate_mode == "learned":
            # Try to get hidden_size from original layer
            hidden_size = self._get_hidden_size()
            self.gate_proj = nn.Linear(hidden_size, 1, bias=True)
            # Initialize to output ~0.5 (neutral gate)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.zeros_(self.gate_proj.bias)

    def _get_hidden_size(self) -> int:
        """Extract hidden_size from the original layer."""
        # Try various attribute names
        for attr in ["hidden_size", "embed_dim", "d_model"]:
            if hasattr(self.original_layer, attr):
                return getattr(self.original_layer, attr)

        # Try to get from self_attn
        if hasattr(self.original_layer, "self_attn"):
            attn = self.original_layer.self_attn
            for attr in ["hidden_size", "embed_dim", "head_dim"]:
                if hasattr(attn, attr):
                    size = getattr(attn, attr)
                    if attr == "head_dim" and hasattr(attn, "num_heads"):
                        size *= attn.num_heads
                    return size

        # Fallback: try to infer from weight shapes
        for name, param in self.original_layer.named_parameters():
            if "weight" in name and param.dim() == 2:
                return param.shape[-1]

        raise ValueError("Could not determine hidden_size from original layer")

    def set_routing_signal(self, signal: Optional[Tensor]):
        """
        Set the routing signal for this forward pass.

        Args:
            signal: Routing tensor of shape [batch*seq, num_experts] or
                   [batch, seq, num_experts]
        """
        self._routing_signal = signal

    def set_alpha(self, alpha: float):
        """Update the fixed alpha value."""
        self.alpha.fill_(alpha)

    def compute_gate(
        self,
        hidden_states: Tensor,
        routing_signals: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute per-token gate values.

        Args:
            hidden_states: [batch, seq, hidden]
            routing_signals: [batch*seq, num_experts] or [batch, seq, num_experts]

        Returns:
            gate: [batch, seq, 1] values in [0, 1]
                  0 = use Mamba (cheap), 1 = use Attention (expensive)
        """
        batch, seq_len, hidden = hidden_states.shape

        if self.gate_mode == "fixed":
            # Constant gate for all tokens
            return self.alpha.expand(batch, seq_len, 1)

        elif self.gate_mode == "routing":
            # Use routing sparsity as gate signal
            if routing_signals is None:
                routing_signals = self._routing_signal

            if routing_signals is None:
                # No routing signal available, fall back to fixed alpha
                return self.alpha.expand(batch, seq_len, 1)

            # Reshape if needed: [batch*seq, experts] -> [batch, seq, experts]
            if routing_signals.dim() == 2:
                num_experts = routing_signals.shape[-1]
                routing_signals = routing_signals.view(batch, seq_len, num_experts)

            # Move to same device as hidden_states
            routing_signals = routing_signals.to(hidden_states.device)

            # Compute sparsity: fraction of zeros per token
            sparsity = (routing_signals == 0).float().mean(dim=-1, keepdim=True)

            if self.hard_gating:
                # Binary: high sparsity -> Mamba (gate=0), low sparsity -> Attention (gate=1)
                gate = (sparsity < self.sparsity_threshold).float()
            else:
                # Soft: higher sparsity -> more Mamba
                # Apply temperature and sigmoid for smooth gating
                # Gate = 1 - sparsity means low sparsity = high gate = attention
                raw_gate = (1.0 - sparsity) * self.temperature
                gate = torch.sigmoid(raw_gate - 0.5 * self.temperature)

            return gate

        elif self.gate_mode == "learned":
            # Learn gate from hidden states
            return torch.sigmoid(self.gate_proj(hidden_states))

        else:
            raise ValueError(f"Unknown gate_mode: {self.gate_mode}")

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Gated hybrid forward pass.

        Computes both attention and Mamba outputs, then blends based on gate.
        This is for research purposes - actual speedup would require skipping
        attention for Mamba-routed tokens.

        Args:
            hidden_states: [batch, seq, hidden] input tensor
            attention_mask: Attention mask (passed to original layer)
            position_ids: Position IDs (passed to original layer)
            past_key_value: KV cache (passed to original layer)
            output_attentions: Whether to output attention weights
            use_cache: Whether to use KV cache
            **kwargs: Additional arguments for the original layer

        Returns:
            Output tensor or tuple (output, cache, attention_weights, ...)
        """
        # Compute gate
        gate = self.compute_gate(hidden_states)

        # Path 1: Original transformer layer (attention + MLP)
        original_outputs = self.original_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        # Path 2: Mamba sidecar
        mamba_output = self.mamba_sidecar(hidden_states)

        # Blend outputs based on gate
        if isinstance(original_outputs, tuple):
            attn_hidden = original_outputs[0]
            blended = gate * attn_hidden + (1 - gate) * mamba_output
            # Preserve other outputs (attention weights, cache, etc.)
            return (blended,) + original_outputs[1:]
        else:
            return gate * original_outputs + (1 - gate) * mamba_output

    def extra_repr(self) -> str:
        return (
            f"gate_mode={self.gate_mode}, "
            f"alpha={self.alpha.item():.2f}, "
            f"hard_gating={self.hard_gating}, "
            f"temperature={self.temperature}"
        )


def attach_mamba_sidecar(
    model: nn.Module,
    layer_indices: Optional[List[int]] = None,
    alpha: float = 0.5,
    gate_mode: str = "routing",
    mamba_config: Optional[Dict[str, Any]] = None,
    **hybrid_kwargs,
) -> nn.Module:
    """
    Attach Mamba sidecars to specified layers of a BlockFFN model.

    This function monkey-patches the model's layers, replacing them with
    GatedHybridLayer wrappers that combine the original layer with Mamba.

    Args:
        model: The BlockFFN model (must have model.model.layers structure)
        layer_indices: Which layers to modify (None = middle third)
        alpha: Default gate value (0=Mamba, 1=Attention)
        gate_mode: "fixed", "routing", or "learned"
        mamba_config: Config dict for MambaSidecar (d_state, d_conv, etc.)
        **hybrid_kwargs: Additional kwargs for GatedHybridLayer

    Returns:
        The modified model (same object, mutated in place)

    Example:
        model = load_model()
        model = attach_mamba_sidecar(model, layer_indices=[10, 11, 12])
    """
    # Default Mamba config
    config = model.config
    default_mamba_config = {
        "d_model": config.hidden_size,
        "d_state": 64,
        "d_conv": 4,
        "expand": 2,
        "headdim": 64,
    }
    if mamba_config:
        default_mamba_config.update(mamba_config)

    # Determine device and dtype from model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Determine target layers
    num_layers = len(model.model.layers)
    if layer_indices is None:
        # Default: middle third
        start = num_layers // 3
        end = 2 * num_layers // 3
        layer_indices = list(range(start, end))

    print(f"Attaching Mamba sidecars to layers: {layer_indices}")
    print(f"Mamba config: {default_mamba_config}")
    print(f"Gate mode: {gate_mode}, alpha: {alpha}")

    # Attach sidecars
    for idx in layer_indices:
        if idx < 0 or idx >= num_layers:
            warnings.warn(f"Layer index {idx} out of range, skipping")
            continue

        original_layer = model.model.layers[idx]

        # Create Mamba sidecar
        sidecar = MambaSidecar(
            **default_mamba_config,
            device=str(device),
            dtype=dtype,
        )

        # Create hybrid wrapper
        hybrid_layer = GatedHybridLayer(
            original_layer=original_layer,
            mamba_sidecar=sidecar,
            alpha=alpha,
            gate_mode=gate_mode,
            **hybrid_kwargs,
        )

        # Replace the layer
        model.model.layers[idx] = hybrid_layer

    return model


def set_all_alphas(model: nn.Module, alpha: float):
    """
    Set alpha value for all GatedHybridLayer instances in the model.

    Args:
        model: Model with GatedHybridLayer layers
        alpha: New alpha value
    """
    for layer in model.model.layers:
        if isinstance(layer, GatedHybridLayer):
            layer.set_alpha(alpha)


def get_hybrid_layers(model: nn.Module) -> List[Tuple[int, GatedHybridLayer]]:
    """
    Get all GatedHybridLayer instances in the model.

    Returns:
        List of (index, layer) tuples
    """
    hybrid_layers = []
    for idx, layer in enumerate(model.model.layers):
        if isinstance(layer, GatedHybridLayer):
            hybrid_layers.append((idx, layer))
    return hybrid_layers


def detach_mamba_sidecars(model: nn.Module) -> nn.Module:
    """
    Remove Mamba sidecars and restore original layers.

    Args:
        model: Model with GatedHybridLayer layers

    Returns:
        Model with original layers restored
    """
    for idx, layer in enumerate(model.model.layers):
        if isinstance(layer, GatedHybridLayer):
            model.model.layers[idx] = layer.original_layer
    return model
