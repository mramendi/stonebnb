#!/usr/bin/env python3
"""
Lightweight Weight References for StoneBnB

This module provides WeightReference, a lightweight reference object that
replaces heavy quantized tensors in autograd context.

Memory savings:
- Old: ~2GB quantized tensor saved to ctx.saved_tensors
- New: ~50 bytes WeightReference saved to ctx

WeightReference stores only (layer_idx, weight_type) metadata and fetches
the actual expert weights from the global registry on demand.
"""

from typing import Optional
import torch
from expert_weight_storage import ExpertWeightRegistry, ExpertWeightSource


class WeightReference:
    """
    Lightweight reference to expert weights.

    This is saved to autograd context instead of the full quantized tensor,
    providing dramatic memory savings:
    - Quantized tensor: ~2GB (uint8, 72 experts)
    - WeightReference: ~50 bytes (2 ints, 1 string)

    The reference fetches actual weights from ExpertWeightRegistry on demand,
    with automatic LRU caching via the source's get_expert() method.
    """

    def __init__(self, layer_idx: int, weight_type: str):
        """
        Create a lightweight weight reference.

        Args:
            layer_idx: Layer index (0 to num_layers-1)
            weight_type: "input_linear" or "output_linear"
        """
        self.layer_idx = layer_idx        # int (4-8 bytes)
        self.weight_type = weight_type    # str (~40 bytes)
        # Total size: ~50 bytes (vs ~2GB for quantized tensor!)

    def get_source(self) -> ExpertWeightSource:
        """
        Retrieve source from global registry.

        Returns:
            ExpertWeightSource instance

        Raises:
            ValueError: If source not registered for this layer/type
        """
        source = ExpertWeightRegistry.get(self.layer_idx, self.weight_type)
        if source is None:
            raise ValueError(
                f"No weight source registered for layer {self.layer_idx}, "
                f"weight_type '{self.weight_type}'. "
                f"Did you forget to call register_weight_source()?"
            )
        return source

    def get_expert(self, expert_id: int) -> torch.Tensor:
        """
        Fetch expert from source (uses cache internally).

        Args:
            expert_id: Expert index (0 to num_experts-1)

        Returns:
            Expert weight tensor (BF16, 2D: [out_features, in_features])
        """
        return self.get_source().get_expert(expert_id)

    def get_all_experts(self) -> torch.Tensor:
        """
        Fetch all experts as 3D tensor (for compatibility).

        Returns:
            All expert weights (BF16, 3D: [num_experts, out_features, in_features])
        """
        return self.get_source().get_all_experts()

    def __repr__(self):
        """String representation for debugging."""
        return f"WeightReference(layer_idx={self.layer_idx}, weight_type='{self.weight_type}')"


def create_weight_reference(layer_idx: int, weight_type: str) -> WeightReference:
    """
    Helper function to create a WeightReference.

    Args:
        layer_idx: Layer index (0 to num_layers-1)
        weight_type: "input_linear" or "output_linear"

    Returns:
        WeightReference instance
    """
    return WeightReference(layer_idx, weight_type)


def extract_layer_info_from_name(param_name: str) -> Optional[tuple]:
    """
    Extract layer index and weight type from parameter name.

    Examples:
        "model.layers.0.block_sparse_moe.input_linear.weight" -> (0, "input_linear")
        "model.layers.12.block_sparse_moe.output_linear.weight" -> (12, "output_linear")

    Args:
        param_name: Full parameter name from model.named_parameters()

    Returns:
        (layer_idx, weight_type) tuple or None if not a MoE expert weight
    """
    # Check if this is a MoE expert weight
    if "block_sparse_moe" not in param_name:
        return None

    # Check if it's input_linear or output_linear
    if "input_linear.weight" in param_name:
        weight_type = "input_linear"
    elif "output_linear.weight" in param_name:
        weight_type = "output_linear"
    else:
        return None

    # Extract layer index
    # Format: "model.layers.{layer_idx}.block_sparse_moe.{weight_type}.weight"
    parts = param_name.split(".")
    try:
        layers_idx = parts.index("layers")
        layer_idx = int(parts[layers_idx + 1])
        return (layer_idx, weight_type)
    except (ValueError, IndexError):
        return None
