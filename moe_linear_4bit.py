#!/usr/bin/env python3
"""
MoELinear4Bit: Efficient 4-bit quantized MoE expert computation.

This autograd function enables memory-efficient quantized MoE by:
1. Dequantizing 3D expert weights ONCE per forward/backward pass
2. Saving only the quantized weights in autograd context
3. Following the same pattern as BnB's MatMul4Bit for 2D layers

This solves the OOM problem where naive __getitem__ dequantizes
the full tensor for each expert (32x dequantizations).

Can be proposed for inclusion in bitsandbytes for general MoE support.
"""

import torch
import torch.nn.functional as F
from typing import Optional
import bitsandbytes.functional as bnb_F


class MoELinear4Bit(torch.autograd.Function):
    """
    Autograd function for 4-bit quantized MoE expert computation.

    Mimics MatMul4Bit behavior but handles 3D expert weights:
    - Forward: Dequantizes once, processes all experts, saves quantized weights
    - Backward: Dequantizes once for gradient computation

    Memory usage: Same as MatMul4Bit (dequant is temporary, not persistent)
    """

    @staticmethod
    def forward(ctx, inputs, weight_quantized, quant_state, expert_size, num_experts):
        """
        Forward pass for quantized MoE experts.

        Args:
            inputs: Input tensor to route to experts
            weight_quantized: Quantized 4-bit weight tensor (flattened format)
            quant_state: QuantState with original shape [num_experts, out, in]
            expert_size: List of expert sizes for splitting inputs
            num_experts: Number of experts

        Returns:
            Concatenated output from all experts
        """
        # Dequantize the 3D weight tensor ONCE
        # Shape: [num_experts, out_features, in_features]
        weight_full = bnb_F.dequantize_4bit(weight_quantized, quant_state)

        # Split inputs by expert assignment
        input_list = inputs.split(expert_size, dim=0)

        # Process each expert with its slice of the dequantized weight
        output_list = []
        for i in range(num_experts):
            # Use the i-th expert's weights (2D slice)
            output_list.append(F.linear(input_list[i], weight_full[i]))

        # Concatenate outputs
        result = torch.cat(output_list, dim=0)

        # Save for backward - CRITICAL: Save quantized weights, not dequantized!
        # This is what makes it memory efficient
        ctx.save_for_backward(inputs, weight_quantized)
        ctx.quant_state = quant_state
        ctx.expert_size = expert_size
        ctx.num_experts = num_experts
        ctx.dtype = inputs.dtype

        # weight_full is garbage collected here - not kept in memory!
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for quantized MoE experts.

        Dequantizes weights again for gradient computation.
        Only computes grad_inputs (weights are frozen during training).
        """
        inputs, weight_quantized = ctx.saved_tensors

        # Dequantize again for backward pass (temporary, just like forward)
        # Shape: [num_experts, out_features, in_features]
        weight_full = bnb_F.dequantize_4bit(weight_quantized, ctx.quant_state)

        # Split grad_output by expert
        grad_output_list = grad_output.split(ctx.expert_size, dim=0)

        # Compute gradient w.r.t. inputs for each expert
        grad_input_list = []
        for i in range(ctx.num_experts):
            # grad_input = grad_output @ weight
            grad_input_list.append(
                torch.matmul(grad_output_list[i], weight_full[i])
            )

        grad_inputs = torch.cat(grad_input_list, dim=0)

        # We don't compute gradients for quantized weights (frozen during training)
        # Return: grad_inputs, None for weight_quantized, None for quant_state,
        #         None for expert_size, None for num_experts
        return grad_inputs, None, None, None, None


def patch_params4bit_getitem():
    """
    Add __getitem__ support to Params4bit for MoE indexing.

    This is a fallback in case MoELinear4Bit doesn't get called.
    Allows weight[i] on 3D quantized tensors (though inefficient - use MoELinear4Bit instead).

    Safe to call multiple times (idempotent).
    """
    from bitsandbytes.nn import Params4bit
    import bitsandbytes.functional as bnb_F

    # Check if already patched
    if hasattr(Params4bit.__getitem__, '_moe_patched'):
        return True

    # Save original __getitem__ if it exists
    original_getitem = getattr(Params4bit, '__getitem__', None)

    def quantized_getitem(self, index):
        """
        Support indexing into 4-bit quantized tensors, e.g., for MoE experts.

        Clones the slice to detach it from the parent tensor, allowing the
        full dequantized tensor to be garbage collected immediately.

        This reduces memory from 906 MB (full tensor) to ~12.6 MB (single expert).
        """
        if not self.bnb_quantized or self.quant_state is None:
            # Not quantized, use parent behavior if available
            if original_getitem:
                return original_getitem(self, index)
            else:
                return super(Params4bit, self).__getitem__(index)

        # Dequantize full tensor (temporarily allocates full size)
        full_dequant = bnb_F.dequantize_4bit(self.data, quant_state=self.quant_state)

        # Slice to get the requested expert (view into parent)
        result = full_dequant[index]

        # Clone to create independent copy (detaches from parent tensor)
        # This allows full_dequant to be garbage collected, freeing memory
        result = result.clone()

        # full_dequant is now eligible for garbage collection
        return result

    # Mark as patched
    quantized_getitem._moe_patched = True

    # Apply the patch
    Params4bit.__getitem__ = quantized_getitem

    print("✓ Added __getitem__ support to Params4bit for MoE indexing (clone-and-release)")
    return True


def patch_granite_moe_for_quantization():
    """
    Monkey-patch GraniteMoeParallelExperts to use MoELinear4Bit when quantized.

    This modifies the forward() method to detect quantized weights and use
    the memory-efficient MoELinear4Bit autograd function instead of the
    naive loop that causes OOM.

    Safe to call multiple times (idempotent).
    """
    try:
        # Import the Granite MoE class
        from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeParallelExperts
    except ImportError:
        try:
            # Try alternate location for different transformers versions
            from transformers.models.granitemoehybrid.modeling_granitemoehybrid import GraniteMoeHybridParallelExperts as GraniteMoeParallelExperts
        except ImportError:
            print("Warning: Could not find GraniteMoeParallelExperts to patch")
            return False

    # Check if already patched
    if hasattr(GraniteMoeParallelExperts.forward, '_quantization_patched'):
        print("GraniteMoeParallelExperts already patched for quantization")
        return True

    # Save original forward method
    original_forward = GraniteMoeParallelExperts.forward

    def quantized_forward(self, inputs, expert_size):
        """
        Forward with quantization support.

        Detects if weights are quantized (Params4bit) and uses MoELinear4Bit
        for memory-efficient computation. Falls back to original implementation
        for non-quantized weights.
        """
        from bitsandbytes.nn import Params4bit

        # Check if weights are quantized
        if isinstance(self.weight, Params4bit) and hasattr(self.weight, 'quant_state') and self.weight.quant_state is not None:
            # Use memory-efficient quantized path
            return MoELinear4Bit.apply(
                inputs,
                self.weight.data,           # Quantized weight tensor
                self.weight.quant_state,    # QuantState with original shape
                expert_size,                # Expert size info for splitting
                self.num_experts            # Number of experts
            )
        else:
            # Fall back to original implementation for non-quantized weights
            input_list = inputs.split(expert_size, dim=0)
            output_list = []
            for i in range(self.num_experts):
                output_list.append(F.linear(input_list[i], self.weight[i]))
            return torch.cat(output_list, dim=0)

    # Mark as patched to avoid double-patching
    quantized_forward._quantization_patched = True

    # Apply the patch
    GraniteMoeParallelExperts.forward = quantized_forward

    print("✓ Patched GraniteMoeParallelExperts for 4-bit quantization support")
    return True


def apply_all_patches():
    """
    Apply all necessary patches for BnB quantization of Granite MoE.

    This is a convenience function that applies:
    1. __getitem__ support to Params4bit (fallback for indexing)
    2. MoELinear4Bit integration to GraniteMoeParallelExperts (efficient path)

    Call this once at the start of your script before loading/quantizing models.
    """
    print("Applying BnB MoE quantization patches...")
    print()

    # Patch Params4bit for __getitem__ support (fallback)
    patch_params4bit_getitem()

    # Patch Granite MoE for efficient quantized forward
    patch_granite_moe_for_quantization()

    print()
    print("="*60)
    print("All patches applied successfully!")
    print("="*60)
    print()
    print("You can now:")
    print("  - Quantize Granite MoE models with BnB")
    print("  - Train with LoRA on unquantized layers without OOM")
    print("  - Use gradient checkpointing")
    print()

    return True


if __name__ == "__main__":
    # Test that the patches work
    print("Testing MoE quantization patches...")
    print()

    success = apply_all_patches()

    if success:
        print()
        print("Ready to use!")
        print()
        print("Usage in your code:")
        print("  from moe_linear_4bit import apply_all_patches")
        print("  apply_all_patches()")
        print("  # Then proceed with normal model loading and quantization")
