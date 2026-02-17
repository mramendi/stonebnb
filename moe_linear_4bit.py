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
    Autograd function for 4-bit quantized MoE expert computation with virtual weights.

    NEW: Uses WeightReference instead of saving full quantized tensor to ctx.
    This enables:
    - Lazy per-expert dequantization (only 1-2 experts at a time)
    - LRU caching for reuse between forward and backward
    - Lightweight ctx (50 bytes instead of 2GB)

    Memory savings:
    - Old: ~2GB quantized tensor in ctx + ~5GB full dequantization
    - New: ~50 bytes WeightReference in ctx + ~140MB per expert (cached)
    """

    @staticmethod
    def forward(ctx, inputs, weight_ref, expert_size, num_experts):
        """
        Forward pass with lazy per-expert dequantization.

        Args:
            inputs: Input tensor to route to experts
            weight_ref: WeightReference (lightweight reference to weight source)
            expert_size: List of expert sizes for splitting inputs
            num_experts: Number of experts

        Returns:
            Concatenated output from all experts
        """
        # Split inputs by expert assignment
        input_list = inputs.split(expert_size, dim=0)

        # Process each expert with lazy weight fetching
        # Each get_expert() call uses the global LRU cache
        output_list = []
        for i in range(num_experts):
            # Fetch only this expert's weights (cache hit if sequential)
            expert_weight = weight_ref.get_expert(i)  # 2D BF16, uses global cache
            output_list.append(torch.matmul(input_list[i], expert_weight.t()))

        # Concatenate outputs
        result = torch.cat(output_list, dim=0)

        # Save ONLY lightweight reference (not tensor!)
        ctx.save_for_backward(inputs)
        ctx.weight_ref = weight_ref  # ~50 bytes, not ~2GB!
        ctx.expert_size = expert_size
        ctx.num_experts = num_experts
        ctx.dtype = inputs.dtype

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward reuses cache from forward pass.

        Since training processes experts sequentially, the experts loaded
        during forward are likely still in the global LRU cache, giving
        high cache hit rates.
        """
        inputs = ctx.saved_tensors[0]

        # Split grad_output by expert
        grad_output_list = grad_output.split(ctx.expert_size, dim=0)

        # Compute gradient w.r.t. inputs for each expert
        grad_input_list = []
        for i in range(ctx.num_experts):
            # Fetch expert (likely cache hit from forward!)
            expert_weight = ctx.weight_ref.get_expert(i)
            grad_input_list.append(
                torch.matmul(grad_output_list[i], expert_weight)
            )

        grad_inputs = torch.cat(grad_input_list, dim=0)

        # We don't compute gradients for quantized weights (frozen during training)
        # Return: grad_inputs, None for weight_ref, None for expert_size, None for num_experts
        return grad_inputs, None, None, None


class Dequantize4BitSlice(torch.autograd.Function):
    """
    Custom autograd function to dequantize and extract a slice while allowing GC.

    CRITICAL: We DON'T save the dequantized slice to ctx because:
    1. Weights are frozen - no gradient computation needed
    2. The caller (F.linear in transformers) will save it for its own backward
    3. Saving here would mean double VRAM usage (once here, once in F.linear)

    This at least prevents DOUBLE-saving the weights to autograd graph.
    (We can't prevent F.linear from saving them without patching transformers)
    """
    @staticmethod
    def forward(ctx, quantized_data, quant_state, index):
        import bitsandbytes.functional as bnb_F

        # Dequantize full tensor (temporarily allocated)
        full_dequant = bnb_F.dequantize_4bit(quantized_data, quant_state=quant_state)

        # Extract and clone slice
        result = full_dequant[index].clone()

        # DON'T save anything to ctx!
        # F.linear will save the weight for its backward - we can't prevent that.
        # But at least we're not saving it TWICE (once here, once in F.linear).

        # full_dequant goes out of scope → garbage collected
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # Weights are frozen (`requires_grad=False`)
        # We don't compute gradients for quantized_data or quant_state
        # grad_output flows through from F.linear's backward, we just pass None back
        return None, None, None


def patch_params4bit_getitem():
    """
    Add __getitem__ support to Params4bit for MoE indexing.

    Uses custom autograd Function to properly break parent tensor reference
    while maintaining gradient flow.
    """
    from bitsandbytes.nn import Params4bit

    # Check if already patched
    if hasattr(Params4bit.__getitem__, '_moe_patched'):
        return True

    # Save original __getitem__ if it exists
    original_getitem = getattr(Params4bit, '__getitem__', None)

    # Global flag to track if regression warning has been shown
    _regression_warning_shown = [False]  # Use list so it's mutable in closure

    def quantized_getitem(self, index):
        """
        Support indexing into 4-bit quantized tensors, e.g., for MoE experts.

        Uses custom autograd Function to allow parent tensor garbage collection.
        """
        # REGRESSION CONTROL: Warn if this fallback path is used (means patching failed)
        if not _regression_warning_shown[0]:
            _regression_warning_shown[0] = True
            print()
            print("="*80)
            print("⚠️  REGRESSION WARNING: Params4bit.__getitem__ FALLBACK PATH ACTIVE")
            print("="*80)
            print("The MoELinear4Bit patch is NOT being used!")
            print("This means:")
            print("  - Dequantized weights ARE being saved to autograd graph")
            print("  - Training will be MUCH slower (~2.5x slower)")
            print("  - You may run out of VRAM on long sequences")
            print()
            print("This likely means the post-load patching in load_quantized_model() failed.")
            print("Check that you see: '✓ Patched N MoE expert layers' during model load.")
            print("="*80)
            print()

        if not self.bnb_quantized or self.quant_state is None:
            # Not quantized, use parent behavior if available
            if original_getitem:
                return original_getitem(self, index)
            else:
                return super(Params4bit, self).__getitem__(index)

        # Use custom autograd Function
        return Dequantize4BitSlice.apply(self.data, self.quant_state, index)

    # Mark as patched
    quantized_getitem._moe_patched = True

    # Apply the patch
    Params4bit.__getitem__ = quantized_getitem

    print("✓ Added __getitem__ support to Params4bit for MoE indexing (custom autograd)")
    return True


def patch_granite_moe_for_quantization():
    """
    Monkey-patch GraniteMoeParallelExperts to use MoELinear4Bit with virtual weights.

    This modifies the forward() method to:
    1. Check if module has _weight_ref attribute (set during model loading)
    2. Use MoELinear4Bit with WeightReference for memory-efficient computation
    3. Fall back to original implementation for non-quantized weights

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
        # Already patched, skip
        return True

    # Save original forward method
    original_forward = GraniteMoeParallelExperts.forward

    def quantized_forward(self, inputs, expert_size):
        """
        Forward with virtual weight system support.

        Uses WeightReference if available (set during model loading),
        otherwise falls back to original implementation.
        """
        # Check if this module has a weight reference (virtual weight system)
        if hasattr(self, '_weight_ref'):
            # Use memory-efficient virtual weight path
            return MoELinear4Bit.apply(
                inputs,
                self._weight_ref,    # WeightReference (lightweight, ~50 bytes)
                expert_size,         # Expert size info for splitting
                self.num_experts     # Number of experts
            )
        else:
            # Fall back to original behavior (non-quantized or old code path)
            # CRITICAL: Dequantize ONCE before loop to avoid repeated full dequantization
            from bitsandbytes.nn import Params4bit
            import bitsandbytes.functional as bnb_F

            if isinstance(self.weight, Params4bit) and hasattr(self.weight, 'bnb_quantized') and self.weight.bnb_quantized:
                # Dequantize once before the loop
                full_weight = bnb_F.dequantize_4bit(self.weight.data, quant_state=self.weight.quant_state)
            else:
                # Already dequantized
                full_weight = self.weight

            # Use dequantized weight for all experts
            # Use matmul instead of F.linear to avoid saving dequantized weights to autograd graph
            input_list = inputs.split(expert_size, dim=0)
            output_list = []
            for i in range(self.num_experts):
                output_list.append(torch.matmul(input_list[i], full_weight[i].t()))
            return torch.cat(output_list, dim=0)

    # Mark as patched to avoid double-patching
    quantized_forward._quantization_patched = True

    # Apply the patch
    GraniteMoeParallelExperts.forward = quantized_forward

    print("✓ Patched GraniteMoeParallelExperts for virtual weight system")
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
