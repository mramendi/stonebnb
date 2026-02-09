#!/usr/bin/env python3
"""
Quantize Granite model to BnB 4-bit and save in StoneBnB format.

Run this on a machine with enough VRAM to load the full bf16 model.
For Granite 8B (Tiny): ~17GB VRAM
For Granite 32B (Small): ~65GB VRAM

The quantized model can then be loaded on smaller GPUs for memory-efficient LoRA training.

Output format: StoneBnB (BnB quantization with 3D MoE tensor support)
See STONEBNB_FORMAT.md for details.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb
from bitsandbytes.nn import Params4bit

from moe_linear_4bit import apply_all_patches
from save_quantized_model import save_quantized_model


def quantize_granite(model_name, output_dir, quantize_shared_mlp=False, list_all_layers=False):
    """
    Quantize Granite model MoE experts to 4-bit.

    Args:
        model_name: HuggingFace model name or path
        output_dir: Output directory for quantized model
        quantize_shared_mlp: Whether to also quantize shared_mlp layers (default: False)
        list_all_layers: List every single layer before and after quantization (default: False)
    """
    print("="*80)
    print("Quantizing Granite Model to BnB 4-bit")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print()

    # Apply MoE patches
    print("Applying MoE quantization patches...")
    apply_all_patches()
    print()

    # Load model
    print("Loading model in bf16...")
    print("(This requires enough VRAM for the full model)")
    print()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ Model loaded")
    print()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()

    # Quantize layers
    print("Quantizing to 4-bit NF4...")
    print()

    quantized_count = 0
    quantized_params = 0

    def quantize_layer(module, name, weight_shape):
        nonlocal quantized_count, quantized_params

        weight = module.weight.data
        if weight.device.type != 'cuda':
            weight = weight.cuda()

        print(f"  Quantizing {name}: {weight_shape}")

        quantized_weight, quant_state = bnb.functional.quantize_4bit(
            weight,
            blocksize=64,
            compress_statistics=True,
            quant_type="nf4"
        )

        param = Params4bit(
            data=quantized_weight,
            requires_grad=False,
            quant_state=quant_state,
            blocksize=64,
            compress_statistics=True,
            quant_type="nf4",
            bnb_quantized=True
        )

        module.weight = param
        quantized_count += 1
        quantized_params += weight.numel()

    # Quantize MoE experts (always)
    for layer_idx in range(len(model.model.layers)):
        layer = model.model.layers[layer_idx]

        if hasattr(layer, 'block_sparse_moe'):
            if hasattr(layer.block_sparse_moe, 'input_linear'):
                experts = layer.block_sparse_moe.input_linear
                quantize_layer(experts,
                             f"layers.{layer_idx}.block_sparse_moe.input_linear",
                             experts.weight.shape)
            if hasattr(layer.block_sparse_moe, 'output_linear'):
                experts = layer.block_sparse_moe.output_linear
                quantize_layer(experts,
                             f"layers.{layer_idx}.block_sparse_moe.output_linear",
                             experts.weight.shape)

        # Optionally quantize shared_mlp
        if quantize_shared_mlp and hasattr(layer, 'shared_mlp'):
            if hasattr(layer.shared_mlp, 'input_linear'):
                quantize_layer(layer.shared_mlp.input_linear,
                             f"layers.{layer_idx}.shared_mlp.input_linear",
                             layer.shared_mlp.input_linear.weight.shape)
            if hasattr(layer.shared_mlp, 'output_linear'):
                quantize_layer(layer.shared_mlp.output_linear,
                             f"layers.{layer_idx}.shared_mlp.output_linear",
                             layer.shared_mlp.output_linear.weight.shape)

    print()
    print(f"✓ Quantized {quantized_count} weight tensors")
    print(f"  Quantized parameters: {quantized_params:,} ({100*quantized_params/total_params:.1f}%)")
    print()

    # List all layers if requested
    if list_all_layers:
        print("="*80)
        print("LISTING ALL MODEL LAYERS (PRE-SAVE)")
        print("="*80)
        print()
        print("All parameters:")
        for i, (name, param) in enumerate(model.named_parameters(), 1):
            param_type = "Params4bit" if isinstance(param, Params4bit) else param.dtype
            print(f"  {i:4d}. {name:80s} {str(param.shape):30s} {param_type}")
        print()
        print("All buffers:")
        for i, (name, buffer) in enumerate(model.named_buffers(), 1):
            print(f"  {i:4d}. {name:80s} {str(buffer.shape):30s} {buffer.dtype}")
        print()
        print("="*80)
        print()

    # Save
    save_quantized_model(model, tokenizer, output_dir, list_all_layers=list_all_layers)
    print()

    print("="*80)
    print("QUANTIZATION COMPLETE")
    print("="*80)
    print()
    print(f"Quantized model saved to: {output_dir}")
    print()
    print("To use for training:")
    print("  from bnb_granite.load_quantized_model import load_quantized_model")
    print(f"  model, tokenizer = load_quantized_model('{output_dir}')")
    print()
    print("Then add LoRA and train!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize Granite model to BnB 4-bit")
    parser.add_argument("model_name", help="HuggingFace model name or path")
    parser.add_argument("output_dir", help="Output directory for quantized model")
    parser.add_argument("--quantize-shared-mlp", action="store_true",
                       help="Also quantize shared_mlp layers (default: MoE experts only)")
    parser.add_argument("--list-all-layers", action="store_true",
                       help="List every single layer before and after quantization (for debugging)")

    args = parser.parse_args()

    quantize_granite(args.model_name, args.output_dir, args.quantize_shared_mlp, args.list_all_layers)
