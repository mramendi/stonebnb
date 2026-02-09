#!/usr/bin/env python3
"""
Save BnB 4-bit quantized Granite MoE model in StoneBnB format.

StoneBnB format = BnB quantization + embedded quant_state metadata + 3D shape tracking.

This differs from standard save_pretrained():
- Preserves quantization (doesn't dequantize to BF16)
- Embeds quant_state metadata (.absmax, .blocksize, .code, etc.)
- Tracks 3D tensor shapes for MoE experts
- Explicitly saves lm_head.weight

See STONEBNB_FORMAT.md for full specification.
"""

import torch
import json
from pathlib import Path
from bitsandbytes.nn import Params4bit


def save_quantized_model(model, tokenizer, output_dir, list_all_layers=False):
    """
    Save quantized model with metadata for 3D MoE weights.

    Args:
        model: Quantized model (with Params4bit weights)
        tokenizer: Tokenizer to save
        output_dir: Output directory path
        list_all_layers: List every layer being saved (for debugging)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving quantized model to: {output_dir}")
    print()

    # Save model state_dict + quant_state metadata
    print("Saving state_dict with quant_state metadata...")
    state_dict = {}

    # Manually build state_dict including quant_state for Params4bit
    for name, param in model.named_parameters():
        if isinstance(param, Params4bit) and param.quant_state is not None:
            # Save the quantized data
            state_dict[name] = param.data

            # Save quant_state components
            qs = param.quant_state
            prefix = name + "."

            state_dict[prefix + "quant_type"] = qs.quant_type
            state_dict[prefix + "absmax"] = qs.absmax
            state_dict[prefix + "blocksize"] = torch.tensor(qs.blocksize)
            state_dict[prefix + "quant_map"] = qs.code
            state_dict[prefix + "dtype"] = str(qs.dtype)
            state_dict[prefix + "shape"] = torch.tensor(list(qs.shape))

            if qs.offset is not None:
                state_dict[prefix + "offset"] = qs.offset

            # Nested quantization (double quant)
            if qs.nested and qs.state2 is not None:
                state_dict[prefix + "nested_absmax"] = qs.state2.absmax
                state_dict[prefix + "nested_blocksize"] = torch.tensor(qs.state2.blocksize)
                state_dict[prefix + "nested_quant_map"] = qs.state2.code
        else:
            # Regular parameter
            state_dict[name] = param

    # Also save buffers
    for name, buffer in model.named_buffers():
        state_dict[name] = buffer

    # CRITICAL: Ensure lm_head is saved (it's at top level, not in model.model)
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        if 'lm_head.weight' not in state_dict:
            print("  Adding lm_head.weight (was missing from named_parameters)")
            state_dict['lm_head.weight'] = model.lm_head.weight
            # Add bias if it exists
            if hasattr(model.lm_head, 'bias') and model.lm_head.bias is not None:
                state_dict['lm_head.bias'] = model.lm_head.bias

    # List all layers being saved if requested
    if list_all_layers:
        print()
        print("="*80)
        print("LISTING ALL LAYERS BEING SAVED TO STATE_DICT")
        print("="*80)
        print()
        for i, (key, value) in enumerate(state_dict.items(), 1):
            # Skip quant_state metadata keys
            if any(key.endswith(suffix) for suffix in ['.quant_type', '.absmax', '.blocksize', '.quant_map',
                                                       '.dtype', '.shape', '.offset', '.nested_absmax',
                                                       '.nested_blocksize', '.nested_quant_map']):
                continue

            value_type = type(value).__name__
            if hasattr(value, 'shape'):
                value_info = f"{str(value.shape):30s} {value.dtype if hasattr(value, 'dtype') else value_type}"
            else:
                value_info = value_type

            print(f"  {i:4d}. {key:80s} {value_info}")
        print()
        print("="*80)
        print()

    torch.save(state_dict, output_dir / "pytorch_model.bin")
    print(f"✓ Saved state_dict: {len(state_dict)} keys")
    print()

    # Save config
    print("Saving config...")
    model.config.save_pretrained(output_dir)
    print("✓ Saved config")
    print()

    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)
    print("✓ Saved tokenizer")
    print()

    # Save quantization metadata (track which weights are quantized and their original shapes)
    print("Saving quantization metadata...")
    quant_metadata = {
        "quantized_weights": [],
        "quantization_method": "bitsandbytes_nf4",
        "moe_patch_required": True,
    }

    for name, param in model.named_parameters():
        if isinstance(param, Params4bit) and param.quant_state is not None:
            # Store original shape from quant_state
            original_shape = param.quant_state.shape.tolist() if hasattr(param.quant_state.shape, 'tolist') else list(param.quant_state.shape)

            quant_metadata["quantized_weights"].append({
                "name": name,
                "original_shape": original_shape,
                "data_shape": list(param.data.shape),
                "is_3d": len(original_shape) == 3,
            })

    with open(output_dir / "quantization_metadata.json", "w") as f:
        json.dump(quant_metadata, f, indent=2)

    print(f"✓ Saved quantization metadata: {len(quant_metadata['quantized_weights'])} quantized weights")
    print()

    # Print summary
    num_3d = sum(1 for w in quant_metadata["quantized_weights"] if w["is_3d"])
    num_2d = len(quant_metadata["quantized_weights"]) - num_3d

    print("="*60)
    print("SAVE COMPLETE")
    print("="*60)
    print(f"Output: {output_dir}")
    print(f"Quantized weights: {len(quant_metadata['quantized_weights'])}")
    print(f"  3D (MoE): {num_3d}")
    print(f"  2D (Linear): {num_2d}")
    print()
    print("To load:")
    print(f"  from bnb_granite.load_quantized_model import load_quantized_model")
    print(f"  model = load_quantized_model('{output_dir}')")


if __name__ == "__main__":
    import sys
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="Save BnB quantized Granite model")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--model", help="Model object (for programmatic use)")
    parser.add_argument("--tokenizer", help="Tokenizer object (for programmatic use)")

    args = parser.parse_args()

    if args.model is None:
        print("Error: This script is meant to be called programmatically")
        print()
        print("Usage in Python:")
        print("  from bnb_granite.save_quantized_model import save_quantized_model")
        print("  save_quantized_model(model, tokenizer, 'output_dir')")
        sys.exit(1)

    save_quantized_model(args.model, args.tokenizer, args.output_dir)
