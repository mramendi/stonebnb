#!/usr/bin/env python3
"""
Load BnB 4-bit quantized Granite MoE model from StoneBnB format.

This loads models saved with save_quantized_model.py, properly reconstructing
Params4bit weights with correct shapes for both 2D and 3D tensors.

StoneBnB format features:
- Embedded quant_state metadata in state_dict
- 3D tensor shape preservation for MoE experts
- Explicit lm_head.weight saving

See STONEBNB_FORMAT.md for details.
"""

import torch
import json
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from bitsandbytes.nn import Params4bit
from bitsandbytes.functional import QuantState

from moe_linear_4bit import apply_all_patches


def load_quantized_model(model_path, device="cuda", apply_moe_patch=True):
    """
    Load BnB 4-bit quantized Granite MoE model.

    Args:
        model_path: Path to saved model directory OR HuggingFace model ID
        device: Device to load to ("cuda" or specific like "cuda:0")
        apply_moe_patch: Whether to apply MoE quantization patches

    Returns:
        (model, tokenizer) tuple
    """
    # Handle both local paths and HuggingFace model IDs
    model_path_str = str(model_path)

    # Check if it's a local path
    local_path = Path(model_path)
    if local_path.exists():
        model_path = local_path
    else:
        # Try to download from HuggingFace Hub
        print(f"Local path not found, attempting to download from HuggingFace Hub: {model_path_str}")
        try:
            from huggingface_hub import snapshot_download
            model_path = Path(snapshot_download(repo_id=model_path_str))
            print(f"✓ Downloaded to: {model_path}")
        except Exception as e:
            raise ValueError(f"Model path not found locally and failed to download from HuggingFace Hub: {model_path_str}\nError: {e}")

    print("="*80)
    print("LOADING BNB 4-BIT QUANTIZED GRANITE MOE")
    print("="*80)
    print(f"Path: {model_path}")
    print()

    # Apply MoE patches if requested
    if apply_moe_patch:
        print("Applying MoE quantization patches...")
        apply_all_patches()
        print()

    # Load config
    print("Loading config...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print("✓ Config loaded")
    print()

    # Load quantization metadata
    print("Loading quantization metadata...")
    metadata_file = model_path / "quantization_metadata.json"

    if not metadata_file.exists():
        raise ValueError(
            f"Quantization metadata not found: {metadata_file}\n"
            "This model was not saved with save_quantized_model.py"
        )

    with open(metadata_file) as f:
        quant_metadata = json.load(f)

    quantized_weight_info = {w["name"]: w for w in quant_metadata["quantized_weights"]}
    print(f"✓ Found {len(quantized_weight_info)} quantized weights")
    print()

    # Create model structure (on meta device to avoid memory allocation)
    print("Creating model structure...")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    print("✓ Model structure created")
    print()

    # Load state_dict
    print("Loading state_dict...")
    state_dict = torch.load(model_path / "pytorch_model.bin", map_location="cpu")
    print(f"✓ Loaded {len(state_dict)} parameters")
    print()

    # NOTE: Do NOT call .to_empty(device) here!
    # That would allocate the full unquantized model structure on GPU before we've
    # had a chance to reconstruct the Params4bit tensors. Instead, we keep the model
    # on CPU and move parameters to device as we load them.

    # Load weights
    print(f"Loading quantized weights to {device}...")
    loaded_count = 0
    quantized_count = 0

    for name, param in model.named_parameters():
        if name not in state_dict:
            print(f"Warning: {name} not in state_dict")
            continue

        weight_data = state_dict[name]

        # Check if this is a quantized weight
        if name in quantized_weight_info:
            info = quantized_weight_info[name]

            # Get original shape from quant_state saved data if available
            shape_key = name + ".shape"
            if shape_key in state_dict:
                shape_tensor = state_dict[shape_key]
                if isinstance(shape_tensor, torch.Tensor):
                    original_shape = torch.Size(shape_tensor.tolist())
                else:
                    original_shape = torch.Size(shape_tensor)
            else:
                # Fallback to metadata
                original_shape = torch.Size(info["original_shape"])

            # Collect quant_state keys
            prefix = name + "."
            quant_state_dict = {}

            for key in state_dict.keys():
                if key.startswith(prefix) and key != name:
                    suffix = key[len(prefix):]
                    quant_state_dict[suffix] = state_dict[key]

            # Debug: print what we found
            if len(quant_state_dict) == 0:
                print(f"  Warning: No quant_state keys found for {name}")
                print(f"  Looking for keys starting with: {prefix}")
                print(f"  Available keys sample: {list(state_dict.keys())[:5]}")
                continue

            # Reconstruct QuantState (move all components to device)
            quant_state = reconstruct_quant_state(quant_state_dict, original_shape)

            # Move quant_state components to device
            quant_state.absmax = quant_state.absmax.to(device)
            quant_state.code = quant_state.code.to(device)
            if quant_state.offset is not None:
                quant_state.offset = quant_state.offset.to(device)
            if quant_state.state2 is not None:
                quant_state.state2.absmax = quant_state.state2.absmax.to(device)
                quant_state.state2.code = quant_state.state2.code.to(device)

            # Create Params4bit
            quantized_param = Params4bit(
                data=weight_data.to(device),
                requires_grad=False,
                quant_state=quant_state,
                blocksize=quant_state.blocksize,
                compress_statistics=quant_state.nested,
                quant_type=quant_state.quant_type,
                bnb_quantized=True
            )

            # Replace the entire parameter (not just .data)
            # Need to get parent module and replace the parameter
            parent_module = model
            param_path = name.split('.')

            for part in param_path[:-1]:
                # Handle both attribute access and list indexing
                if part.isdigit():
                    parent_module = parent_module[int(part)]
                else:
                    parent_module = getattr(parent_module, part)

            param_name = param_path[-1]
            setattr(parent_module, param_name, quantized_param)

            quantized_count += 1

            if info["is_3d"]:
                print(f"  Loaded 3D quantized: {name} {original_shape}")
            else:
                print(f"  Loaded 2D quantized: {name} {original_shape}")

        else:
            # Regular parameter (not quantized)
            # Create new parameter with loaded data (matches dtype and device)
            new_param = torch.nn.Parameter(
                weight_data.to(device=device, dtype=param.dtype),
                requires_grad=param.requires_grad
            )

            # Replace the parameter entirely (same as quantized path)
            parent_module = model
            param_path = name.split('.')

            for part in param_path[:-1]:
                if part.isdigit():
                    parent_module = parent_module[int(part)]
                else:
                    parent_module = getattr(parent_module, part)

            param_name = param_path[-1]
            setattr(parent_module, param_name, new_param)

        loaded_count += 1

    print()
    print(f"✓ Loaded {loaded_count} parameters ({quantized_count} quantized)")
    print()

    # Move any remaining components (buffers, etc.) to device
    # The Params4bit are already on the device from reconstruction above
    print(f"Moving model to {device}...")
    model = model.to(device)
    print("✓ Model ready on device")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")
    print()

    print("="*80)
    print("LOAD COMPLETE")
    print("="*80)
    print(f"Model: {model.__class__.__name__}")
    print(f"Quantized weights: {quantized_count}")
    print(f"Device: {device}")
    print()

    return model, tokenizer


def reconstruct_quant_state(quant_state_dict, original_shape):
    """
    Reconstruct QuantState from saved dict.

    Args:
        quant_state_dict: Dict with quant_state keys (absmax, blocksize, etc.)
        original_shape: Original tensor shape

    Returns:
        QuantState object
    """
    # Parse dtype (might be string or torch.dtype)
    dtype_val = quant_state_dict.get("dtype", "torch.bfloat16")
    if isinstance(dtype_val, str):
        dtype = getattr(torch, dtype_val.replace("torch.", ""), torch.bfloat16)
    else:
        dtype = dtype_val

    # Handle nested quantization (double quant)
    state2 = None
    if "nested_absmax" in quant_state_dict:
        nested_blocksize = quant_state_dict["nested_blocksize"]
        if isinstance(nested_blocksize, torch.Tensor):
            nested_blocksize = int(nested_blocksize.item())

        state2 = QuantState(
            absmax=quant_state_dict["nested_absmax"],
            blocksize=nested_blocksize,
            code=quant_state_dict["nested_quant_map"],
            dtype=dtype,
        )

    # Parse blocksize (might be tensor or int)
    blocksize = quant_state_dict["blocksize"]
    if isinstance(blocksize, torch.Tensor):
        blocksize = int(blocksize.item())

    # Create main quant_state
    quant_state = QuantState(
        quant_type=quant_state_dict.get("quant_type", "nf4"),
        absmax=quant_state_dict["absmax"],
        blocksize=blocksize,
        code=quant_state_dict["quant_map"],
        dtype=dtype,
        shape=original_shape,
        offset=quant_state_dict.get("offset"),
        state2=state2,
    )

    return quant_state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load BnB quantized Granite model")
    parser.add_argument("model_path", help="Path to saved model")
    parser.add_argument("--device", default="cuda", help="Device to load to")
    parser.add_argument("--test-generation", action="store_true", help="Test with generation")

    args = parser.parse_args()

    model, tokenizer = load_quantized_model(args.model_path, device=args.device)

    if args.test_generation:
        print("Testing generation...")
        model.eval()

        inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: '{generated}'")
        print()
        print("✓ Generation works!")
