#!/usr/bin/env python3
"""
Test BnB 4-bit quantization with Granite MoE for forward AND backward pass.

This tests if BnB can handle:
1. Forward pass with 3D MoE expert weights
2. Backward pass (computing gradients) for training
3. Gradient checkpointing

If this works, we just need a custom save/load format.
If this fails, BnB is fundamentally incompatible.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import GraniteMoeHybridParallelExperts

import bitsandbytes as bnb
from bitsandbytes.nn import Params4bit

# Import and apply ALL MoE quantization patches (self-contained solution)
from moe_linear_4bit import apply_all_patches

print("="*80)
print("Testing BnB 4-bit Quantization with Granite MoE")
print("="*80)
print()

# Apply all necessary patches for BnB + MoE
apply_all_patches()

# Load model
print("Loading Granite Tiny in bf16...")
model = AutoModelForCausalLM.from_pretrained(
    "ibm-granite/granite-3.1-1b-a400m-instruct",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "ibm-granite/granite-3.1-1b-a400m-instruct",
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✓ Model loaded")
print()

# Count parameters before quantization
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print()

# Quantize MLPs (both shared_mlp and MoE experts)
print("Quantizing MLPs to 4-bit...")
quantized_count = 0

def quantize_layer(module, name, weight_shape):
    global quantized_count
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

# Iterate through all layers
for layer_idx in range(len(model.model.layers)):
    layer = model.model.layers[layer_idx]

    # Quantize shared_mlp Linear layers
    if hasattr(layer, 'shared_mlp'):
        if hasattr(layer.shared_mlp, 'input_linear'):
            quantize_layer(
                layer.shared_mlp.input_linear,
                f"layers.{layer_idx}.shared_mlp.input_linear",
                layer.shared_mlp.input_linear.weight.shape
            )
        if hasattr(layer.shared_mlp, 'output_linear'):
            quantize_layer(
                layer.shared_mlp.output_linear,
                f"layers.{layer_idx}.shared_mlp.output_linear",
                layer.shared_mlp.output_linear.weight.shape
            )

    # Quantize MoE expert weights (3D tensors)
    if hasattr(layer, 'block_sparse_moe'):
        if hasattr(layer.block_sparse_moe, 'input_linear'):
            experts = layer.block_sparse_moe.input_linear
            quantize_layer(
                experts,
                f"layers.{layer_idx}.block_sparse_moe.input_linear",
                experts.weight.shape
            )
        if hasattr(layer.block_sparse_moe, 'output_linear'):
            experts = layer.block_sparse_moe.output_linear
            quantize_layer(
                experts,
                f"layers.{layer_idx}.block_sparse_moe.output_linear",
                experts.weight.shape
            )

print()

print(f"✓ Quantized {quantized_count} weight tensors")
print()

# Test 1: Forward pass
print("TEST 1: Forward pass")
print("-" * 40)
try:
    inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    print(f"✓ Forward pass successful")
    print(f"  Output shape: {logits.shape}")
    print(f"  Output dtype: {logits.dtype}")
    print()
except Exception as e:
    print(f"✗ Forward pass FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Backward pass (gradient computation)
print("TEST 2: Backward pass (gradient computation)")
print("-" * 40)
try:
    # Create a simple loss
    inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)

    # Enable gradients for testing
    model.train()

    # Forward pass
    outputs = model(**inputs, labels=inputs.input_ids)
    loss = outputs.loss

    print(f"  Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Check if gradients were computed for non-quantized parameters
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_count += 1

    print(f"✓ Backward pass successful")
    print(f"  Parameters with gradients: {grad_count}")
    print()

    # Clear gradients
    model.zero_grad()

except Exception as e:
    print(f"✗ Backward pass FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: With gradient checkpointing
print("TEST 3: Gradient checkpointing")
print("-" * 40)
try:
    model.gradient_checkpointing_enable()

    inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
    outputs = model(**inputs, labels=inputs.input_ids)
    loss = outputs.loss

    print(f"  Loss: {loss.item():.4f}")

    loss.backward()

    print(f"✓ Gradient checkpointing works")
    print()

    model.zero_grad()
    model.gradient_checkpointing_disable()

except Exception as e:
    print(f"✗ Gradient checkpointing FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Generation
print("TEST 4: Generation")
print("-" * 40)
try:
    model.eval()
    inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✓ Generation works")
    print(f"  Prompt: 'Hello, world!'")
    print(f"  Generated: '{generated}'")
    print()

except Exception as e:
    print(f"✗ Generation FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("="*80)
print("ALL TESTS PASSED!")
print("="*80)
print()
print("BnB 4-bit quantization works with Granite MoE for:")
print("  ✓ Forward pass")
print("  ✓ Backward pass (gradients)")
print("  ✓ Gradient checkpointing")
print("  ✓ Generation")
print()
print("Next step: Create custom save/load format for 3D MoE weights")
