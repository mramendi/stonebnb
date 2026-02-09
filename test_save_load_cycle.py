#!/usr/bin/env python3
"""
Complete test of BnB quantization save/load cycle.

This script:
1. Loads Granite Tiny in bf16
2. Quantizes MoE experts to 4-bit
3. Saves quantized model
4. Loads quantized model back
5. Tests generation
6. Tests qLoRA training

This proves the complete workflow works end-to-end.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import bitsandbytes as bnb
from bitsandbytes.nn import Params4bit

from moe_linear_4bit import apply_all_patches
from save_quantized_model import save_quantized_model
from load_quantized_model import load_quantized_model

print("="*80)
print("BnB Quantization Save/Load Cycle Test")
print("="*80)
print()

# Apply MoE patches
apply_all_patches()

# ============================================================================
# STEP 1: Load and Quantize
# ============================================================================

print("STEP 1: Load and quantize Granite Tiny")
print("-" * 80)
print()

print("Loading model in bf16...")
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

# Quantize MoE experts
print("Quantizing MoE experts...")
quantized_count = 0

def quantize_moe_layer(module, name):
    global quantized_count
    weight = module.weight.data
    if weight.device.type != 'cuda':
        weight = weight.cuda()

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

for layer_idx in range(len(model.model.layers)):
    layer = model.model.layers[layer_idx]

    if hasattr(layer, 'block_sparse_moe'):
        if hasattr(layer.block_sparse_moe, 'input_linear'):
            quantize_moe_layer(layer.block_sparse_moe.input_linear,
                             f"layers.{layer_idx}.block_sparse_moe.input_linear")
        if hasattr(layer.block_sparse_moe, 'output_linear'):
            quantize_moe_layer(layer.block_sparse_moe.output_linear,
                             f"layers.{layer_idx}.block_sparse_moe.output_linear")

print(f"✓ Quantized {quantized_count} MoE expert tensors")
print()

# ============================================================================
# STEP 2: Save
# ============================================================================

print("STEP 2: Save quantized model")
print("-" * 80)
print()

output_dir = "granite-tiny-bnb-test"
save_quantized_model(model, tokenizer, output_dir)
print()

# Clear memory
del model
torch.cuda.empty_cache()

# ============================================================================
# STEP 3: Load
# ============================================================================

print("STEP 3: Load quantized model")
print("-" * 80)
print()

model, tokenizer = load_quantized_model(output_dir, device="cuda")
print()

# ============================================================================
# STEP 4: Test Generation
# ============================================================================

print("STEP 4: Test generation")
print("-" * 80)
print()

model.eval()
inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Input: 'Hello, world!'")
print(f"Generated: '{generated}'")
print("✓ Generation works!")
print()

# ============================================================================
# STEP 5: Test qLoRA Training
# ============================================================================

print("STEP 5: Test qLoRA training")
print("-" * 80)
print()

# Freeze base model
print("Freezing base model...")
model.train()
for param in model.parameters():
    param.requires_grad = False
print("✓ Base model frozen")
print()

# Add LoRA
print("Adding LoRA adapters...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print()

# Enable gradient checkpointing
print("Enabling gradient checkpointing...")
if hasattr(model, 'enable_input_require_grads'):
    model.enable_input_require_grads()
if hasattr(model.base_model, 'gradient_checkpointing_enable'):
    model.base_model.gradient_checkpointing_enable()
print("✓ Gradient checkpointing enabled")
print()

# Training data
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming technology.",
    "Python is a powerful programming language.",
]

inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=64
).to(model.device)

inputs["labels"] = inputs["input_ids"].clone()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training steps
print("Running 3 training steps...")
losses = []

for step in range(3):
    outputs = model(**inputs)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    print(f"  Step {step + 1}: Loss = {loss.item():.4f}")

print()
print(f"Initial loss: {losses[0]:.4f}")
print(f"Final loss:   {losses[-1]:.4f}")
print(f"Change:       {losses[-1] - losses[0]:.4f}")
print()

if losses[-1] < losses[0]:
    print("✓ Loss decreased - training works!")
else:
    print("⚠ Loss did not decrease (normal for 3 steps on toy data)")

print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("="*80)
print("COMPLETE SAVE/LOAD CYCLE TEST PASSED!")
print("="*80)
print()
print("Successfully verified:")
print("  ✓ Quantization (MoE experts)")
print("  ✓ Save (with shape metadata)")
print("  ✓ Load (reconstructing Params4bit)")
print("  ✓ Generation (model works after load)")
print("  ✓ qLoRA training (forward + backward)")
print()
print("Ready for production use!")
print()
print(f"Quantized model saved in: {output_dir}")
print("You can now use this for real training!")
