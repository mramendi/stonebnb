#!/usr/bin/env python3
"""
Check if lm_head.weight is in the saved model file.
"""

import sys
import torch
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python check_lm_head_in_saved_model.py <model_path>")
    sys.exit(1)

model_path = Path(sys.argv[1])

if not model_path.exists():
    print(f"Error: {model_path} does not exist")
    sys.exit(1)

state_dict_file = model_path / "pytorch_model.bin"

if not state_dict_file.exists():
    print(f"Error: {state_dict_file} does not exist")
    sys.exit(1)

print(f"Loading state_dict from: {state_dict_file}")
state_dict = torch.load(state_dict_file, map_location="cpu")

print(f"\nTotal keys in state_dict: {len(state_dict)}")
print()

# Check for lm_head
lm_head_keys = [k for k in state_dict.keys() if 'lm_head' in k]

print("="*80)
print("LM_HEAD KEYS:")
print("="*80)

if lm_head_keys:
    print(f"Found {len(lm_head_keys)} lm_head keys:")
    for key in lm_head_keys:
        value = state_dict[key]
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, all_zeros={value.eq(0).all().item()}")
        else:
            print(f"  {key}: {type(value).__name__}")
else:
    print("NO lm_head keys found!")

print()

# Look for top-level keys that might include lm_head
print("="*80)
print("TOP-LEVEL KEYS (first 20):")
print("="*80)

top_level_keys = sorted([k for k in state_dict.keys() if not any(suffix in k for suffix in
    ['.quant_type', '.absmax', '.blocksize', '.quant_map', '.dtype', '.shape', '.offset',
     '.nested_absmax', '.nested_blocksize', '.nested_quant_map'])])

for i, key in enumerate(top_level_keys[:20]):
    value = state_dict[key]
    if isinstance(value, torch.Tensor):
        print(f"  {key}: shape={value.shape}")
    else:
        print(f"  {key}: {type(value).__name__}")

if len(top_level_keys) > 20:
    print(f"  ... and {len(top_level_keys) - 20} more")

print()
print(f"Total parameter keys (excluding quant_state metadata): {len(top_level_keys)}")
