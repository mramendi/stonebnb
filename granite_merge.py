#!/usr/bin/env python3
"""
Merge LoRA adapter with Granite base model.

This loads a BF16 Granite base model and a trained LoRA adapter,
merges them, and saves the merged model.

VRAM Requirements:
- Granite 1B (Tiny): ~17GB
- Granite 8B: ~35GB
- Granite 32B (Small): ~65GB

The merged model will be in BF16 format and can be:
1. Used directly for inference
2. Re-quantized with quantize_and_save_granite.py for efficient deployment

Usage:
    python granite_merge.py \\
        --base-model ibm-granite/granite-3.1-1b-a400m-instruct \\
        --adapter ./results/train-r128-a256/checkpoint-500 \\
        --output ./granite-tiny-merged
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_adapter(base_model_name, adapter_path, output_dir):
    """
    Merge LoRA adapter with base model.

    Args:
        base_model_name: HuggingFace model name or path to BF16 base model
        adapter_path: Path to LoRA adapter checkpoint
        output_dir: Output directory for merged model
    """
    output_dir = Path(output_dir)
    adapter_path = Path(adapter_path)

    print("="*80)
    print("Merging LoRA Adapter with Granite Base Model")
    print("="*80)
    print(f"Base model: {base_model_name}")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {output_dir}")
    print()

    if not adapter_path.exists():
        raise ValueError(f"Adapter path not found: {adapter_path}")

    # Load base model
    print("Loading base model in BF16...")
    print("(This requires VRAM for full model)")
    print()

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ Base model loaded")
    print()

    # Check VRAM
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"GPU Memory after base model load:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print()

    # Load adapter
    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    print("✓ Adapter loaded")
    print()

    # Merge adapter
    print("Merging adapter into base model...")
    model = model.merge_and_unload()
    print("✓ Adapter merged")
    print()

    # Save merged model
    print(f"Saving merged model to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("✓ Merged model saved")
    print()

    # Final summary
    print("="*80)
    print("MERGE COMPLETE")
    print("="*80)
    print(f"Merged model saved to: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Use directly for inference")
    print("  2. Or re-quantize for efficient deployment:")
    print(f"     python quantize_and_save_granite.py {output_dir} {output_dir}-bnb")
    print()


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with Granite base model")
    parser.add_argument("--base-model", required=True,
                       help="HuggingFace model name or path to BF16 base model")
    parser.add_argument("--adapter", required=True,
                       help="Path to LoRA adapter checkpoint directory")
    parser.add_argument("--output", required=True,
                       help="Output directory for merged model")

    args = parser.parse_args()

    merge_adapter(args.base_model, args.adapter, args.output)


if __name__ == "__main__":
    main()
