#!/usr/bin/env python3
"""
Merge LoRA adapter into the original unquantized Granite model.

This script merges a trained LoRA adapter back into the original BF16 model
for deployment. This is the final step in the workflow:

1. Quantize model (on high-VRAM machine)
2. Train LoRA adapter (on low-VRAM machine with quantized model)
3. Merge adapter into original model (on high-VRAM machine) ← THIS SCRIPT
4. Deploy merged model

IMPORTANT: This merges into the ORIGINAL unquantized model, not the quantized one.
The quantized model was just used for training; we merge into the full precision
model for best inference quality.

Requirements:
- High-VRAM machine (same as quantization step)
- Original model (ibm-granite/granite-8b-code-base or similar)
- Trained LoRA adapter (from training step)

Usage:
    python merge_adapter.py \\
        --base-model ibm-granite/granite-8b-code-base \\
        --adapter ./output/final \\
        --output ./granite-8b-merged

The merged model can then be:
- Shared on HuggingFace Hub
- Used for inference without needing the adapter
- Converted to other formats (GGUF, etc.)
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_adapter(
    base_model_name: str,
    adapter_path: str,
    output_dir: str,
    device_map: str = "auto",
):
    """
    Merge LoRA adapter into base model and save.

    Args:
        base_model_name: Original model name or path (e.g., "ibm-granite/granite-8b-code-base")
        adapter_path: Path to trained LoRA adapter
        output_dir: Output directory for merged model
        device_map: Device map for model loading (default: "auto")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("MERGING LORA ADAPTER INTO BASE MODEL")
    print("="*80)
    print()
    print(f"Base model: {base_model_name}")
    print(f"LoRA adapter: {adapter_path}")
    print(f"Output: {output_dir}")
    print()

    # ========================================================================
    # Step 1: Load the ORIGINAL base model (unquantized, BF16)
    # ========================================================================
    print("="*80)
    print("STEP 1: LOADING ORIGINAL BASE MODEL")
    print("="*80)
    print()
    print("Loading unquantized BF16 model...")
    print("⚠️  This requires the same VRAM as the original model:")
    print("   Granite 8B: ~17GB")
    print("   Granite 32B: ~65GB")
    print()

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ Base model loaded")

    # Show memory usage
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated(0) / 1e9
        print(f"✓ GPU memory: {allocated_gb:.2f} GB")
    print()

    # ========================================================================
    # Step 2: Load the LoRA adapter
    # ========================================================================
    print("="*80)
    print("STEP 2: LOADING LORA ADAPTER")
    print("="*80)
    print()

    print(f"Loading adapter from: {adapter_path}")
    model_with_adapter = PeftModel.from_pretrained(
        base_model,
        adapter_path,
    )
    print("✓ Adapter loaded")
    print()

    # Show adapter info
    print("Adapter configuration:")
    if hasattr(model_with_adapter, 'peft_config'):
        for adapter_name, config in model_with_adapter.peft_config.items():
            print(f"  Adapter: {adapter_name}")
            print(f"  Rank: {config.r}")
            print(f"  Alpha: {config.lora_alpha}")
            print(f"  Target modules: {config.target_modules}")
    print()

    # ========================================================================
    # Step 3: Merge adapter into base model
    # ========================================================================
    print("="*80)
    print("STEP 3: MERGING ADAPTER INTO BASE MODEL")
    print("="*80)
    print()

    print("Merging LoRA weights into base model...")
    print("This combines the adapter weights with the base weights.")
    print()

    # Merge and unload adapter
    merged_model = model_with_adapter.merge_and_unload()

    print("✓ Adapter merged successfully")
    print()

    # ========================================================================
    # Step 4: Save the merged model
    # ========================================================================
    print("="*80)
    print("STEP 4: SAVING MERGED MODEL")
    print("="*80)
    print()

    print(f"Saving merged model to: {output_dir}")
    print()

    # Save model
    print("Saving model weights...")
    try:
        merged_model.save_pretrained(
            str(output_dir),
            safe_serialization=True,  # Use safetensors (recommended)
            max_shard_size="5GB",     # Shard large models
        )
        print("✓ Model saved with safetensors format")
    except Exception as e:
        print(f"⚠️  Safetensors save failed: {e}")
        print("Falling back to pytorch_model.bin format...")
        merged_model.save_pretrained(
            str(output_dir),
            safe_serialization=False,
            max_shard_size="5GB",
        )
        print("✓ Model saved with pytorch_model.bin format")

    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer.save_pretrained(str(output_dir))
    print("✓ Tokenizer saved")
    print()

    # Show final size
    total_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
    total_size_gb = total_size / 1e9
    print(f"Total size: {total_size_gb:.2f} GB")
    print()

    # ========================================================================
    # Done!
    # ========================================================================
    print("="*80)
    print("MERGE COMPLETE!")
    print("="*80)
    print()
    print(f"Merged model saved to: {output_dir}")
    print()
    print("You can now:")
    print(f"  1. Test it:")
    print(f"     python -c \"from transformers import AutoModelForCausalLM; \\")
    print(f"                  model = AutoModelForCausalLM.from_pretrained('{output_dir}')\"")
    print()
    print(f"  2. Share it on HuggingFace Hub:")
    print(f"     huggingface-cli upload {output_dir} <your-username>/<model-name>")
    print()
    print(f"  3. Convert to other formats:")
    print(f"     llama.cpp: convert.py {output_dir}")
    print(f"     GGUF: python convert-hf-to-gguf.py {output_dir}")
    print()
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into original Granite model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge adapter trained on quantized Granite 8B
  python merge_adapter.py \\
      --base-model ibm-granite/granite-8b-code-base \\
      --adapter ./output/final \\
      --output ./granite-8b-custom

  # Merge adapter for Granite 32B
  python merge_adapter.py \\
      --base-model ibm-granite/granite-32b-code-base \\
      --adapter ./lora-adapter \\
      --output ./granite-32b-custom

IMPORTANT:
  - Use the ORIGINAL model name (e.g., ibm-granite/granite-8b-code-base)
  - NOT the quantized model path (that was just for training)
  - Requires same VRAM as original model (17GB for 8B, 65GB for 32B)
        """
    )

    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Original base model name or path (e.g., ibm-granite/granite-8b-code-base)"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to trained LoRA adapter directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for merged model"
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for model loading (default: auto)"
    )

    args = parser.parse_args()

    # Validate paths
    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"Error: Adapter path does not exist: {adapter_path}")
        sys.exit(1)

    # Check for adapter config
    adapter_config = adapter_path / "adapter_config.json"
    if not adapter_config.exists():
        print(f"Error: No adapter_config.json found in {adapter_path}")
        print("This doesn't appear to be a valid LoRA adapter directory.")
        sys.exit(1)

    # Merge
    merge_adapter(
        base_model_name=args.base_model,
        adapter_path=args.adapter,
        output_dir=args.output,
        device_map=args.device_map,
    )


if __name__ == "__main__":
    main()
