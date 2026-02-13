#!/usr/bin/env python3
"""
LoRA Fine-tuning for BnB 4-bit Quantized Granite MoE Models

This script demonstrates memory-efficient LoRA fine-tuning of IBM Granite MoE models:
- BitsAndBytes (BnB) 4-bit NF4 quantization for frozen MoE expert layers
- LoRA adapters on unquantized attention and Mamba layers
- Standard HuggingFace Transformers workflow

Key approach:
- **Quantize what we don't train**: MoE experts (frozen, 60-70% of parameters)
- **Train what we don't quantize**: Attention + Mamba (unquantized, where LoRA goes)

This is NOT qLoRA. We quantize frozen layers to save VRAM, while training on
unquantized layers for full precision gradients.

Usage:
    python train_lora.py \\
        --model-name ./granite-8b-quantized \\
        --dataset train_data.jsonl \\
        --output-dir ./output \\
        --batch-size 2 \\
        --gradient-accumulation-steps 8 \\
        --epochs 3 \\
        --learning-rate 2e-4 \\
        --rank 128 \\
        --alpha 256

Dataset Format:
    JSONL file with one conversation per line:
    {"messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."}
    ]}

Author: Community contribution for IBM Granite
License: Apache 2.0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from bitsandbytes.nn import Params4bit

# Import our custom quantized model loader
from load_quantized_model import load_quantized_model


# ============================================================================
# Dataset Processing
# ============================================================================

def load_jsonl_dataset(file_path: str) -> List[Dict]:
    """
    Load dataset from JSONL file.

    Each line should be a JSON object with a "messages" field containing
    a list of message dicts with "role" and "content" fields.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of conversation dicts
    """
    conversations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "messages" not in data:
                    print(f"Warning: Line {line_num} missing 'messages' field, skipping")
                    continue
                conversations.append(data)
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON: {e}")
                continue

    print(f"Loaded {len(conversations)} conversations from {file_path}")
    return conversations


def tokenize_conversation(
    conversation: Dict,
    tokenizer,
    max_length: int = 2048,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a conversation and create labels with proper masking.

    This function:
    1. Applies the chat template to format the conversation
    2. Tokenizes the formatted text
    3. Creates labels where system/user messages are masked with -100
       (so only assistant responses contribute to loss)

    Args:
        conversation: Dict with "messages" field
        tokenizer: HuggingFace tokenizer with chat template
        max_length: Maximum sequence length (default: 2048)

    Returns:
        Dict with input_ids, attention_mask, and labels tensors
    """
    messages = conversation["messages"]

    # Apply chat template to format the conversation
    # This converts the message list into the model's expected format
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # Tokenize the full conversation
    tokenized = tokenizer(
        formatted_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = tokenized["input_ids"][0]
    attention_mask = tokenized["attention_mask"][0]

    # Create labels: copy input_ids but mask non-assistant tokens
    labels = input_ids.clone()

    # We need to mask system and user messages, keeping only assistant responses
    # Strategy: tokenize each message separately to find token boundaries
    current_pos = 0

    for i, message in enumerate(messages):
        # Create a partial conversation up to this message
        partial_messages = messages[:i+1]
        partial_text = tokenizer.apply_chat_template(
            partial_messages,
            tokenize=False,
            add_generation_prompt=False
        )
        partial_tokens = tokenizer(partial_text, add_special_tokens=False)["input_ids"]
        end_pos = len(partial_tokens)

        # If this is not an assistant message, mask its tokens
        if message["role"] != "assistant":
            labels[current_pos:end_pos] = -100

        current_pos = end_pos

        # Stop if we've exceeded the input length (due to truncation)
        if current_pos >= len(input_ids):
            break

    # Mask padding tokens
    labels[attention_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def prepare_dataset(
    conversations: List[Dict],
    tokenizer,
    max_length: int = 2048,
) -> Dataset:
    """
    Prepare HuggingFace Dataset from conversations.

    Args:
        conversations: List of conversation dicts
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length

    Returns:
        HuggingFace Dataset ready for training
    """
    print(f"Tokenizing {len(conversations)} conversations...")

    tokenized_data = []
    for i, conv in enumerate(conversations):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(conversations)} conversations")

        try:
            tokenized = tokenize_conversation(conv, tokenizer, max_length)
            tokenized_data.append(tokenized)
        except Exception as e:
            print(f"Warning: Failed to tokenize conversation {i}: {e}")
            continue

    print(f"Successfully tokenized {len(tokenized_data)} conversations")

    # Create HuggingFace Dataset
    dataset = Dataset.from_dict({
        "input_ids": [item["input_ids"] for item in tokenized_data],
        "attention_mask": [item["attention_mask"] for item in tokenized_data],
        "labels": [item["labels"] for item in tokenized_data],
    })

    return dataset


# ============================================================================
# Model Setup
# ============================================================================

def setup_model_for_training(
    model,
    lora_rank: int = 128,
    lora_alpha: int = 256,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
):
    """
    Set up model for memory-efficient LoRA training.

    This function:
    1. Enables gradient checkpointing for memory efficiency
    2. Enables input gradients (required when frozen layers exist)
    3. Applies LoRA adapters to target modules (attention + mamba)

    Our approach:
    - MoE experts: Frozen and quantized (4-bit) → saves VRAM, no training
    - Attention/Mamba: Unquantized (BF16) with LoRA adapters → full precision training

    The model has quantized frozen layers (experts) and unquantized trainable layers
    (attention/mamba with LoRA). Gradients flow:

        input -> [frozen quantized experts] -> [attention/mamba + LoRA] -> output
                  (no gradients)                  (trainable adapters)

    We train only attention/mamba because that's typical for style or task adaptation.
    MoE experts stay frozen (they're knowledge storage, rarely need training).

    Args:
        model: Loaded quantized model
        lora_rank: LoRA rank (r) - size of low-rank bottleneck
        lora_alpha: LoRA alpha - scaling factor (typically 2*rank)
        lora_dropout: Dropout rate for LoRA layers
        target_modules: List of module names to apply LoRA to
                       (default: attention + mamba for Granite)

    Returns:
        Model with LoRA adapters applied
    """
    print("\n" + "="*80)
    print("SETTING UP MODEL FOR TRAINING")
    print("="*80)

    # Step 1: Enable gradient checkpointing
    # This saves memory by not storing all intermediate activations,
    # recomputing them during backward pass instead
    print("\n1. Enabling gradient checkpointing...")
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("   ✓ Gradient checkpointing enabled")
    else:
        print("   ⚠ Model doesn't support gradient checkpointing")

    # Step 2: Enable input gradients
    # CRITICAL: This is required for gradient flow through frozen quantized layers
    # Without this, gradients won't flow from the LoRA adapters back to the inputs
    print("\n2. Enabling input gradients (required for frozen base model)...")
    model.enable_input_require_grads()
    print("   ✓ Input gradients enabled")

    # Step 3: Configure LoRA target modules
    if target_modules is None:
        # Default: Target attention and mamba layers (Granite's hybrid architecture)
        # For Granite models, we target:
        # - Attention: q_proj, k_proj, v_proj, o_proj (standard transformer attention)
        # - Mamba: in_proj only (out_proj excluded - fused kernel bypasses LoRA)
        #
        # NOTE: mamba.out_proj is excluded because the mamba_split_conv1d_scan_combined
        # fused CUDA kernel passes out_proj.weight directly to the kernel, never calling
        # the nn.Module wrapper, so LoRA adapters on out_proj are never used during training.
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "mamba.in_proj",                          # Mamba in_proj (out_proj excluded)
        ]
        print("\n3. Using default target modules (Attention + Mamba in_proj):")
        print("   Note: mamba.out_proj excluded - fused kernel bypasses LoRA")
    else:
        print("\n3. Using custom target modules:")

    for module in target_modules:
        print(f"   - {module}")

    # Step 4: Create LoRA configuration
    print(f"\n4. Creating LoRA configuration...")
    print(f"   Rank (r): {lora_rank}")
    print(f"   Alpha: {lora_alpha}")
    print(f"   Dropout: {lora_dropout}")

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Step 5: Apply LoRA to model
    print("\n5. Applying LoRA adapters...")
    model = get_peft_model(model, lora_config)

    # Re-enable input gradients after PEFT (PEFT might modify the model)
    model.enable_input_require_grads()
    print("   ✓ LoRA adapters applied")
    print("   ✓ Input gradients re-enabled")

    # Step 6: Print training statistics
    print("\n6. Training configuration:")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())

    print(f"   Total parameters: {all_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Trainable %: {100 * trainable_params / all_params:.2f}%")

    # Step 7: Verify LoRA was applied
    lora_count = sum(1 for name, _ in model.named_parameters() if "lora" in name.lower())
    if lora_count == 0:
        raise ValueError(
            "ERROR: No LoRA adapters found! "
            "This means target_modules didn't match any layers in the model. "
            "Check your target_modules configuration."
        )

    print(f"   LoRA adapter parameters: {lora_count}")
    print("\n" + "="*80)

    return model


# ============================================================================
# Callbacks
# ============================================================================

class MemoryLoggingCallback(TrainerCallback):
    """Log GPU memory usage during training."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated(0) / 1e9
            reserved_gb = torch.cuda.memory_reserved(0) / 1e9
            logs["gpu_memory_allocated_gb"] = allocated_gb
            logs["gpu_memory_reserved_gb"] = reserved_gb


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for BnB 4-bit quantized Granite models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python train_lora.py \\
      --model-name ./granite-8b-quantized \\
      --dataset train_data.jsonl \\
      --eval-dataset eval_data.jsonl \\
      --output-dir ./output \\
      --batch-size 2 \\
      --gradient-accumulation-steps 8 \\
      --epochs 3 \\
      --learning-rate 2e-4 \\
      --rank 128 \\
      --alpha 256

Dataset format (JSONL):
  {"messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"},
      {"role": "assistant", "content": "Hi! How can I help you?"}
  ]}
        """
    )

    # Model arguments
    parser.add_argument("--model-name", type=str, required=True,
                       help="Path to BnB 4-bit quantized Granite model")

    # Data arguments
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to training dataset (JSONL file)")
    parser.add_argument("--eval-dataset", type=str, default=None,
                       help="Path to evaluation dataset (JSONL file, optional)")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                       help="Maximum sequence length (default: 2048)")

    # Training arguments
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for checkpoints and logs")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Per-device batch size (default: 2)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8,
                       help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs (default: 3)")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate (default: 2e-4)")
    parser.add_argument("--warmup-steps", type=int, default=100,
                       help="Warmup steps (default: 100)")
    parser.add_argument("--logging-steps", type=int, default=10,
                       help="Log every N steps (default: 10)")
    parser.add_argument("--save-steps", type=int, default=500,
                       help="Save checkpoint every N steps (default: 500)")
    parser.add_argument("--eval-steps", type=int, default=500,
                       help="Evaluate every N steps (default: 500)")

    # LoRA arguments
    parser.add_argument("--rank", type=int, default=128,
                       help="LoRA rank (default: 128)")
    parser.add_argument("--alpha", type=int, default=256,
                       help="LoRA alpha (default: 256)")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                       help="LoRA dropout (default: 0.05)")

    # Additional options
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--wandb-project", type=str, default=None,
                       help="Weights & Biases project name (optional)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("BnB 4-BIT GRANITE LORA TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output: {args.output_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  LoRA rank: {args.rank}")
    print(f"  LoRA alpha: {args.alpha}")
    print()

    # ========================================================================
    # Step 1: Load the quantized model
    # ========================================================================
    print("="*80)
    print("STEP 1: LOADING QUANTIZED MODEL")
    print("="*80)
    print()
    print("Loading BnB 4-bit quantized Granite model...")
    print("This preserves the 4-bit quantization for memory efficiency.")
    print()

    model, tokenizer = load_quantized_model(args.model_name, device="cuda")

    # Verify quantization was preserved
    bnb_count = sum(1 for _, p in model.named_parameters() if isinstance(p, Params4bit))
    print(f"✓ Model loaded with {bnb_count} quantized (4-bit) parameters")

    # Check memory usage
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated(0) / 1e9
        print(f"✓ GPU memory: {allocated_gb:.2f} GB")

    print()

    # ========================================================================
    # Step 2: Load and tokenize datasets
    # ========================================================================
    print("="*80)
    print("STEP 2: LOADING AND TOKENIZING DATASETS")
    print("="*80)
    print()

    # Load training data
    train_conversations = load_jsonl_dataset(args.dataset)
    train_dataset = prepare_dataset(
        train_conversations,
        tokenizer,
        max_length=args.max_seq_length
    )

    # Load evaluation data if provided
    eval_dataset = None
    if args.eval_dataset:
        eval_conversations = load_jsonl_dataset(args.eval_dataset)
        eval_dataset = prepare_dataset(
            eval_conversations,
            tokenizer,
            max_length=args.max_seq_length
        )
        print(f"Loaded {len(eval_dataset)} eval examples")

    print()

    # ========================================================================
    # Step 3: Set up model for training (apply LoRA)
    # ========================================================================
    model = setup_model_for_training(
        model,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.lora_dropout,
    )

    # ========================================================================
    # Step 4: Configure training
    # ========================================================================
    print("="*80)
    print("STEP 4: CONFIGURING TRAINING")
    print("="*80)
    print()

    # Set up Weights & Biases if requested
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        report_to = "wandb"
        print(f"Weights & Biases: {args.wandb_project}")
    else:
        report_to = "none"
        print("Weights & Biases: disabled")

    # Create training arguments
    training_args = TrainingArguments(
        # Output
        output_dir=str(output_dir),

        # Training schedule
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Optimization
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,

        # Mixed precision
        bf16=True,  # Use bfloat16 for Ampere+ GPUs

        # Logging
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to=report_to,

        # Checkpointing
        save_steps=args.save_steps,
        save_total_limit=3,  # Keep only last 3 checkpoints

        # Evaluation
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        per_device_eval_batch_size=args.batch_size,

        # Memory optimization
        gradient_checkpointing=True,

        # Other
        remove_unused_columns=False,
        dataloader_num_workers=0,  # 0 for safety with pickled data
    )

    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Total training steps: ~{len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps) * args.epochs}")
    print()

    # ========================================================================
    # Step 5: Create Trainer and start training
    # ========================================================================
    print("="*80)
    print("STEP 5: CREATING TRAINER")
    print("="*80)
    print()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[MemoryLoggingCallback()],
    )

    print("Trainer created successfully")
    print()

    # ========================================================================
    # Step 6: Train!
    # ========================================================================
    print("="*80)
    print("STEP 6: STARTING TRAINING")
    print("="*80)
    print()

    # Show memory before training
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated(0) / 1e9
        reserved_gb = torch.cuda.memory_reserved(0) / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory before training:")
        print(f"  Total: {total_gb:.2f} GB")
        print(f"  Allocated: {allocated_gb:.2f} GB")
        print(f"  Reserved: {reserved_gb:.2f} GB")
        print(f"  Free: {total_gb - allocated_gb:.2f} GB")
        print()

    # Train
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print()
    print("="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print()

    # ========================================================================
    # Step 7: Save the final model
    # ========================================================================
    print("Saving final model...")

    # Save the LoRA adapter
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print(f"✓ Model saved to {final_dir}")
    print()
    print("To use the trained model:")
    print(f"  from peft import PeftModel")
    print(f"  from load_quantized_model import load_quantized_model")
    print(f"  ")
    print(f"  base_model, tokenizer = load_quantized_model('{args.model_name}')")
    print(f"  model = PeftModel.from_pretrained(base_model, '{final_dir}')")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
