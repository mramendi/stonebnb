#!/usr/bin/env python3
"""
Simple inference script for BnB quantized Granite models with LoRA adapters.

Usage:
    # Interactive mode
    python inference.py --model-name ./granite-8b-quantized --adapter ./output/final

    # Single prompt
    python inference.py --model-name ./granite-8b-quantized --adapter ./output/final --prompt "What is AI?"

    # Batch inference from file
    python inference.py --model-name ./granite-8b-quantized --adapter ./output/final --input prompts.txt --output responses.txt
"""

import argparse
import sys
from pathlib import Path

import torch
from load_quantized_model import load_quantized_model
from peft import PeftModel


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    system_prompt: str = None,
) -> str:
    """
    Generate a response for a single prompt.

    Args:
        model: The model (with or without adapter)
        tokenizer: Tokenizer
        prompt: User prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy, 1 = creative)
        top_p: Nucleus sampling threshold
        system_prompt: Optional system prompt

    Returns:
        Generated text
    """
    # Format as chat messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant's response (remove the prompt)
    # The response includes the formatted prompt + generation
    # We want only the new tokens after the generation prompt
    response = full_response[len(text):].strip()

    return response


def interactive_mode(model, tokenizer, system_prompt=None):
    """
    Interactive chat mode.

    Args:
        model: The model
        tokenizer: Tokenizer
        system_prompt: Optional system prompt
    """
    print("="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print()
    print("Type your prompts below. Commands:")
    print("  /quit or /exit - Exit")
    print("  /clear - Clear conversation history")
    print("  /system <text> - Set system prompt")
    print()

    conversation_history = []
    current_system = system_prompt

    while True:
        try:
            prompt = input("\nYou: ").strip()

            if not prompt:
                continue

            # Handle commands
            if prompt.lower() in ["/quit", "/exit"]:
                print("\nGoodbye!")
                break

            if prompt.lower() == "/clear":
                conversation_history = []
                print("Conversation cleared.")
                continue

            if prompt.lower().startswith("/system "):
                current_system = prompt[8:].strip()
                print(f"System prompt set to: {current_system}")
                continue

            # Generate response
            print("\nAssistant: ", end="", flush=True)

            response = generate_response(
                model,
                tokenizer,
                prompt,
                system_prompt=current_system
            )

            print(response)

            # Store in history (optional, for future multi-turn support)
            conversation_history.append({"role": "user", "content": prompt})
            conversation_history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.")
        except Exception as e:
            print(f"\nError: {e}")


def batch_mode(model, tokenizer, input_file, output_file, system_prompt=None):
    """
    Batch inference from file.

    Args:
        model: The model
        tokenizer: Tokenizer
        input_file: Input file with one prompt per line
        output_file: Output file for responses
        system_prompt: Optional system prompt
    """
    print("="*80)
    print("BATCH INFERENCE MODE")
    print("="*80)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print()

    # Read prompts
    with open(input_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(prompts)} prompts...")
    print()

    # Generate responses
    responses = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Processing: {prompt[:50]}..." if len(prompt) > 50 else f"[{i}/{len(prompts)}] Processing: {prompt}")

        try:
            response = generate_response(
                model,
                tokenizer,
                prompt,
                system_prompt=system_prompt
            )
            responses.append(response)
        except Exception as e:
            print(f"  Error: {e}")
            responses.append(f"ERROR: {e}")

    # Write responses
    with open(output_file, 'w', encoding='utf-8') as f:
        for response in responses:
            f.write(response + "\n")

    print()
    print(f"✓ Wrote {len(responses)} responses to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Inference with BnB quantized Granite models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Model arguments
    parser.add_argument("--model-name", type=str, required=True,
                       help="Path to quantized Granite model")
    parser.add_argument("--adapter", type=str, default=None,
                       help="Path to LoRA adapter (optional)")

    # Inference mode
    parser.add_argument("--prompt", type=str, default=None,
                       help="Single prompt (if not specified, enters interactive mode)")
    parser.add_argument("--input", type=str, default=None,
                       help="Input file for batch mode (one prompt per line)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for batch mode")

    # Generation parameters
    parser.add_argument("--system-prompt", type=str, default=None,
                       help="System prompt")
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="Maximum tokens to generate (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Nucleus sampling top-p (default: 0.9)")

    args = parser.parse_args()

    # Validate batch mode arguments
    if args.input and not args.output:
        print("Error: --output required when using --input")
        sys.exit(1)
    if args.output and not args.input:
        print("Error: --input required when using --output")
        sys.exit(1)

    # Load model
    print("="*80)
    print("LOADING MODEL")
    print("="*80)
    print()

    print(f"Loading base model: {args.model_name}")
    model, tokenizer = load_quantized_model(args.model_name, device="cuda")

    # Load adapter if specified
    if args.adapter:
        print(f"Loading LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()
    print("✓ Model ready")
    print()

    # Show memory usage
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated(0) / 1e9
        print(f"GPU memory: {allocated_gb:.2f} GB")
        print()

    # Select mode
    if args.input:
        # Batch mode
        batch_mode(
            model,
            tokenizer,
            args.input,
            args.output,
            system_prompt=args.system_prompt
        )
    elif args.prompt:
        # Single prompt mode
        print("="*80)
        print("SINGLE PROMPT MODE")
        print("="*80)
        print()
        print(f"Prompt: {args.prompt}")
        print()
        print("Response:")
        print("-"*80)

        response = generate_response(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            system_prompt=args.system_prompt
        )

        print(response)
        print("-"*80)
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, system_prompt=args.system_prompt)


if __name__ == "__main__":
    main()
