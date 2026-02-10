# StoneBnB: BitsAndBytes Quantization for Granite MoE Models


For most small(ish) LLMs, you can use quantization to reduce VRAM requirements while training. Typically, you just quantize the entire model to a 4 or 8 bit format, train a LoRA  adapter on that wuantized model, then apply it to the original BF16 model. This method is called qLoRA and Unsloth supports it directly, using bitsandbytes (BnB) quantization. Another popular method of quantization is torchao. 

However, with the IBM Granite 4 Small and Tiny models (Mamba2-hybrid, MoE) qLoRA is not available, at least in 4 bits. Stock BnB fails to compress these models to any significant degree, because it does not natively support the 3D tensors that are used in the MoE MLP layers that comprise the bulk of the models. torchao 4-bit methods can be adapted to compress the tensors and infer, but don't support the backward pass for training.

The result is that one normally needs a GPU with VRAM to fit the entire unquantized BF16 model (so 64 Gb just for the weights of Granite 4 small, before any overhead for actual training). Or maybe a Hopper/Blackwell chip allows  FP8 qLoRA - but for Granite 4 Small, that is still 32 Gb, leaving zero space for training on a 5090.

We propose a method of using bitsandbytes with some custom scaffolding and a custom model saving format to quantize the MLP layers in the experts to 4 bits, enabling memory-efficient training. **This is not qLoRA** - the quantized layers cannot be trained, the layers you train are not quantized. However, training only the attention and Mamba layers is a very common fine-tuning approach with Granite. StoneBnB enables this method with significantly less VRAM; the model is pre-quantized on a large-VRAM GPU, then can be trained and tested on a much smaller one, with the final merge of the adapter into the original model on the large GPU again.

The main drawback we observe so far is a significant speed penalty. However this frammerosk was NOT properly tested yet. Try at your own risk.

![mamba in a stone wall](https://d25f54r5k7x61w.archive.is/kGCOh/2311be5f87eb726cb68ecdc961879ff82d11b33a.jpg)
(image credit: [Nick Evans](https://www.citizen.co.za/news/south-africa/black-mamba-caught-after-hiding-in-wall-of-house-for-weeks/))


This repository contains complete proof-of-concept code for this approach. We hope it can help fine-tune Grenite models and inform improvement of training frameworks.

**Key Innovation**: Patches BitsAndBytes to handle 3D MoE expert tensors, enabling 4-bit quantization of frozen layers while training LoRA adapters on unquantized attention and Mamba layers.

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Quantize a Model](#1-quantize-a-model)
  - [2. Train with LoRA](#2-train-with-lora)
  - [3. Use the Trained Model](#3-use-the-trained-model)
- [How It Works](#how-it-works)
- [Memory Requirements](#memory-requirements)
- [Files](#files)
- [StoneBnB Format](#stonebnb-format)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

This package solves a critical problem: **Granite MoE models use 3D weight tensors for their expert layers**, which BitsAndBytes didn't originally support for quantization.

**What we do**:
1. Patch BitsAndBytes to handle 3D tensors in MoE layers
2. Quantize MoE expert layers to 4-bit NF4 (60-70% of model parameters → 4x compression)
3. Apply LoRA adapters to **unquantized** attention and Mamba layers (the layers we actually train)
4. MoE experts stay frozen and quantized, saving massive amounts of VRAM

**This is NOT qLoRA** (which trains on quantized layers). Instead, we:
- **Quantize what we don't train**: MoE experts (frozen, too risky to train anyway)
- **Train what we don't quantize**: Attention + Mamba (unquantized, where LoRA adapters go)

**Benefits**:
- **Memory efficient**: Train 32B Granite models on 24GB GPUs
- **Safe**: No quality degradation from quantization (we don't train through quantized layers)
- **Practical**: Matches typical training patterns (most people only train attention/mamba for style or task adaptation)

## Quick Start

```bash
# 1. Quantize your Granite model (requires high-VRAM machine: A100 40GB+)
python quantize_and_save_granite.py \
    ibm-granite/granite-4.0-h-small \
    ./granite-small-quantized

# Alternatively, you can use versions on HuggingFace:

# Tiny: https://huggingface.co/ramendik/granite-4.0-h-tiny-stonebnb
# Small: https://huggingface.co/ramendik/granite-4.0-h-small-stonebnb

# 2. Fine-tune with LoRA (works on 32GB GPU for Granite 4-h small, 8 Gb for Granite 4-h Tiny)
python train_lora.py \
    --model-name ./granite-small-quantized \
    --dataset train_data.jsonl \
    --output-dir ./output \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --epochs 3 \
    --rank 128 \
    --alpha 256

# you can also use --model-name ramendik/granite-4.0-h-small-stonebnb
# Also supports Granite 4-h Tiny

# 3. Test the trained model by inferring, applying the adapter in ./output/final

python
>>> from load_quantized_model import load_quantized_model
>>> from peft import PeftModel
>>> base_model, tokenizer = load_quantized_model("./granite-small-quantized") 
# or: base_model, tokenizer = load_quantized_model("ramendik/granite-4.0-h-small-stonebnb")
>>> model = PeftModel.from_pretrained(base_model, "./output/final")

# 4. Merge adapter into original model (requires high-VRAM machine again)


# 4. Use the merged model (now works anywhere)
python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> model = AutoModelForCausalLM.from_pretrained("./granite-small-custom")
>>> tokenizer = AutoTokenizer.from_pretrained("./granite-small-custom")
```

## Installation

### Requirements

- Python 3.10+
- CUDA 12.x+
- PyTorch 2.0+
- Transformers 4.40+
- BitsAndBytes 0.43+
- PEFT 0.10+

### Install Dependencies

```bash
pip install torch transformers bitsandbytes peft datasets accelerate

# HIGHLY RECOMMENDED for long sequences (>4K tokens):
pip install flash-attn --no-build-isolation
# Or download pre-built wheels from:
#   - Official: https://github.com/Dao-AILab/flash-attention/releases
#   - Community (convenient): https://github.com/mjun0812/flash-attention-prebuild-wheels
```

**Flash Attention 2** is critical if training with sequences longer than ~4K tokens. It reduces attention memory from O(n²) to O(n), preventing OOM errors on long sequences.

### Get the Code

```bash
# Clone or download the stonebnb directory
git clone https://github.com/mramendi/stonebnb
stonebnb stonebnb
```


## Usage

### 1. Quantize a Model

**⚠️ Quantization requires enough VRAM to load the full BF16 model:**
- Granite 4-h Tiny (8B): ~17GB VRAM
- Granite 40h Small (32B): ~65GB VRAM

On a high-VRAM machine:

```bash
python quantize_and_save_granite.py \
    ibm-granite/granite-8b-code-base \
    ./granite-8b-quantized
```

**What this does**:
1. Loads the full BF16 model
2. Quantizes MoE expert layers to 4-bit NF4
3. Saves quantized model with metadata

**Output**: A directory with:
- `pytorch_model.bin` - Quantized weights + quant_state
- `config.json` - Model configuration
- `quantization_metadata.json` - Info about quantized layers
- Tokenizer files

**Options**:

- `--quantize-shared-mlp` - also quantize the shared MLP layers to save a bit more VRAM. (The MoE router is never quantized)

The quantized model can now be copied to machines with less VRAM for training. Alretharivetly you can use a ready quantized model from HuggingFace:

- Tiny: https://huggingface.co/ramendik/granite-4.0-h-tiny-stonebnb
- Small: https://huggingface.co/ramendik/granite-4.0-h-small-stonebnb

### 2. Train with LoRA

On a 8GB+ GPU for Tiny, 32GB+ GPU for Small:

```bash
python train_lora.py \
    --model-name ./granite-8b-quantized \
    --dataset train_data.jsonl \
    --eval-dataset eval_data.jsonl \
    --output-dir ./output \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --epochs 3 \
    --learning-rate 2e-4 \
    --rank 128 \
    --alpha 256
```

NOTE: This is a standard sample training script. Please modify and improve it as necessary for your case, including the addition of an evaluation dataset.

While teh quantized model is stored in the custom StoneBnB format, *the resulting adapter is in the standard format*.

**Dataset Format** (JSONL):
```json
{"messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
]}
```

Each line in the JSONL dataset is one conversation. The script:
1. Applies the model's chat template
2. Tokenizes the conversation
3. Masks system/user tokens (labels = -100) so only assistant responses are trained

**Training Parameters**:
- `--batch-size`: Per-device batch size (start with 2)
- `--gradient-accumulation-steps`: Accumulate gradients (effective batch = batch_size × this)
- `--rank`: LoRA rank (64-256, higher = more capacity but more memory)
- `--alpha`: LoRA alpha (typically 2×rank)
- `--epochs`: Number of training epochs

**Memory Usage** (Granite 8B with rank=128):
- Model: ~4-5GB
- Activations: ~6-8GB per batch
- LoRA adapters: ~200-300MB
- **Total**: ~12-15GB for batch_size=2

### 3. Test inference on the Trained Model

```python
from load_quantized_model import load_quantized_model
from peft import PeftModel

# Load quantized base model
base_model, tokenizer = load_quantized_model("./granite-8b-quantized", device="cuda")

# Load LoRA adapter on top
model = PeftModel.from_pretrained(base_model, "./output/final")

# Use for inference
messages = [
    {"role": "user", "content": "Hello!"}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### 4. Merge Adapter into the Unquantized Model (requires high VRAM again)

```
python merge_adapter.py \
    --base-model ibm-granite/granite-4.0-h-small  \
    --adapter ./output/final \
    --output ./granite-small-custom
```

You can use the resulting standard Safetensors checkpoint like any other model of the same architecture.


## How It Works

### The Problem

Granite MoE models have a unique architecture:
- **Hybrid**: Mamba (SSM) + Attention + MoE in each layer
- **3D Expert Weights**: MoE experts use shape `(num_experts, hidden_size, intermediate_size)`

Standard BitsAndBytes only supports 2D matrices. When you try to quantize 3D tensors:
1. BnB flattens them to 2D for quantization
2. The flattened shape is saved in `quant_state.shape`
3. On load, BnB tries to `reshape()` back to original shape
4. **But**: uint8 tensors (quantized data) don't support 3D reshape in PyTorch!

### Our Solution

We patch three BitsAndBytes components:

1. **`quantize_4bit`** (`functional.py`):
   - Reshape 3D → 2D before quantization
   - Store original shape in `quant_state.shape`

2. **`Params4bit.cuda()`** (`nn/modules.py`):
   - After dequantizing, reshape back to original 3D shape
   - Uses `quant_state.shape` to know the target shape

3. **`Int4Tensor` indexing** (custom patch):
   - Add `__getitem__` support for tensor indexing
   - Allows `weight[expert_idx]` to work on quantized experts

**File**: `moe_linear_4bit.py` - Contains all patches

### Enabling Gradient Flow Through Frozen Layers

The key challenge: How do you train LoRA adapters when they sit on top of frozen quantized layers?

**The mechanism**:

When you have frozen layers followed by trainable layers, PyTorch's autograd needs to know that even though the frozen layer's weights don't have `requires_grad=True`, the **inputs** to those layers still need gradients computed (so they can flow back to the trainable layers before them).

**Our solution**:

```python
# CRITICAL: Enable input gradient computation
model.enable_input_require_grads()
```

This tells PyTorch: "Even though these layer weights are frozen (`requires_grad=False`), still compute gradients with respect to the layer's **inputs**."

**How it works in the computation graph**:

```
Input activations
    ↓ (requires_grad=True)
Frozen quantized MoE experts
    ↓ (weight.requires_grad=False, but input grads ARE computed)
Attention/Mamba layers
    ↓ (requires_grad=True)
LoRA adapters
    ↓ (requires_grad=True, these ARE updated)
Output & Loss

Backward pass:
    grad flows back through LoRA adapters ✓ (updated)
    grad flows back through attention/mamba ✓ (updated via LoRA)
    grad flows back through frozen experts ✓ (input grads computed, weights not updated)
    grad flows to earlier layers if needed ✓
```

**Why this is necessary**:

Without `enable_input_require_grads()`:
- PyTorch sees frozen layer weights (`requires_grad=False`)
- Assumes no gradients needed for this layer at all
- Stops computing gradients, breaking the graph
- LoRA adapters get zero gradients → no training

With `enable_input_require_grads()`:
- PyTorch computes input gradients even for frozen layers
- Gradient flows through frozen layers (like a passthrough)
- Reaches LoRA adapters which ARE trainable
- Training works correctly

**Technical details**:

The frozen quantized layers act like deterministic functions during training:
1. **Forward**: `dequantize_4bit(weight) @ input` (weight is frozen, input has grad)
2. **Backward**: Compute `grad_input = grad_output @ weight.T` (weight frozen, grad flows to input)
3. **No weight update** (frozen)

This is why we can train through quantized layers safely - they're just frozen operations that pass gradients through.

### Training Setup

The model has three types of layers:

1. **MoE Experts** (quantized, frozen):
   - MLP (Multilayer Perceptron)
   - `Params4bit` (uint8 data + quant_state)
   - 60-70% of model parameters
   - Frozen during training (no gradients)
   - Saves ~16GB VRAM for Granite 32B
   - Note: a small number of MLP layers is outside experts as `shared_mlp`; you can quantize them if you want

2. **Attention + Mamba** (unquantized, LoRA-adapted):
   - Full BF16 precision - important for noise-sensitive Mamba
   - LoRA adapters attached (trainable)
   - Where all training happens

3. **Other layers** (unquantized, frozen):
   - Embeddings, layer norms, MoE router, lm_head
   - Full BF16 precision
   - Frozen during training

**Gradient flow** (forward pass):
```
input → [frozen quantized experts] → [attention + mamba + LoRA] → output
          (no training here)              (training happens here)
```

**Why this works**:
- Attention/Mamba are typically the ONLY layers trained for style or task adaptation
- MoE experts rarely need training (they're knowledge storage, not behavior)
- Quantizing frozen layers is safe (no gradient noise from quantization)

**Key requirements**:
1. `model.enable_input_require_grads()` - Allows gradients to flow back through frozen layers
2. `gradient_checkpointing=True` - Saves memory by recomputing activations
3. LoRA adapters on attention/mamba - Where actual training happens

## StoneBnB Format

Quantized models are saved in the **StoneBnB format** - a custom serialization format that preserves BitsAndBytes quantization state with 3D tensor support.

**What it includes**:
- Quantized weights (uint8) + embedded quant_state metadata
- Original 3D tensor shapes for MoE experts
- Explicit lm_head saving
- Quantization metadata (which layers, shapes, etc.)

**Why custom format?**
- Standard `save_pretrained()` loses quant_state → model dequantizes on load
- 3D MoE tensor shapes not preserved → reshape fails
- `lm_head.weight` sometimes missing → broken generation

**File structure**:
```
granite-8b-stonebnb/
├── pytorch_model.bin              # Weights + quant_state
├── quantization_metadata.json     # Layer info
├── config.json                    # Standard HF config
└── tokenizer files                # Standard HF tokenizer
```

**📖 Full specification**: See [STONEBNB_FORMAT.md](STONEBNB_FORMAT.md) for technical details.

## Files

### Core Files

- **`train_lora.py`** - Main training script (vanilla, well-documented)
- **`quantize_and_save_granite.py`** - Quantize and save Granite models
- **`save_quantized_model.py`** - Save quantized model with metadata
- **`load_quantized_model.py`** - Load quantized model for training/inference
- **`moe_linear_4bit.py`** - BitsAndBytes patches for 3D MoE tensors

### Documentation

- **`README.md`** - Main documentation (this file)
- **`APPROACH.md`** - Technical deep-dive on the approach
- **`STONEBNB_FORMAT.md`** - Serialization format specification

### Utilities

- **`check_lm_head_in_model.py`** - Verify lm_head is saved correctly
- **`verify_saved_model_complete.py`** - List all layers in saved model
- **`fix_missing_lm_head.py`** - Repair models missing lm_head (if needed)

### Testing

- **`test_quantized_model_repl.py`** - Interactive testing
- **`test_forward_vs_generate.py`** - Verify forward/generate work
- **`test_bnb_moe_indexing.py`** - Test MoE expert indexing

## Troubleshooting

### "No quantized tensors found after loading"

The model was dequantized during loading. Causes:
- Using `device_map="auto"` instead of `device_map={"": "cuda"}`
- Old version of BitsAndBytes
- Patches not applied

**Solution**: Use `load_quantized_model()` which handles this correctly.

### "All logits are zero" or "Constant loss"

The `lm_head.weight` is missing or all zeros.

**Solution**: Re-quantize with the latest `save_quantized_model.py` which explicitly saves lm_head.

### "CUDA out of memory" during training

**Solutions**:
1. **Install Flash Attention 2** (most important for long sequences!):
   ```bash
   pip install flash-attn --no-build-isolation
   # Or use pre-built wheels: https://github.com/mjun0812/flash-attention-prebuild-wheels
   ```
   Flash Attention 2 reduces attention memory from O(n²) to O(n). Without it, sequences >4K tokens will likely OOM.
2. Check for the model dequantizing on load (see "No quantized tensors" above)
3. Reduce `--batch-size` (try 1)
4. Reduce `--max-seq-length` (try 2048 or 4096)
5. Reduce `--rank` (try 64)
6. Enable CPU offloading (advanced)

### "No gradients flowing" (grad_norm=0)

**Check**:
1. LoRA adapters applied? Look for "lora" in parameter names
2. `model.enable_input_require_grads()` called?
3. Labels properly masked? (non-assistant tokens should be -100)
4. lm_head weights non-zero?

## Technical Details

### Quantization Format

- **Method**: NF4 (4-bit NormalFloat)
- **Block size**: 64 (quantizes in 64-element blocks)
- **Double quantization**: Yes (quantizes the scaling factors too)
- **Storage**: ~0.5 bytes per parameter (vs 2 bytes for BF16)

### What Gets Quantized

By default, only MoE expert layers:
- `block_sparse_moe.input_linear` (3D: experts → hidden)
- `block_sparse_moe.output_linear` (3D: experts → hidden)

**Not quantized** (kept in BF16):
- Embedding layers
- Attention projections (q/k/v/o_proj)
- Mamba layers
- Layer norms
- lm_head

You can optionally quantize more layers with `--quantize-shared-mlp` flag.

### Why Only Quantize Experts?

We only quantize MoE experts because:

1. **They're frozen anyway**: Typical LoRA training only targets attention/mamba (for style or task adaptation), not MoE experts
2. **Largest component**: ~60-70% of model parameters → biggest VRAM savings
3. **Safe to quantize**: Since they're frozen, no gradient noise from quantization
4. **Other layers are risky**:
   - Attention: very small layers that should not be diluted further
   - Mamba: Very noise-sensitive, need full precision for LoRA
   - Embeddings/norms: Too small to matter
   - lm_head: Critical for output quality

Quantizing the layers we train (attention/mamba) would be qLoRA, which we don't do because:
- Full implementation of the backwards pass for quantized 3D tensors would be too complex 
- Adds gradient noise from quantization errors
- Marginal memory savings (attention/mamba are small compared to experts)

## Contributing

This is a community contribution to make Granite models more accessible. If you find bugs or have improvements, please share them!

## License

MIT (same as BitsandBytes)

## Citation

If you use this in your work:

```bibtex
@software{stonebnb_2026,
  title = {StoneBnB: BitsAndBytes Quantization for Granite MoE Models},
  author = {Misha Ramendik},
  year = {2026},
  url = {https://github.com/mramendi/stonebnb}
}
```

## Acknowledgments

- IBM Research for the Granite model family
- BitsAndBytes team for the quantization library
- HuggingFace for PEFT and Transformers
- Red Hat for encouraging associates to research AI

NOTE: Claude Code was extensively used in the development of this solution.

---

**Questions?** Open an issue or check the [Troubleshooting](#troubleshooting) section.
