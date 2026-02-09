# StoneBnB Format Specification

**StoneBnB** is a serialization format for BitsAndBytes 4-bit quantized Granite MoE models with 3D tensor support.

**Name origin**: Stone (Granite) + BnB (BitsAndBytes) - a community format for sharing quantized Granite models.

## Why a Custom Format?

Standard HuggingFace `save_pretrained()` doesn't preserve BitsAndBytes quantization state properly for models with 3D tensors:

1. **Missing quant_state**: Standard serialization doesn't save BnB's `quant_state` metadata (quantization parameters, block sizes, codebooks)
2. **3D tensor shapes lost**: MoE experts use 3D weights `(num_experts, hidden, intermediate)` - shape information gets lost
3. **lm_head omission**: `named_parameters()` sometimes misses top-level `lm_head.weight`
4. **Dequantization on load**: Without quant_state, models dequantize to BF16 on load (16GB → 32GB VRAM)

## Format Structure

A StoneBnB checkpoint is a directory containing:

```
granite-8b-stonebnb/
├── pytorch_model.bin              # Weights + embedded quant_state
├── quantization_metadata.json     # Quantization configuration
├── config.json                    # Standard HF model config
├── tokenizer.json                 # Tokenizer
├── tokenizer_config.json          # Tokenizer config
├── special_tokens_map.json        # Special tokens
└── ...                            # Other tokenizer files
```

### 1. pytorch_model.bin

Standard PyTorch state dict with **embedded BnB metadata** for quantized parameters.

#### Structure

For each parameter in the model:

**Unquantized parameters** (embeddings, norms, lm_head, etc.):
```python
{
    "model.embed_tokens.weight": <torch.Tensor>,  # Standard tensor
    "model.norm.weight": <torch.Tensor>,
    "lm_head.weight": <torch.Tensor>,  # Explicitly included
    # ... etc
}
```

**Quantized parameters** (MoE experts):
```python
{
    # The quantized data (uint8)
    "model.layers.0.block_sparse_moe.input_linear.weight": <torch.Tensor dtype=uint8>,

    # BnB quant_state metadata (one entry per metadata field)
    "model.layers.0.block_sparse_moe.input_linear.weight.quant_type": "nf4",
    "model.layers.0.block_sparse_moe.input_linear.weight.blocksize": torch.tensor(64),
    "model.layers.0.block_sparse_moe.input_linear.weight.absmax": <torch.Tensor>,
    "model.layers.0.block_sparse_moe.input_linear.weight.quant_map": <torch.Tensor>,
    "model.layers.0.block_sparse_moe.input_linear.weight.dtype": "torch.bfloat16",
    "model.layers.0.block_sparse_moe.input_linear.weight.shape": torch.tensor([8, 2048, 8192]),  # Original 3D shape!

    # Optional: nested quantization (double quant)
    "model.layers.0.block_sparse_moe.input_linear.weight.nested_absmax": <torch.Tensor>,
    "model.layers.0.block_sparse_moe.input_linear.weight.nested_blocksize": torch.tensor(256),
    "model.layers.0.block_sparse_moe.input_linear.weight.nested_quant_map": <torch.Tensor>,
}
```

#### Key Differences from Standard save_pretrained()

| Aspect | Standard HF | StoneBnB |
|--------|-------------|----------|
| Quantized weights | Dequantized to BF16 | Kept as uint8 + quant_state |
| quant_state | Lost | Embedded as `.quant_type`, `.absmax`, etc. |
| 3D shapes | Flattened, lost | Preserved in `.shape` |
| lm_head | Sometimes missing | Explicitly saved |
| File size | Large (BF16) | Small (~4x smaller) |

### 2. quantization_metadata.json

High-level metadata about which layers are quantized and how.

```json
{
  "quantization_method": "bitsandbytes_nf4",
  "moe_patch_required": true,
  "quantized_weights": [
    {
      "name": "model.layers.0.block_sparse_moe.input_linear.weight",
      "original_shape": [8, 2048, 8192],
      "data_shape": [16384, 8192],
      "is_3d": true
    },
    {
      "name": "model.layers.0.block_sparse_moe.output_linear.weight",
      "original_shape": [8, 8192, 2048],
      "data_shape": [65536, 2048],
      "is_3d": true
    }
    // ... more quantized weights
  ]
}
```

**Fields**:
- `quantization_method`: Always `"bitsandbytes_nf4"`
- `moe_patch_required`: `true` - indicates 3D MoE patches needed
- `quantized_weights`: Array of quantized layer info
  - `name`: Full parameter name
  - `original_shape`: Original 3D shape `[num_experts, hidden, intermediate]`
  - `data_shape`: Flattened 2D shape used for quantization `[num_experts * hidden, intermediate]`
  - `is_3d`: `true` for MoE experts, `false` for 2D layers

**Purpose**:
- Quick inspection of what's quantized
- Validation during loading
- Documentation for users

### 3. Standard HuggingFace Files

**config.json**: Standard Granite model configuration (unchanged)

**Tokenizer files**: Standard HuggingFace tokenizer (unchanged)

These are identical to non-quantized models.

## How StoneBnB Differs from Standard Formats

### vs. HuggingFace save_pretrained()

**Standard HuggingFace**:
```python
model.save_pretrained("./model")
# Result: All weights in BF16, ~16GB for Granite 8B
# quant_state lost, loads as unquantized model
```

**StoneBnB**:
```python
from bnb_granite.save_quantized_model import save_quantized_model
save_quantized_model(model, tokenizer, "./model")
# Result: Quantized weights preserved, ~4GB for Granite 8B
# quant_state embedded, loads as quantized model
```

### vs. safetensors

**safetensors** (HF default for new models):
- Safe, fast loading with memory mapping
- Doesn't support custom metadata like quant_state
- StoneBnB uses `pytorch_model.bin` instead

**Future**: Could extend safetensors to support quant_state metadata

### vs. GGUF/GPTQ/AWQ

These are different quantization formats:

| Format | Precision | Framework | Use Case |
|--------|-----------|-----------|----------|
| GGUF | 4-bit | llama.cpp | CPU inference |
| GPTQ | 4-bit | AutoGPTQ | GPU inference |
| AWQ | 4-bit | AutoAWQ | GPU inference |
| **StoneBnB** | **4-bit** | **BitsAndBytes** | **Training + inference** |

**StoneBnB advantages**:
- Supports gradient flow (training)
- Works with PEFT/LoRA
- Handles 3D tensors (MoE experts)

## Creating StoneBnB Checkpoints

### From Scratch

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from bnb_granite.quantize_and_save_granite import quantize_granite

# Quantize and save
quantize_granite(
    model_name="ibm-granite/granite-8b-code-base",
    output_dir="./granite-8b-stonebnb",
    quantize_shared_mlp=False  # Only quantize MoE experts
)
```

This produces a complete StoneBnB checkpoint.

### Converting Existing Checkpoints

If you have a quantized model in memory:

```python
from bnb_granite.save_quantized_model import save_quantized_model

# model = ... (your quantized model)
# tokenizer = ... (your tokenizer)

save_quantized_model(
    model=model,
    tokenizer=tokenizer,
    output_dir="./granite-8b-stonebnb",
    list_all_layers=True  # Optional: debug output
)
```

## Loading StoneBnB Checkpoints

### Standard Loading

```python
from bnb_granite.load_quantized_model import load_quantized_model

model, tokenizer = load_quantized_model(
    "./granite-8b-stonebnb",
    device="cuda"
)

# Verify quantization preserved
from bitsandbytes.nn import Params4bit
num_quantized = sum(
    1 for p in model.parameters()
    if isinstance(p, Params4bit)
)
print(f"Quantized parameters: {num_quantized}")  # Should be ~80 for Granite 8B
```

### What Happens During Loading

1. **Apply MoE patches**: Enable 3D tensor support
2. **Load state dict**: Read `pytorch_model.bin`
3. **Reconstruct Params4bit**:
   - Find all `.quant_type` keys
   - Group related metadata (`.absmax`, `.blocksize`, etc.)
   - Create `QuantState` objects
   - Wrap quantized data in `Params4bit`
4. **Restore 3D shapes**: Use `.shape` metadata
5. **Verify lm_head**: Check critical layers present

## Validation

### Check Quantization Preserved

```python
from bnb_granite.load_quantized_model import load_quantized_model
from bitsandbytes.nn import Params4bit

model, tokenizer = load_quantized_model("./granite-8b-stonebnb")

# Count quantized params
quantized = sum(1 for p in model.parameters() if isinstance(p, Params4bit))
print(f"Quantized parameters: {quantized}")

# Expected: 80 for Granite 8B (2 per MoE layer × 40 layers)
# Expected: 80 for Granite 32B (2 per MoE layer × 40 layers)
```

### Verify All Layers Present

```python
from bnb_granite.verify_saved_model_complete import verify_saved_model

verify_saved_model("./granite-8b-stonebnb")
```

This lists all layers and checks for critical components (embeddings, lm_head, etc.).

### Check lm_head

```python
from bnb_granite.check_lm_head_in_model import check_lm_head

check_lm_head("./granite-8b-stonebnb")
```

Ensures `lm_head.weight` is present (critical for text generation).

## File Size Comparison

For Granite 8B:

| Format | Size | Notes |
|--------|------|-------|
| Original BF16 | ~16 GB | Full precision |
| StoneBnB | ~4.5 GB | Experts 4-bit, rest BF16 |
| Full 4-bit | ~4 GB | Everything quantized (can't train) |

**Breakdown** (StoneBnB):
- Quantized experts: ~2.5 GB (was ~10 GB in BF16)
- Unquantized layers: ~2 GB (embeddings, attention, mamba, norms, lm_head)
- **Compression ratio**: ~3.5x vs. full BF16

## Compatibility

### ✅ Compatible With

- PEFT (LoRA, QLoRA adapters)
- HuggingFace Transformers Trainer
- Gradient checkpointing
- BF16 mixed precision training
- Standard inference APIs

### ❌ Not Compatible With

- Standard `AutoModelForCausalLM.from_pretrained()` without patches
- safetensors (uses pytorch_model.bin)
- Models without MoE layers (3D tensor patches not needed)
- BitsAndBytes < 0.43 (needs recent version)

### 🔧 Requires

- `moe_linear_4bit.py` patches applied before loading
- `load_quantized_model()` loader (not standard `from_pretrained()`)
- CUDA-capable GPU (BnB quantization is CUDA-only)

## Technical Details

### quant_state Structure

BitsAndBytes `QuantState` object contains:

```python
QuantState(
    absmax=<torch.Tensor>,          # Absolute max values per block
    shape=<tuple>,                   # Original tensor shape (2D or 3D!)
    code=<torch.Tensor>,            # Quantization codebook (NF4 mapping)
    blocksize=64,                   # Block size for quantization
    quant_type="nf4",               # Quantization type
    dtype=torch.bfloat16,           # Original dtype
    offset=<torch.Tensor or None>,  # Optional offset
    state2=<QuantState or None>     # Nested quantization (double quant)
)
```

**Critical field**: `shape` - StoneBnB stores 3D shape here, enabling reshape after dequantization.

### 3D Tensor Handling

**Quantization** (3D → 2D → quantize):
```python
# Original: (num_experts, hidden, intermediate) = (8, 2048, 8192)
original_shape = weight.shape  # (8, 2048, 8192)

# Reshape to 2D for BnB
weight_2d = weight.reshape(8 * 2048, 8192)  # (16384, 8192)

# Quantize
quantized, quant_state = quantize_4bit(weight_2d, ...)

# Store original 3D shape
quant_state.shape = original_shape  # (8, 2048, 8192)
```

**Dequantization** (dequantize → reshape to 3D):
```python
# Load quantized data + quant_state
quantized = state_dict["layer.weight"]  # uint8, shape (16384, 8192)
quant_state = reconstruct_quant_state(state_dict, "layer.weight")

# Dequantize to 2D BF16
weight_2d = dequantize_4bit(quantized, quant_state)  # (16384, 8192)

# Reshape back to original 3D
weight_3d = weight_2d.reshape(quant_state.shape)  # (8, 2048, 8192)
```

### Why pytorch_model.bin vs safetensors?

**pytorch_model.bin** allows arbitrary metadata:
- Can store multiple tensors per parameter (`.absmax`, `.blocksize`, etc.)
- No strict schema requirements
- Easy to extend

**safetensors** is stricter:
- One tensor per key
- Metadata must be in header
- Would need format extension for quant_state

**Future**: Extend safetensors to support StoneBnB metadata natively.

## Example: Full Workflow

```python
# 1. Create StoneBnB checkpoint (on high-VRAM machine)
from bnb_granite.quantize_and_save_granite import quantize_granite

quantize_granite(
    "ibm-granite/granite-8b-code-base",
    "./granite-8b-stonebnb"
)

# 2. Transfer to training machine (24GB GPU)
# scp -r granite-8b-stonebnb user@training-machine:~/

# 3. Load and train (on training machine)
from bnb_granite.load_quantized_model import load_quantized_model
from peft import LoraConfig, get_peft_model

model, tokenizer = load_quantized_model("./granite-8b-stonebnb")

# Apply LoRA
lora_config = LoraConfig(r=128, lora_alpha=256, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

# Train...

# 4. Save adapter (not the base model)
model.save_pretrained("./lora-adapter")

# 5. Load for inference
base, tokenizer = load_quantized_model("./granite-8b-stonebnb")
model = PeftModel.from_pretrained(base, "./lora-adapter")
```

## FAQ

### Can I convert StoneBnB to safetensors?

Not directly. Safetensors doesn't preserve quant_state. Converting would dequantize to BF16.

### Can I use with standard HuggingFace APIs?

Requires `load_quantized_model()` instead of `from_pretrained()`. After loading, works with standard APIs.

### Does it work with Granite 3.1, 3.2, etc.?

Yes, works with all Granite MoE models (any version with MoE architecture).

### Can I quantize non-Granite models?

Only if they have 3D MoE tensors needing the patches. For standard transformers, use regular BnB.

### Will this be upstreamed to BitsAndBytes?

We hope so! This format demonstrates 3D tensor quantization support that could benefit the broader community.

## Version History

**v1.0 (2025-01)**: Initial StoneBnB format
- 4-bit NF4 quantization with quant_state embedding
- 3D tensor shape preservation
- Explicit lm_head saving
- quantization_metadata.json

## License

StoneBnB format specification: Apache 2.0 (same as Granite and BitsAndBytes)

---

**Questions?** See the main [README.md](README.md) or [APPROACH.md](APPROACH.md) for more details.

**Using StoneBnB?** Let us know! We'd love to see what you build.
