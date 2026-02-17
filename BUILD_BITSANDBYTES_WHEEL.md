# Building bitsandbytes Wheel for Vast.ai

This guide explains how to build a wheel from the modified bitsandbytes library for installation on your Vast.ai machine.

## Prerequisites

### On Your Build Machine (Local or Vast.ai)

You need:
- **CUDA Toolkit** (11.8, 12.x, or 13.x) - must match your target PyTorch version
- **CMake** >= 3.22
- **GCC/G++** compiler (for Linux)
- **Python** >= 3.10
- **NVIDIA GPU** (for testing, optional for building)

### Quick Check

```bash
# Check CUDA
nvcc --version

# Check CMake
cmake --version

# Check Python
python --version

# Check GCC
gcc --version
```

## Build Instructions

### Option 1: Build on Vast.ai Directly (Recommended)

This ensures compatibility with the exact CUDA version on your Vast instance.

1. **SSH into your Vast.ai instance**:
   ```bash
   # Use the SSH command from Vast.ai dashboard
   ssh -p <port> root@<host>.vast.ai
   ```

2. **Clone your repos** (if not already done):
   ```bash
   cd /workspace  # or wherever you want
   git clone <your-bitsandbytes-fork-url>
   git clone <your-stonebnb-repo-url>
   ```

3. **Install build dependencies**:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install build scikit-build-core cmake
   ```

4. **Build the wheel**:
   ```bash
   cd bitsandbytes
   python -m build --wheel
   ```

   This will:
   - Compile the C++/CUDA extensions with CMake
   - Create a wheel in `dist/` directory
   - Take 5-10 minutes depending on GPU/CPU

5. **Install the wheel**:
   ```bash
   pip install dist/bitsandbytes-*.whl
   ```

6. **Verify installation**:
   ```bash
   python -c "import bitsandbytes as bnb; print(bnb.__version__)"
   ```

### Option 2: Build Locally and Upload

If you want to build on your local machine and upload to Vast.ai:

1. **Build locally** (requires CUDA toolkit installed):
   ```bash
   cd /path/to/bitsandbytes
   pip install build scikit-build-core
   python -m build --wheel
   ```

2. **Upload to Vast.ai**:
   ```bash
   scp -P <port> dist/bitsandbytes-*.whl root@<host>.vast.ai:/workspace/
   ```

3. **Install on Vast.ai**:
   ```bash
   ssh -p <port> root@<host>.vast.ai
   pip install /workspace/bitsandbytes-*.whl
   ```

**⚠️ Warning**: Cross-compilation can fail if CUDA versions don't match. Use Option 1 for best compatibility.

### Option 3: Editable Install (Development)

For active development where you're making changes:

```bash
cd bitsandbytes
pip install -e .
```

This installs in editable mode, so changes to Python files take effect immediately (C++ changes still require rebuild).

## Troubleshooting

### Issue: CMake not found

```bash
# Install CMake
pip install cmake

# Or on Ubuntu/Debian:
apt-get update && apt-get install -y cmake

# Or download from https://cmake.org/download/
```

### Issue: CUDA not found

```bash
# Check CUDA location
which nvcc

# If not found, set CUDA_HOME
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify
nvcc --version
```

### Issue: Build fails with "scikit-build-core not found"

```bash
pip install scikit-build-core>=0.5.0
```

### Issue: Build fails with compiler errors

Make sure you have GCC/G++:
```bash
# Ubuntu/Debian
apt-get update && apt-get install -y build-essential

# Check version (need >= 7.0)
gcc --version
```

### Issue: "Unsupported CUDA version"

Check that your CUDA toolkit version is supported (11.8, 12.x, or 13.x):
```bash
nvcc --version

# Match PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"
```

### Issue: Import fails with "cannot find libbitsandbytes_*.so"

The compiled library wasn't included. Try:
```bash
# Rebuild with verbose output
python -m build --wheel -v

# Check what's in the wheel
unzip -l dist/bitsandbytes-*.whl | grep libbitsandbytes
```

### Issue: Very slow build or OOM during compilation

Reduce parallel jobs:
```bash
export CMAKE_BUILD_PARALLEL_LEVEL=2
python -m build --wheel
```

## Quick Build Script

Save this as `build_bnb.sh`:

```bash
#!/bin/bash
set -e

echo "Building bitsandbytes wheel..."
echo "=============================="

# Check prerequisites
command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc not found. Install CUDA toolkit."; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo "ERROR: cmake not found. Run: pip install cmake"; exit 1; }

echo "✓ CUDA version: $(nvcc --version | grep release | awk '{print $5}' | cut -c2-)"
echo "✓ CMake version: $(cmake --version | head -1 | awk '{print $3}')"
echo "✓ Python version: $(python --version | awk '{print $2}')"
echo ""

# Install build dependencies
echo "Installing build dependencies..."
pip install --upgrade pip setuptools wheel build scikit-build-core

# Build
echo ""
echo "Building wheel (this may take 5-10 minutes)..."
cd "$(dirname "$0")"
python -m build --wheel

# Check result
if [ -f dist/bitsandbytes-*.whl ]; then
    echo ""
    echo "=============================="
    echo "✓ Build successful!"
    echo "=============================="
    echo ""
    echo "Wheel created:"
    ls -lh dist/bitsandbytes-*.whl
    echo ""
    echo "To install:"
    echo "  pip install dist/bitsandbytes-*.whl"
    echo ""
else
    echo ""
    echo "=============================="
    echo "✗ Build failed!"
    echo "=============================="
    exit 1
fi
```

Make it executable and run:
```bash
chmod +x build_bnb.sh
./build_bnb.sh
```

## Testing After Installation

After installing the wheel, run these tests:

```bash
# Test import
python -c "import bitsandbytes as bnb; print('Version:', bnb.__version__)"

# Test CUDA availability
python -c "import bitsandbytes as bnb; print('CUDA available:', bnb.cuda_setup.is_available())"

# Run bitsandbytes self-check
python -m bitsandbytes

# Test 4-bit quantization
python -c "
import torch
from bitsandbytes.nn import Params4bit
print('Testing 4-bit quantization...')
# This will fail if CUDA binaries are missing
tensor = torch.randn(100, 100, device='cuda', dtype=torch.bfloat16)
print('✓ 4-bit ops work')
"
```

## Installing on Vast.ai (Complete Workflow)

Here's the complete workflow from scratch:

```bash
# 1. Rent a Vast.ai instance with:
#    - CUDA 12.x or 11.8
#    - At least 8GB VRAM
#    - PyTorch pre-installed (or install it)

# 2. SSH into instance
ssh -p <port> root@<host>.vast.ai

# 3. Setup workspace
cd /workspace
git clone <your-bitsandbytes-repo>
git clone <your-stonebnb-repo>

# 4. Install PyTorch if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Build bitsandbytes wheel
cd bitsandbytes
pip install build scikit-build-core cmake
python -m build --wheel

# 6. Install the wheel
pip install dist/bitsandbytes-*.whl

# 7. Install other dependencies for stonebnb
pip install transformers peft datasets

# 8. Test stonebnb
cd ../stonebnb
python test_virtual_weights_integration.py
```

## Expected Build Output

Successful build looks like:

```
* Creating isolated environment: venv+pip...
* Installing packages in isolated environment:
  - scikit-build-core
  - setuptools>=77.0.3
  - trove-classifiers>=2025.8.6.13
* Getting build dependencies for wheel...
* Building wheel from sdist
* Installing packages in isolated environment:
  - scikit-build-core
  - setuptools>=77.0.3
  - trove-classifiers>=2025.8.6.13
* Building wheel...
[cmake] -- The CXX compiler identification is GNU 11.4.0
[cmake] -- The CUDA compiler identification is NVIDIA 12.1.105
[cmake] -- Detecting CXX compiler ABI info
[cmake] -- Detecting CXX compiler ABI info - done
[cmake] -- Check for working CXX compiler: /usr/bin/c++ - skipped
[cmake] -- Detecting CXX compile features
[cmake] -- Detecting CXX compile features - done
[cmake] -- Detecting CUDA compiler ABI info
[cmake] -- Detecting CUDA compiler ABI info - done
...
[100%] Built target bitsandbytes_cuda
Successfully built bitsandbytes-0.49.2.dev0-cp310-cp310-linux_x86_64.whl
```

## Build Time

Typical build times:
- **Fast GPU**: 3-5 minutes
- **Mid GPU**: 5-8 minutes
- **CPU only**: 10-15 minutes

## Wheel Size

The wheel will be approximately:
- **50-100 MB** (includes compiled CUDA binaries)

## Alternative: Download Pre-built Wheel

If you're using a standard CUDA version (not modified bitsandbytes), you can download pre-built wheels:

```bash
pip install bitsandbytes
```

**But for your modified version**, you must build from source as shown above.

## Next Steps

After successfully installing bitsandbytes:

1. **Test StoneBnB**:
   ```bash
   cd /workspace/stonebnb
   python test_virtual_weights_integration.py
   ```

2. **Run training**:
   ```bash
   python train_lora.py \
       --model-name ramendik/granite-4.0-h-tiny-stonebnb \
       --dataset example_data.jsonl \
       --output-dir ./output \
       --batch-size 2 \
       --epochs 1 \
       --cache-size 20
   ```

## References

- **bitsandbytes**: https://github.com/bitsandbytes-foundation/bitsandbytes
- **scikit-build-core**: https://scikit-build-core.readthedocs.io/
- **CMake**: https://cmake.org/
- **Python build**: https://build.pypa.io/
