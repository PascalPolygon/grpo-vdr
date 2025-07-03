#!/bin/bash
#SBATCH --job-name=debug_env
#SBATCH --output=slurm/debug_%j.log
#SBATCH --error=slurm/debug_error_%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=hpg-turin
#SBATCH --time=00:30:00

# Load modules
module load conda

# Activate the conda environment
conda activate vllm_env

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

echo "=== Environment Debug ==="
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Create a debug script
cat > debug_imports.py << 'EOF'
import sys
print("Python executable:", sys.executable)
print("Python path:", sys.path[:3], "...")

# Test basic imports
print("\n1. Testing torch import...")
try:
    import torch
    print(f"✓ Torch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n2. Testing torchvision import...")
try:
    import torchvision
    print(f"✓ Torchvision version: {torchvision.__version__}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n3. Testing transformers import (minimal)...")
try:
    # Try to import just the core without triggering image utils
    import transformers.models
    import transformers.tokenization_utils
    print("✓ Core transformers imports work")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n4. Testing TRL import...")
try:
    # First try basic TRL import
    import trl
    print(f"✓ TRL version: {trl.__version__}")
    
    # Then try GRPO specifically
    print("\n5. Testing GRPO import...")
    from trl import GRPOConfig, GRPOTrainer
    print("✓ GRPOConfig and GRPOTrainer imported successfully!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n6. Package versions check:")
import subprocess
result = subprocess.run(['pip', 'list', '|', 'grep', '-E', 'torch|transformers|trl'], 
                       shell=True, capture_output=True, text=True)
print(result.stdout)
EOF

# Run the debug script
python debug_imports.py

# Also check conda environment
echo -e "\n=== Conda Environment Info ==="
conda list | grep -E "torch|transformers|trl|vision" 