#!/bin/bash
#SBATCH --job-name=grpo_l4_gpu
#SBATCH --output=slurm/output_%j.log
#SBATCH --error=slurm/error_%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:l4:1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --mem-per-gpu=240G
#SBATCH --cpus-per-task=12
#SBATCH --partition=hwgui
#SBATCH --time=4-00:00:00

# Load modules
module load conda

# Activate the conda environment that now has torchvision
conda activate vllm_env

# Set working directory
WORKDIR=/home/mdao1/grpo-vdr
cd $WORKDIR

# Memory optimization settings - keeping required values
BATCH_SIZE=8            # Required for num_generations=8
GRAD_ACCUM=1            # Keep at 1 to minimize memory
MAX_STEPS=1500
NUM_GENERATIONS=8      # Required value
MAX_NEW_TOKENS=64       # Reduced from 128 to save memory
MAX_LENGTH=256          # Reduced from 512 to save memory

LOGGING_STEPS=50        # Less frequent logging
SAVE_STEPS=500          # Less frequent saving

# Set environment variables
export WANDB_API_KEY=
export CUDA_VISIBLE_DEVICES=0

# Balanced memory optimization settings (less aggressive for better speed)
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
# Remove CUDA_LAUNCH_BLOCKING=1 for async execution (much faster)
# export CUDA_LAUNCH_BLOCKING=1  # COMMENTED OUT FOR SPEED
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Moderate memory optimizations
# export CUDA_EMPTY_CACHE_THRESHOLD_MB=128  # COMMENTED OUT - let PyTorch manage
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1   # COMMENTED OUT - caching improves speed

# Keep garbage collection optimization
export PYTHONOPTIMIZE=1

echo "Starting GRPO training with balanced performance/memory settings..."
echo "NOTE: Using hwgui partition instead of hpg-turin due to maintenance"
echo ""
echo "Configuration:"
echo "  Batch size: $BATCH_SIZE (required for num_generations=8)"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Num generations: 8 (required)"
echo "  epi_reward_num_samples: 8 (required)"
echo "  Max new tokens: $MAX_NEW_TOKENS"
echo "  Max length: $MAX_LENGTH"
echo "  Max steps: $MAX_STEPS"
echo ""
echo "Performance optimizations:"
echo "  - Increased chunk size to 4 (from 2) in intrinsic reward computation"
echo "  - Removed CUDA_LAUNCH_BLOCKING for async execution"
echo "  - Enabled PyTorch memory caching"
echo "  - Smart memory cleanup only when needed (>18GB)"

# Run training with balanced settings
OMP_NUM_THREADS=12 python -u scripts/multinode_train.py \
    --explore_beta 1 \
    --per_device_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_steps $MAX_STEPS \
    --use_intrinsic_rewards \
    --num_generations $NUM_GENERATIONS \
    --epi_reward_num_samples $NUM_GENERATIONS \
    --epi_reward_mode eos
