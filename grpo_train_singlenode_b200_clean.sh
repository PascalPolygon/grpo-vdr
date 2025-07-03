#!/bin/bash
#SBATCH --job-name=grpo_b200_gpu
#SBATCH --output=slurm/output_%j.log
#SBATCH --error=slurm/error_%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:b200:1    # Using B200 GPU instead of L4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --mem-per-gpu=80G    # B200 GPUs have 80GB memory each
#SBATCH --cpus-per-task=12
#SBATCH --partition=hpg-b200
#SBATCH --time=14-00:00:00

# Load modules
module load conda

# Activate the conda environment that now has torchvision
conda activate vllm_env

# Set working directory
WORKDIR=/home/mdao1/grpo-vdr
cd $WORKDIR

# Memory optimization settings
BATCH_SIZE=8    # With B200's larger memory, could potentially increase this
GRAD_ACCUM=1    # Keep low to avoid increasing effective batch size
MAX_STEPS=1500

# Set environment variables
export WANDB_API_KEY=2c252ca0e83bb5a2c8873ebc2b865c0cc61c1cf5
export CUDA_VISIBLE_DEVICES=0

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "Starting GRPO training with B200 GPU..."
echo "Configuration:"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Num generations: 8  # Must be <= batch_size to avoid reshape errors"
echo "  Max steps: $MAX_STEPS"
echo "  Effective batch size: $((BATCH_SIZE * GRAD_ACCUM))"
echo ""
echo "Note: B200 GPUs have 80GB memory (vs 24GB for L4)"
echo "You can potentially increase batch_size or epi_reward_num_samples"
echo ""

# Run training directly with Python (no torchrun needed for single GPU)
OMP_NUM_THREADS=12 python scripts/multinode_train.py \
    --explore_beta 0.5 \
    --per_device_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_steps $MAX_STEPS \
    --use_intrinsic_rewards \
    --epi_reward_num_samples 8 \
    --num_generations 8 