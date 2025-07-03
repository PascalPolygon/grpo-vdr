#!/bin/bash
#SBATCH --job-name=grpo_l4_gpu_debug
#SBATCH --output=slurm/output_%j.log
#SBATCH --error=slurm/error_%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:l4:1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --mem-per-gpu=240G
#SBATCH --cpus-per-task=12
#SBATCH --partition=hpg-turin
#SBATCH --time=14-00:00:00

# Start timing
START_TIME=$(date +%s)
echo "Job started at: $(date)"

# Load modules
echo "Loading modules..."
module load conda

# Activate the conda environment
echo "Activating conda environment..."
conda activate vllm_env

# Set working directory
WORKDIR=/home/mdao1/grpo-vdr
cd $WORKDIR

# Memory optimization settings
BATCH_SIZE=8
GRAD_ACCUM=2
MAX_STEPS=1500

# Set environment variables
export WANDB_API_KEY=2c252ca0e83bb5a2c8873ebc2b865c0cc61c1cf5
export CUDA_VISIBLE_DEVICES=0

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Enable Python verbose mode for imports
export PYTHONVERBOSE=1

# Add timing
echo "Environment setup complete at: $(date)"
echo "Starting GRPO training with single GPU (no distributed setup)..."

# Run training using the debug version of the script
OMP_NUM_THREADS=12 python scripts/multinode_train_debug.py \
    --explore_beta 0.0 \
    --per_device_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_steps $MAX_STEPS \
    --use_intrinsic_rewards \
    --epi_reward_num_samples 8 \
    --num_generations 8 

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Job completed at: $(date)"
echo "Total duration: $DURATION seconds" 