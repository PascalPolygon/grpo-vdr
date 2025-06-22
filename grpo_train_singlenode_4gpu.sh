#!/bin/bash
#SBATCH --job-name=grpo_s_surprisal_4gpu
#SBATCH --output=slurm/output_%j.log
#SBATCH --error=slurm/error_%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --mem-per-gpu=30G
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu
#SBATCH --time=14-00:00:00

# Load modules
module load conda
module load nccl/20.11

# Initialize conda properly
conda activate vllm_env

# add the missing metrics libs
pip install -U evaluate sacrebleu rouge-score

export WANDB_API_KEY=2c252ca0e83bb5a2c8873ebc2b865c0cc61c1cf5

# For single-node, MASTER_ADDR is localhost and torchrun handles it.
export MASTER_ADDR="localhost"
export MASTER_PORT=$(shuf -i 49152-65535 -n 1)

# Set CUDA visible devices to all GPUs on the node
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',')

#NCCL debugging
export NCCL_DEBUG=INFO
export LOGLEVEL=INFO

# Run training
cd /home/mdao1/grpo-vdr

> gpu_usage.log  # This clears the file before logging starts
nvidia-smi -l 5 > gpu_usage.log 2>&1 &

# Memory optimization settings
BATCH_SIZE=4                         # Smaller batch size to avoid OOM
GRAD_ACCUM=4                         # Accumulate gradients over multiple steps
MAX_STEPS=1500                       # Limit training to 3000 iterations

# Set NCCL and CUDA memory management environment variables to avoid OOM
export NCCL_ASYNC_ERROR_HANDLING=1   # Better error handling
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # More flexible memory allocation
export CUDA_LAUNCH_BLOCKING=0        # Async CUDA operations
export CUDA_DEVICE_MAX_CONNECTIONS=1 # Limit connections to avoid fragmentation

# Run with memory-optimized settings for a single node
OMP_NUM_THREADS=12 srun torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    scripts/multinode_train.py \
    --per_device_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_steps $MAX_STEPS \
    --use_intrinsic_rewards
