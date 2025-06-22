#!/bin/bash
#SBATCH --job-name=grpo_m_surprisal
#SBATCH --output=slurm/output_%j.log
#SBATCH --error=slurm/error_%j.log
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=2 
#SBATCH --mem-per-gpu=30G
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu
#SBATCH --time=14-00:00:00
#SBATCH --array=1-4

# Load modules
# module load mpich
module load conda
# module load cuda/12.2.2  # Comment out to avoid CUDA driver conflicts
module load nccl/20.11
# module load cuda/11.8.0-gcc-9.4.0-dmftitd

# Initialize conda properly
# source ~/.bashrc
# source /usr/local/spack/opt/spack/linux-ubuntu20.04-cascadelake/gcc-9.4.0/anaconda3-2022.10-qnnpc2ciyw76yyntaq6mxyim6eh4axd6/etc/profile.d/conda.sh
# conda init --all
conda activate vllm_env

# add the missing metrics libs
pip install -U evaluate sacrebleu rouge-score

export WANDB_API_KEY=2c252ca0e83bb5a2c8873ebc2b865c0cc61c1cf5

# find the MASTER_ADDR:
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}

# Alternative way to get IP address
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" /sbin/ip -4 -o addr show | grep -v 127.0.0.1 | grep -v docker | awk '{print $4}' | cut -d/ -f1 | head -1)
echo Master addr: $MASTER_ADDR
echo NCCL_IB_TIMEOUT: $NCCL_IB_TIMEOUT
echo NCCL_IB_RETRY_CNT: $NCCL_IB_RETRY_CNT
echo NCCL_IB_DISABLE: $NCCL_IB_DISABLE
echo GLOO_SOCKET_IFNAME: $GLOO_SOCKET_IFNAME
echo NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME

# Validation check
echo "Master: $MASTER_ADDR, My Rank: $SLURM_NODEID, Node List: $SLURM_JOB_NODELIST"
if [ -z "$MASTER_ADDR" ]; then
    echo "CRITICAL ERROR: Could not determine MASTER_ADDR"
    exit 1
fi

# export MASTER_PORT=29500
export MASTER_PORT=$(shuf -i 49152-65535 -n 1)

#dynamic port assignment
# export MASTER_PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
# Automatically set machine rank
# export MACHINE_RANK=$SLURM_NODEID

# export RANK=$SLURM_PROCID
# export LOCAL_RANK=$SLURM_LOCALID
# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',')



#NCCL debugging
export NCCL_DEBUG=INFO
export LOGLEVEL=INFO
# export NCCL_DEBUG_FILE=slurm/nccl_%j.log
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=3

# Set GPU and rank environment variables
# export RANK=$SLURM_PROCID
# export LOCAL_RANK=$SLURM_LOCALID
# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# NCCL configuration to handle cross-node GPU bus ID conflicts
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_IGNORE_DISABLED_P2P=1
# export NCCL_SOCKET_IFNAME=eth0
# export GLOO_SOCKET_IFNAME=eth0

# Run training
cd /home/mdao1/grpo-vdr

> gpu_usage.log  # This clears the file before logging starts
nvidia-smi -l 5 > gpu_usage.log 2>&1 &

# Generate accelerate config from template
# Commented out since the template file doesn't exist
# envsubst < config/accelerate_config_template.yml > config/accelerate_config_${SLURM_JOBID}.yml

# Add this before the training command
echo "Testing network connectivity..."
ping -c 3 $MASTER_ADDR
# nmap command not available on compute nodes
# nmap -p $MASTER_PORT $MASTER_ADDR

echo "nnodes: $SLURM_NNODES"
# NCCL connectivity test
# srun torchrun --nnodes=$SLURM_NNODES \
#     --nproc_per_node=3 \
#     --rdzv_id $RANDOM \
#     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#     --rdzv_backend=c10d \
#     scripts/nccl_test.py

# srun torchrun --nnodes=$SLURM_NNODES \
#     --nproc_per_node=3 \
#     --rdzv_id $RANDOM \
#     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#     --rdzv_backend=c10d \
#     scripts/multinode_train.py

# Memory optimization settings
BATCH_SIZE=4                         # Smaller batch size to avoid OOM
GRAD_ACCUM=4                         # Accumulate gradients over multiple steps
MAX_STEPS=1500                       # Limit training to 3000 iterations
# LAMBDAS=(0 1e-3 1 1e3 1e7)
BETAS=(0.00 0.05 0.20 5.0)

# Set NCCL and CUDA memory management environment variables to avoid OOM
export NCCL_ASYNC_ERROR_HANDLING=1   # Better error handling
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # More flexible memory allocation
export CUDA_LAUNCH_BLOCKING=0        # Async CUDA operations
export CUDA_DEVICE_MAX_CONNECTIONS=1 # Limit connections to avoid fragmentation

# Run with memory-optimized settings
OMP_NUM_THREADS=12 srun torchrun \
    --nnodes $SLURM_NNODES \
    --nproc_per_node 4 \
    --rdzv_id $RANDOM \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    scripts/multinode_train.py \
    --explore_beta ${BETAS[$SLURM_ARRAY_TASK_ID]} \
    --per_device_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_steps $MAX_STEPS \
    --use_intrinsic_rewards


# OMP_NUM_THREADS=12 srun python -m torch.distributed.run  \
#     --nnodes $SLURM_NNODES \
#     --nproc_per_node 1 \
#     --rdzv_id $RANDOM \
#     --rdzv_backend c10d \
#     --rdzv_endpoint localhost:$MASTER_PORT \
#     scripts/multinode_train.py