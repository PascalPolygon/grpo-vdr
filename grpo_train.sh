#!/bin/bash
#SBATCH --job-name=grpo_surprisal
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=pdao2015@my.fit.edu    # Where to send mail
#SBATCH --output=slurm/output_%j.log
#SBATCH --error=slurm/error_%j.log
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=a100:8
#SBATCH --ntasks-per-node=8

# export NUM_MACHINES=1
# export NUM_PROCESSES=8
# export MACHINE_RANK=0

# Load modules
# module load mpich
module load conda
# module load cuda/12.2.2

# Initialize conda properly
# source ~/.bashrc
# source /usr/local/spack/opt/spack/linux-ubuntu20.04-cascadelake/gcc-9.4.0/anaconda3-2022.10-qnnpc2ciyw76yyntaq6mxyim6eh4axd6/etc/profile.d/conda.sh
# conda init --all
conda activate vllm_env


# Run training
# cd /home1/pdao2015/PhD/GRPO
cd /home/mdao1/grpo-vdr

> gpu_usage.log  # This clears the file before logging starts
nvidia-smi -l 5 > gpu_usage.log 2>&1 &

# Generate accelerate config from template
# envsubst < config/accelerate_config_template.yml > config/accelerate_config_${SLURM_JOBID}.yml

# torchrun --nnodes=$SLURM_NNODES \
#     --nproc_per_node=3 \
#     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#     --rdzv_backend=c10d \
#     --max_restarts=0 \
#     --network_interface=eth0 \
#     scripts/train.py

# Run with generated config
# /home1/pdao2015/.conda/envs/vllm_env/bin/accelerate launch \
#     --config_file config/accelerate_config_${SLURM_JOBID}.yml \
#     scripts/train.py

# srun /home/mdao1/.conda/envs/vllm_env/bin/torchrun \
#     --nproc_per_node=8 \
#     scripts/train.py

# Generate a random port between 10000 and 65535 to avoid conflicts
export MASTER_PORT=$((10000 + RANDOM % 55535))

OMP_NUM_THREADS=12 srun torchrun \
    --nproc_per_node=8 \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    scripts/train.py

# /home1/pdao2015/.conda/envs/vllm_env/bin/accelerate launch --config_file config/accelerate_config.yml scripts/train.py

# Cleanup (optional)
# rm config/accelerate_config_${SLURM_JOBID}.yml