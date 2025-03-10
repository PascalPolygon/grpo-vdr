#!/bin/bash
#SBATCH --job-name=grpo_surprisal
#SBATCH --output=slurm/output_%j.log
#SBATCH --error=slurm/error_%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=4
#SBATCH --mem=160G
#SBATCH --time=7-00:00:00    
#SBATCH --exclusive
#SBATCH --partition=gpu2

export NUM_MACHINES=1
export NUM_PROCESSES=4
export MACHINE_RANK=0

# Load modules
# module load mpich

module load cuda/12.3.0-gcc-9.4.0-fvbwiov

# Initialize conda properly
# source ~/.bashrc
source /usr/local/spack/opt/spack/linux-ubuntu20.04-cascadelake/gcc-9.4.0/anaconda3-2022.10-qnnpc2ciyw76yyntaq6mxyim6eh4axd6/etc/profile.d/conda.sh
# conda init --all
conda activate vllm_env


# Run training
cd /home1/pdao2015/PhD/GRPO

> gpu_usage.log  # This clears the file before logging starts
nvidia-smi -l 5 > gpu_usage.log 2>&1 &

# Generate accelerate config from template
envsubst < config/accelerate_config_template.yml > config/accelerate_config_${SLURM_JOBID}.yml

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

srun /home1/pdao2015/.conda/envs/vllm_env/bin/torchrun \
    --nproc_per_node=4 \
    scripts/train.py

# /home1/pdao2015/.conda/envs/vllm_env/bin/accelerate launch --config_file config/accelerate_config.yml scripts/train.py

# Cleanup (optional)
# rm config/accelerate_config_${SLURM_JOBID}.yml