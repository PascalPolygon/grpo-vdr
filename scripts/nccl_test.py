import torch
import torch.distributed as dist
import os
import subprocess

def check_port(port):
    try:
        result = subprocess.check_output(["lsof", "-i", f":{port}"])
        print(f"Port {port} is already in use by:\n{result.decode()}")
    except subprocess.CalledProcessError:
        print(f"Port {port} appears free.")

master_port = os.environ["MASTER_PORT"]

def main():
    
    # Get SLURM variables
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    
    check_port(master_port)

    # Initialize process group
    try:
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://{master_addr}:{master_port}',
            world_size=world_size,
            rank=rank
        )
    except Exception as e:
        print(f"Failed to initialize process group on {master_addr}:{master_port}")
        print("Error details:", e)
        check_port(master_port)
        raise

    # NCCL test
    tensor = torch.ones(1).cuda() * rank
    dist.all_reduce(tensor)
    print(f"Rank {rank}: NCCL test passed!", flush=True)

if __name__ == "__main__":
    main()