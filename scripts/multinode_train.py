import argparse
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import logging
import os
from torch.distributed import init_process_group, destroy_process_group

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ddp_setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    init_process_group(backend="nccl")
    logger.info(f"Initialized process group on GPU {local_rank}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epistemic_mode", type=str, default="none", 
                        choices=["none","per_token","end_of_sequence"],
                        help="If 'none', no epistemic bonus is used; if 'per_token', compute bonus each token; else end-of-sequence.")
    parser.add_argument("--bald_weight", type=float, default=0.0,
                        help="Scaling factor for the BALD disagreement intrinsic reward.")
    # Any other arguments you want to expose
    args, unknown = parser.parse_known_args()
    return args

def reward_len(completions, **kwargs):
    """Reward function that gives higher scores to longer completions."""
    return [float(len(c)) for c in completions]

def main():
    ddp_setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    logger.info(f"local_rank: {local_rank}, global_rank: {os.environ['RANK']}")

    args = parse_args()

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    logger.info(f"Using model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # We create a small GRPOConfig with the final usage
    training_args = GRPOConfig(
        output_dir="Qwen2-0.5B-GRPO",
        bf16=True,
        logging_steps=10,
        use_vllm=False,
        report_to=["wandb"],
        per_device_train_batch_size=8,
        # Additional relevant flags...
    )
    # We'll inject our new flags into training_args
    training_args.epistemic_mode = args.epistemic_mode
    training_args.bald_weight    = args.bald_weight

    # Load dataset
    logger.info("Loading dataset 'trl-lib/tldr' (train split)")
    dataset = load_dataset("trl-lib/tldr", split="train")

    # Build the model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = model.to(f"cuda:{local_rank}")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    model.config = model.module.config
    model.warnings_issued = getattr(model.module, "warnings_issued", {})
    model.add_model_tags = model.module.add_model_tags
    
    # Create the trainer
    trainer = GRPOTrainer(
        model=model,                
        reward_funcs=reward_len,    
        args=training_args,
        train_dataset=dataset
    )

    logger.info(f"Starting training with epistemic_mode={args.epistemic_mode}, bald_weight={args.bald_weight}")
    trainer.train()
    destroy_process_group()

if __name__ == "__main__":
    main()
