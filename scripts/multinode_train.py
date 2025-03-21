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
import sys
from torch.distributed import init_process_group, destroy_process_group

# Add the parent directory to path to import our custom model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.qwen_with_dropout import create_qwen_with_dropout

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
    parser.add_argument("--dropout_rate", type=float, default=0.1,
                        help="Dropout rate to use throughout the model.")
    # Any other arguments you want to expose
    args, unknown = parser.parse_known_args()
    return args

def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

# def reward_len(completions, **kwargs):
#     """Reward function that gives higher scores to longer completions."""
#     return [float(len(c)) for c in completions]

def main():
    ddp_setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    logger.info(f"local_rank: {local_rank}, global_rank: {os.environ['RANK']}")

    args = parse_args()

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    logger.info(f"Using model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Define dropout parameters for model initialization
    dropout_config = {
        "attn_pdrop": 0.1,            # Attention dropout (GPT-style models)
        "resid_pdrop": 0.1,           # Residual connection dropout (GPT-style models)
        "embd_pdrop": 0.1,            # Embedding dropout (GPT-style models)
        "hidden_dropout_prob": 0.1,   # Hidden layer dropout (BERT-style models)
        "attention_probs_dropout_prob": 0.1  # Attention probs dropout (BERT-style models)
    }
    
    # We create a small GRPOConfig with the final usage
    training_args = GRPOConfig(
        output_dir="Qwen2-0.5B-GRPO",
        bf16=True,
        logging_steps=10,
        use_vllm=False,
        report_to=["wandb"],
        per_device_train_batch_size=8,
        # epi_reward_lambda=0.01,
        epi_reward_lambda=1e30,
        # aleatoric_reward_lambda=0.01,
        aleatoric_reward_lambda=1e30,
        epi_reward_mode="all",
        intrinsic_reward_type="epistemic",
        
        # We don't need model_init_kwargs anymore as we're using our custom model
        # But let's keep a reference to dropout settings for documentation
        model_init_kwargs={"dropout_rate": dropout_rate},
        
        # Additional relevant flags...
    )
    # We'll inject our new flags into training_args
    training_args.epistemic_mode = args.epistemic_mode
    training_args.bald_weight    = args.bald_weight

    # Load dataset
    logger.info("Loading dataset 'trl-lib/tldr' (train split)")
    dataset = load_dataset("trl-lib/tldr", split="train")

    # Build our custom Qwen model with dropout layers properly injected
    logger.info(f"Creating custom Qwen model with dropout layers injected")
    dropout_rate = args.dropout_rate
    
    # Use our helper function to create a model with dropout layers
    model = create_qwen_with_dropout(
        model_name,
        dropout_rate=dropout_rate,
        torch_dtype=torch.bfloat16
    )
    
    # Log the model architecture
    logger.info(f"Model config: {model.config}")
    logger.info(f"Using model with dropout_rate={dropout_rate}")
    
    # Continue with DDP setup
    model = model.to(f"cuda:{local_rank}")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    model.config = model.module.config
    model.warnings_issued = getattr(model.module, "warnings_issued", {})
    model.add_model_tags = model.module.add_model_tags
    
    # Create the trainer
    trainer = GRPOTrainer(
        model=model,                
        reward_funcs=reward_len,    
        # reward_funcs=format_reward_func,
        args=training_args,
        train_dataset=dataset,
    )

    logger.info(f"Starting training with epistemic_mode={args.epistemic_mode}, bald_weight={args.bald_weight}, dropout_rate={args.dropout_rate}")
    trainer.train()
    destroy_process_group()

if __name__ == "__main__":
    main()
