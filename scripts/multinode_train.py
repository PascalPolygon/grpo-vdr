import argparse
from datasets import load_dataset
import torch
from trl import GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import logging
import os
import sys
from torch.distributed import init_process_group, destroy_process_group
import torch.nn as nn # Added for force_dropout
import inspect # Added for force_dropout
import torch.nn.functional as F # Added for force_dropout

# Import our custom trainer
from custom_grpo_trainer import TLDRGRPOTrainer

# Add the parent directory to path to import our custom model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.qwen_with_dropout import create_qwen_with_dropout

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# def ddp_setup():
#     local_rank = int(os.environ["LOCAL_RANK"])
#     torch.cuda.set_device(local_rank)
#     init_process_group(backend="nccl")
#     logger.info(f"Initialized process group on GPU {local_rank}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epistemic_mode", type=str, default="all", 
                        choices=["all","none","per_token","end_of_sequence"],
                        help="If 'none', no epistemic bonus is used; if 'per_token', compute bonus each token; else end-of-sequence.")
    parser.add_argument("--bald_weight", type=float, default=0.0,
                        help="Scaling factor for the BALD disagreement intrinsic reward.")
    # parser.add_argument("--explore_lambda", type=float, default=1e-3,
    #                     help="Max multiplicative boost from z-scored BALD (λ in the paper)")
    parser.add_argument("--use_intrinsic_rewards", action="store_true",
                        help="Whether to use intrinsic rewards. Disabling saves memory.")
    parser.add_argument("--per_device_batch_size", type=int, default=8,
                        help="Per device batch size (smaller values use less memory)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=3000,
                        help="Maximum number of training steps")
    parser.add_argument("--explore_beta", type=float, default=0.5,
                        help="Max multiplicative boost from MI (0=off)")
    parser.add_argument("--epi_reward_lambda", type=float, default=0,
                        help="Max multiplicative boost from z-scored BALD (λ in the paper)")
    parser.add_argument("--mi_cap", type=float, default=3.0,
                        help="Clip abs(z-scored BALD) to this value")
    parser.add_argument(
        "--epi_reward_num_samples",
        type=int,
        default=8,           # ← force 8 MC-Dropout passes
        help="Forward passes used in BALD intrinsic reward"
    )
    # Any other arguments you want to expose
    args, unknown = parser.parse_known_args()
    return args

# def reward_len(completions, **kwargs):
#     return [-abs(20 - len(completion)) for completion in completions]

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format.
    Works with both chat-style completions (list of list[dict]) and plain strings."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    # Determine the representation of completions
    if completions and isinstance(completions[0], str):
        completion_contents = completions  # list of raw strings
    else:
        # Assume chat-format: [[{"role": "assistant", "content": "..."}], ...]
        completion_contents = [c[0]["content"] for c in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def reward_len(completions, **kwargs):
    """Reward function that gives higher scores to longer completions."""
    return [float(len(c)) for c in completions]

def mc_variance(model, tokenizer, local_rank, text="Hello world for variance check.", n_samples=6):
    """Computes Monte-Carlo variance for the last token logits."""
    model.train() # Ensure dropout is active for MC passes
    logger.info(f"Running mc_variance with n_samples={n_samples} on rank {local_rank}...")
    
    device = f"cuda:{local_rank}"
    inputs = tokenizer(text, return_tensors="pt")
    ids = inputs.input_ids.to(device)
    attn_mask = inputs.attention_mask.to(device)

    try:
        with torch.no_grad(): # No gradients needed for variance calculation
            outs = []
            for i in range(n_samples):
                output = model(ids, attention_mask=attn_mask)
                if hasattr(output, 'logits'):
                    logits = output.logits
                elif isinstance(output, tuple) and len(output) > 0 and torch.is_tensor(output[0]):
                    logits = output[0]
                else:
                    logger.error(f"Could not extract logits from model output: {type(output)}")
                    return 0.0
                outs.append(logits[:, -1]) 
        
        if not outs:
            logger.warning("No logits collected for mc_variance.")
            return 0.0

        stacked_logits = torch.stack(outs, dim=0)  # (n_samples, B, V)
        variance = stacked_logits.var(dim=0).mean().item()
        logger.info(f"mc_variance result: {variance:.6e}")
        return variance
    except Exception as e:
        logger.error(f"Error in mc_variance: {e}", exc_info=True)
        return 0.0

def force_dropout(model, p: float = 0.1):
    """
    • sets p for every nn.Dropout *module*  
    • overwrites float attributes like `.dropout`, `.attention_dropout`  
    • leaves the model in train() mode so dropout is active in MC passes
    """
    logger.info(f"Applying force_dropout with p={p} to model {type(model).__name__}")
    
    global_F = torch.nn.functional 

    if not hasattr(global_F, '_original_dropout_cascade_patch'):
        global_F._original_dropout_cascade_patch = global_F.dropout
        logger.info("Original torch.nn.functional.dropout backed up as _original_dropout_cascade_patch.")

    patched_nn_dropout_count = 0
    patched_float_attr_count = 0

    for module_name, m in model.named_modules():
        if isinstance(m, nn.Dropout):
            logger.info(f"Found nn.Dropout module: {module_name}")
            if m.p != p:
                m.p = p
            patched_nn_dropout_count +=1

        for attr_name in ("dropout", "attention_dropout", "ffn_dropout",
                     "hidden_dropout_prob", "attention_probs_dropout_prob"):
            if hasattr(m, attr_name):
                logger.info(f"Found float dropout attribute: {attr_name}")
                current_val = getattr(m, attr_name)
                if isinstance(current_val, float) and current_val != p:
                    setattr(m, attr_name, p)
                    patched_float_attr_count +=1
    
    forced_p_for_F_patch = p
    def _custom_functional_dropout_wrapper(input, p_callsite_arg=0.5, training=True, inplace=False):
        return global_F._original_dropout_cascade_patch(input, p=forced_p_for_F_patch, training=training, inplace=inplace)

    global_F.dropout = _custom_functional_dropout_wrapper
    
    logger.info(f"Patched {patched_nn_dropout_count} nn.Dropout layers and {patched_float_attr_count} float dropout attributes.")
    logger.info(f"torch.nn.functional.dropout is now globally patched to use p={forced_p_for_F_patch} when training=True.")
    
    model.train()
    logger.info("Model set to train() mode after force_dropout.")

def main():
    # ddp_setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    logger.info(f"local_rank: {local_rank}, global_rank: {global_rank}, world_size: {world_size}")

    # Ensure all processes are initialized before proceeding
    torch.distributed.barrier()
    
    args = parse_args()

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    logger.info(f"Using model: {model_name}")
    logger.info(f"epistemic_mode: {args.epistemic_mode}")
    logger.info(f"bald_weight: {args.bald_weight}")
    logger.info(f"epi_reward_lambda: {args.epi_reward_lambda}")
    logger.info(f"use_intrinsic_rewards: {args.use_intrinsic_rewards}")
    logger.info(f"per_device_batch_size: {args.per_device_batch_size}")
    logger.info(f"gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    logger.info(f"max_steps: {args.max_steps}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Create GRPOConfig with only supported parameters
    training_args = GRPOConfig(
        output_dir="Qwen2-0.5B-GRPO",
        bf16=True,
        logging_steps=10,
        save_total_limit=3,
        use_vllm=False,
        report_to=["wandb"],
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epi_reward_alpha=1.0,
        epi_reward_lambda=args.epi_reward_lambda,
        explore_beta=args.explore_beta,
        aleatoric_reward_lambda=0,
        epi_reward_mode="all",
        intrinsic_reward_type="epistemic",
        max_steps=args.max_steps,  # Limit training to 3000 steps
    )
    
    # Set additional attributes that aren't part of GRPOConfig constructor
    training_args.logging_first_step = True
    training_args.dataloader_num_workers = 1
    
    # Only set gradient checkpointing if available
    try:
        training_args.gradient_checkpointing = True
        logger.info("Gradient checkpointing enabled")
    except:
        logger.info("Gradient checkpointing not available for this model/config")
    
    # Add the use_intrinsic_rewards as a custom attribute after initialization
    setattr(training_args, 'use_intrinsic_rewards', args.use_intrinsic_rewards)
    setattr(training_args, "epi_reward_num_samples", args.epi_reward_num_samples)
    assert training_args.epi_reward_num_samples >= 2, "Need multiple passes for BALD (epi_reward_num_samples must be >= 2)"
    logger.info(f"epi_reward_num_samples set to: {training_args.epi_reward_num_samples} (assertion passed)")
    # We'll inject our new flags into training_args
    training_args.epistemic_mode = args.epistemic_mode
    # training_args.bald_weight    = args.bald_weight
    setattr(training_args, "explore_beta", args.explore_beta)
    setattr(training_args, "mi_cap",      args.mi_cap)
    # setattr(training_args, "explore_lambda", args.explore_lambda)
    
    # Log memory settings
    logger.info(f"Memory optimization settings:")
    logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * torch.distributed.get_world_size()}")
    logger.info(f"  Intrinsic rewards enabled: {getattr(training_args, 'use_intrinsic_rewards', False)}")
    logger.info(f"  Gradient checkpointing: {training_args.gradient_checkpointing}")
    logger.info(f"  BF16 precision: {training_args.bf16}")

    # Load dataset
    logger.info("Loading dataset 'trl-lib/tldr' (train split)")
    dataset = load_dataset("trl-lib/tldr", split="train")

    # Build the model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(f"cuda:{local_rank}")
    logger.info(f"Model {model_name} loaded on cuda:{local_rank}")

    # 1. Check mc_variance BEFORE patch
    logger.info("Calculating mc_variance BEFORE force_dropout patch...")
    # mc_variance function handles tokenization and device placement internally
    variance_before = mc_variance(model, tokenizer, local_rank, n_samples=8)
    logger.info(f"mc_variance BEFORE patch: {variance_before:.6e}")

    # 2. Apply force_dropout patch
    dropout_p_to_force = 0.1 # Define the dropout probability
    logger.info(f"Applying force_dropout with p={dropout_p_to_force}...")
    force_dropout(model, p=dropout_p_to_force) #Set dropout to p=0.1
    
    # 3. Check mc_variance AFTER patch
    logger.info("Calculating mc_variance AFTER force_dropout patch...")
    variance_after = mc_variance(model, tokenizer, local_rank, n_samples=8)
    logger.info(f"mc_variance AFTER patch: {variance_after:.6e}")

    # Note: The manual config changes for dropout below are now largely superseded by force_dropout.
    # logger.info(f"For reference, model.config.hidden_dropout_prob: {getattr(model.config, 'hidden_dropout_prob', 'N/A')}")
    # logger.info(f"For reference, model.config.attention_dropout: {getattr(model.config, 'attention_dropout', 'N/A')}")

    # Create the trainer - using our custom TLDRGRPOTrainer instead of GRPOTrainer
    trainer = TLDRGRPOTrainer(
        model=model,                
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=dataset,
    )

    logger.info(f"Starting training with epistemic_mode={args.epistemic_mode}, bald_weight={args.bald_weight}, dropout_rate={args.dropout_rate}")
    trainer.train()
    destroy_process_group()

if __name__ == "__main__":
    main()
