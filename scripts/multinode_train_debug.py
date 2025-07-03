print("Script started - importing modules...")
print("1. Importing standard libraries...")
import argparse
import time
start_time = time.time()

print("2. Importing datasets...")
from datasets import load_dataset

print("3. Importing torch...")
import torch

print("4. Importing trl...")
from trl import GRPOConfig

print("5. Importing transformers...")
from transformers import AutoTokenizer, AutoModelForCausalLM

print("6. Importing remaining modules...")
import re
import logging
import os
import sys
from torch.distributed import init_process_group, destroy_process_group
import torch.nn as nn
import inspect
import torch.nn.functional as F

print("7. Setting up logging...")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Logger configured")

print("8. Importing custom modules...")
# Import our custom trainer
from custom_grpo_trainer import TLDRGRPOTrainer

# Try to import wandb for logging
try:
    import wandb
    print("9. Wandb imported successfully")
except ImportError:
    wandb = None
    print("9. Wandb not available")

# Add the parent directory to path to import our custom model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.qwen_with_dropout import create_qwen_with_dropout

# Import our quality-weighted exploration patch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from grpo_quality_weighted_patch import apply_quality_weighted_exploration_patch
    HAS_QUALITY_PATCH = True
except ImportError:
    logger.warning("Could not import quality-weighted exploration patch")
    HAS_QUALITY_PATCH = False

print(f"All imports completed in {time.time() - start_time:.2f} seconds")

# Rest of the original script continues from here...
def ddp_setup():
    """Initialize distributed training"""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        init_process_group(backend="nccl")
        logger.info(f"Initialized process group on GPU {local_rank}")
    else:
        logger.warning("LOCAL_RANK not found, running in non-distributed mode")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epistemic_mode", type=str, default="all", 
                        choices=["all","none","per_token","end_of_sequence"],
                        help="If 'none', no epistemic bonus is used; if 'per_token', compute bonus each token; else end-of-sequence.")
    parser.add_argument("--bald_weight", type=float, default=0.0,
                        help="Scaling factor for the BALD disagreement intrinsic reward.")
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
        default=8,
        help="Forward passes used in BALD intrinsic reward"
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=None,
        help="Number of generations per prompt (must evenly divide batch size)"
    )
    args, unknown = parser.parse_known_args()
    return args

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format.
    Works with both chat-style completions (list of list[dict]) and plain strings."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    if completions and isinstance(completions[0], str):
        completion_contents = completions
    else:
        completion_contents = [c[0]["content"] for c in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def reward_len(completions, **kwargs):
    """Reward function that gives higher scores to longer completions."""
    return [float(len(c)) for c in completions]

def mc_variance(model, tokenizer, local_rank, text="Hello world for variance check.", n_samples=6):
    """Computes Monte-Carlo variance for the last token logits."""
    model.train()
    logger.info(f"Running mc_variance with n_samples={n_samples} on rank {local_rank}...")
    
    device = f"cuda:{local_rank}"
    inputs = tokenizer(text, return_tensors="pt")
    ids = inputs.input_ids.to(device)
    attn_mask = inputs.attention_mask.to(device)

    try:
        with torch.no_grad():
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

        stacked_logits = torch.stack(outs, dim=0)
        variance = stacked_logits.var(dim=0).mean().item()
        logger.info(f"mc_variance result: {variance:.6e}")
        return variance
    except Exception as e:
        logger.error(f"Error in mc_variance: {e}", exc_info=True)
        return 0.0

def force_dropout(model, p: float = 0.1):
    """
    Sets dropout probability for all dropout layers and attributes
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
    print("10. Starting main function...")
    
    # Initialize distributed training
    ddp_setup()
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    logger.info(f"local_rank: {local_rank}, global_rank: {global_rank}, world_size: {world_size}")

    # Ensure all processes are initialized before proceeding
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    else:
        logger.info("Running in non-distributed mode, skipping barrier")
    
    print("11. Parsing arguments...")
    args = parse_args()

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    logger.info(f"Using model: {model_name}")
    logger.info(f"Arguments: {vars(args)}")
    
    print("12. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully")

    # Create GRPOConfig
    print("13. Creating GRPOConfig...")
    grpo_config_kwargs = {
        "output_dir": "Qwen2-0.5B-GRPO",
        "bf16": True,
        "logging_steps": 10,
        "save_total_limit": 3,
        "use_vllm": False,
        "report_to": ["wandb"],
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "epi_reward_alpha": 1.0,
        "epi_reward_lambda": args.epi_reward_lambda,
        "explore_beta": args.explore_beta,
        "aleatoric_reward_lambda": 0,
        "epi_reward_mode": "all",
        "intrinsic_reward_type": "epistemic",
        "max_steps": args.max_steps,
    }
    
    if args.num_generations is not None:
        grpo_config_kwargs["num_generations"] = args.num_generations
        logger.info(f"Setting num_generations to: {args.num_generations}")
    
    training_args = GRPOConfig(**grpo_config_kwargs)
    
    # Set additional attributes
    training_args.logging_first_step = True
    training_args.dataloader_num_workers = 1
    
    try:
        training_args.gradient_checkpointing = True
        logger.info("Gradient checkpointing enabled")
    except:
        logger.info("Gradient checkpointing not available for this model/config")
    
    setattr(training_args, 'use_intrinsic_rewards', args.use_intrinsic_rewards)
    setattr(training_args, "epi_reward_num_samples", args.epi_reward_num_samples)
    assert training_args.epi_reward_num_samples >= 2, "Need multiple passes for BALD"
    
    training_args.epistemic_mode = args.epistemic_mode
    setattr(training_args, "explore_beta", args.explore_beta)
    setattr(training_args, "mi_cap", args.mi_cap)
    
    logger.info(f"Memory optimization settings:")
    logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")

    # Load dataset
    print("14. Loading dataset...")
    dataset = load_dataset("trl-lib/tldr", split="train")
    print(f"Dataset loaded: {len(dataset)} samples")

    # Build the model
    print("15. Loading model...")
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    logger.info(f"Model {model_name} loaded on {device}")

    # Check mc_variance BEFORE patch
    print("16. Checking mc_variance before patch...")
    variance_before = mc_variance(model, tokenizer, local_rank, n_samples=8)
    logger.info(f"mc_variance BEFORE patch: {variance_before:.6e}")

    # Apply force_dropout patch
    print("17. Applying force_dropout...")
    dropout_p_to_force = 0.1
    force_dropout(model, p=dropout_p_to_force)
    
    # Check mc_variance AFTER patch
    print("18. Checking mc_variance after patch...")
    variance_after = mc_variance(model, tokenizer, local_rank, n_samples=8)
    logger.info(f"mc_variance AFTER patch: {variance_after:.6e}")

    # Apply quality-weighted exploration patch if available
    if HAS_QUALITY_PATCH and args.explore_beta > 0:
        logger.info("Applying reward-quality weighted exploration patch...")
        apply_quality_weighted_exploration_patch()
        logger.info("Patch applied")
    
    # Create the trainer
    print("19. Creating trainer...")
    trainer = TLDRGRPOTrainer(
        model=model,                
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=dataset,
    )
    print("Trainer created successfully")

    logger.info(f"Starting training with explore_beta={args.explore_beta}")
    
    # Log hyperparameters to wandb
    if wandb is not None and wandb.run is not None:
        wandb.config.update({
            "explore_beta": args.explore_beta,
            "epi_reward_lambda": args.epi_reward_lambda,
            "epi_reward_num_samples": args.epi_reward_num_samples,
            "epi_reward_mode": args.epi_reward_mode,
            "mi_cap": args.mi_cap,
            "use_intrinsic_rewards": args.use_intrinsic_rewards,
            "epistemic_mode": args.epistemic_mode,
            "batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_generations": args.num_generations if args.num_generations else training_args.num_generations,
        })
    
    print("20. Starting training...")
    trainer.train()
    
    # Clean up distributed training if initialized
    if torch.distributed.is_initialized():
        destroy_process_group()

if __name__ == "__main__":
    print("=== GRPO Training Script Starting ===")
    main()
    print("=== GRPO Training Script Completed ===") 