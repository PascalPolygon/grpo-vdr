from datasets import load_dataset
import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Loading dataset")
dataset = load_dataset("trl-lib/tldr", split="train")

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def reward_func(completions, ground_truth, **kwargs):
    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
logger.info(f'model_name: {model_name}')

batch_size = 4
training_args = GRPOConfig(
        output_dir="Qwen2-0.5B-GRPO", 
        learning_rate=5e-4, 
        bf16=True, 
        logging_steps=10, 
        use_vllm=False, 
        report_to=["wandb"], 
        per_device_train_batch_size=batch_size,
        # epi_reward_lambda=0.01,
        epi_reward_lambda=1e7,
        # aleatoric_reward_lambda=0.01,
        aleatoric_reward_lambda=1e7,
        epi_reward_mode="all",
        intrinsic_reward_type="epistemic",)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     device_map=None
# ).to("cuda")
logger.info('Creating tokenizer')
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

logger.info("Setting up trainer")

trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=reward_len,
    # reward_funcs=[format_reward_func, reward_func],
    args=training_args,
    train_dataset=dataset,
    # fp16=True,                
)

# use peft at your own risk; not working for me with multi-GPU training
# trainer = GRPOTrainer(
#     model=model,
#     reward_funcs=reward_len,
#     processing_class=tokenizer,
#     args=training_args,
#     train_dataset=dataset,
#     #peft_config=peft_config
# )
trainer.train()
# trainer.train()