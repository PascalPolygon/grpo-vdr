# train.py
from datasets import load_dataset
import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer
import re
import logging

# Import our custom Qwen model with dropout helper
from models.qwen_with_dropout import create_qwen_with_dropout

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading dataset")
dataset = load_dataset("trl-lib/tldr", split="train")

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

# You can also define other reward functions (commented out example)
# def format_reward_func(completions, **kwargs):
#     pattern = r"^<think>.*?</think><answer>.*?</answer>$"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, content) for content in completion_contents]
#     return [1.0 if match else 0.0 for match in matches]

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
logger.info(f"Model name: {model_name}")

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

logger.info("Creating tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Create our custom Qwen model with dropout
dropout_rate = 0.1  # set your desired dropout rate here
logger.info("Creating custom Qwen model with dropout")
model = create_qwen_with_dropout(
    model_name,
    dropout_rate=dropout_rate,
    torch_dtype=torch.bfloat16
)
# Make sure the model is moved to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

logger.info("Setting up trainer")

trainer = GRPOTrainer(
    model=model,                
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)

logger.info("Starting training")
trainer.train()
