from trl.trainer.grpo_trainer import GRPOTrainer
import wandb
import torch
import logging

logger = logging.getLogger(__name__)

class TLDRGRPOTrainer(GRPOTrainer):
    """Custom GRPO Trainer for the TLDR dataset which has 'prompt' and 'completion' fields
    instead of 'content' and 'summary' fields expected by the base trainer.
    Also implements memory optimization and better error handling.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Disable expensive intrinsic rewards by default to save memory
        use_intrinsic = getattr(self.args, 'use_intrinsic_rewards', False)
        if not use_intrinsic:
            logger.info("Intrinsic rewards disabled to save memory")
            # Override these values to disable intrinsic rewards
            self.args.epi_reward_lambda = 0.0
            self.args.aleatoric_reward_lambda = 0.0
        else:
            logger.info("Intrinsic rewards ENABLED - watch for memory usage")
        logger.info(f"epi_reward_lambda after __init__ logic: {self.args.epi_reward_lambda}")
            
        # Enable gradient accumulation if not specified
        if self.args.gradient_accumulation_steps < 2:
            logger.info(f"Setting gradient_accumulation_steps from {self.args.gradient_accumulation_steps} to 4 to reduce memory usage")
            self.args.gradient_accumulation_steps = 4
    
    def compute_intrinsic_reward(self, prompt_ids, completion_ids, attention_mask):
        """Override to avoid OOM errors - return 0 rewards if intrinsic rewards are disabled"""
        use_intrinsic = getattr(self.args, 'use_intrinsic_rewards', False)
        if not use_intrinsic or self.args.epi_reward_lambda == 0.0:
            # Return zero tensor of appropriate shape instead of computing expensive intrinsic rewards
            batch_size = prompt_ids.shape[0]
            return torch.zeros(batch_size, dtype=torch.float, device=prompt_ids.device)
        
        # If intrinsic rewards are enabled, call the parent implementation
        try:
            intrinsic_reward_values = super().compute_intrinsic_reward(prompt_ids, completion_ids, attention_mask)
            if intrinsic_reward_values is not None and torch.is_tensor(intrinsic_reward_values) and intrinsic_reward_values.numel() > 0:
                # Ensure it's a float tensor for mean/std
                float_rewards = intrinsic_reward_values.float()
                logger.info(f"[debug] BALD mean={float_rewards.mean().item():.4f}, std={float_rewards.std().item():.4f}, shape={float_rewards.shape}")
            else:
                logger.warning(f"[debug] Intrinsic reward computation did not return a valid tensor. Got: {intrinsic_reward_values}")
            return intrinsic_reward_values
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM in compute_intrinsic_reward: {e}")
            # Return zero tensor as fallback
            batch_size = prompt_ids.shape[0]
            return torch.zeros(batch_size, dtype=torch.float, device=prompt_ids.device)
    
    def _prepare_inputs(self, inputs):
        """Override to catch OOM errors"""
        try:
            return super()._prepare_inputs(inputs)
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM in _prepare_inputs: {e}")
            # Just return inputs as-is and try to continue
            return inputs
    
    def _offline_eval(self, max_batches=10):  # reduced from 40 to 10 to save memory
        """Override the offline eval to use the correct field names and handle errors better."""
        if self.args.local_rank != 0:  # Only run eval on main process
            return
            
        try:
            prev_mode = self.model.training  
            self.model.eval()

            raw_model = self.accelerator.unwrap_model(self.model)   # returns the underlying AutoModel
            device = self.accelerator.device

            preds, refs = [], []
            if not hasattr(self, "_val_subset"):
                # Take a smaller subset to reduce memory usage
                self._val_subset = self.val_data.select(range(min(10, len(self.val_data))))
            
            # Set a smaller max_new_tokens for evaluation
            eval_max_tokens = min(32, self.max_completion_length)
            
            for ex in self._val_subset:
                # Use 'prompt' instead of 'content' and 'completion' instead of 'summary'
                prompt = ex["prompt"]
                ref = ex["completion"]
                
                # Use smaller batch size for evaluation
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.args.bf16):
                    ids = self.processing_class(prompt, return_tensors="pt").to(device)["input_ids"]
                    
                    try:
                        # Use more memory-efficient generation parameters
                        gen_ids = raw_model.generate(
                            ids,
                            max_new_tokens=eval_max_tokens,  # Use fewer tokens for eval
                            temperature=0.7,
                            do_sample=False,  # Use greedy decoding to save memory
                        )
                        gen_txt = self.processing_class.decode(gen_ids[0], skip_special_tokens=True)
                        preds.append(gen_txt)
                        refs.append(ref)
                    except torch.cuda.OutOfMemoryError as e:
                        logger.warning(f"OOM during generation in eval: {e}")
                        # Skip this example and continue
                        continue
                    
            # Only log metrics if we have predictions
            if preds and self.args.local_rank == 0:  # Only log from main process
                try:
                    from trl.trainer.utils import rouge_l, self_bleu
                    rl = rouge_l(preds, refs)      # ROUGE-L F1
                    sbleu = self_bleu(preds)       # self-BLEU (lower is better)

                    # Log memory usage too
                    gpu_mem_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    wandb.log({
                        "eval/rougeL": rl, 
                        "eval/selfBLEU": sbleu,
                        "eval/gpu_memory_GB": gpu_mem_used,
                        "global_step": self.state.global_step
                    })
                except Exception as e:
                    logger.warning(f"Error computing metrics: {e}")
            
            self.model.train(prev_mode)  # restore dropout state
        except Exception as e:
            logger.error(f"Error in _offline_eval: {e}")
            if self.args.local_rank == 0:  # Only log from main process
                wandb.log({"eval/error": str(e), "global_step": self.state.global_step})
            # Continue training
