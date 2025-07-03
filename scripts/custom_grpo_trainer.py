from trl.trainer.grpo_trainer import GRPOTrainer
import wandb
import torch
import logging
import gc
import os

logger = logging.getLogger(__name__)

class TLDRGRPOTrainer(GRPOTrainer):
    """Custom GRPO Trainer for the TLDR dataset which has 'prompt' and 'completion' fields
    instead of 'content' and 'summary' fields expected by the base trainer.
    Also implements memory optimization and better error handling.
    """
    
    def __init__(self, *args, **kwargs):
        # Note: num_generations adjustment is now handled in multinode_train.py
        # to ensure the batch size constraint is satisfied
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
            
    def compute_intrinsic_reward(self, prompt_ids, completion_ids, attention_mask):
        """Compute intrinsic rewards with chunked processing to save memory."""
        use_intrinsic = getattr(self.args, 'use_intrinsic_rewards', False)
        if not use_intrinsic or self.args.epi_reward_lambda == 0.0:
            # Return zero tensor of appropriate shape instead of computing expensive intrinsic rewards
            batch_size = prompt_ids.shape[0]
            return torch.zeros(batch_size, dtype=torch.float, device=prompt_ids.device)
        
        # If batch size is small enough, use parent implementation
        batch_size = prompt_ids.size(0)
        
        # Allow chunk size to be controlled via environment variable
        chunk_size = int(os.environ.get('GRPO_CHUNK_SIZE', '4'))
        
        if batch_size <= chunk_size:
            # Small batch, process normally
            try:
                result = super().compute_intrinsic_reward(prompt_ids, completion_ids, attention_mask)
                
                # Store the raw BALD scores for use in quality weighting
                if result is not None and torch.is_tensor(result):
                    self._last_bald_raw = result.detach()
                
                return result
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"OOM even with small batch size {batch_size}: {e}")
                # Retry with smaller chunks
                if chunk_size > 2:
                    logger.info("Retrying with chunk_size=2...")
                    os.environ['GRPO_CHUNK_SIZE'] = '2'
                    return self.compute_intrinsic_reward(prompt_ids, completion_ids, attention_mask)
                return torch.zeros(batch_size, dtype=torch.float, device=prompt_ids.device)
        
        # Large batch - process in chunks
        logger.info(f"Processing intrinsic rewards in chunks of {chunk_size} (batch_size={batch_size})")
        all_rewards = []
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            
            # Process chunk
            chunk_prompt = prompt_ids[i:end_idx]
            chunk_completion = completion_ids[i:end_idx]
            chunk_mask = attention_mask[i:end_idx] if attention_mask is not None else None
            
            try:
                # Call parent method on chunk
                chunk_reward = super().compute_intrinsic_reward(
                    chunk_prompt, chunk_completion, chunk_mask
                )
                all_rewards.append(chunk_reward)
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"OOM in chunk {i//chunk_size}: {e}")
                # Try with smaller chunk size
                if chunk_size > 2:
                    logger.info("Reducing chunk size and retrying...")
                    os.environ['GRPO_CHUNK_SIZE'] = str(chunk_size // 2)
                    # Reprocess entire batch with smaller chunks
                    return self.compute_intrinsic_reward(prompt_ids, completion_ids, attention_mask)
                else:
                    # Return zeros for this chunk
                    chunk_size_actual = end_idx - i
                    all_rewards.append(torch.zeros(chunk_size_actual, dtype=torch.float, device=prompt_ids.device))
            
            # Only clear cache if we're getting close to OOM
            if i < batch_size - chunk_size:  # Not the last chunk
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                if allocated > 18.0:  # If using more than 18GB, clean up
                    torch.cuda.empty_cache()
                    # Skip gc.collect() as it's very slow
        
        # Concatenate results
        result = torch.cat(all_rewards, dim=0)
        
        # Store the raw BALD scores for use in quality weighting
        if result is not None and torch.is_tensor(result):
            self._last_bald_raw = result.detach()
        
        if result.numel() > 0:
            logger.info(f"[debug] Chunked BALD mean={result.mean().item():.4f}, std={result.std().item():.4f}, shape={result.shape}")
        
        return result
    
    def _prepare_inputs(self, inputs):
        """Override to catch OOM errors and implement reward-quality weighted exploration coefficient"""
        try:
            # Get all the standard processing from parent
            result = super()._prepare_inputs(inputs)
            
            # Apply quality-weighted coefficient if we're using exploration
            if hasattr(self.args, 'explore_beta') and self.args.explore_beta > 0:
                # The parent class computes advantages but we need to modify them
                # Extract the key values we need
                advantages = result.get('advantages')
                
                # Check if we have the necessary metrics from parent computation
                if hasattr(self, '_last_rewards') and hasattr(self, '_last_bald_raw'):
                    rewards = self._last_rewards
                    bald_raw = self._last_bald_raw
                    
                    # Compute reward quality per prompt group
                    rewards_grouped = rewards.view(-1, self.num_generations)
                    reward_quality = torch.sigmoid(rewards_grouped - rewards_grouped.mean(1, keepdim=True))
                    reward_quality = reward_quality.view_as(rewards)
                    
                    # Compute BALD factor
                    bald_grouped = bald_raw.view(-1, self.num_generations)
                    bald_z = (bald_grouped - bald_grouped.mean(1, keepdim=True)) / (bald_grouped.std(1, keepdim=True) + 1e-6)
                    bald_z = bald_z.view_as(bald_raw).clamp(-self.args.mi_cap, self.args.mi_cap)
                    bald_factor = torch.sigmoid(torch.relu(bald_z))
                    
                    # Compute quality-weighted coefficient
                    coef = 1.0 + self.args.explore_beta * reward_quality * bald_factor
                    
                    # Get the local slice for this process
                    process_slice = slice(
                        self.accelerator.process_index * (len(rewards) // self.accelerator.num_processes),
                        (self.accelerator.process_index + 1) * (len(rewards) // self.accelerator.num_processes),
                    )
                    coef_local = coef[process_slice]
                    
                    # Recompute advantages with quality-weighted coefficient
                    # First, remove the old coefficient (parent applies coef uniformly)
                    # Then apply our quality-weighted coefficient
                    if advantages is not None:
                        # The parent already applied some coefficient, we need to replace it
                        # Since we can't easily undo it, we'll just apply our factor on top
                        result['advantages'] = advantages * (coef_local.detach() / coef_local.mean().detach())
                        
                        # Log additional metrics
                        if not hasattr(self._metrics, 'reward_quality_mean'):
                            self._metrics['reward_quality_mean'] = []
                            self._metrics['bald_factor_mean'] = []
                        self._metrics['reward_quality_mean'].append(reward_quality.mean().item())
                        self._metrics['bald_factor_mean'].append(bald_factor.mean().item())
                        
                        logger.info(f"Applied quality-weighted exploration: reward_quality={reward_quality.mean():.3f}, bald_factor={bald_factor.mean():.3f}")
            
            return result
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
