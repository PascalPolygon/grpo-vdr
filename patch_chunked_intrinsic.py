#!/usr/bin/env python3
"""
Patch to enable chunked intrinsic reward computation to reduce memory usage.
Run this before training to modify the GRPO trainer.
"""

import os
import sys

def create_chunked_compute_patch():
    """Create a patch for chunked intrinsic reward computation."""
    
    patch_content = '''
# Add this to custom_grpo_trainer.py to override compute_intrinsic_reward

def compute_intrinsic_reward(self, prompt_ids, completion_ids, attention_mask):
    """Compute intrinsic rewards with chunked processing to save memory."""
    import torch
    import gc
    
    if not self.use_intrinsic_rewards:
        return torch.zeros(completion_ids.size(0), device=completion_ids.device)
    
    # Process in smaller chunks to avoid OOM
    chunk_size = 2  # Process 2 samples at a time instead of full batch
    batch_size = prompt_ids.size(0)
    all_rewards = []
    
    for i in range(0, batch_size, chunk_size):
        end_idx = min(i + chunk_size, batch_size)
        
        # Process chunk
        chunk_prompt = prompt_ids[i:end_idx]
        chunk_completion = completion_ids[i:end_idx]
        chunk_mask = attention_mask[i:end_idx] if attention_mask is not None else None
        
        # Call parent method on chunk
        chunk_reward = super().compute_intrinsic_reward(
            chunk_prompt, chunk_completion, chunk_mask
        )
        
        all_rewards.append(chunk_reward)
        
        # Force memory cleanup between chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Concatenate results
    return torch.cat(all_rewards, dim=0)
'''
    
    print("Chunked intrinsic reward computation patch created.")
    print("\nTo use this patch, add the above method to your custom_grpo_trainer.py")
    print("This will process intrinsic rewards in chunks of 2 samples at a time.")
    
    # Save to file
    with open("chunked_intrinsic_patch.txt", "w") as f:
        f.write(patch_content)
    
    print(f"\nPatch saved to: chunked_intrinsic_patch.txt")
    print("Add this method to your CustomGRPOTrainer class.")

if __name__ == "__main__":
    create_chunked_compute_patch()
    
    print("\nAdditional memory tips:")
    print("1. Monitor GPU memory during training: watch -n 1 nvidia-smi")
    print("2. If OOM occurs at a specific step, note the pattern")
    print("3. Consider using gradient accumulation if effective batch size is important") 