"""
Custom implementation of Qwen2 model with dropout layers.
This module creates subclasses of Qwen2 components to add dropout
layers in key places for Monte Carlo dropout inference.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM, 
    Qwen2Model, 
    Qwen2MLP, 
    Qwen2DecoderLayer,
    Qwen2Attention,
    Qwen2RMSNorm,
    Cache,
    Unpack, 
    FlashAttentionKwargs
)


class Qwen2MLPWithDropout(Qwen2MLP):
    """Extends Qwen2MLP to add dropout layers."""
    
    def __init__(self, config):
        super().__init__(config)
        # Add dropout layers
        self.dropout_rate = getattr(config, "hidden_dropout", 0.1)
        self.mlp_dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x):
        # Apply dropout to intermediate activations
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        # Apply dropout before multiplication
        gate_output = self.mlp_dropout(gate_output)
        intermediate = gate_output * up_output
        # Final projection with dropout
        down_proj = self.down_proj(intermediate)
        return down_proj


class Qwen2AttentionWithDropout(Qwen2Attention):
    """Extends Qwen2Attention to add more explicit dropout control."""
    
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # Add explicit dropout layer for attention output
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Get the base output from parent class
        attn_output, attn_weights = super().forward(
            hidden_states, 
            position_embeddings, 
            attention_mask, 
            past_key_value, 
            cache_position, 
            **kwargs
        )
        
        # Apply dropout to attention output if model is in training mode
        if self.training:
            attn_output = self.attn_dropout(attn_output)
            
        return attn_output, attn_weights


class Qwen2DecoderLayerWithDropout(Qwen2DecoderLayer):
    """Extends Qwen2DecoderLayer to add dropout after attention and feed-forward."""
    
    def __init__(self, config, layer_idx: int):
        # Initialize parent class without attention and MLP
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        
        # Create custom attention and MLP modules with dropout
        self.self_attn = Qwen2AttentionWithDropout(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLPWithDropout(config)
        
        # Create normalization layers
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Add explicit residual dropouts
        self.hidden_dropout = getattr(config, "hidden_dropout", 0.1)
        self.residual_dropout = nn.Dropout(self.hidden_dropout)
        
        # Handle sliding window warning
        if getattr(config, "sliding_window", None) and config._attn_implementation != "flash_attention_2":
            from transformers.utils import logging
            logger = logging.get_logger(__name__)
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention with dropout
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        # Apply residual dropout
        hidden_states = self.residual_dropout(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected with dropout
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Apply residual dropout
        hidden_states = self.residual_dropout(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Qwen2ModelWithDropout(Qwen2Model):
    """Qwen2Model with dropout layers added to each decoder layer."""
    
    def __init__(self, config):
        super().__init__(config)
        # Replace the decoder layers with our custom ones
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayerWithDropout(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # Add embedding dropout
        self.embd_dropout = nn.Dropout(getattr(config, "embd_pdrop", 0.1))
        
        # Initialize weights
        self.post_init()
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ):
        # Use parent's forward, but with embedding dropout
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
            
            # Apply dropout to embeddings when in training mode
            if self.training:
                inputs_embeds = self.embd_dropout(inputs_embeds)
                
        # Continue with normal forward pass
        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **flash_attn_kwargs,
        )


class Qwen2ForCausalLMWithDropout(Qwen2ForCausalLM):
    """
    Qwen2ForCausalLM with dropout layers added throughout the model.
    This can be used as a drop-in replacement for Qwen2ForCausalLM.
    """
    
    def __init__(self, config):
        # Initialize with parent but replace model with our custom one
        super(Qwen2ForCausalLM, self).__init__(config)  # Call grandparent's init
        self.model = Qwen2ModelWithDropout(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Add final dropout before output projection
        self.output_dropout = nn.Dropout(getattr(config, "hidden_dropout", 0.1))

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,  # Force return_dict to always be True
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        logits_to_keep: int = None,  # Added for compatibility with GRPO trainer
        **kwargs,
    ):
        # Get transformer outputs from our modified model - force return_dict=True
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,  # Force this to be True to get an object
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        
        # Apply dropout before projection if in training mode
        if self.training:
            hidden_states = self.output_dropout(hidden_states)
            
        # Handle both num_logits_to_keep and logits_to_keep (for GRPO)
        actual_logits_to_keep = logits_to_keep if logits_to_keep is not None else num_logits_to_keep
            
        # Apply modified logic for logits calculation, based on logits_to_keep
        if actual_logits_to_keep == 0:
            logits = self.lm_head(hidden_states)
        else:
            # Only compute logits for the last logits_to_keep tokens
            # This is particularly useful during generation to save memory
            logits = torch.cat(
                [
                    # Pad with zeros for all but the last logits_to_keep tokens
                    torch.zeros(
                        hidden_states.shape[0],
                        hidden_states.shape[1] - actual_logits_to_keep,
                        self.vocab_size,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    ),
                    # Compute logits only for the last logits_to_keep tokens
                    self.lm_head(hidden_states[:, -actual_logits_to_keep:, :]),
                ],
                dim=1,
            )
            
        # Handle loss calculation if labels are provided
        loss = None
        if labels is not None:
            # Here we implement the modified loss calculation similar to parent class
            # This handles cases where we compute logits only for specific tokens
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Handle num_logits_to_keep case
            if num_logits_to_keep > 0:
                cutoff = shift_labels.shape[1] - num_logits_to_keep + 1
                if cutoff > 0:
                    # Zero out loss for tokens outside the num_logits_to_keep window
                    temp_shift_labels = shift_labels.clone()
                    temp_shift_labels[:, :cutoff] = -100
                    shift_labels = temp_shift_labels
                    
            # Calculate cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        # Always return a CausalLMOutputWithPast object for compatibility with the GRPO trainer
        from transformers.modeling_outputs import CausalLMOutputWithPast
        
        # Create a proper CausalLMOutputWithPast object that will have the right attributes
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs[1] if len(outputs) > 1 else None,
            hidden_states=outputs[2] if output_hidden_states and len(outputs) > 2 else None,
            attentions=outputs[3] if output_attentions and len(outputs) > 3 else None,
        )


def create_qwen_with_dropout(pretrained_model_name_or_path, dropout_rate=0.1, **kwargs):
    """
    Helper function to create a Qwen2ForCausalLMWithDropout model from a pretrained model.
    
    Args:
        pretrained_model_name_or_path: Path or name of the pretrained model to load
        dropout_rate: Rate of dropout to use throughout the model
        **kwargs: Additional arguments to pass to from_pretrained
        
    Returns:
        A Qwen2ForCausalLMWithDropout model with weights loaded from pretrained model
    """
    import torch
    from transformers import AutoConfig
    
    # Load original config
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    
    # Add dropout parameters
    config.attention_dropout = dropout_rate
    config.hidden_dropout = dropout_rate
    config.embd_pdrop = dropout_rate
    config.resid_pdrop = dropout_rate
    
    # Create custom model with these dropout rates
    model = Qwen2ForCausalLMWithDropout.from_pretrained(
        pretrained_model_name_or_path,
        config=config,
        **kwargs
    )
    
    return model
