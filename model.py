from typing import Callable, Optional, Union, Tuple

import torch
import torch.nn as nn

import copy

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast

from utils import trunc_normal_init_
from loss import compute_lm_loss


class Qwen3RecurrentModule(nn.Module):
    """
    Implementation of the net() function in https://arxiv.org/pdf/2510.04871 Figure 3.
    Except this uses the Qwen3 architecture instead of the one in the paper, very similar though.
    `net` takes two forms of arguments:
    1) z = net(x, y, z) # gives new latent state
    2) y = net(y, z) # gives new output state
    """
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing -> PretrainedModel only
        # self.post_init()

    def forward(
        self,
        # output_states: torch.FloatTensor, # y
        # latent_states: torch.FloatTensor, # z
        states: torch.FloatTensor,
        original_input: Optional[torch.FloatTensor] = None, # x
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        
        if original_input is None:
            inputs_embeds = states
        else:
            inputs_embeds = states + original_input

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type], # type: ignore[assignment]
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class ModelWithRecurrentHead(nn.Module):
    """
    Wrapper that combines a base model with a recurrent transformer decoder head.
    Implementation of the forward pass in https://arxiv.org/pdf/2510.04871 Figure 3.
    Except that this first gets the embeddings for the input tokens via a pretrained language model
    then builds on top of that.
    """

    def __init__(self, base_model, custom_head):
        """
        Args:
            base_model: Pretrained base model (e.g., from transformers)
            custom_head: Custom transformer recurrent head
        """
        super().__init__()
        self.base_model = base_model
        self.custom_head = custom_head

        hidden_size = custom_head.config.hidden_size

        # Random initialization math happens in fp32 but stored buffers are bf16 
        # self.z_init = nn.Buffer(trunc_normal_init_(torch.empty(hidden_size, dtype=torch.float32), std=1).to(torch.bfloat16), persistent=True)

        self.halting_head = nn.Linear(in_features=hidden_size, out_features=1).to(torch.bfloat16)

    # def get_inits(self, input_ids: torch.LongTensor):
    #     # Expand initial states to match batch size and sequence length
    #     # the states should have shape: (batch_size, seq_len, hidden_size)
    #     batch_size, seq_len = input_ids.shape
    #     output_init = self.y_init.expand(batch_size, seq_len, -1)
    #     #latent_init = self.z_init.expand(batch_size, seq_len, -1)
    #     return output_init #, latent_init

    # def latent_recursion(
    #     self,
    #     input_states: torch.FloatTensor,
    #     # latent_states: torch.FloatTensor,
    #     # original_input: torch.FloatTensor,
    #     attention_mask: Optional[torch.Tensor] = None, 
    #     n: int = 6) -> torch.FloatTensor:

    #     for _ in range(n + 1):
    #         head_output = self.custom_head(
    #             input_states=input_states, attention_mask=attention_mask
    #         )
    #         input_states = head_output.last_hidden_state
        
    #     head_output = self.custom_head(
    #         input_states=input_states, attention_mask=attention_mask)
    #     output_states = head_output.last_hidden_state
        
    #     return output_states
    
    def deep_recursion_ACT(
        self,
        states: torch.FloatTensor,
        # original_input: torch.FloatTensor, # x
        # output_states: torch.FloatTensor, # y
        # latent_states: torch.FloatTensor, # z
        attention_mask: torch.Tensor,
        halt_mask: Optional[torch.Tensor] = None,
        n: int = 6,
        threshold: float = 0.99,
        delta: float = 1e-6, # for rounding errors on halting probabilties
    ):
        """
        Forward pass: extract hidden states from base model and
        apply custom current head.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for input
            **kwargs: Additional arguments for base model

        Returns:
            output_states: Detached output states after recurrent process (for use in deep supervision process)
            latent_states: Detached latent states after recurrent process (for use in deep supervision process)
            logits: Output logits from piping output_states through the base LLM's lm head 
        """

        # with torch.no_grad():
        #     for _ in range(T-1):
        #         for _ in range(n+1): # n+1 for compute invariance vs. previous runs 
        #             head_output = self.custom_head(
        #                 states=states, original_input=original_input, attention_mask=attention_mask
        #             )
        #             states = head_output.last_hidden_state
        
        batch, length, _ = states.size()
        # ACT accumulation in higher precision
        halting_probs = torch.zeros((batch, length), device=states.device, dtype=torch.float32)
        remainders = torch.zeros((batch, length), device=states.device, dtype=torch.float32)
        n_updates = torch.zeros((batch, length), device=states.device, dtype=torch.int32)
        weighted_states = torch.zeros_like(states, dtype=torch.float32)

        # For auto-regressive generation we only care about the pred at last position
        if halt_mask is None:
            halt_mask = (torch.arange(length, device=states.device) < (length - 1)).float()
            halt_mask = halt_mask.unsqueeze(0).expand(batch, -1)

        for step in range(n+1): # n+1 for compute invariance vs. previous runs
            # Calculate probs based on states
            p = torch.sigmoid(self.halting_head(states).squeeze(-1).float())  # [batch, len], float32

            # Masks
            potentially_running = halting_probs < (1.0 - delta) # [batch, len]
            newly_halted = ((halting_probs + p * potentially_running) > threshold) * potentially_running # [batch, len]
            still_running = ((halting_probs + p * potentially_running) <= threshold) * potentially_running # [batch, len]

            halting_probs += p * still_running # [batch, len]
            remainders += newly_halted * (1 - halting_probs) # [batch, len]
            halting_probs += newly_halted * remainders # [batch, len]

            n_updates += still_running + newly_halted

            update_weights = (still_running * p + newly_halted * remainders).unsqueeze(-1) # [batch, len, 1]
            # states used in update same as states used to compute probs
            weighted_states = states * update_weights + weighted_states * (1 - update_weights)

            # Could happen that it halts while not having gone into the recurrent head even once
            # If all halted, then terminate early, with prompt and pad tokens excluded
            if ((halting_probs + halt_mask) >= threshold).all():
                break

            head_output = self.custom_head(
                states=states, original_input=None, attention_mask=attention_mask
            )
            states = head_output.last_hidden_state

        # input/output embeds might be tied here for Qwen3 dense models
        logits = self.base_model.lm_head(weighted_states.to(states.dtype))

        return states.detach(), logits, step, n_updates.detach()

def instantiate_model(base_model_name: str, num_recurrent_layers: int, device: torch.device):
    # Load tokenizer and base model
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    for param in base_model.parameters():
        param.requires_grad = False
    for param in base_model.lm_head.parameters():
        param.requires_grad = True

    new_config = copy.deepcopy(base_model.config)
    new_config.num_hidden_layers = num_recurrent_layers
    new_config._attn_implementation = "flash_attention_2"
    print(f'the NEW config is {new_config}')

    custom_head = Qwen3RecurrentModule(new_config).to(device).to(torch.bfloat16) # type: ignore
    model = ModelWithRecurrentHead(base_model, custom_head).to(device)

    base_model_trainable_params = sum(p.numel() for p in model.base_model.parameters() if p.requires_grad)
    print(f"Base model frozen. Base model trainable parameters: {base_model_trainable_params:,}")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created. Trainable parameters: {trainable_params:,} / {total_params:,}")


    return tokenizer, model

def main():
    # Set random seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    base_model_name = "Qwen/Qwen3-0.6B"
    num_recurrent_layers = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = instantiate_model(base_model_name, num_recurrent_layers, device)

    print('Prepare model input...')
    # prepare the model input
    prompt = "What is 1 + 1 = ?"
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print('DEBUG: input text is >> ' + text)

    model_inputs = tokenizer([text], return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    # output_states, latent_states = model.get_inits(model_inputs['input_ids'])

    base_outputs = model.base_model.model(
        input_ids=model_inputs['input_ids'],
        attention_mask=model_inputs['attention_mask'],
        use_cache=False,
    )

    original_input = base_outputs.last_hidden_state

    print('ORIGINAL INPUT --->')
    print(original_input)

    _, logits, _, _ = model.deep_recursion_ACT(
        states=original_input,
        attention_mask=model_inputs['attention_mask'],
    )
    print('logits --->')
    print(logits)
    
    indices = torch.argmax(logits, dim=-1)
    print(f'RESULT is >> {indices}')
    print(f'DECODES to >> {tokenizer.batch_decode(indices)}')

if __name__ == "__main__":
    main()