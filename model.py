from typing import Callable, Optional, Union, Tuple

import torch
import torch.nn as nn

import copy

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3PreTrainedModel, Qwen3RMSNorm, Qwen3RotaryEmbedding
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast

from utils import trunc_normal_init_


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
        output_states: torch.FloatTensor, # y
        latent_states: torch.FloatTensor, # z
        original_input: Optional[torch.FloatTensor] = None, # x
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        
        if original_input is None:
            inputs_embeds = output_states + latent_states
        else:
            inputs_embeds = original_input + output_states + latent_states

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

        # Initialize non-trainable hidden states from truncated normal distribution
        # Shape: (1, 1, hidden_size) - will be expanded to match batch size during forward
        hidden_size = base_model.config.hidden_size

        # Random initialization math happens in fp32 but stored buffers are bf16
        self.y_init = nn.Buffer(trunc_normal_init_(torch.empty(hidden_size, dtype=torch.float32), std=1).to(torch.bfloat16), persistent=True)
        self.z_init = nn.Buffer(trunc_normal_init_(torch.empty(hidden_size, dtype=torch.float32), std=1).to(torch.bfloat16), persistent=True)

        # output_init = torch.empty(1, 1, hidden_size, dtype=torch.float32)
        # latent_init = torch.empty(1, 1, hidden_size, dtype=torch.float32)

        # # Initialize with truncated normal: mean=0, std=1, truncated at ±2*std
        # nn.init.trunc_normal_(output_init, mean=0.0, std=1.0, a=-2.0, b=2.0)
        # nn.init.trunc_normal_(latent_init, mean=0.0, std=1.0, a=-2.0, b=2.0)

        # # Random initialization math happens in fp32 but stored buffers are bf16
        # output_init = output_init.to(torch.bfloat16)
        # latent_init = latent_init.to(torch.bfloat16)

        # # Register as non-trainable buffers (automatically handles device placement)
        # self.register_buffer("output_init", output_init, persistent=True)
        # self.register_buffer("latent_init", latent_init, persistent=True)

    def get_inits(self, input_ids: torch.LongTensor):
        # Expand initial states to match batch size and sequence length
        # the states should have shape: (batch_size, seq_len, hidden_size)
        batch_size, seq_len = input_ids.shape
        output_init = self.y_init.expand(batch_size, seq_len, -1)
        latent_init = self.z_init.expand(batch_size, seq_len, -1)
        return output_init, latent_init

    def latent_recursion(
        self,
        output_states: torch.FloatTensor,
        latent_states: torch.FloatTensor,
        original_input: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None, 
        n: int = 6) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        for _ in range(n):
            head_output = self.custom_head(
                output_states=output_states, latent_states=latent_states, 
                original_input=original_input, attention_mask=attention_mask
            )
            latent_states = head_output.last_hidden_state
        
        head_output = self.custom_head(
            output_states=output_states, latent_states=latent_states, attention_mask=attention_mask)
        output_states = head_output.last_hidden_state
        
        return output_states, latent_states
    
    def compute_loss(self, logits, labels, vocab_size):
        """
        Compute cross-entropy loss for language modeling.
        """
        logits = logits.to(torch.float32) # upcast for loss calculation

        # Labels is actually passed in as labels=input_ids
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute cross-entropy loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct( # per token loss
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1)
        )

        return loss
    
    def forward(
        self,
        output_states: torch.FloatTensor, # y
        latent_states: torch.FloatTensor, # z
        input_ids: Optional[torch.LongTensor] = None, # x
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        n: int = 6,
        T: int = 3, 
        # logits_to_keep: Union[int, torch.Tensor] = 0, 
        **kwargs
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
        # Get hidden states from base model (before custom head)
        with torch.no_grad():
            base_outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            # Is it really the last index?
            original_input = base_outputs.last_hidden_state

            for _ in range(T-1):
                output_states, latent_states = self.latent_recursion(
                    output_states, latent_states, original_input,
                    attention_mask=attention_mask, n=n 
                )
        
        output_states, latent_states = self.latent_recursion(
            output_states, latent_states, original_input,
            attention_mask=attention_mask, n=n 
        )

        with torch.no_grad(): # really want no grad here??
            logits = self.base_model.lm_head(output_states)

        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels, self.base_model.config.vocab_size)

        return output_states.detach(), latent_states.detach(), logits, loss

def main():
    base_model_name = "Qwen/Qwen3-0.6B-Base"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    new_config = copy.deepcopy(base_model.config)
    new_config.num_hidden_layers = 2
    print(f'the NEW config is {new_config}')

    # Cast weights of the recurrent module to bf16 just like base model
    custom_head = Qwen3RecurrentModule(new_config).to(device).to(torch.bfloat16)

    # Create the complete model with custom head
    model = ModelWithRecurrentHead(base_model, custom_head).to(device)

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
    print('INPUT TEXT IS >> ' + text)

    model_inputs = tokenizer([text], return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    _, _, res = model(**model_inputs)
    indices = torch.argmax(res, dim=-1)
    print(f'RESULT is >> {indices}')
    print(f'DECODES to >> {tokenizer.batch_decode(indices)}')

if __name__ == "__main__":
    main()