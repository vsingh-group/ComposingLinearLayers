from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Cache,
    apply_rotary_pos_emb,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS
)

from typing_extensions import Unpack
from transformers.models.llama.modeling_llama import FlashAttentionKwargs

class CustomQwen2Attention(Qwen2Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        self.pre_qkv_input = hidden_states.detach().clone()

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q_proj_raw = self.q_proj(hidden_states)
        k_proj_raw = self.k_proj(hidden_states)
        v_proj_raw = self.v_proj(hidden_states)

        self.q_proj_output = q_proj_raw.detach().clone()
        self.k_proj_output = k_proj_raw.detach().clone()
        self.v_proj_output = v_proj_raw.detach().clone()

        query_states = q_proj_raw.view(hidden_shape).transpose(1, 2)
        key_states = k_proj_raw.view(hidden_shape).transpose(1, 2)
        value_states = v_proj_raw.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                pass
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        self.pre_o_proj_input = attn_output.detach().clone()
        attn_output = self.o_proj(attn_output)
        self.o_proj_output = attn_output.detach().clone()
        return attn_output, attn_weights
