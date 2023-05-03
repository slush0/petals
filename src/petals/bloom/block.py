"""
Bloom intermediate layer
Based on https://github.com/huggingface/transformers/commit/ca2a55e9dfb245527b5e1c954fec6ffbb7aef07b
See commit history for authorship.
"""
import os
from typing import Optional, Tuple

import torch.nn.quantized.dynamic.modules.linear
import transformers
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, _make_causal_mask, _expand_mask

if not os.getenv("PETALS_IGNORE_DEPENDENCY_VERSION"):
    assert transformers.__version__.startswith("4.25."), "Please install transformers 4.25.1"


class WrappedLlamaBlock(LlamaDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        alibi: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ):
        assert attention_mask is None
        batch_size, seq_length = hidden_states.shape[:2]
        past_length = 0 if layer_past is None else layer_past[0].shape[-1]
        seq_length_with_past = seq_length + past_length
        attention_mask = torch.ones((batch_size, seq_length_with_past),
                                    device=hidden_states.device,
                                    dtype=hidden_states.dtype)
        attention_mask = self._prepare_attn_mask(attention_mask, (batch_size, seq_length), past_length)

        print("BLOCK FORWARD", hidden_states, attention_mask, layer_past)
        return super().forward(
            hidden_states, *args, attention_mask=attention_mask, past_key_value=layer_past, **kwargs
        )

    def _prepare_attn_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
    ) -> torch.Tensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        device = attention_mask.device
        dtype = attention_mask.dtype
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, dtype=dtype, past_key_values_length=past_key_values_length
            ).to(device)

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, dtype=dtype).to(device)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )
        return combined_attention_mask

