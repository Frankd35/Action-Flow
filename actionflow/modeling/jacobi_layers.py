"""
Jacobi decode layer for packed pipeline: mirrors ``layers.py`` / ``LlamaPIPEDecodeLayer`` layout,
with Jacobi KV segments (see ``kernels/jacobi_ops.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func

try:
    import bitsandbytes.functional as bnb_F
    from bitsandbytes.nn import Linear4bit

    HAS_BNB = True
except ImportError:
    HAS_BNB = False

from actionflow.kernels.ops import TritonLlamaRMSNorm
from actionflow.kernels.jacobi_ops import fused_rope_write_kv_jacobi_wrapper, shift_jacobi_kv_cache_torch


class LlamaPIPEJacobiDecodeLayer(nn.Module):
    """Jacobi pipeline stage: Q segments [prefill | J | J | ...], K segments [P | P+J | P+J | ...]."""

    def __init__(self, original_layer: nn.Module, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = original_layer.self_attn.scaling
        self.num_key_value_heads = config.num_key_value_heads
        self.original_layer = original_layer

        self.input_layernorm = TritonLlamaRMSNorm(config.hidden_size, eps=original_layer.input_layernorm.variance_epsilon)
        self.input_layernorm.weight = original_layer.input_layernorm.weight

        self.post_attention_layernorm = TritonLlamaRMSNorm(config.hidden_size, eps=original_layer.post_attention_layernorm.variance_epsilon)
        self.post_attention_layernorm.weight = original_layer.post_attention_layernorm.weight

    def _optimized_linear(self, layer, x):
        if HAS_BNB and isinstance(layer, Linear4bit) and not x.requires_grad:
            shape = x.shape
            x_2d = x.view(-1, shape[-1])
            out = bnb_F.gemv_4bit(x_2d, layer.weight.t(), state=layer.weight.quant_state)
            if layer.bias is not None:
                out += layer.bias
            return out.view(*shape[:-1], -1)
        return layer(x)

    def packed_forward(
        self,
        batch_hidden_states,
        kv_ring_buffer,
        global_position_embeddings,
        seq_lens,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        prefill_len: int,
        jacobi_tokens: int,
        num_stages: int,
        **flash_attn_kwargs,
    ):
        residual = torch.cat(batch_hidden_states, dim=1)
        B, L, D = residual.shape
        if B != 1:
            raise ValueError("Jacobi ActionFlow currently only supports batch_size=1")

        normed = self.input_layernorm(residual)
        queries = self._optimized_linear(self.original_layer.self_attn.q_proj, normed)
        keys = self._optimized_linear(self.original_layer.self_attn.k_proj, normed)
        values = self._optimized_linear(self.original_layer.self_attn.v_proj, normed)

        queries = queries.view(B, L, -1, self.head_dim)
        keys = keys.view(B, L, self.num_key_value_heads, self.head_dim)
        values = values.view(B, L, self.num_key_value_heads, self.head_dim)

        # Triton wrapper requires contiguous q_varlen; empty_like(squeezed Q) can inherit non-contiguous strides.
        qs = queries.squeeze(0)
        q_varlen = torch.empty(qs.shape, dtype=qs.dtype, device=qs.device)
        cos_full, sin_full = global_position_embeddings

        fused_rope_write_kv_jacobi_wrapper(
            Q_new=queries,
            K_new=keys,
            V_new=values,
            kv_ring_buffer=kv_ring_buffer,
            cos=cos_full.squeeze(0),
            sin=sin_full.squeeze(0),
            prefill_len=prefill_len,
            jacobi_tokens=jacobi_tokens,
            num_stages=num_stages,
            q_varlen=q_varlen,
        )

        k_varlen = kv_ring_buffer[0]
        v_varlen = kv_ring_buffer[1]

        attn_output = flash_attn_varlen_func(
            q=q_varlen,
            k=k_varlen,
            v=v_varlen,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=self.scaling,
            causal=True,
        )

        shift_jacobi_kv_cache_torch(
            kv_cache=kv_ring_buffer,
            num_stages=num_stages,
            prefill_len=prefill_len,
            jacobi_tokens=jacobi_tokens,
        )

        attn_concat = attn_output.view(B, L, D)
        attn_concat = self._optimized_linear(self.original_layer.self_attn.o_proj, attn_concat)

        hidden_states = residual + attn_concat
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        mlp = self.original_layer.mlp
        gate_out = self._optimized_linear(mlp.gate_proj, hidden_states)
        up_out = self._optimized_linear(mlp.up_proj, hidden_states)
        hidden_states = self._optimized_linear(mlp.down_proj, mlp.act_fn(gate_out) * up_out)

        hidden_states = residual + hidden_states

        outputs = []
        start = 0
        for seq_len in seq_lens:
            outputs.append(hidden_states[:, start : start + seq_len, :])
            start += seq_len

        return outputs
