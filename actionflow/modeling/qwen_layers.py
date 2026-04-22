"""
Qwen2.5-VL Action-Flow Layer Implementation
Adapted from LlamaPIPEDecodeLayer for Qwen architecture.
"""
import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func
from typing import List, Tuple

try:
    import bitsandbytes.functional as bnb_F
    from bitsandbytes.nn import Linear4bit
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

from actionflow.kernels.ops import TritonLlamaRMSNorm, fused_rope_write_kv_wrapper, shift_varlen_kv_cache_wrapper


class QwenPIPEDecodeLayer(nn.Module):
    """
    Wraps Qwen2_5_VLDecoderLayer for packed pipelined execution.

    Key differences from Llama:
    - Uses Qwen2RMSNorm (compatible with Llama RMSNorm)
    - SwiGLU MLP with gate_proj, up_proj, down_proj
    - mrope for multimodal positions (simplified for decode)

    For decode stage, we treat it similar to Llama since:
    - RoPE positions are sequential after prefill
    - Attention pattern is the same
    """

    def __init__(self, original_layer: nn.Module, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.scaling = 1.0 / (self.head_dim ** 0.5)

        # Zero-copy reference to original layer
        self.original_layer = original_layer

        # RMSNorm wrappers (Qwen2RMSNorm is compatible)
        eps = getattr(original_layer.input_layernorm, 'variance_epsilon',
                     getattr(original_layer.input_layernorm, 'eps', 1e-6))

        self.input_layernorm = TritonLlamaRMSNorm(config.hidden_size, eps=eps)
        self.input_layernorm.weight = original_layer.input_layernorm.weight

        self.post_attention_layernorm = TritonLlamaRMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm.weight = original_layer.post_attention_layernorm.weight

    def _optimized_linear(self, layer, x):
        """BNB INT4 optimization."""
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
        batch_hidden_states: List[torch.Tensor],
        kv_ring_buffer: torch.Tensor,
        global_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        seq_lens: List[int],
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        **kwargs,
    ):
        """
        Execute packed forward pass for Qwen layer.

        Same logic as LlamaPIPEDecodeLayer:
        1. Concat hidden states
        2. RMSNorm + QKV projection
        3. Fused RoPE + Write KV (Triton)
        4. Flash Attention Varlen
        5. Shift KV cache (Triton)
        6. Output projection + MLP
        7. Split outputs
        """
        # Step 1: Concat hidden states
        residual = torch.cat(batch_hidden_states, dim=1)
        B, L, D = residual.shape

        if B != 1:
            raise ValueError("ActionFlow only supports batch_size=1")

        # Step 2: RMSNorm + QKV projection
        normed = self.input_layernorm(residual)

        queries = self._optimized_linear(self.original_layer.self_attn.q_proj, normed)
        keys = self._optimized_linear(self.original_layer.self_attn.k_proj, normed)
        values = self._optimized_linear(self.original_layer.self_attn.v_proj, normed)

        # Reshape to (B, L, H, D_h)
        queries = queries.view(B, L, self.num_attention_heads, self.head_dim)
        keys = keys.view(B, L, self.num_key_value_heads, self.head_dim)
        values = values.view(B, L, self.num_key_value_heads, self.head_dim)

        # Step 3: Fused RoPE + Write KV (using same Triton kernel as Llama)
        q_varlen = torch.empty_like(queries.squeeze(0))
        cos_full, sin_full = global_position_embeddings
        prefill_len = seq_lens[0]

        # Apply RoPE and write KV to ring buffer
        fused_rope_write_kv_wrapper(
            Q_new=queries,
            K_new=keys,
            V_new=values,
            kv_ring_buffer=kv_ring_buffer,
            cos=cos_full.squeeze(0),
            sin=sin_full.squeeze(0),
            prefill_len=prefill_len,
            q_varlen=q_varlen
        )

        # Step 4: Flash Attention Varlen
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
            causal=True
        )

        # Step 5: Shift KV cache (in-place for next iteration)
        shift_varlen_kv_cache_wrapper(
            kv_cache=kv_ring_buffer,
            B_stages=len(seq_lens),
            prefill_len=prefill_len
        )

        # Step 6: Output projection + MLP
        attn_concat = attn_output.view(B, L, D)
        attn_concat = self._optimized_linear(self.original_layer.self_attn.o_proj, attn_concat)

        # First residual
        hidden_states = residual + attn_concat

        # MLP block (SwiGLU)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        mlp = self.original_layer.mlp
        gate_out = self._optimized_linear(mlp.gate_proj, hidden_states)
        up_out = self._optimized_linear(mlp.up_proj, hidden_states)
        hidden_states = self._optimized_linear(mlp.down_proj, mlp.act_fn(gate_out) * up_out)

        hidden_states = residual + hidden_states

        # Step 7: Split outputs
        outputs = []
        start = 0
        for seq_len in seq_lens:
            outputs.append(hidden_states[:, start:start+seq_len, :])
            start += seq_len

        return outputs
