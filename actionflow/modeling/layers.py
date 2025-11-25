import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func

from actionflow.kernels.ops import (
    TritonLlamaRMSNorm,
    fused_rope_write_kv_wrapper,
    shift_varlen_kv_cache_wrapper
)

class LlamaPIPEDecodeLayer(nn.Module):
    def __init__(self, original_layer: nn.Module, config):
        """
        Wraps an original LlamaDecoderLayer to enable packed pipelined execution.
        Uses Zero-Copy references to the original weights.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = original_layer.self_attn.scaling
        self.num_key_value_heads = config.num_key_value_heads
        
        # --- Zero-Copy References ---
        # We hold a reference to the original layer to access its projections
        self.original_layer = original_layer
        
        # Initialize Triton Norms wrapping the ORIGINAL weights
        # Note: We must ensure the weight tensors are shared
        self.input_layernorm = TritonLlamaRMSNorm(
            config.hidden_size, 
            eps=original_layer.input_layernorm.variance_epsilon
        )
        self.input_layernorm.weight = original_layer.input_layernorm.weight

        self.post_attention_layernorm = TritonLlamaRMSNorm(
            config.hidden_size, 
            eps=original_layer.post_attention_layernorm.variance_epsilon
        )
        self.post_attention_layernorm.weight = original_layer.post_attention_layernorm.weight

    def packed_forward(
        self,
        batch_hidden_states,   # List[Tensor]
        kv_ring_buffer,        # Tensor
        global_position_embeddings, # Tuple[Tensor, Tensor]
        seq_lens,              # List[int]
        cu_seqlens_q,          # Tensor
        cu_seqlens_k,          # Tensor
        max_seqlen_q,          # int
        max_seqlen_k,          # int
        **flash_attn_kwargs,
    ):
        """
        Executes the 1P + (K-1)D packed forward pass.
        """
        # === Step 1: Concat Hidden States ===
        # Shape: (1, total_L, D)
        residual = torch.cat(batch_hidden_states, dim=1)
        B, L, D = residual.shape
        
        if B != 1:
            raise ValueError("ActionFlow currently only supports batch_size=1")

        # === Step 2: Norm + QKV Projection ===
        # Apply Triton RMSNorm
        normed = self.input_layernorm(residual)
        
        # Use original projections
        queries = self.original_layer.self_attn.q_proj(normed)
        keys = self.original_layer.self_attn.k_proj(normed)
        values = self.original_layer.self_attn.v_proj(normed)

        # Reshape to (B, L, H, D_h)
        queries = queries.view(B, L, -1, self.head_dim)
        keys = keys.view(B, L, self.num_key_value_heads, self.head_dim)
        values = values.view(B, L, self.num_key_value_heads, self.head_dim)

        # === Step 3: Fused RoPE + Write KV (Triton) ===
        # Prepare output tensor for Q (RoPE applied)
        q_varlen = torch.empty_like(queries.squeeze(0))
        cos_full, sin_full = global_position_embeddings
        prefill_len = seq_lens[0]
        
        # Call your custom kernel wrapper
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

        # === Step 4: Flash Attention Varlen ===
        # Retrieve K/V from Ring Buffer (Zero-copy slice)
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

        # === Step 5: Shift Ring Buffer (Triton) ===
        # In-place shift for next iteration
        shift_varlen_kv_cache_wrapper(
            kv_cache=kv_ring_buffer,
            B_stages=len(seq_lens),
            prefill_len=prefill_len
        )
        
        # === Step 6: Output Projection & MLP ===
        attn_concat = attn_output.view(B, L, D)
        attn_concat = self.original_layer.self_attn.o_proj(attn_concat)
        
        # First Residual
        hidden_states = residual + attn_concat

        # MLP Block (Norm -> MLP -> Residual)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.original_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # === Step 7: Split Outputs ===
        outputs = []
        start = 0
        for seq_len in seq_lens:
            outputs.append(hidden_states[:, start:start+seq_len, :])
            start += seq_len

        return outputs