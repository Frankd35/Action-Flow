import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func

from transformers.models.llama.modeling_llama import eager_attention_forward, apply_rotary_pos_emb
from transformers.cache_utils import Cache
from typing import Optional, Dict, Any, Tuple, List

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


class PipeStaticCache(Cache):
    is_compileable = True
    def __init__(self, config, max_batch_size: int, max_cache_len: int, device=None, dtype=torch.float32) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("meta")
        self.num_key_value_heads = config.num_key_value_heads

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        cache_shape = (self.max_batch_size, self.max_cache_len, self.num_key_value_heads, self.head_dim)
        
        for idx in range(config.num_hidden_layers):
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=self.device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=self.device)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def update(self, key_states, value_states, layer_idx: int, cache_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_position = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        
        if cache_position.dim() == 2:
            cache_position = cache_position.squeeze(0)
            
        k_out.index_copy_(1, cache_position, key_states.to(k_out.dtype))
        v_out.index_copy_(1, cache_position, value_states.to(v_out.dtype))
        return k_out, v_out

class ActionFlowBaselineDecodeLayer(LlamaPIPEDecodeLayer):
    """
    Baseline Layer: 无融合算子，纯 PyTorch eage Attention + for 循环
    继承自当前你的 LlamaPIPEDecodeLayer，复用 norm 和 original_layer
    """
    def packed_forward(
        self,
        batch_hidden_states,          # List[Tensor]
        batch_past_key_values,        # List[PipeStaticCache] (注意：Baseline不传kv_ring_buffer)
        batch_position_ids,           # List[Tensor]
        global_position_embeddings,   # Tuple[Tensor, Tensor]
        seq_lens,                     # List[int]
        **kwargs
    ):
        # 1. Concat
        residual = torch.cat(batch_hidden_states, dim=1)
        B, L, D = residual.shape

        # 2. Norm + QKV Proj
        normed = self.input_layernorm(residual)
        queries = self.original_layer.self_attn.q_proj(normed)
        keys = self.original_layer.self_attn.k_proj(normed)
        values = self.original_layer.self_attn.v_proj(normed)

        # 3. Reshape
        queries = queries.view(B, L, -1, self.head_dim).transpose(1, 2)
        keys = keys.view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 4. RoPE
        cos_full, sin_full = global_position_embeddings
        queries_rot, keys_rot = apply_rotary_pos_emb(queries, keys, cos_full, sin_full)

        # 5. Eager Attention Loop
        attn_outputs = []
        start_pos = 0
        for i in range(len(batch_hidden_states)):
            seq_len = seq_lens[i]
            end_pos = start_pos + seq_len

            query = queries_rot[:, :, start_pos:end_pos, :]
            key = keys_rot[:, :, start_pos:end_pos, :]
            value = values[:, :, start_pos:end_pos, :]

            past_kv = batch_past_key_values[i]
            pos_id = batch_position_ids[i]

            key_for_cache = key.transpose(1, 2) 
            value_for_cache = value.transpose(1, 2)

            full_k, full_v = past_kv.update(
                key_for_cache, 
                value_for_cache, 
                self.original_layer.self_attn.layer_idx,
                cache_kwargs={"cache_position": pos_id}
            )

            key_states = full_k[:, :end_pos, :, :].transpose(1, 2)
            value_states = full_v[:, :end_pos, :, :].transpose(1, 2)

            # Eager Attention
            attn_output, _ = eager_attention_forward(
                module=self.original_layer.self_attn,
                query=query,
                key=key_states,
                value=value_states, 
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0,
                **kwargs,
            )
            attn_outputs.append(attn_output)
            start_pos = end_pos

        # 6. Concat
        # eager_attention_forward return -- (B, L_q, H, D_h)
        attn_concat = torch.cat(attn_outputs, dim=1) 
        attn_concat = attn_concat.reshape(B, L, -1).contiguous()
        attn_concat = self.original_layer.self_attn.o_proj(attn_concat)

        hidden_states = residual + attn_concat

        # 7. MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.original_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # 8. Split
        outputs = []
        start = 0
        for seq_len in seq_lens:
            outputs.append(hidden_states[:, start:start+seq_len, :])
            start += seq_len

        return outputs
    
    
# === 以下代码追加到 actionflow/modeling/layers.py 的末尾 ===

# 确保导入（如果你的文件头部没有的话）
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    pass # 跑 baseline 的时候可能会报错，但保证代码能加载

class ActionFlowVarlenBaselineDecodeLayer(ActionFlowBaselineDecodeLayer):
    """
    Varlen Baseline Layer: 
    使用 flash_attn_varlen_func 优化 Attention，但 KV Cache 依然使用 PipeStaticCache，
    存在拼接 (torch.cat) 和拷贝开销。
    继承自 ActionFlowBaselineDecodeLayer 以复用相同的输入/输出处理逻辑。
    """
    def packed_forward(
        self,
        batch_hidden_states,          # List[Tensor]
        batch_past_key_values,        # List[PipeStaticCache]
        batch_position_ids,           # List[Tensor]
        global_position_embeddings,   # Tuple[Tensor, Tensor]
        seq_lens,                     # List[int]
        **kwargs
    ):
        # === Step 1: 拼接 hidden_states ===
        residual = torch.cat(batch_hidden_states, dim=1) 
        padded_hidden_states = residual.clone()
        B, L, D = padded_hidden_states.shape 
        
        if B != 1:
            raise ValueError("packed_forward with Varlen-Attention currently only supports batch_size=1")

        # === Step 2: Norm + QKV Proj ===
        normed = self.input_layernorm(padded_hidden_states)
        queries = self.original_layer.self_attn.q_proj(normed)
        keys = self.original_layer.self_attn.k_proj(normed)
        values = self.original_layer.self_attn.v_proj(normed)

        # === Step 3: Reshape to multi-head ===
        queries = queries.view(B, L, -1, self.head_dim)
        keys = keys.view(B, L, self.num_key_value_heads, self.head_dim)
        values = values.view(B, L, self.num_key_value_heads, self.head_dim)

        # === Step 4: Apply RoPE ===
        cos_full, sin_full = global_position_embeddings
        queries_rot, keys_rot = apply_rotary_pos_emb(queries, keys, cos_full, sin_full, unsqueeze_dim=2)

        # === Step 5: 准备 Varlen-Attention 输入 ===
        # 1. 准备 Q 和 cu_seqlens_q
        q_varlen = queries_rot.squeeze(0).contiguous()
        
        cu_seqlens_q_list = [0]
        current_q_sum = 0
        for s in seq_lens:
            current_q_sum += s
            cu_seqlens_q_list.append(current_q_sum)
            
        cu_seqlens_q = torch.tensor(cu_seqlens_q_list, dtype=torch.int32, device=q_varlen.device)
        max_seqlen_q = max(seq_lens) if seq_lens else 0

        # 2. 准备 K, V 和 cu_seqlens_k
        k_seq_lens_list = []
        L_prefill = seq_lens[0]
        for i in range(len(seq_lens)):
            k_seq_lens_list.append(L_prefill + i)

        all_k_list = []
        all_v_list = []
        start_pos = 0

        for i in range(len(batch_hidden_states)):
            seq_len = seq_lens[i] 
            end_pos = start_pos + seq_len

            new_k = keys_rot[:, start_pos:end_pos, :, :]
            new_v = values[:, start_pos:end_pos, :, :] # V 未旋转
            
            past_kv = batch_past_key_values[i]
            
            # .update() 获取完整缓存
            pos_ids_i = batch_position_ids[i]
            cache_kwargs = {"cache_position": pos_ids_i}
            full_k, full_v = past_kv.update(new_k, new_v, self.original_layer.self_attn.layer_idx, cache_kwargs=cache_kwargs)

            # 切片出实际有效长度
            sliced_k = full_k[:, :end_pos, :, :]
            sliced_v = full_v[:, :end_pos, :, :]

            all_k_list.append(sliced_k.squeeze(0)) 
            all_v_list.append(sliced_v.squeeze(0))
            
            start_pos = end_pos

        # 拼接所有 K/V
        k_cat = torch.cat(all_k_list, dim=0) 
        v_cat = torch.cat(all_v_list, dim=0) 
        k_varlen = k_cat.contiguous()
        v_varlen = v_cat.contiguous()       

        cu_seqlens_k_list = [0]
        current_k_sum = 0
        for s in k_seq_lens_list: 
            current_k_sum += s
            cu_seqlens_k_list.append(current_k_sum)
            
        cu_seqlens_k = torch.tensor(cu_seqlens_k_list, dtype=torch.int32, device=k_varlen.device)
        max_seqlen_k = max(k_seq_lens_list) if k_seq_lens_list else 0

        # 3. 调用 flash_attn_varlen_func
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
        
        # === Step 6: 调整 attn 输出，O-Proj, MLP ===
        attn_concat = attn_output.reshape(L, D) 
        attn_concat = attn_concat.unsqueeze(0) 
        
        attn_concat = self.original_layer.self_attn.o_proj(attn_concat)
        padded_hidden_states = residual + attn_concat

        # 残差加和在 O-Proj 和 MLP 外面处理，保持和 Baseline 一致
        residual = padded_hidden_states
        normed_attn_out = self.post_attention_layernorm(padded_hidden_states)
        mlp_out = self.original_layer.mlp(normed_attn_out)
        output_hidden_states = residual + mlp_out

        # === Step 7: 拆分输出 ===
        outputs = []
        start = 0
        for seq_len in seq_lens:
            outputs.append(output_hidden_states[:, start:start+seq_len, :])
            start += seq_len

        return outputs