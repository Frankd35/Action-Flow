import torch
import torch.nn as nn
from .layers import LlamaPIPEDecodeLayer

class ActionFlowPipeline(nn.Module):
    """
    Manages the ActionFlow execution pipeline. 
    It wraps the original LlamaModel and orchestrates the packed forward pass.
    """
    def __init__(self, llama_model: nn.Module, max_token: int = 8):
        super().__init__()
        self.base_model = llama_model
        self.config = llama_model.config
        self.device = llama_model.device
        self.dtype = llama_model.dtype
        self.hidden_size = self.config.hidden_size
        
        # Components extracted from base model
        self.lm_head = self.base_model.lm_head if hasattr(self.base_model, 'lm_head') else self.base_model.get_output_embeddings()
        self.embed_tokens = self.base_model.model.get_input_embeddings()
        self.norm = self.base_model.model.norm
        self.rotary_emb = self.base_model.model.rotary_emb
        
        # Replace layers with ActionFlow wrappers
        # Note: This creates a parallel list of wrappers, not replacing inplace in base_model
        self.layers = nn.ModuleList([
            LlamaPIPEDecodeLayer(layer, self.config) 
            for layer in self.base_model.model.layers
        ])
        
        # Pipeline State
        self.max_token = max_token
        self._initialized = False
        self._kv_ring_buffer = None
        self._stage_hidden_states = None
        self._stage_ids = None
        self._global_position_embeddings = None
        
        # Constants
        self.TOTAL_LEN_BUFFER = 256 + 256 + 32 # Safety buffer size

    def init_resources(self, prefill_len: int, max_new_tokens: int):
        """
        Lazy initialization of buffers and RoPE embeddings based on runtime input shape.
        """
        if self._initialized and self.max_token == max_new_tokens:
            # Check if prefill len changed significantly enough to require re-allocation?
            # For simplicity, we assume re-init if explicit call or params change.
            # In dynamic scenarios, you might want to only re-init if size grows.
            self.prefill_len = prefill_len # Update prefill len
            return

        self.max_token = max_new_tokens
        self.decode_steps = max_new_tokens - 1
        self.prefill_len = prefill_len
        self.total_seq_len = prefill_len + self.decode_steps
        
        # 1. Allocate Unified KV Ring Buffer
        H_kv = self.config.num_key_value_heads
        D_h = self.config.hidden_size // self.config.num_attention_heads
        
        # Allocating for all layers
        # Shape: [num_layers, 2, total_capacity, H_kv, D_h] flattened/managed per layer
        self._kv_ring_buffer = [
            torch.empty(
                (2 * self.max_token * self.TOTAL_LEN_BUFFER * H_kv * D_h),
                dtype=self.dtype,
                device=self.device
            ) for _ in range(len(self.layers))
        ]
        
        # 2. Pipeline Stage States
        self._stage_hidden_states = [None for _ in range(self.max_token)]
        self._stage_ids = [torch.zeros(i, device=self.device, dtype=torch.long) for i in range(self.max_token)]
        
        # 3. Precompute Global RoPE
        # Ensure we cover the max possible sequence length
        dummy_input = torch.zeros(1, 4096, self.hidden_size, device=self.device, dtype=self.dtype)
        dummy_ids = torch.arange(4096, device=self.device).unsqueeze(0)
        self._global_position_embeddings = self.rotary_emb(dummy_input, dummy_ids)
        
        self._initialized = True

    def _compute_next_token_and_embedding(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Helper: HS -> Norm -> Head -> Argmax -> Embed
        align to transformer implementation:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
        """
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        next_embedding = self.embed_tokens(next_token_id.unsqueeze(-1))
        return next_embedding

    def pipe_forward(self, new_prefill_inputs_embeds: torch.Tensor):
        """
        Main execution entry point.
        Args:
            new_prefill_inputs_embeds: (1, prefill_len, D)
        Returns:
            output_ids: The result token IDs from the oldest request in the pipeline (Stage K-1)
        """
        self.prefill_len = new_prefill_inputs_embeds.shape[1]
        self.total_seq_len = self.prefill_len + self.decode_steps
        
        # === Step 1: Prepare Batch Inputs ===
        # Stage 0: New Prefill
        batch_hidden_states = [new_prefill_inputs_embeds]
        
        # Stage 1..K-1: Decode steps from historical requests
        # We need to compute the embedding for the NEXT token based on previous stage output
        for stage in range(1, self.max_token):
            prev_hs = self._stage_hidden_states[stage]
            if prev_hs is not None:
                next_emb = self._compute_next_token_and_embedding(prev_hs)
                batch_hidden_states.append(next_emb)
            else:
                # Cold start / bubble: use random or zero embedding
                rand_emb = torch.randn(1, 1, self.hidden_size, device=self.device, dtype=self.dtype)
                batch_hidden_states.append(rand_emb)
        
        seq_lens = [self.prefill_len] + [1] * self.decode_steps
        
        # === Step 2: Prepare Varlen Metadata (cu_seqlens) ===
        # Q: [0, prefill_len, prefill_len+1, ..., total]
        q_lens = torch.tensor(seq_lens, device=self.device)
        _cu_seqlens_q = torch.cat([torch.tensor([0], device=self.device), torch.cumsum(q_lens, 0)]).int()
        _max_seqlen_q = self.prefill_len
        
        # K: [prefill_len, prefill_len+1, ...] -> accumulative logic for Ring Buffer
        # Note: This logic depends on how your Ring Buffer indexes are laid out.
        # Based on your snippet: k_seq_lens = torch.arange(B_stages) + self.prefill_len
        B_stages = self.max_token
        k_seq_lens = torch.arange(B_stages, device=self.device) + self.prefill_len
        _cu_seqlens_k = torch.cat([torch.tensor([0], device=self.device), torch.cumsum(k_seq_lens, 0)]).int()
        _max_seqlen_k = int(k_seq_lens[-1])
        
        # Calc Total KV Elements for reshaping buffer
        H_kv = self.config.num_key_value_heads
        D_h = self.config.hidden_size // self.config.num_attention_heads
        total_L_kv = _cu_seqlens_k[-1].item()
        
        # === Step 3: Packed Forward Loop ===
        current_hidden_states = batch_hidden_states
        
        # RoPE Slice for this iteration
        # We need enough RoPE for the longest sequence
        rope_slice = (
            self._global_position_embeddings[0][:, :self.total_seq_len, :],
            self._global_position_embeddings[1][:, :self.total_seq_len, :]
        )

        for layer_idx, layer in enumerate(self.layers):
            # View the buffer for this layer
            raw_buffer = self._kv_ring_buffer[layer_idx]
            # Careful slicing to match the shape expected by the kernel
            active_buffer = raw_buffer[: 2 * total_L_kv * H_kv * D_h].view(2, total_L_kv, H_kv, D_h)
            
            current_hidden_states = layer.packed_forward(
                batch_hidden_states=current_hidden_states,
                kv_ring_buffer=active_buffer,
                global_position_embeddings=rope_slice,
                seq_lens=seq_lens,
                cu_seqlens_q=_cu_seqlens_q,
                cu_seqlens_k=_cu_seqlens_k,
                max_seqlen_q=_max_seqlen_q,
                max_seqlen_k=_max_seqlen_k
            )

        # === Step 4: Pipeline State Update ===
        # Shift hidden states: Stage i -> Stage i+1
        # The output of Stage 0 becomes the input state for Stage 1 next time
        self._stage_hidden_states = [None, *current_hidden_states[:-1]]
        
        # Collect Token IDs for the final output
        # Update stage_ids accumulators
        for stage in range(self.max_token):
            hs = current_hidden_states[stage]
            # Decode token
            hs = self.norm(hs)
            logits = self.lm_head(hs)
            token_id = torch.argmax(logits[:, -1, :], dim=-1) # (1,)
            token_id = token_id.squeeze(0)  # scalar tensor
            
            if stage == 0:
                self._stage_ids[stage] = token_id.unsqueeze(0)
            else:
                self._stage_ids[stage] = torch.cat([self._stage_ids[stage], token_id.unsqueeze(0)])
        
        # Pop the oldest request (Stage K-1) as the final output
        final_output_ids = self._stage_ids[self.max_token - 1]
        
        # Shift stage_ids list
        self._stage_ids = [None, *self._stage_ids[:-1]]
        
        return final_output_ids.unsqueeze(0) # (1, seq_len)