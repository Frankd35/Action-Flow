"""
Qwen2.5-VL Action-Flow Pipeline Implementation
"""
import torch
import torch.nn as nn

from .qwen_layers import QwenPIPEDecodeLayer


class ActionFlowQwenPipeline(nn.Module):
    """
    ActionFlow Pipeline for Qwen2.5-VL models.

    Similar to ActionFlowPipeline for Llama, but adapted for Qwen architecture:
    - Handles Qwen's embedded vision encoder
    - Uses QwenPIPEDecodeLayer wrappers
    """

    def __init__(self, qwen_model: nn.Module, max_token: int = 35):
        """
        Initialize ActionFlow pipeline for Qwen.

        Args:
            qwen_model: Qwen2_5_VLForConditionalGeneration model
            max_token: Maximum number of tokens to generate (default: 35 for HORIZON=1)
        """
        super().__init__()
        self.base_model = qwen_model
        self.config = qwen_model.config
        self.device = qwen_model.device
        self.dtype = qwen_model.dtype
        self.hidden_size = self.config.hidden_size

        # Components from base model
        self.lm_head = qwen_model.lm_head
        self.embed_tokens = qwen_model.model.embed_tokens
        self.norm = qwen_model.model.norm
        self.rotary_emb = qwen_model.model.rotary_emb

        # Wrap layers
        self.layers = nn.ModuleList([
            QwenPIPEDecodeLayer(layer, self.config)
            for layer in qwen_model.model.layers
        ])

        # Pipeline state
        self.max_token = max_token
        self._initialized = False
        self._kv_ring_buffer = None
        self._stage_hidden_states = None
        self._stage_ids = None
        self._global_rope_cos = None
        self._global_rope_sin = None

        # Buffer size
        self.TOTAL_LEN_BUFFER = 256 + 256 + 32

    def init_resources(self, prefill_len: int, max_new_tokens: int):
        """Initialize KV buffers and RoPE embeddings."""
        if self._initialized and self.max_token == max_new_tokens:
            self.prefill_len = prefill_len
            return

        self.max_token = max_new_tokens
        self.decode_steps = max_new_tokens - 1
        self.prefill_len = prefill_len
        self.total_seq_len = prefill_len + self.decode_steps

        # Allocate KV ring buffer
        H_kv = self.config.num_key_value_heads
        D_h = self.config.hidden_size // self.config.num_attention_heads

        self._kv_ring_buffer = [
            torch.empty(
                (2 * self.max_token * self.TOTAL_LEN_BUFFER * H_kv * D_h),
                dtype=self.dtype,
                device=self.device
            ) for _ in range(len(self.layers))
        ]

        # Pipeline stage states
        self._stage_hidden_states = [None for _ in range(self.max_token)]
        self._stage_ids = [torch.zeros(i, device=self.device, dtype=torch.long) for i in range(self.max_token)]

        # Precompute global RoPE for text-only decode
        # Qwen mrope expects position_ids shape (3, batch, seq_len)
        # For text-only decode, all 3 dimensions use same sequential positions
        # Output shape: cos/sin each (3, 1, 4096, D)
        dummy_input = torch.zeros(1, 4096, self.hidden_size, device=self.device, dtype=self.dtype)
        dummy_ids = torch.arange(4096, device=self.device).unsqueeze(0).unsqueeze(0).expand(3, -1, -1)
        rope_cos, rope_sin = self.rotary_emb(dummy_input, dummy_ids)
        # Take dim 0 of mrope (all identical for text) and squeeze batch -> (4096, D)
        self._global_rope_cos = rope_cos[0, 0, :, :].contiguous()  # (4096, D)
        self._global_rope_sin = rope_sin[0, 0, :, :].contiguous()  # (4096, D)

        self._initialized = True

    def _compute_next_token_and_embedding(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Hidden states -> Norm -> Head -> Argmax -> Embed"""
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        next_embedding = self.embed_tokens(next_token_id.unsqueeze(-1))
        return next_embedding

    def pipe_forward(self, new_prefill_inputs_embeds: torch.Tensor):
        """
        Execute pipeline forward pass.

        Args:
            new_prefill_inputs_embeds: (1, prefill_len, D) multimodal embeddings

        Returns:
            output_ids: Token IDs from the oldest pipeline stage
        """
        self.prefill_len = new_prefill_inputs_embeds.shape[1]
        self.total_seq_len = self.prefill_len + self.decode_steps

        # Prepare batch inputs
        batch_hidden_states = [new_prefill_inputs_embeds]

        for stage in range(1, self.max_token):
            prev_hs = self._stage_hidden_states[stage]
            if prev_hs is not None:
                next_emb = self._compute_next_token_and_embedding(prev_hs)
                batch_hidden_states.append(next_emb)
            else:
                rand_emb = torch.randn(1, 1, self.hidden_size, device=self.device, dtype=self.dtype)
                batch_hidden_states.append(rand_emb)

        seq_lens = [self.prefill_len] + [1] * self.decode_steps

        # Prepare varlen metadata
        q_lens = torch.tensor(seq_lens, device=self.device)
        cu_seqlens_q = torch.cat([torch.tensor([0], device=self.device), torch.cumsum(q_lens, 0)]).int()
        max_seqlen_q = self.prefill_len

        B_stages = self.max_token
        k_seq_lens = torch.arange(B_stages, device=self.device) + self.prefill_len
        cu_seqlens_k = torch.cat([torch.tensor([0], device=self.device), torch.cumsum(k_seq_lens, 0)]).int()
        max_seqlen_k = int(k_seq_lens[-1])

        H_kv = self.config.num_key_value_heads
        D_h = self.config.hidden_size // self.config.num_attention_heads
        total_L_kv = cu_seqlens_k[-1].item()

        # Packed forward
        current_hidden_states = batch_hidden_states

        rope_slice = (
            self._global_rope_cos[:self.total_seq_len, :],
            self._global_rope_sin[:self.total_seq_len, :]
        )

        for layer_idx, layer in enumerate(self.layers):
            raw_buffer = self._kv_ring_buffer[layer_idx]
            active_buffer = raw_buffer[: 2 * total_L_kv * H_kv * D_h].view(2, total_L_kv, H_kv, D_h)

            current_hidden_states = layer.packed_forward(
                batch_hidden_states=current_hidden_states,
                kv_ring_buffer=active_buffer,
                global_position_embeddings=rope_slice,
                seq_lens=seq_lens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k
            )

        # Pipeline state update
        self._stage_hidden_states = [None, *current_hidden_states[:-1]]

        # Collect token IDs
        for stage in range(self.max_token):
            hs = current_hidden_states[stage]
            hs = self.norm(hs)
            logits = self.lm_head(hs)
            token_id = torch.argmax(logits[:, -1, :], dim=-1).squeeze(0)

            if stage == 0:
                self._stage_ids[stage] = token_id.unsqueeze(0)
            else:
                self._stage_ids[stage] = torch.cat([self._stage_ids[stage], token_id.unsqueeze(0)])

        # Pop oldest output
        final_output_ids = self._stage_ids[self.max_token - 1]
        self._stage_ids = [None, *self._stage_ids[:-1]]

        return final_output_ids.unsqueeze(0)
