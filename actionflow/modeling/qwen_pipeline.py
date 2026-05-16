"""
Qwen2.5-VL Action-Flow Pipeline Implementation
"""
from typing import Optional

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
        self._stage_request_ids = None
        self._stage_token_steps = None
        self._request_counter = 0
        self._request_prefill_ids = {}
        self._global_rope_cos = None
        self._global_rope_sin = None
        self._global_rope_cache_len = 0

        # Buffer size
        self.TOTAL_LEN_BUFFER = 256 + 256 + 32

    def _ensure_global_rope_cache(self, seq_len: int):
        """
        Build/expand RoPE cache up to `seq_len`.
        Qwen mRoPE returns shape (3, 1, L, D) for dummy ids; for decode we use dim-0.
        """
        if self._global_rope_cache_len >= seq_len:
            return

        dummy_input = torch.zeros(1, seq_len, self.hidden_size, device=self.device, dtype=self.dtype)
        dummy_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).unsqueeze(0).expand(3, -1, -1)
        rope_cos, rope_sin = self.rotary_emb(dummy_input, dummy_ids)
        self._global_rope_cos = rope_cos[0, 0, :, :].contiguous()  # (L, D)
        self._global_rope_sin = rope_sin[0, 0, :, :].contiguous()  # (L, D)
        self._global_rope_cache_len = seq_len

    def _build_qwen_mrope_embeddings(
        self,
        prefill_len: int,
        decode_steps: int,
        position_ids: torch.Tensor = None,
    ):
        """
        Build Qwen mRoPE cos/sin with shape (3, total_seq_len, head_dim).
        If model-provided position_ids are unavailable, fallback to sequential text positions.
        """
        total_seq_len = prefill_len + decode_steps
        if position_ids is not None and position_ids.dim() == 3 and position_ids.shape[0] == 3:
            prefill_pos_3d = position_ids[:, 0, :prefill_len].to(device=self.device, dtype=torch.long)
        elif position_ids is not None and position_ids.dim() == 2:
            prefill_pos_1d = position_ids[0, :prefill_len].to(device=self.device, dtype=torch.long)
            prefill_pos_3d = prefill_pos_1d.unsqueeze(0).expand(3, -1)
        elif position_ids is not None and position_ids.dim() == 1:
            prefill_pos_1d = position_ids[:prefill_len].to(device=self.device, dtype=torch.long)
            prefill_pos_3d = prefill_pos_1d.unsqueeze(0).expand(3, -1)
        else:
            prefill_pos_1d = torch.arange(prefill_len, device=self.device, dtype=torch.long)
            prefill_pos_3d = prefill_pos_1d.unsqueeze(0).expand(3, -1)

        if decode_steps > 0:
            decode_start_3d = prefill_pos_3d[:, -1:] + 1 if prefill_len > 0 else torch.zeros(
                (3, 1), device=self.device, dtype=torch.long
            )
            decode_offsets = torch.arange(decode_steps, device=self.device, dtype=torch.long).unsqueeze(0)
            decode_pos_3d = decode_start_3d + decode_offsets
            rope_pos_3d = torch.cat([prefill_pos_3d, decode_pos_3d], dim=1)
        else:
            rope_pos_3d = prefill_pos_3d

        if rope_pos_3d.shape[1] != total_seq_len:
            raise RuntimeError(
                f"Invalid rope position length: {rope_pos_3d.shape[1]} != {total_seq_len}"
            )

        dummy_input = torch.zeros(
            1, total_seq_len, self.hidden_size, device=self.device, dtype=self.dtype
        )
        rotary_pos_ids = rope_pos_3d.unsqueeze(1)  # (3, 1, L)
        rope_cos, rope_sin = self.rotary_emb(dummy_input, rotary_pos_ids)
        return (
            rope_cos[:, 0, :, :].contiguous(),  # (3, L, D)
            rope_sin[:, 0, :, :].contiguous(),  # (3, L, D)
            rope_pos_3d,
        )

    def _validate_pipeline_invariants(
        self,
        seq_lens,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        current_hidden_states,
    ):
        expected_total_q = sum(seq_lens)
        if int(cu_seqlens_q[-1].item()) != expected_total_q:
            raise RuntimeError(
                f"Invalid cu_seqlens_q tail: {int(cu_seqlens_q[-1].item())} != {expected_total_q}"
            )
        expected_k_lens = [self.prefill_len + i for i in range(self.max_token)]
        expected_total_k = sum(expected_k_lens)
        if int(cu_seqlens_k[-1].item()) != expected_total_k:
            raise RuntimeError(
                f"Invalid cu_seqlens_k tail: {int(cu_seqlens_k[-1].item())} != {expected_total_k}"
            )
        if len(current_hidden_states) != self.max_token:
            raise RuntimeError(
                f"Invalid hidden stage count: {len(current_hidden_states)} != {self.max_token}"
            )
        if current_hidden_states[0].shape[1] != self.prefill_len:
            raise RuntimeError(
                f"Stage-0 hidden length mismatch: {current_hidden_states[0].shape[1]} != {self.prefill_len}"
            )
        for stage in range(1, self.max_token):
            if current_hidden_states[stage].shape[1] != 1:
                raise RuntimeError(
                    f"Decode stage hidden length mismatch at stage {stage}: "
                    f"{current_hidden_states[stage].shape[1]} != 1"
                )

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
        self._stage_request_ids = [None for _ in range(self.max_token)]
        self._stage_token_steps = [None for _ in range(self.max_token)]
        self._request_counter = 0
        self._request_prefill_ids = {}

        # Precompute global RoPE cache for decode positions.
        self._ensure_global_rope_cache(4096)

        self._initialized = True

    def _compute_next_token_and_embedding(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Hidden states -> Norm -> Head -> Argmax -> Embed"""
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        next_embedding = self.embed_tokens(next_token_id.unsqueeze(-1))
        return next_embedding

    def _embed_selected_token(self, token_id: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(token_id.view(1, 1))

    def _apply_repetition_penalty_to_logits(
        self,
        logits_last: torch.Tensor,
        history_ids: torch.Tensor,
        repetition_penalty: float,
    ) -> torch.Tensor:
        if repetition_penalty is None or repetition_penalty == 1.0:
            return logits_last
        if history_ids is None or history_ids.numel() == 0:
            return logits_last
        out = logits_last.clone()
        unique_hist = torch.unique(history_ids.to(device=out.device, dtype=torch.long))
        selected = out.index_select(0, unique_hist)
        selected = torch.where(selected < 0, selected * repetition_penalty, selected / repetition_penalty)
        out.index_copy_(0, unique_hist, selected)
        return out

    def pipe_forward(
        self,
        new_prefill_inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor = None,
        rope_position_mode: str = "sequential",
        prefill_input_ids: torch.Tensor = None,
        logit_mask: Optional[torch.Tensor] = None,
    ):
        """
        Execute pipeline forward pass.

        Args:
            new_prefill_inputs_embeds: (1, prefill_len, D) multimodal embeddings

        Returns:
            output_ids: Token IDs from the oldest pipeline stage
        """
        self.prefill_len = new_prefill_inputs_embeds.shape[1]
        self.total_seq_len = self.prefill_len + self.decode_steps
        if self.total_seq_len > self._global_rope_cache_len:
            self._ensure_global_rope_cache(self.total_seq_len + 256)
        current_request_id = self._request_counter
        self._request_counter += 1
        if prefill_input_ids is not None:
            self._request_prefill_ids[current_request_id] = prefill_input_ids[0].detach().to(device=self.device, dtype=torch.long)

        # Prepare batch inputs
        batch_hidden_states = [new_prefill_inputs_embeds]
        input_stage_request_ids = [current_request_id] + [self._stage_request_ids[s] for s in range(1, self.max_token)]
        input_stage_token_steps = [0] + [self._stage_token_steps[s] for s in range(1, self.max_token)]
        generated_stage_request_ids = [None for _ in range(self.max_token)]
        generated_stage_token_steps = [None for _ in range(self.max_token)]

        for stage in range(1, self.max_token):
            prev_hs = self._stage_hidden_states[stage]
            if prev_hs is not None:
                prev_selected_ids = self._stage_ids[stage]
                if prev_selected_ids is not None and prev_selected_ids.numel() > 0:
                    next_emb = self._embed_selected_token(prev_selected_ids[-1])
                else:
                    next_emb = self._compute_next_token_and_embedding(prev_hs)
                batch_hidden_states.append(next_emb)
            else:
                rand_emb = torch.randn(1, 1, self.hidden_size, device=self.device, dtype=self.dtype)
                batch_hidden_states.append(rand_emb)
            req_id = input_stage_request_ids[stage]
            step = input_stage_token_steps[stage]
            if req_id is not None and step is not None:
                generated_stage_request_ids[stage] = req_id
                generated_stage_token_steps[stage] = int(step) + 1

        generated_stage_request_ids[0] = current_request_id
        generated_stage_token_steps[0] = 1

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

        if rope_position_mode == "model":
            rope_slice = self._build_qwen_mrope_embeddings(
                prefill_len=self.prefill_len,
                decode_steps=self.decode_steps,
                position_ids=position_ids,
            )
            rope_cos, rope_sin, rope_pos_3d = rope_slice
            rope_slice = (rope_cos, rope_sin)
            rope_indices = rope_pos_3d[0]
        else:
            rope_indices = torch.arange(self.total_seq_len, device=self.device, dtype=torch.long)
            rope_slice = (
                self._global_rope_cos.index_select(0, rope_indices),
                self._global_rope_sin.index_select(0, rope_indices),
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
                max_seqlen_k=max_seqlen_k,
            )

        self._validate_pipeline_invariants(
            seq_lens=seq_lens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            current_hidden_states=current_hidden_states,
        )

        # Pipeline state update
        next_stage_hidden_states = [None for _ in range(self.max_token)]
        for stage in range(1, self.max_token):
            next_stage_hidden_states[stage] = current_hidden_states[stage - 1]
        self._stage_hidden_states = next_stage_hidden_states
        next_stage_request_ids = [None for _ in range(self.max_token)]
        next_stage_token_steps = [None for _ in range(self.max_token)]
        for stage in range(1, self.max_token):
            next_stage_request_ids[stage] = generated_stage_request_ids[stage - 1]
            next_stage_token_steps[stage] = generated_stage_token_steps[stage - 1]
        self._stage_request_ids = next_stage_request_ids
        self._stage_token_steps = next_stage_token_steps

        # Collect token IDs
        stage_token_ids_this_round = []
        repetition_penalty = getattr(getattr(self.base_model, "generation_config", None), "repetition_penalty", 1.0)
        for stage in range(self.max_token):
            hs = current_hidden_states[stage]
            hs = self.norm(hs)
            logits = self.lm_head(hs)
            logits_last = logits[0, -1, :]
            req_id = generated_stage_request_ids[stage]
            stage_generated_ids = self._stage_ids[stage]
            prefill_ids = self._request_prefill_ids.get(req_id, None) if req_id is not None else None
            if prefill_ids is not None and stage_generated_ids is not None and stage_generated_ids.numel() > 0:
                history_ids = torch.cat([prefill_ids, stage_generated_ids.to(device=self.device, dtype=torch.long)], dim=0)
            elif prefill_ids is not None:
                history_ids = prefill_ids
            elif stage_generated_ids is not None and stage_generated_ids.numel() > 0:
                history_ids = stage_generated_ids.to(device=self.device, dtype=torch.long)
            else:
                history_ids = None
            logits_last = self._apply_repetition_penalty_to_logits(
                logits_last=logits_last,
                history_ids=history_ids,
                repetition_penalty=float(repetition_penalty) if repetition_penalty is not None else 1.0,
            )
            if logit_mask is not None:
                logits_last = logits_last + logit_mask.to(device=logits_last.device, dtype=logits_last.dtype)
            token_id = torch.argmax(logits_last, dim=-1)
            stage_token_ids_this_round.append(int(token_id.detach().cpu().item()))

            if stage == 0:
                self._stage_ids[stage] = token_id.unsqueeze(0)
            else:
                self._stage_ids[stage] = torch.cat([self._stage_ids[stage], token_id.unsqueeze(0)])

        # Pop oldest output
        final_output_ids = self._stage_ids[self.max_token - 1]
        next_stage_ids = [None for _ in range(self.max_token)]
        for stage in range(1, self.max_token):
            next_stage_ids[stage] = self._stage_ids[stage - 1]
        self._stage_ids = next_stage_ids

        return final_output_ids.unsqueeze(0)
