"""
Jacobi pipeline for ActionFlow: mirrors ``pipeline.py`` / ``ActionFlowPipeline`` structure.

Each ``pipe_forward`` runs one packed forward with a fixed number of stages:
- ``max_depth_K``: Jacobi iteration depth K
- ``jacobi_tokens``: Jacobi window length J
"""

from __future__ import annotations

import torch
import torch.nn as nn

from actionflow.kernels.jacobi_ops import build_jacobi_cu_seqlens, jacobi_total_kv_elements
from actionflow.modeling.jacobi_layers import LlamaPIPEJacobiDecodeLayer


class ActionFlowJacobiPipeline(nn.Module):
    """
    Packed Jacobi execution: ``pipe_forward`` matches ``ActionFlowPipeline`` naming; decode segments
    are ``jacobi_tokens`` wide. Always uses ``max_depth_K`` stages (full shape).
    """

    def __init__(self, llama_model: nn.Module, max_depth_K: int = 8, jacobi_tokens: int = 22):
        super().__init__()
        self.base_model = llama_model
        self.config = llama_model.config
        _p = next(llama_model.parameters())
        self.device = getattr(llama_model, "device", _p.device)
        self.dtype = getattr(llama_model, "dtype", _p.dtype)
        self.hidden_size = self.config.hidden_size

        self.lm_head = self.base_model.lm_head if hasattr(self.base_model, "lm_head") else self.base_model.get_output_embeddings()
        self.embed_tokens = self.base_model.model.get_input_embeddings()
        self.norm = self.base_model.model.norm
        self.rotary_emb = self.base_model.model.rotary_emb

        self.layers = nn.ModuleList([LlamaPIPEJacobiDecodeLayer(layer, self.config) for layer in self.base_model.model.layers])

        self.max_depth_K = max_depth_K
        self.jacobi_tokens = jacobi_tokens

        self._initialized = False
        self._kv_ring_buffer = None
        self._stage_hidden_states = None
        self._stage_points = None
        self._global_position_embeddings = None
        self.prefill_len = 0

    def init_resources(self, prefill_len: int, max_depth_K: int | None = None, jacobi_tokens: int | None = None):
        if jacobi_tokens is not None:
            self.jacobi_tokens = jacobi_tokens
        if max_depth_K is not None:
            self.max_depth_K = max_depth_K

        self.prefill_len = prefill_len

        H_kv = self.config.num_key_value_heads
        D_h = self.config.hidden_size // self.config.num_attention_heads

        total_L_kv = jacobi_total_kv_elements(self.max_depth_K, prefill_len, self.jacobi_tokens)
        self._kv_ring_buffer = [
            torch.empty((2, total_L_kv, H_kv, D_h), dtype=self.dtype, device=self.device) for _ in range(len(self.layers))
        ]

        self._stage_hidden_states = [None for _ in range(self.max_depth_K)]
        self._stage_points = [None for _ in range(self.max_depth_K)]

        max_pos = max(prefill_len + self.jacobi_tokens, 4096)
        dummy_input = torch.zeros(1, max_pos, self.hidden_size, device=self.device, dtype=self.dtype)
        dummy_ids = torch.arange(max_pos, device=self.device).unsqueeze(0)
        self._global_position_embeddings = self.rotary_emb(dummy_input, dummy_ids)

        self._initialized = True

    def reset_state(self) -> None:
        """Clear in-flight pipeline (new episode). Does not reallocate buffers."""
        self._stage_hidden_states = [None for _ in range(self.max_depth_K)]
        self._stage_points = [None for _ in range(self.max_depth_K)]

    def _compute_next_token_and_embedding(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        next_embedding = self.embed_tokens(next_token_id.unsqueeze(-1))
        return next_embedding

    def _make_jacobi_chunk(self, stage: int) -> torch.Tensor:
        prev_hs = self._stage_hidden_states[stage]
        if prev_hs is not None:
            seed = prev_hs[:, -1:, :]
            return self._compute_next_token_and_embedding(seed).expand(1, self.jacobi_tokens, self.hidden_size)
        return torch.randn(1, self.jacobi_tokens, self.hidden_size, device=self.device, dtype=self.dtype)

    def _shift_stage_hidden_states(self, current_hidden_states: list[torch.Tensor], num_stages: int) -> None:
        new_hs: list[torch.Tensor | None] = [None] * self.max_depth_K
        if self.max_depth_K <= 1:
            self._stage_hidden_states = new_hs
            return
        if num_stages == 1:
            new_hs[1] = current_hidden_states[0]
        else:
            for s in range(1, num_stages):
                new_hs[s] = current_hidden_states[s - 1]
        for s in range(num_stages, self.max_depth_K):
            new_hs[s] = None
        self._stage_hidden_states = new_hs

    def _shift_stage_points(self, current_points: list[torch.Tensor], num_stages: int) -> None:
        new_points: list[torch.Tensor | None] = [None] * self.max_depth_K
        if self.max_depth_K <= 1:
            self._stage_points = new_points
            return
        if num_stages == 1:
            new_points[1] = current_points[0]
        else:
            for s in range(1, num_stages):
                new_points[s] = current_points[s - 1]
        for s in range(num_stages, self.max_depth_K):
            new_points[s] = None
        self._stage_points = new_points

    def pipe_forward(self, new_prefill_inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        One packed forward with exactly ``max_depth_K`` stages (Prefill + Jacobi segments).

        Returns:
            ``(1, jacobi_tokens)`` token ids from the oldest stage (index ``max_depth_K - 1``).
        """
        if not self._initialized:
            raise RuntimeError("Call init_resources(...) before pipe_forward.")

        self.prefill_len = new_prefill_inputs_embeds.shape[1]
        num_stages = self.max_depth_K
        J = self.jacobi_tokens

        batch_hidden_states = [new_prefill_inputs_embeds]
        for stage in range(1, num_stages):
            batch_hidden_states.append(self._make_jacobi_chunk(stage))

        seq_lens = [self.prefill_len] + [J] * (num_stages - 1)
        cu_q, cu_k, max_sq, max_sk = build_jacobi_cu_seqlens(num_stages, self.prefill_len, J, self.device)

        total_L_q = sum(seq_lens)
        rope_len = max(self.prefill_len + J, total_L_q)
        rope_slice = (
            self._global_position_embeddings[0][:, :rope_len, :],
            self._global_position_embeddings[1][:, :rope_len, :],
        )

        needed_kv = jacobi_total_kv_elements(num_stages, self.prefill_len, J)

        current_hidden_states = batch_hidden_states
        for layer_idx, layer in enumerate(self.layers):
            raw_buffer = self._kv_ring_buffer[layer_idx]
            active_buffer = raw_buffer[:, :needed_kv, :, :].contiguous()

            with torch.cuda.nvtx.range("JacobiPipeline.packed_forward"):
                current_hidden_states = layer.packed_forward(
                    batch_hidden_states=current_hidden_states,
                    kv_ring_buffer=active_buffer,
                    global_position_embeddings=rope_slice,
                    seq_lens=seq_lens,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_k,
                    max_seqlen_q=max_sq,
                    max_seqlen_k=max_sk,
                    prefill_len=self.prefill_len,
                    jacobi_tokens=J,
                    num_stages=num_stages,
                )

            raw_buffer[:, :needed_kv, :, :].copy_(active_buffer)

        self._shift_stage_hidden_states(current_hidden_states, num_stages)

        stage_next_points: list[torch.Tensor] = []
        for stage in range(num_stages):
            hs = self.norm(current_hidden_states[stage])
            logits = self.lm_head(hs).float()
            # 与 CEED jacobi_forward 对齐：softmax(logits/0.01) 后再 argmax
            all_shift_one_token = torch.argmax(torch.nn.functional.softmax(logits / 0.01, dim=-1), dim=-1)

            prev_point = self._stage_points[stage]
            if prev_point is not None and prev_point.shape[1] >= J:
                # 与 CEED 对齐：next_point = [current_point首token] + predicted[-J:-1]
                next_tokens = all_shift_one_token[:, -J:-1]
                next_point = torch.cat((prev_point[:, :1], next_tokens), dim=-1)
            else:
                # 冷启动阶段没有 current_point 时，退化为直接取长度 J 的预测窗口
                next_point = all_shift_one_token[:, -J:]
            stage_next_points.append(next_point)

        oldest = num_stages - 1
        final_output_ids = stage_next_points[oldest]
        self._shift_stage_points(stage_next_points, num_stages)
        return final_output_ids
