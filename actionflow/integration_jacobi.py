"""
Optional Jacobi pipeline integration (additive). Does not modify ``integration.py``.

Usage::

    from actionflow.integration_jacobi import enable_actionflow_jacobi
    # max_iter -> AF pipeline depth K
    # max_token -> AF jacobi_tokens J
    enable_actionflow_jacobi(vla_model, max_iter=5, max_token=22, action_chunk=3)

- **Jacobi path only**: one ``pipe_forward`` per ``predict_action`` (full ``max_depth_K`` stages).
- ``max_iter`` -> AF pipeline depth ``K``.
- ``max_token`` -> AF jacobi segment length ``J``.
- ``action_chunk>1`` 时，对最后 ``7*action_chunk`` 个 token 反词表并反归一化，得到 ``(action_chunk, action_dim)``。
"""

from __future__ import annotations

import types
import time

import numpy as np
import torch

from actionflow.modeling.jacobi_pipeline import ActionFlowJacobiPipeline


def enable_actionflow_jacobi(
    vla_model,
    max_iter: int | None = None,
    max_token: int | None = None,
    enable_timing: bool = False,
    action_chunk: int = 1,
):
    """
    Patch ``predict_action`` to use ``ActionFlowJacobiPipeline``.

    Args:
        max_iter: AF packed depth K; if ``None``, uses ``7 * action_chunk + 1``.
        max_token: AF jacobi segment length J; if ``None``, uses ``7 * action_chunk + 1``.
        action_chunk: 每调一次 ``predict_action`` 输出的步数（LIBERO 常为 3）；与 CEED ``max_new_tokens=7*chunk+1`` 对齐。
    """

    if max_token is None:
        max_token = 7 * action_chunk + 1
    print(
        f"[ActionFlow-Jacobi] Initializing max_depth_K={max_iter}, jacobi_tokens={max_token}, "
        f"action_chunk={action_chunk}..."
    )

    print("[ActionFlow-Jacobi] USING packed Jacobi pipeline (full depth K)")
    pipeline_engine = ActionFlowJacobiPipeline(
        vla_model.language_model, max_depth_K=max_iter, jacobi_tokens=max_token
    )

    vla_model.actionflow_engine = pipeline_engine
    vla_model._enable_timing = enable_timing
    vla_model._jacobi_tokens = max_token
    vla_model._jacobi_max_depth_K = max_iter
    vla_model._jacobi_action_chunk = action_chunk

    if enable_timing:
        vla_model._timing_stats = {
            "vision_backbone": [],
            "projector": [],
            "text_embed": [],
            "llm_actionflow": [],
            "total": [],
        }
    else:
        vla_model._timing_stats = None

    def step_jacobi_pipeline(self, multimodal_embeddings: torch.Tensor):
        """One full-depth ``pipe_forward``; returns ``(1, seq_len)`` token ids from oldest stage."""
        eng = self.actionflow_engine
        pl = multimodal_embeddings.shape[1]
        if (not eng._initialized) or (eng.prefill_len != pl):
            eng.init_resources(prefill_len=pl, max_depth_K=self._jacobi_max_depth_K, jacobi_tokens=self._jacobi_tokens)
        return eng.pipe_forward(multimodal_embeddings)

    def reset_jacobi_pipeline(self) -> None:
        eng = self.actionflow_engine
        eng.reset_state()

    def predict_action_accelerated(self, input_ids=None, pixel_values=None, unnorm_key=None, **kwargs):
        if self._enable_timing:
            torch.cuda.synchronize()
        total_start = time.perf_counter()

        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        multimodal_embeddings = self._get_multimodal_embeddings(input_ids, pixel_values)
        prefill_len = multimodal_embeddings.shape[1]

        eng = self.actionflow_engine
        eng.init_resources(prefill_len=prefill_len, max_depth_K=self._jacobi_max_depth_K, jacobi_tokens=self._jacobi_tokens)

        if self._enable_timing:
            torch.cuda.synchronize()
        llm_start = time.perf_counter()

        output_ids = eng.pipe_forward(multimodal_embeddings)

        if self._enable_timing:
            torch.cuda.synchronize()
            llm_elapsed = (time.perf_counter() - llm_start) * 1000
            self._timing_stats["llm_actionflow"].append(llm_elapsed)

        action_dim = self.get_action_dim(unnorm_key)
        chunk = getattr(self, "_jacobi_action_chunk", 1)
        n_action_tokens = action_dim * chunk
        predicted_ids = output_ids[0, -n_action_tokens:].cpu().numpy()

        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high = np.array(action_norm_stats["q99"])
        action_low = np.array(action_norm_stats["q01"])

        def _decode_one_row(row_ids: np.ndarray) -> np.ndarray:
            discretized_actions = self.vocab_size - row_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
            normalized_actions = self.bin_centers[discretized_actions]
            return np.where(
                mask,
                0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
                normalized_actions,
            )

        if chunk > 1:
            predicted_ids = predicted_ids.reshape(chunk, action_dim)
            actions = np.stack([_decode_one_row(predicted_ids[i]) for i in range(chunk)], axis=0)
        else:
            actions = _decode_one_row(predicted_ids)

        if self._enable_timing:
            torch.cuda.synchronize()
            total_elapsed = (time.perf_counter() - total_start) * 1000
            self._timing_stats["total"].append(total_elapsed)

        return actions

    def _get_multimodal_embeddings(self, input_ids, pixel_values):
        if self._enable_timing:
            torch.cuda.synchronize()
        vision_start = time.perf_counter()
        patch_features = self.vision_backbone(pixel_values)
        if self._enable_timing:
            torch.cuda.synchronize()
            self._timing_stats["vision_backbone"].append((time.perf_counter() - vision_start) * 1000)

        if self._enable_timing:
            torch.cuda.synchronize()
        proj_start = time.perf_counter()
        projected_patches = self.projector(patch_features)
        if self._enable_timing:
            torch.cuda.synchronize()
            self._timing_stats["projector"].append((time.perf_counter() - proj_start) * 1000)

        if self._enable_timing:
            torch.cuda.synchronize()
        text_start = time.perf_counter()
        input_embeddings = self.get_input_embeddings()(input_ids)
        embeddings = torch.cat([input_embeddings[:, :1, :], projected_patches, input_embeddings[:, 1:, :]], dim=1)
        if self._enable_timing:
            torch.cuda.synchronize()
            self._timing_stats["text_embed"].append((time.perf_counter() - text_start) * 1000)

        return embeddings

    vla_model._get_multimodal_embeddings = types.MethodType(_get_multimodal_embeddings, vla_model)
    vla_model.predict_action = types.MethodType(predict_action_accelerated, vla_model)
    vla_model.step_jacobi_pipeline = types.MethodType(step_jacobi_pipeline, vla_model)
    vla_model.reset_jacobi_pipeline = types.MethodType(reset_jacobi_pipeline, vla_model)

    print("[ActionFlow-Jacobi] Model patched. `predict_action` uses the selected pipeline engine.")
    return vla_model
