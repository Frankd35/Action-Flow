import torch
import types
import time
import numpy as np
from .modeling.pipeline import ActionFlowPipeline, ActionFlowBaselinePipeline, ActionFlowVarlenBaselinePipeline


def enable_actionflow(vla_model, max_new_tokens=7, pipeline_type="optimized", enable_timing=False):
    """
    Enables ActionFlow acceleration on an existing OpenVLA model instance.
    This performs in-place monkey-patching of the model's prediction methods.

    Args:
        vla_model: The Prismatic/OpenVLA model instance.
        max_new_tokens: The depth of the pipeline (K).
        pipeline_type (str):
            - "optimized": Triton in-place shift + flatten kv buffer
            - "varlen": FlashAttn Varlen + PipeStaticCache + torch.cat
            - "eager": Eager Attention with loop
        enable_timing: Whether to enable torch.cuda.synchronize() and timing collection.
    """
    print(f"[ActionFlow] Initializing pipeline with depth K={max_new_tokens}...")

    if pipeline_type == "eager":
        print("[ActionFlow] USING BASELINE: Eager Attention (Slowest)")
        pipeline_engine = ActionFlowBaselinePipeline(vla_model.language_model, max_token=max_new_tokens)
    elif pipeline_type == "varlen":
        print("[ActionFlow] USING BASELINE: Varlen Attention + Concat (Medium)")
        pipeline_engine = ActionFlowVarlenBaselinePipeline(vla_model.language_model, max_token=max_new_tokens)
    elif pipeline_type == "optimized":
        print("[ActionFlow] USING OPTIMIZED: Triton In-place + FA2 (Fastest)")
        pipeline_engine = ActionFlowPipeline(vla_model.language_model, max_token=max_new_tokens)
    else:
        raise ValueError(f"Unknown pipeline_type: {pipeline_type}")

    # Attach engine to model to prevent garbage collection
    vla_model.actionflow_engine = pipeline_engine
    vla_model._enable_timing = enable_timing

    # Initialize timing stats
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

    # 2. Define the new accelerated predict method
    def predict_action_accelerated(self, input_ids=None, pixel_values=None, unnorm_key=None, **kwargs):
        """
        Accelerated action prediction using ActionFlow.
        """
        if self._enable_timing:
            torch.cuda.synchronize()
        total_start = time.perf_counter()

        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        # A. Vision + Text Embedding (Standard VLA frontend)
        # We manually run the frontend to get the fused embeddings
        multimodal_embeddings = self._get_multimodal_embeddings(input_ids, pixel_values)

        # B. Initialize Resources (Lazy init based on input shape)
        # We check the shape of the incoming prefill
        prefill_len = multimodal_embeddings.shape[1]
        self.actionflow_engine.init_resources(prefill_len=prefill_len, max_new_tokens=max_new_tokens)

        # C. Run Pipeline
        # Note: This executes one step of the macro-pipeline.
        # It takes the current frame's embeddings and returns the finished action
        # from (K-1) frames ago.
        if self._enable_timing:
            torch.cuda.synchronize()
        llm_start = time.perf_counter()
        output_ids = self.actionflow_engine.pipe_forward(multimodal_embeddings)
        if self._enable_timing:
            torch.cuda.synchronize()
            llm_elapsed = (time.perf_counter() - llm_start) * 1000
            self._timing_stats["llm_actionflow"].append(llm_elapsed)

        # D. Decode Action (Standard VLA backend)
        # Convert IDs back to continuous actions
        action_dim = self.get_action_dim(unnorm_key)

        # Slice the last `action_dim` tokens to ensure we get the action part
        # Logic matches OpenVLA: vocab_size - token_id
        predicted_ids = output_ids[0, -action_dim:].cpu().numpy()

        discretized_actions = self.vocab_size - predicted_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]

        # Un-normalize
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high = np.array(action_norm_stats["q99"])
        action_low = np.array(action_norm_stats["q01"])

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        if self._enable_timing:
            torch.cuda.synchronize()
            total_elapsed = (time.perf_counter() - total_start) * 1000
            self._timing_stats["total"].append(total_elapsed)

        return actions

    # 3. Helper for Multimodal Embeddings
    def _get_multimodal_embeddings(self, input_ids, pixel_values):
        if self._enable_timing:
            torch.cuda.synchronize()
        vision_start = time.perf_counter()
        patch_features = self.vision_backbone(pixel_values)
        if self._enable_timing:
            torch.cuda.synchronize()
            vision_elapsed = (time.perf_counter() - vision_start) * 1000
            self._timing_stats["vision_backbone"].append(vision_elapsed)

        # Projector
        if self._enable_timing:
            torch.cuda.synchronize()
        proj_start = time.perf_counter()
        projected_patches = self.projector(patch_features)
        if self._enable_timing:
            torch.cuda.synchronize()
            proj_elapsed = (time.perf_counter() - proj_start) * 1000
            self._timing_stats["projector"].append(proj_elapsed)

        # Text Embeddings
        if self._enable_timing:
            torch.cuda.synchronize()
        text_start = time.perf_counter()
        input_embeddings = self.get_input_embeddings()(input_ids)

        # Concatenate: <BOS> + Image + Instruction
        # Assuming standard OpenVLA formatting
        embeddings = torch.cat([input_embeddings[:, :1, :], projected_patches, input_embeddings[:, 1:, :]], dim=1)
        if self._enable_timing:
            torch.cuda.synchronize()
            text_elapsed = (time.perf_counter() - text_start) * 1000
            self._timing_stats["text_embed"].append(text_elapsed)

        return embeddings

    # 4. Apply Patches
    import numpy as np  # Need numpy inside for the closure

    # Bind methods
    vla_model._get_multimodal_embeddings = types.MethodType(_get_multimodal_embeddings, vla_model)
    vla_model.predict_action = types.MethodType(predict_action_accelerated, vla_model)

    print("[ActionFlow] Model successfully patched! `predict_action` is now accelerated.")
    return vla_model


def print_timing_stats(vla_model):
    """Print timing statistics for ActionFlow model."""
    if not hasattr(vla_model, "_timing_stats") or vla_model._timing_stats is None:
        print("[ActionFlow] No timing stats available (timing was disabled)")
        return

    stats = vla_model._timing_stats
    n_runs = len(stats.get("total", []))

    if n_runs == 0:
        print("[ActionFlow] No timing data collected yet")
        return

    print("\n" + "=" * 60)
    print(f"📊 ActionFlow Timing Stats (averaged over {n_runs} runs)")
    print("=" * 60)

    def avg_ms(key):
        vals = stats.get(key, [])
        return np.mean(vals) if vals else 0.0

    def std_ms(key):
        vals = stats.get(key, [])
        return np.std(vals) if vals else 0.0

    vision = avg_ms("vision_backbone")
    proj = avg_ms("projector")
    text = avg_ms("text_embed")
    llm = avg_ms("llm_actionflow")
    total = avg_ms("total")

    print(f"  Vision backbone:  {vision:7.1f} ms")
    print(f"  Projector:        {proj:7.1f} ms")
    print(f"  Text embed:       {text:7.1f} ms")
    print(f"  LLM (ActionFlow): {llm:7.1f} ms")
    print(f"  ─────────────────────────────────")
    print(f"  Total:            {total:7.1f} ms")
    print(f"  FPS:              {1000.0 / total:7.2f}")
    print("=" * 60)


def reset_timing_stats(vla_model):
    """Reset timing statistics."""
    if hasattr(vla_model, "_timing_stats") and vla_model._timing_stats is not None:
        for key in vla_model._timing_stats:
            vla_model._timing_stats[key] = []
