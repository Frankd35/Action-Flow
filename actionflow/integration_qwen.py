"""
ActionFlow Integration for VLA-0 (Qwen2.5-VL based models)

Key insight: Qwen2.5-VL embeds vision tokens INTO text token positions
(using image_token_id as placeholder), unlike OpenVLA's concatenation approach.

For ActionFlow:
1. Pre-compute multimodal embeddings (vision + text merged)
2. Pass embeddings to pipeline
3. Pipeline outputs token IDs
4. Decode token IDs to text, parse numbers
"""
import torch
import types
import numpy as np
from typing import Optional

from .modeling.qwen_pipeline import ActionFlowQwenPipeline


def enable_actionflow_qwen(
    vla_model,
    max_new_tokens: int = 35,
    enable_timing: bool = False,
    rope_position_mode: str = "model",
):
    """
    Enable ActionFlow acceleration on a VLA-0 (Qwen2.5-VL) model.

    Args:
        vla_model: Qwen2_5_VLForConditionalGeneration model
        max_new_tokens: Max tokens to generate (35 for HORIZON=1)
        enable_timing: Enable timing collection

    Returns:
        Patched model with ActionFlow acceleration
    """
    print(f"[ActionFlow-Qwen] Initializing pipeline with depth K={max_new_tokens}...")

    # Create pipeline
    pipeline_engine = ActionFlowQwenPipeline(vla_model, max_token=max_new_tokens)

    # Attach to model
    vla_model.actionflow_engine = pipeline_engine
    vla_model._enable_timing = enable_timing
    vla_model._rope_position_mode = rope_position_mode
    vla_model._af_max_new_tokens = max_new_tokens

    if enable_timing:
        vla_model._timing_stats = {
            "embed": [],
            "llm_actionflow": [],
            "total": [],
        }
    else:
        vla_model._timing_stats = None

    def get_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute multimodal embeddings for Qwen.

        Qwen embeds vision tokens into specific positions in text embeddings
        (marked by image_token_id). This replicates the logic from
        Qwen2_5_VLForConditionalGeneration.forward().
        """
        # Text embeddings
        inputs_embeds = self.model.embed_tokens(input_ids)

        # Vision embeddings (if present)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)

            # Find image token positions and replace with vision embeddings
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]

            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image tokens ({n_image_tokens}) != image features ({n_image_features})"
                )

            # Replace image token embeddings with vision embeddings
            image_mask = input_ids == self.config.image_token_id
            image_mask_unsqueezed = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueezed, image_embeds)

        return inputs_embeds

    def patched_generate(self, **kwargs):
        """
        ActionFlow-accelerated generate (replaces model.generate).

        Extracts inputs from HF generate kwargs, builds logit_mask from
        logits_processor, and runs the ActionFlow pipeline.

        Returns:
            torch.Tensor: [input_ids | generated_ids] matching HF generate convention.
        """
        input_ids = kwargs["input_ids"]
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")
        attention_mask = kwargs.get("attention_mask")
        logits_processor = kwargs.get("logits_processor")
        position_ids = kwargs.get("position_ids")
        rope_mode = kwargs.get("rope_position_mode", self._rope_position_mode)

        # Build logit_mask from logits_processor
        logit_mask = None
        if logits_processor is not None:
            vocab_size = self.config.vocab_size
            dummy_ids = torch.zeros(1, 1, dtype=torch.long, device=input_ids.device)
            dummy_scores = torch.zeros(1, vocab_size, device=input_ids.device)
            for p in logits_processor:
                dummy_scores = p(dummy_ids, dummy_scores)
            logit_mask = dummy_scores[0, :]  # (vocab_size,)

        # Compute multimodal embeddings
        inputs_embeds = self.get_multimodal_embeddings(input_ids, pixel_values, image_grid_thw)
        prefill_len = inputs_embeds.shape[1]

        # Initialize pipeline
        self.actionflow_engine.init_resources(prefill_len, max_new_tokens=self._af_max_new_tokens)

        # Position IDs
        if position_ids is None and rope_mode == "model" and hasattr(self, "get_rope_index"):
            position_ids, _ = self.get_rope_index(input_ids, image_grid_thw, attention_mask)
        if position_ids is None:
            position_ids = torch.arange(prefill_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)

        # Run pipeline forward
        output_ids = self.actionflow_engine.pipe_forward(
            inputs_embeds,
            position_ids=position_ids,
            rope_position_mode=rope_mode,
            prefill_input_ids=input_ids,
            logit_mask=logit_mask,
        )

        # Return [input_ids | generated] matching HF generate convention
        return torch.cat([input_ids, output_ids.to(device=input_ids.device, dtype=input_ids.dtype)], dim=1)

    # Bind methods
    vla_model.get_multimodal_embeddings = types.MethodType(get_multimodal_embeddings, vla_model)

    # Save original and replace model.generate
    vla_model._original_generate = vla_model.generate
    vla_model.generate = types.MethodType(patched_generate, vla_model)

    print(f"[ActionFlow-Qwen] `model.generate` patched with ActionFlow pipeline (K={max_new_tokens})")
    return vla_model


def parse_action_text(action_text: str, num_bins: int = 1000, expected_len: int = 7) -> np.ndarray:
    """
    Parse action text to normalized numpy array.

    Args:
        action_text: Space-separated numbers (e.g., "500 503 500 421...")
        num_bins: Discretization bins (default 1000)
        expected_len: Expected number of values (default 7)

    Returns:
        Normalized action array in [0, 1]
    """
    parts = action_text.strip().split()
    numbers = []
    for x in parts:
        try:
            numbers.append(int(x))
        except ValueError:
            continue

    if len(numbers) >= expected_len:
        return np.array(numbers[:expected_len]) / num_bins

    # Pad with zeros if needed
    result = np.zeros(expected_len)
    if len(numbers) > 0:
        result[:len(numbers)] = np.array(numbers) / num_bins
    return result


def print_timing_stats(vla_model):
    """Print timing statistics."""
    if not hasattr(vla_model, "_timing_stats") or vla_model._timing_stats is None:
        print("[ActionFlow-Qwen] No timing stats available")
        return

    stats = vla_model._timing_stats
    n_runs = len(stats.get("total", []))

    if n_runs == 0:
        print("[ActionFlow-Qwen] No timing data collected yet")
        return

    print("\n" + "=" * 60)
    print(f"📊 ActionFlow-Qwen Timing Stats ({n_runs} runs)")
    print("=" * 60)

    def avg_ms(key):
        vals = stats.get(key, [])
        return np.mean(vals) if vals else 0.0

    embed = avg_ms("embed")
    llm = avg_ms("llm_actionflow")
    total = avg_ms("total")

    print(f"  Embedding:       {embed:7.1f} ms")
    print(f"  LLM (AF):       {llm:7.1f} ms")
    print(f"  ─────────────────────────────────")
    print(f"  Total:          {total:7.1f} ms")
    print(f"  FPS:            {1000.0 / total:7.2f}")
    print("=" * 60)
