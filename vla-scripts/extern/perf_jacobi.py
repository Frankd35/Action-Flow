#!/usr/bin/env python3
"""
Jacobi 性能脚本（简化版）：CEED Jacobi-KV vs ActionFlow(Jacobi pipeline)。

计时口径：
- `torch.cuda.synchronize()` + wall-time

对齐口径（默认）：LIBERO 风格 token 预算
- `max_token = 7*action_chunk + 1`
- `max_iter` CEED Jacobi max_iter & AF pipeline depth K.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from types import ModuleType
from typing import Any, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CEED_ROOT = os.environ.get("CEED_VLA_ROOT") or os.path.abspath(os.path.join(PROJECT_ROOT, "..", "CEED-VLA"))


def _ensure_path_front(path: str) -> None:
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


_ensure_path_front(PROJECT_ROOT)

try:
    from actionflow.integration_jacobi import enable_actionflow_jacobi
except ImportError as e:
    print(f"❌ [ActionFlow] integration_jacobi import failed: {e}")
    sys.exit(1)


def _install_ceed_hf_bypass(ceed_root: str) -> None:
    """只挂 CEED-VLA/prismatic/extern/hf，避免拉全套依赖。"""
    hf_dir = os.path.join(ceed_root, "prismatic", "extern", "hf")
    if not os.path.isdir(hf_dir):
        raise FileNotFoundError(f"CEED-VLA HF 目录不存在: {hf_dir}")

    for _pkg in ["prismatic", "prismatic.extern", "prismatic.extern.hf"]:
        _mod = ModuleType(_pkg)
        _mod.__path__ = [hf_dir]
        _mod.__package__ = _pkg
        sys.modules[_pkg] = _mod

    _IGNORE_INDEX = -100
    _ACTION_DIM = 7
    _ACTION_TOKEN_BEGIN_IDX = 32001

    def _get_current_action_mask(token_ids):
        newline_positions = token_ids != _IGNORE_INDEX
        cumsum = torch.cumsum(newline_positions, dim=1)
        mask = (1 <= cumsum) & (cumsum <= _ACTION_DIM)
        return (token_ids > _ACTION_TOKEN_BEGIN_IDX) * mask

    def _get_next_actions_mask(token_ids):
        newline_positions = token_ids != _IGNORE_INDEX
        cumsum = torch.cumsum(newline_positions, dim=1)
        return (token_ids > _ACTION_TOKEN_BEGIN_IDX) * (cumsum > _ACTION_DIM)

    _train_utils = ModuleType("prismatic.training.train_utils")
    _train_utils.get_current_action_mask = _get_current_action_mask
    _train_utils.get_next_actions_mask = _get_next_actions_mask
    sys.modules["prismatic.training"] = ModuleType("prismatic.training")
    sys.modules["prismatic.training.train_utils"] = _train_utils

    if hf_dir not in sys.path:
        sys.path.insert(0, hf_dir)


def _get_openvla_prompt(instruction: str, model_name_or_path: str) -> str:
    if "openvla-v01" in model_name_or_path:
        system_prompt = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
        return f"{system_prompt} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def _load_ceed_vla(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    quantize: str,
    num_images_in_input: int,
) -> Tuple[Any, Any]:
    _install_ceed_hf_bypass(CEED_ROOT)

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    if quantize == "bf16":
        vla = AutoModelForVision2Seq.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        vla = vla.to(device=device)
    elif quantize == "int4":
        vla = AutoModelForVision2Seq.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=dtype,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            ),
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unknown quantize: {quantize}")

    if hasattr(vla, "vision_backbone") and hasattr(vla.vision_backbone, "set_num_images_in_input"):
        vla.vision_backbone.set_num_images_in_input(num_images_in_input)

    stats_path = os.path.join(model_path, "dataset_statistics.json")
    if os.path.isfile(stats_path):
        with open(stats_path, "r") as f:
            vla.norm_stats = json.load(f)

    return vla, processor


def _print_summary(times: list[float], runs: int, label: str, action_chunk: int) -> None:
    arr = np.asarray(times, dtype=np.float64)
    mean_t = float(arr.mean())
    std_t = float(arr.std())
    fps = 1.0 / arr
    mean_fps = float(fps.mean())
    std_fps = float(fps.std())

    print("\n" + "=" * 60)
    print(f"📊 {label} ({runs} timed runs)")
    print(f"   Avg Time: {mean_t:.4f} ± {std_t:.4f} s")
    if action_chunk <= 1:
        print(f"   Avg FPS : {mean_fps:.2f} ± {std_fps:.2f}")
    else:
        print(f"   Avg FPS (per-call): {mean_fps:.2f} ± {std_fps:.2f}")
        print(f"   Effective FPS (×{action_chunk}): {mean_fps * action_chunk:.2f} ± {std_fps * action_chunk:.2f}")
    print("=" * 60)


@torch.inference_mode()
def _run_ceed(
    vla: Any,
    processor: Any,
    device: torch.device,
    dtype: torch.dtype,
    unnorm_key: str | None,
    prompt: str,
    warmup: int,
    runs: int,
    max_new_tokens: int,
    max_iter: int,
    image_size: int,
    seed: int,
    action_chunk: int,
) -> None:
    print(f"[*] CEED: predict_action_jacobi_kv (max_new_tokens={max_new_tokens}, max_iter={max_iter})")
    times: list[float] = []

    for i in range(warmup + runs):
        np.random.seed(seed + i)
        image = Image.fromarray(np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8))
        inputs = processor(prompt, image, return_tensors="pt").to(device, dtype=dtype)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        action, _, _ = vla.predict_action_jacobi_kv(
            **inputs,
            unnorm_key=unnorm_key,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            make_action_mask=False,
            max_iter=max_iter,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= warmup:
            times.append(t1 - t0)
            print(f"\t=>> Iter {len(times):2d}: Time = {times[-1]:.4f}s")
            if i == warmup + runs - 1:
                print(f"\t=>> Last action shape: {np.asarray(action).shape}")

    _print_summary(times, runs, label="CEED Jacobi-KV", action_chunk=action_chunk)


@torch.inference_mode()
def _run_af(
    vla: Any,
    processor: Any,
    device: torch.device,
    dtype: torch.dtype,
    unnorm_key: str | None,
    prompt: str,
    warmup: int,
    runs: int,
    max_iter: int,
    jacobi_tokens: int,
    image_size: int,
    seed: int,
    action_chunk: int,
) -> None:
    print(f"[*] AF: enable_actionflow_jacobi -> predict_action (K={max_iter}, J={jacobi_tokens}, action_chunk={action_chunk})")
    vla = enable_actionflow_jacobi(
        vla,
        max_iter=max_iter,
        max_token=jacobi_tokens,
        action_chunk=action_chunk,
        enable_timing=False,
    )

    times: list[float] = []
    last_action: Any = None

    for i in range(warmup + runs):
        np.random.seed(seed + 100_000 + i)
        image = Image.fromarray(np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8))
        inputs = processor(prompt, image, return_tensors="pt").to(device, dtype=dtype)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= warmup:
            times.append(t1 - t0)
            last_action = action
            print(f"\t=>> Iter {len(times):2d}: Time = {times[-1]:.4f}s")
            if i == warmup + runs - 1:
                print(f"\t=>> Last action shape: {np.asarray(action).shape}")

    _print_summary(times, runs, label="AF Jacobi (predict_action)", action_chunk=action_chunk)


def main() -> None:
    p = argparse.ArgumentParser(description="Jacobi perf (simple): CEED jacobi-kv vs AF jacobi")
    p.add_argument("--model-path", type=str, required=True, help="OpenVLA / CEED checkpoint directory")
    p.add_argument("--unnorm-key", type=str, default=None)
    p.add_argument("--instruction", type=str, default="put spoon on towel")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--quantize", type=str, default="bf16", choices=["bf16", "int4"])
    p.add_argument("--num-images-in-input", type=int, default=1)
    p.add_argument("--backend", type=str, default="both", choices=["af", "ceed", "both"])
    p.add_argument("--action-chunk", type=int, default=1, help="effective env steps = action_chunk * per-call")
    p.add_argument("--max-iter", type=int, default=4, help="CEED Jacobi max_iter & AF pipeline depth K.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.")
        sys.exit(1)

    def _budget(ac: int) -> int:
        return 7 * ac + 1

    max_token = _budget(args.action_chunk)

    print(
        f"[*] Token budget: action_chunk={args.action_chunk} -> "
        f"max_iter={args.max_iter} (CEED iter & AF K), max_token={max_token} (CEED max_new_tokens & AF J)"
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    def one_prompt(vla: Any) -> str:
        model_name_or_path = getattr(vla, "name_or_path", "") or getattr(getattr(vla, "config", None), "_name_or_path", "")
        return _get_openvla_prompt(args.instruction, model_name_or_path)

    if args.backend in ("ceed", "both"):
        print("[*] Loading CEED OpenVLA...")
        vla_c, proc_c = _load_ceed_vla(args.model_path, device, dtype, args.quantize, args.num_images_in_input)
        prompt_c = one_prompt(vla_c)
        _run_ceed(
            vla=vla_c,
            processor=proc_c,
            device=device,
            dtype=dtype,
            unnorm_key=args.unnorm_key,
            prompt=prompt_c,
            warmup=args.warmup,
            runs=args.runs,
            max_new_tokens=max_token,
            max_iter=args.max_iter,
            image_size=args.image_size,
            seed=args.seed,
            action_chunk=args.action_chunk,
        )
        del vla_c, proc_c
        torch.cuda.empty_cache()

    if args.backend in ("af", "both"):
        print("[*] Loading CEED OpenVLA (for AF patched predict_action)...")
        vla_a, proc_a = _load_ceed_vla(args.model_path, device, dtype, args.quantize, args.num_images_in_input)
        prompt_a = one_prompt(vla_a)
        _run_af(
            vla=vla_a,
            processor=proc_a,
            device=device,
            dtype=dtype,
            unnorm_key=args.unnorm_key,
            prompt=prompt_a,
            warmup=args.warmup,
            runs=args.runs,
            max_iter=args.max_iter,
            jacobi_tokens=max_token,
            image_size=args.image_size,
            seed=args.seed,
            action_chunk=args.action_chunk,
        )


if __name__ == "__main__":
    main()

