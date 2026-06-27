"""
bench_e2e_kernel.py

End-to-end A/B benchmark: Triton vs CUDA kernels on the ActionFlow path.

Both the Triton baseline and the CUDA-optimized FusedRoPE+KV / ShiftKV kernels
are exercised in ONE process (single 7B model load) by monkeypatching the two
wrapper references inside `actionflow.modeling.layers` between timing runs.

  - CUDA side  : actionflow.kernels.ops.fused_rope_write_kv_wrapper /
                 shift_varlen_kv_cache_wrapper  (hardcoded to CUDA in ops.py)
  - Triton side: equivalent reference launches reconstructed below from the
                 still-present Triton @triton.jit kernels.

RMSNorm stays on Triton in BOTH cases, so the measured delta isolates exactly
the two kernels we optimized.

Usage:
    python vla-scripts/extern/bench_e2e_kernel.py
"""

import os
import sys
import time
import statistics

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from actionflow import enable_actionflow
import actionflow.modeling.layers as af_layers
from actionflow.kernels.ops import (
    fused_rope_write_kv_wrapper as cuda_fused_wrapper,        # hardcoded CUDA
    shift_varlen_kv_cache_wrapper as cuda_shift_wrapper,      # hardcoded CUDA
    fused_rope_write_kv_kernel,                               # raw Triton jit
    shift_varlen_kv_cache_kernel,                             # raw Triton jit
)

MODEL_PATH = "/home/daiyuntao/jetson-containers/data/models/huggingface/models--openvla--openvla-7b/snapshots/31f090d05236101ebfc381b61c674dd4746d4ce0"
INSTRUCTION = "put spoon on towel"

TEXT_LENS = [16, 128, 256]   # target text-token counts (image tokens ~256 fixed)
DECODE_LENS = [7, 32]
WARMUP = 3
RUNS = 10


# --------------------------------------------------------------------------- #
# Triton reference wrappers (same signatures as the originals in ops.py).
# --------------------------------------------------------------------------- #
def triton_fused_rope_write_kv_wrapper(
    Q_new, K_new, V_new, kv_ring_buffer, cos, sin, prefill_len, q_varlen
):
    B, L_q, H_q, D = Q_new.shape
    _, _, H_kv, _ = K_new.shape
    D_half = D // 2
    Q_new_in = Q_new.squeeze(0).contiguous()
    K_new_in = K_new.squeeze(0).contiguous()
    V_new_in = V_new.squeeze(0).contiguous()
    total_L_kv_buffer = kv_ring_buffer.shape[1]

    grid = (L_q, max(H_q, H_kv))
    fused_rope_write_kv_kernel[grid](
        Q_new_ptr=Q_new_in,
        K_new_ptr=K_new_in,
        V_new_ptr=V_new_in,
        Q_out_ptr=q_varlen,
        kv_ring_ptr=kv_ring_buffer,
        cos_ptr=cos,
        sin_ptr=sin,
        total_L_q=L_q,
        total_max_L=total_L_kv_buffer,
        prefill_len=prefill_len,
        D=D,
        stride_ql=Q_new_in.stride(0),
        stride_qh=Q_new_in.stride(1),
        stride_kl=K_new_in.stride(0),
        stride_kh=K_new_in.stride(1),
        stride_vl=V_new_in.stride(0),
        stride_vh=V_new_in.stride(1),
        stride_qo_l=q_varlen.stride(0),
        stride_qo_h=q_varlen.stride(1),
        stride_ring_k_dim=kv_ring_buffer.stride(0),
        stride_ring_seq=kv_ring_buffer.stride(1),
        stride_ring_h=kv_ring_buffer.stride(2),
        stride_cos_l=cos.stride(0),
        stride_sin_l=sin.stride(0),
        BLOCK_D=D,
        D_half=D_half,
        H_q=H_q,
        H_kv=H_kv,
    )


def triton_shift_varlen_kv_cache_wrapper(kv_cache, B_stages, prefill_len):
    if B_stages <= 1:
        return kv_cache
    _, _, H_kv, D = kv_cache.shape
    kv_cache = kv_cache.contiguous()
    L_max = prefill_len + B_stages - 1
    grid = (L_max, H_kv, 2)
    strides = kv_cache.stride()
    shift_varlen_kv_cache_kernel[grid](
        KV_Cache_ptr=kv_cache,
        stride_kv_k_dim=strides[0],
        stride_kv_seq=strides[1],
        stride_kv_head=strides[2],
        H_kv=H_kv,
        D=D,
        PREFILL_LEN=prefill_len,
        B_STAGES=B_stages,
        BLOCK_D=D,
    )
    return kv_cache


def set_kernel_mode(mode: str):
    """Monkeypatch the wrapper references used inside layers.packed_forward."""
    if mode == "cuda":
        af_layers.fused_rope_write_kv_wrapper = cuda_fused_wrapper
        af_layers.shift_varlen_kv_cache_wrapper = cuda_shift_wrapper
    elif mode == "triton":
        af_layers.fused_rope_write_kv_wrapper = triton_fused_rope_write_kv_wrapper
        af_layers.shift_varlen_kv_cache_wrapper = triton_shift_varlen_kv_cache_wrapper
    else:
        raise ValueError(mode)


def build_prompt(target_len: int, tokenizer) -> str:
    base = f"In: What action should the robot take to {INSTRUCTION.lower()}?\nOut:"
    ids = tokenizer(base, return_tensors="pt")["input_ids"][0]
    if len(ids) >= target_len:
        return tokenizer.decode(ids[:target_len], skip_special_tokens=True)
    prompt = base
    while True:
        prompt += " continue."
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        if len(ids) >= target_len or len(prompt) > 2000:
            return tokenizer.decode(ids[:target_len], skip_special_tokens=True)


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda")

    print(f"[*] Loading OpenVLA: {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    # results[(text_len, decode_len)][mode] = median latency (s)
    results = {}

    for decode_len in DECODE_LENS:
        vla = enable_actionflow(vla, max_new_tokens=decode_len)
        for text_len in TEXT_LENS:
            prompt = build_prompt(text_len, processor.tokenizer)
            img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            inputs = processor(prompt, img, return_tensors="pt").to(device, dtype=torch.bfloat16)

            for mode in ("triton", "cuda"):
                set_kernel_mode(mode)
                # warmup (also triggers Triton/CUDA JIT compile on first use)
                for _ in range(WARMUP):
                    _ = vla.predict_action(**inputs, unnorm_key="bridge_orig")
                torch.cuda.synchronize()

                times = []
                for _ in range(RUNS):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _ = vla.predict_action(**inputs, unnorm_key="bridge_orig")
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
                med = statistics.median(times)
                results.setdefault((text_len, decode_len), {})[mode] = med
                print(f"  text={text_len:>3} decode={decode_len:>2} [{mode:>6}] "
                      f"median={med * 1000:7.2f} ms  fps={1.0 / med:6.2f}")

    # ----------------------------- summary -------------------------------- #
    print("\n" + "=" * 78)
    print("END-TO-END: Triton vs CUDA kernels (FusedRoPE+KV & ShiftKV swapped)")
    print("=" * 78)
    hdr = f"{'text':>5} {'decode':>7} {'Triton ms':>11} {'CUDA ms':>10} {'speedup':>9}"
    print(hdr)
    print("-" * len(hdr))
    for decode_len in DECODE_LENS:
        for text_len in TEXT_LENS:
            r = results[(text_len, decode_len)]
            tr, cu = r["triton"] * 1000, r["cuda"] * 1000
            print(f"{text_len:>5} {decode_len:>7} {tr:>11.2f} {cu:>10.2f} {tr / cu:>8.2f}x")
    print("=" * 78)


if __name__ == "__main__":
    main()
