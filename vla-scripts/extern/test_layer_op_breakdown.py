"""
test_layer_op_breakdown.py

Per-op breakdown of ONE ActionFlow decoder layer during a REAL OpenVLA run,
fixed at prefill=277 (text_tokens=16) + decode=7 (the README's smallest shape).

It replaces LlamaPIPEDecodeLayer.packed_forward with a CUDA-event-instrumented
copy, runs the real `predict_action`, and accumulates each op's GPU time across
all 32 layers (and all timed iterations).  It is run twice — once with the
Triton FusedRoPE+KV / ShiftKV kernels, once with the CUDA kernels — so the table
shows both the per-op share inside a layer and how much the two optimized
kernels improved.

RMSNorm stays on Triton in BOTH modes (unchanged).

Run (inside container dyt_af_ops):
    docker exec dyt_af_ops bash -lc \
        'cd /home/daiyuntao/Action-Flow && python vla-scripts/extern/test_layer_op_breakdown.py'
"""

import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from flash_attn import flash_attn_varlen_func

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from actionflow import enable_actionflow
import actionflow.modeling.layers as af_layers
from actionflow.kernels.ops import (
    fused_rope_write_kv_wrapper as cuda_fused_wrapper,   # hardcoded CUDA in ops.py
    shift_varlen_kv_cache_wrapper as cuda_shift_wrapper,  # hardcoded CUDA in ops.py
    fused_rope_write_kv_kernel,
    shift_varlen_kv_cache_kernel,
)

MODEL_PATH = "/home/daiyuntao/jetson-containers/data/models/huggingface/models--openvla--openvla-7b/snapshots/31f090d05236101ebfc381b61c674dd4746d4ce0"
INSTRUCTION = "put spoon on towel"
TEXT_LEN = 16        # -> prefill ~277
DECODE_LEN = 7       # -> B_stages=7, L_q=283
WARMUP = 3
RUNS = 20

# Order matters for the printed table.
OP_NAMES = [
    "rmsnorm_in", "qkv_proj", "fused_rope_kv", "flash_attn",
    "shift_kv", "o_proj", "rmsnorm_post", "mlp",
]

# --------------------------------------------------------------------------- #
# Triton reference wrappers (identical signatures to ops.py originals).
# --------------------------------------------------------------------------- #
def triton_fused_wrapper(Q_new, K_new, V_new, kv_ring_buffer, cos, sin, prefill_len, q_varlen):
    B, L_q, H_q, D = Q_new.shape
    _, _, H_kv, _ = K_new.shape
    D_half = D // 2
    Q_new_in = Q_new.squeeze(0).contiguous()
    K_new_in = K_new.squeeze(0).contiguous()
    V_new_in = V_new.squeeze(0).contiguous()
    total_L_kv_buffer = kv_ring_buffer.shape[1]
    grid = (L_q, max(H_q, H_kv))
    fused_rope_write_kv_kernel[grid](
        Q_new_ptr=Q_new_in, K_new_ptr=K_new_in, V_new_ptr=V_new_in,
        Q_out_ptr=q_varlen, kv_ring_ptr=kv_ring_buffer, cos_ptr=cos, sin_ptr=sin,
        total_L_q=L_q, total_max_L=total_L_kv_buffer, prefill_len=prefill_len, D=D,
        stride_ql=Q_new_in.stride(0), stride_qh=Q_new_in.stride(1),
        stride_kl=K_new_in.stride(0), stride_kh=K_new_in.stride(1),
        stride_vl=V_new_in.stride(0), stride_vh=V_new_in.stride(1),
        stride_qo_l=q_varlen.stride(0), stride_qo_h=q_varlen.stride(1),
        stride_ring_k_dim=kv_ring_buffer.stride(0), stride_ring_seq=kv_ring_buffer.stride(1),
        stride_ring_h=kv_ring_buffer.stride(2),
        stride_cos_l=cos.stride(0), stride_sin_l=sin.stride(0),
        BLOCK_D=D, D_half=D_half, H_q=H_q, H_kv=H_kv,
    )


def triton_shift_wrapper(kv_cache, B_stages, prefill_len):
    if B_stages <= 1:
        return kv_cache
    _, _, H_kv, D = kv_cache.shape
    kv_cache = kv_cache.contiguous()
    L_max = prefill_len + B_stages - 1
    grid = (L_max, H_kv, 2)
    strides = kv_cache.stride()
    shift_varlen_kv_cache_kernel[grid](
        KV_Cache_ptr=kv_cache,
        stride_kv_k_dim=strides[0], stride_kv_seq=strides[1], stride_kv_head=strides[2],
        H_kv=H_kv, D=D, PREFILL_LEN=prefill_len, B_STAGES=B_stages, BLOCK_D=D,
    )
    return kv_cache


# --------------------------------------------------------------------------- #
# Instrumentation.
# --------------------------------------------------------------------------- #
CUR = {"fused": None, "shift": None}          # active wrappers (mode dependent)
EVENTS = []                                   # (name, start_event, end_event)
META = {"prefill": None, "seq_lens": None}


def timed(name, fn):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = fn()
    e.record()
    EVENTS.append((name, s, e))
    return out


def instrumented_packed_forward(
    self, batch_hidden_states, kv_ring_buffer, global_position_embeddings,
    seq_lens, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kw,
):
    t_start = torch.cuda.Event(enable_timing=True)
    t_end = torch.cuda.Event(enable_timing=True)
    t_start.record()

    residual = torch.cat(batch_hidden_states, dim=1)
    B, L, D = residual.shape
    if META["prefill"] is None:
        META["prefill"] = seq_lens[0]
        META["seq_lens"] = list(seq_lens)

    normed = timed("rmsnorm_in", lambda: self.input_layernorm(residual))

    def _qkv():
        q = self._optimized_linear(self.original_layer.self_attn.q_proj, normed)
        k = self._optimized_linear(self.original_layer.self_attn.k_proj, normed)
        v = self._optimized_linear(self.original_layer.self_attn.v_proj, normed)
        return q, k, v
    queries, keys, values = timed("qkv_proj", _qkv)

    queries = queries.view(B, L, -1, self.head_dim)
    keys = keys.view(B, L, self.num_key_value_heads, self.head_dim)
    values = values.view(B, L, self.num_key_value_heads, self.head_dim)

    q_varlen = torch.empty_like(queries.squeeze(0))
    cos_full, sin_full = global_position_embeddings
    prefill_len = seq_lens[0]

    timed("fused_rope_kv", lambda: CUR["fused"](
        Q_new=queries, K_new=keys, V_new=values, kv_ring_buffer=kv_ring_buffer,
        cos=cos_full.squeeze(0), sin=sin_full.squeeze(0),
        prefill_len=prefill_len, q_varlen=q_varlen))

    k_varlen = kv_ring_buffer[0]
    v_varlen = kv_ring_buffer[1]
    attn_output = timed("flash_attn", lambda: flash_attn_varlen_func(
        q=q_varlen, k=k_varlen, v=v_varlen,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        dropout_p=0.0, softmax_scale=self.scaling, causal=True))

    timed("shift_kv", lambda: CUR["shift"](
        kv_cache=kv_ring_buffer, B_stages=len(seq_lens), prefill_len=prefill_len))

    attn_concat = attn_output.view(B, L, D)
    attn_concat = timed("o_proj", lambda: self._optimized_linear(
        self.original_layer.self_attn.o_proj, attn_concat))
    hidden_states = residual + attn_concat

    residual = hidden_states
    hidden_states = timed("rmsnorm_post", lambda: self.post_attention_layernorm(hidden_states))

    mlp = self.original_layer.mlp

    def _mlp():
        gate_out = self._optimized_linear(mlp.gate_proj, hidden_states)
        up_out = self._optimized_linear(mlp.up_proj, hidden_states)
        return self._optimized_linear(mlp.down_proj, mlp.act_fn(gate_out) * up_out)
    mlp_out = timed("mlp", _mlp)
    hidden_states = residual + mlp_out

    outputs = []
    start = 0
    for seq_len in seq_lens:
        outputs.append(hidden_states[:, start:start + seq_len, :])
        start += seq_len

    t_end.record()
    EVENTS.append(("layer_total", t_start, t_end))
    return outputs


def reduce_events():
    """Sync, then sum elapsed_time per op name. Returns dict name->ms total."""
    torch.cuda.synchronize()
    acc = {}
    for name, s, e in EVENTS:
        acc[name] = acc.get(name, 0.0) + s.elapsed_time(e)  # ms
    return acc


def set_mode(mode):
    if mode == "cuda":
        CUR["fused"], CUR["shift"] = cuda_fused_wrapper, cuda_shift_wrapper
    else:
        CUR["fused"], CUR["shift"] = triton_fused_wrapper, triton_shift_wrapper


def build_prompt(target_len, tokenizer):
    base = f"In: What action should the robot take to {INSTRUCTION.lower()}?\nOut:"
    ids = tokenizer(base, return_tensors="pt")["input_ids"][0]
    prompt = base
    while len(ids) < target_len and len(prompt) < 2000:
        prompt += " continue."
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    return tokenizer.decode(ids[:target_len], skip_special_tokens=True)


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda")

    print(f"[*] Loading OpenVLA: {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH, attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device)

    num_layers = vla.language_model.config.num_hidden_layers

    # Install the instrumented method on the class (affects all layer wrappers).
    af_layers.LlamaPIPEDecodeLayer.packed_forward = instrumented_packed_forward

    vla = enable_actionflow(vla, max_new_tokens=DECODE_LEN)

    prompt = build_prompt(TEXT_LEN, processor.tokenizer)
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    inputs = processor(prompt, img, return_tensors="pt").to(device, dtype=torch.bfloat16)

    per_mode = {}
    for mode in ("triton", "cuda"):
        set_mode(mode)
        for _ in range(WARMUP):
            _ = vla.predict_action(**inputs, unnorm_key="bridge_orig")
        torch.cuda.synchronize()
        EVENTS.clear()
        for _ in range(RUNS):
            _ = vla.predict_action(**inputs, unnorm_key="bridge_orig")
        acc = reduce_events()
        # normalize to per-predict_action (i.e. per full 32-layer pass)
        per_mode[mode] = {k: v / RUNS for k, v in acc.items()}

    pf = META["prefill"]
    sl = META["seq_lens"]
    print(f"\n[shape] prefill_len={pf}  decode={DECODE_LEN}  "
          f"B_stages={len(sl)}  L_q={sum(sl)}  num_layers={num_layers}")

    # ----------------------------- table --------------------------------- #
    tri, cud = per_mode["triton"], per_mode["cuda"]
    tri_total = tri["layer_total"]
    cud_total = cud["layer_total"]

    print("\n" + "=" * 92)
    print("PER-OP BREAKDOWN — summed over all 32 layers, per predict_action (ms)")
    print("RMSNorm = Triton in both modes; only fused_rope_kv & shift_kv differ.")
    print("=" * 92)
    hdr = (f"{'op':<14}{'Triton ms':>11}{'Tri %':>8}{'CUDA ms':>11}"
           f"{'CUDA %':>8}{'speedup':>9}")
    print(hdr)
    print("-" * len(hdr))
    sum_t = sum_c = 0.0
    for name in OP_NAMES:
        t, c = tri.get(name, 0.0), cud.get(name, 0.0)
        sum_t += t
        sum_c += c
        sp = (t / c) if c > 1e-9 else 0.0
        print(f"{name:<14}{t:>11.3f}{100 * t / tri_total:>7.1f}%{c:>11.3f}"
              f"{100 * c / cud_total:>7.1f}%{sp:>8.2f}x")
    other_t = tri_total - sum_t
    other_c = cud_total - sum_c
    print(f"{'other/resid':<14}{other_t:>11.3f}{100 * other_t / tri_total:>7.1f}%"
          f"{other_c:>11.3f}{100 * other_c / cud_total:>7.1f}%{'-':>8}")
    print("-" * len(hdr))
    print(f"{'LAYER TOTAL':<14}{tri_total:>11.3f}{100.0:>7.1f}%{cud_total:>11.3f}"
          f"{100.0:>7.1f}%{tri_total / cud_total:>8.2f}x")
    print("=" * 92)

    # Focused improvement on the two optimized kernels.
    print("\nOptimized kernels (per predict_action, 32 layers):")
    for name in ("fused_rope_kv", "shift_kv"):
        t, c = tri[name], cud[name]
        print(f"  {name:<14} Triton {t:7.3f} ms -> CUDA {c:7.3f} ms "
              f"| {t / c:.2f}x faster | saves {t - c:6.3f} ms/call "
              f"({100 * (t - c) / tri_total:.2f}% of layer total)")
    saved = (tri["fused_rope_kv"] + tri["shift_kv"]) - (cud["fused_rope_kv"] + cud["shift_kv"])
    print(f"  -> combined layer-stack saving: {saved:.3f} ms/predict_action "
          f"({100 * saved / tri_total:.2f}% of total layer time)")


if __name__ == "__main__":
    main()
