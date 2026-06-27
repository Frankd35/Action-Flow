"""
Action-Flow Kernel Benchmark — Triton vs CUDA, two timing regimes.

KEY FINDING (see README): a per-call `record(); fn(); record(); synchronize()`
loop (the old methodology) charges each kernel's LAUNCH overhead to its time,
because every call starts from an idle GPU.  Triton's Python launch is ~50 us;
the CUDA C++ extension's is ~8 us.  That gap alone made the Triton FusedRoPE
kernel look ~1.5x slower than CUDA — a measurement artifact, NOT bandwidth.

This bench therefore reports BOTH regimes:

  * synced    : per-call CUDA event + synchronize()  (old methodology; launch
                overhead included → misleading for launch-heavy Triton kernels)
  * pipelined : ONE event pair around N back-to-back launches, no inner sync
                (matches the real packed_forward pipeline, where the CPU runs
                ahead and launch latency is hidden behind neighbouring ops →
                ≈ pure kernel GPU time = the number that actually matters)

Run:  python actionflow/kernels/bench_real.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import triton
from actionflow.kernels.ops import (
    rmsnorm_fwd_kernel,
    fused_rope_write_kv_kernel,
    shift_varlen_kv_cache_kernel,
)
from actionflow.kernels.cuda_ops import _get_cuda_module

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
PEAK_BW = 238.0          # DRAM peak (GB/s), cudaMemcpy D2D cache-busted
N = 4096
D, H_q, H_kv = 128, 32, 32
WARMUP = 20
ITERS_SYNC = 100
ITERS_PIPE = 200

PRE = [277, 385, 513]
DEC = [7, 16, 24, 32]


# --------------------------------------------------------------------------- #
# Timers
# --------------------------------------------------------------------------- #
def t_synced(fn):
    """Old methodology: per-call event + sync (launch overhead included)."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(ITERS_SYNC):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        xs.append(s.elapsed_time(e))
    xs.sort()
    return xs[len(xs) // 2] * 1000.0  # us


def t_pipelined(fn):
    """Real-pipeline methodology: batch of launches, launch latency hidden."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS_PIPE):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS_PIPE * 1000.0  # us


def gbs(nbytes, us):
    return (nbytes / 1e9) / (us / 1e6)


# --------------------------------------------------------------------------- #
# Kernel fns
# --------------------------------------------------------------------------- #
def make_fused(prefill, decode):
    L_q = prefill + decode - 1
    B_stages = decode
    total_L_kv = B_stages * prefill + (B_stages - 1) * B_stages // 2
    Q = torch.randn(L_q, H_q, D, dtype=DTYPE, device=DEVICE)
    K = torch.randn(L_q, H_kv, D, dtype=DTYPE, device=DEVICE)
    V = torch.randn(L_q, H_kv, D, dtype=DTYPE, device=DEVICE)
    qo = torch.empty(L_q, H_q, D, dtype=DTYPE, device=DEVICE)
    kv = torch.randn(2, total_L_kv, H_kv, D, dtype=DTYPE, device=DEVICE)
    cos = torch.randn(L_q, D, dtype=DTYPE, device=DEVICE)
    sin = torch.randn(L_q, D, dtype=DTYPE, device=DEVICE)
    mod = _get_cuda_module()

    def triton_fn():
        fused_rope_write_kv_kernel[(L_q, H_q)](
            Q_new_ptr=Q, K_new_ptr=K, V_new_ptr=V, Q_out_ptr=qo, kv_ring_ptr=kv,
            cos_ptr=cos, sin_ptr=sin, total_L_q=L_q, total_max_L=total_L_kv,
            prefill_len=prefill, D=D,
            stride_ql=Q.stride(0), stride_qh=Q.stride(1),
            stride_kl=K.stride(0), stride_kh=K.stride(1),
            stride_vl=V.stride(0), stride_vh=V.stride(1),
            stride_qo_l=qo.stride(0), stride_qo_h=qo.stride(1),
            stride_ring_k_dim=kv.stride(0), stride_ring_seq=kv.stride(1), stride_ring_h=kv.stride(2),
            stride_cos_l=cos.stride(0), stride_sin_l=sin.stride(0),
            BLOCK_D=D, D_half=D // 2, H_q=H_q, H_kv=H_kv,
        )

    def cuda_fn():
        mod.fused_rope_write_kv_cuda(Q, K, V, qo, kv, cos, sin, prefill)

    r = (L_q * H_q * D + L_q * H_kv * D + L_q * H_kv * D + L_q * D * 2) * 2
    w = (L_q * H_q * D + L_q * H_kv * D * 2) * 2
    return triton_fn, cuda_fn, L_q, r + w


def make_shift(prefill, B_stages):
    total_L_kv = B_stages * prefill + (B_stages - 1) * B_stages // 2
    kv = torch.randn(2, total_L_kv, H_kv, D, dtype=DTYPE, device=DEVICE)
    L_max = prefill + B_stages - 1
    mod = _get_cuda_module()

    def triton_fn():
        shift_varlen_kv_cache_kernel[(L_max, H_kv, 2)](
            KV_Cache_ptr=kv, stride_kv_k_dim=kv.stride(0), stride_kv_seq=kv.stride(1),
            stride_kv_head=kv.stride(2), H_kv=H_kv, D=D,
            PREFILL_LEN=prefill, B_STAGES=B_stages, BLOCK_D=D,
        )

    def cuda_fn():
        mod.shift_varlen_kv_cache_cuda(kv, B_stages, prefill)

    b = sum(2 * H_kv * (prefill + s) * D * 2 for s in range(B_stages - 1))
    return triton_fn, cuda_fn, b


def make_rmsnorm(prefill, decode):
    M = prefill + decode - 1
    x = torch.randn(M, N, dtype=DTYPE, device=DEVICE)
    w = torch.randn(N, dtype=DTYPE, device=DEVICE)
    out = torch.empty_like(x)
    BLK = triton.next_power_of_2(N)
    mod = _get_cuda_module()

    def triton_fn():
        rmsnorm_fwd_kernel[(M,)](x, w, out, x.stride(0), x.stride(1),
                                 out.stride(0), out.stride(1), M, N, 1e-6, BLOCK_SIZE_N=BLK)

    def cuda_fn():
        mod.rmsnorm_fwd_cuda(x, w, out)

    b = 2 * M * N * 2 + N * 2 + M * N * 2
    return triton_fn, cuda_fn, M, b


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def hdr(cols):
    line = "".join(f"{c:>{w}}" for c, w in cols)
    print(line)
    print("-" * len(line))


def bench_pair(tri_fn, cu_fn, nbytes):
    ts, tp = t_synced(tri_fn), t_pipelined(tri_fn)
    cs, cp = t_synced(cu_fn), t_pipelined(cu_fn)
    return ts, tp, cs, cp, gbs(nbytes, tp), gbs(nbytes, cp)


def main():
    torch.manual_seed(0)
    print(f"Peak DRAM BW = {PEAK_BW} GB/s | regimes: synced (old) vs pipelined (real)\n")

    # ---- FusedRoPE+KV ----
    print("=" * 100)
    print("FusedRoPE+KV — Triton vs CUDA   (us = per-call; BW from pipelined)")
    print("=" * 100)
    cols = [("prefill", 8), ("dec", 5), ("MB", 8), ("Tri_sync", 10), ("Tri_pipe", 10),
            ("CU_sync", 9), ("CU_pipe", 9), ("TriBW_pipe", 12), ("CUBW_pipe", 11), ("CU/Tri_pipe", 13)]
    hdr(cols)
    for p in PRE:
        for d in DEC:
            tri, cu, L_q, nb = make_fused(p, d)
            ts, tp, cs, cp, tbw, cbw = bench_pair(tri, cu, nb)
            print(f"{p:>8}{d:>5}{nb/1e6:>8.2f}{ts:>10.1f}{tp:>10.1f}{cs:>9.1f}{cp:>9.1f}"
                  f"{tbw:>11.0f} {cbw:>10.0f} {cp/tp:>11.2f}x")

    # ---- ShiftKV ----
    print("\n" + "=" * 100)
    print("ShiftKV — Triton vs CUDA   (us = per-call; BW from pipelined)")
    print("=" * 100)
    cols = [("prefill", 8), ("B", 5), ("MB", 8), ("Tri_sync", 10), ("Tri_pipe", 10),
            ("CU_sync", 9), ("CU_pipe", 9), ("TriBW_pipe", 12), ("CUBW_pipe", 11), ("CU/Tri_pipe", 13)]
    hdr(cols)
    for p in PRE:
        for b_stages in DEC:
            tri, cu, nb = make_shift(p, b_stages)
            ts, tp, cs, cp, tbw, cbw = bench_pair(tri, cu, nb)
            print(f"{p:>8}{b_stages:>5}{nb/1e6:>8.2f}{ts:>10.1f}{tp:>10.1f}{cs:>9.1f}{cp:>9.1f}"
                  f"{tbw:>11.0f} {cbw:>10.0f} {tp/cp:>11.2f}x")

    # ---- RMSNorm (Triton-only context; both kernels shown for reference) ----
    print("\n" + "=" * 100)
    print("RMSNorm — Triton vs CUDA   (N=4096; pipelined BW)")
    print("=" * 100)
    cols = [("L", 6), ("MB", 8), ("Tri_sync", 10), ("Tri_pipe", 10),
            ("CU_sync", 9), ("CU_pipe", 9), ("TriBW_pipe", 12), ("CUBW_pipe", 11)]
    hdr(cols)
    for p in PRE:
        for d in (7, 32):
            tri, cu, M, nb = make_rmsnorm(p, d)
            ts, tp, cs, cp, tbw, cbw = bench_pair(tri, cu, nb)
            print(f"{M:>6}{nb/1e6:>8.2f}{ts:>10.1f}{tp:>10.1f}{cs:>9.1f}{cp:>9.1f}"
                  f"{tbw:>11.0f} {cbw:>10.0f}")


if __name__ == "__main__":
    main()
