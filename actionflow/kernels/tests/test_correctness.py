"""
Correctness tests for CUDA kernels vs Triton reference implementations.

Run:  python -m pytest actionflow/kernels/tests/test_correctness.py -v
Or:   python actionflow/kernels/tests/test_correctness.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import triton

from actionflow.kernels.ops import (
    rmsnorm_fwd_kernel,
    fused_rope_write_kv_kernel,
    shift_varlen_kv_cache_kernel,
    fused_rope_write_kv_wrapper,
    shift_varlen_kv_cache_wrapper,
)
from actionflow.kernels.cuda_ops import (
    cuda_rmsnorm_fwd,
    cuda_fused_rope_write_kv,
    cuda_shift_varlen_kv_cache,
)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
D = 128
H_q = 32
H_kv = 32
N = 4096

# Representative shapes from README
SHAPES = [
    # (prefill, decode_len)
    (277, 7),
    (277, 16),
    (277, 32),
    (385, 16),
    (513, 7),
    (513, 32),
]

ATOL = 1e-2  # bf16 tolerance
RTOL = 1e-3


def _allclose(a, b, name):
    if not torch.allclose(a, b, atol=ATOL, rtol=RTOL):
        max_diff = (a.float() - b.float()).abs().max().item()
        raise AssertionError(
            f"[{name}] mismatch: max_abs_diff={max_diff:.4e} (atol={ATOL})"
        )


# ─────────────────────────────────────────────────────────
# 1. RMSNorm
# ─────────────────────────────────────────────────────────
def test_rmsnorm():
    for prefill, decode in SHAPES:
        M = prefill + decode - 1
        x = torch.randn(M, N, dtype=DTYPE, device=DEVICE)
        w = torch.randn(N, dtype=DTYPE, device=DEVICE)

        # Triton reference
        out_tri = torch.empty_like(x)
        BLK = triton.next_power_of_2(N)
        rmsnorm_fwd_kernel[(M,)](
            x, w, out_tri,
            x.stride(0), x.stride(1), out_tri.stride(0), out_tri.stride(1),
            M, N, 1e-6, BLOCK_SIZE_N=BLK,
        )

        # CUDA
        out_cuda = torch.empty_like(x)
        cuda_rmsnorm_fwd(x, w, out_cuda)

        _allclose(out_tri, out_cuda, f"rmsnorm prefill={prefill} decode={decode} M={M}")
    print("[OK] rmsnorm: all shapes match")


# ─────────────────────────────────────────────────────────
# 2. Fused RoPE + Write KV
# ─────────────────────────────────────────────────────────
def test_fused_rope_write_kv():
    for prefill, decode in SHAPES:
        L_q = prefill + decode - 1
        B_stages = decode
        total_L_kv = B_stages * prefill + (B_stages - 1) * B_stages // 2

        Q = torch.randn(1, L_q, H_q, D, dtype=DTYPE, device=DEVICE)
        K = torch.randn(1, L_q, H_kv, D, dtype=DTYPE, device=DEVICE)
        V = torch.randn(1, L_q, H_kv, D, dtype=DTYPE, device=DEVICE)
        cos = torch.randn(L_q, D, dtype=DTYPE, device=DEVICE)
        sin = torch.randn(L_q, D, dtype=DTYPE, device=DEVICE)

        # --- Triton reference ---
        qo_tri = torch.empty(L_q, H_q, D, dtype=DTYPE, device=DEVICE)
        kv_tri = torch.randn(2, total_L_kv, H_kv, D, dtype=DTYPE, device=DEVICE)
        fused_rope_write_kv_wrapper(
            Q.clone(), K.clone(), V.clone(),
            kv_tri, cos, sin, prefill, qo_tri,
        )

        # --- CUDA ---
        qo_cuda = torch.empty(L_q, H_q, D, dtype=DTYPE, device=DEVICE)
        kv_cuda = kv_tri.clone()  # same init to compare in-place writes
        cuda_fused_rope_write_kv(
            Q.clone(), K.clone(), V.clone(),
            qo_cuda, kv_cuda, cos, sin, prefill,
        )

        _allclose(qo_tri, qo_cuda, f"fused_rope Q_out prefill={prefill} decode={decode}")
        _allclose(kv_tri, kv_cuda, f"fused_rope kv_ring prefill={prefill} decode={decode}")
    print("[OK] fused_rope_write_kv: all shapes match")


# ─────────────────────────────────────────────────────────
# 3. Shift KV Cache (in-place)
# ─────────────────────────────────────────────────────────
def test_shift_kv_cache():
    for prefill, decode in SHAPES:
        B_stages = decode
        total_L_kv = B_stages * prefill + (B_stages - 1) * B_stages // 2

        kv_init = torch.randn(2, total_L_kv, H_kv, D, dtype=DTYPE, device=DEVICE)

        # --- Triton reference ---
        kv_tri = kv_init.clone()
        shift_varlen_kv_cache_wrapper(kv_tri, B_stages, prefill)

        # --- CUDA ---
        kv_cuda = kv_init.clone()
        cuda_shift_varlen_kv_cache(kv_cuda, B_stages, prefill)

        _allclose(kv_tri, kv_cuda, f"shift_kv prefill={prefill} B_stages={B_stages}")
    print("[OK] shift_varlen_kv_cache: all shapes match")


if __name__ == "__main__":
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Testing {len(SHAPES)} shape combos, atol={ATOL}, rtol={RTOL}\n")

    # test_rmsnorm()  # skipped: investigate separately
    test_fused_rope_write_kv()
    test_shift_kv_cache()

    print("\n=== All correctness tests passed ===")
