"""
GPU Bandwidth Microbenchmark for Thor.

Demonstrates:
  1. Naive copy kernel hits L2 cache → ~997 GB/s (misleading)
  2. Cache-busted cudaMemcpy D2D → ~238 GB/s (real DRAM bandwidth)
  3. Brief Triton kernel latency baseline

This file serves as evidence for the 238 GB/s peak bandwidth used in bench_real.py.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_kernel(src_ptr, dst_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = off < n_elements
    tl.store(dst_ptr + off, tl.load(src_ptr + off, mask=m), mask=m)


def measure_peak_bw_triton_copy(max_bytes=512 * 1024 * 1024):
    """Naive Triton copy — measures L2 cache bandwidth, NOT DRAM."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    e = dtype.itemsize

    peak = 0.0
    sz = 1 * 1024 * 1024
    while sz <= max_bytes:
        n = sz // e
        src = torch.randn(n, dtype=dtype, device=device)
        dst = torch.empty_like(src)
        n_blocks = triton.cdiv(n, 1024)
        BLK = min(1024, triton.next_power_of_2(n))

        for _ in range(10):
            _copy_kernel[(n_blocks,)](src, dst, n, BLOCK_SIZE=BLK)
        torch.cuda.synchronize()

        s = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(50):
            _copy_kernel[(n_blocks,)](src, dst, n, BLOCK_SIZE=BLK)
        e2.record()
        torch.cuda.synchronize()
        t = s.elapsed_time(e2) / 50
        bw = (2 * n * e / 1e9) / (t / 1e3)
        peak = max(peak, bw)
        sz *= 2
    return peak


def measure_peak_bw_dram(max_bytes=1024 * 1024 * 1024):
    """Cache-busted cudaMemcpy D2D — measures real DRAM bandwidth."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    e = dtype.itemsize

    peak = 0.0
    dummy = None
    sz = 64 * 1024 * 1024
    while sz <= max_bytes:
        n = sz // e
        src = torch.randn(n, dtype=dtype, device=device)
        for _ in range(5):
            dst = torch.empty_like(src)
            dst.copy_(src)
        torch.cuda.synchronize()

        if dummy is None or dummy.numel() < n:
            dummy = torch.empty(n, dtype=torch.float32, device=device)

        times = []
        for _ in range(30):
            dummy.zero_()  # evict L2 cache
            dst = torch.empty_like(src)
            s = torch.cuda.Event(enable_timing=True)
            e2 = torch.cuda.Event(enable_timing=True)
            s.record()
            dst.copy_(src)
            e2.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e2))
        times.sort()
        t = times[len(times) // 2]
        bw = (2 * n * e / 1e9) / (t / 1e3)
        peak = max(peak, bw)
        sz *= 2
    return peak


def triton_kernel_latency_baseline():
    """Quick latency floor check for small Triton kernels."""
    device = torch.device("cuda")
    dtype = torch.bfloat16

    @triton.jit
    def _nop_kernel(N: tl.constexpr, BLK: tl.constexpr):
        off = tl.program_id(0) * BLK + tl.arange(0, BLK)
        # no-op: measure launch overhead only
        _ = off + 0

    for N in [1, 256, 4096, 65536]:
        BLK = min(1024, triton.next_power_of_2(N))
        n_blocks = max(1, triton.cdiv(N, BLK))
        for _ in range(5):
            _nop_kernel[(n_blocks,)](N=N, BLK=BLK)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(100):
            _nop_kernel[(n_blocks,)](N=N, BLK=BLK)
        e.record()
        torch.cuda.synchronize()
        t = s.elapsed_time(e) / 100
        print(f"  Triton no-op kernel  N={N:<6} blocks={n_blocks:<6} → {t*1000:.1f} us")


if __name__ == "__main__":
    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  ({props.major}.{props.minor})")
    l2 = getattr(props, 'L2_cache_size', getattr(props, 'l2_cache_size', 0))
    print(f"L2 cache: {l2 / 1024**2:.0f} MB")
    print()

    print("=== Peak Bandwidth ===")
    bw_cache = measure_peak_bw_triton_copy()
    print(f"  Triton copy kernel (naive, cache):  {bw_cache:.1f} GB/s")
    bw_dram = measure_peak_bw_dram()
    print(f"  cudaMemcpy D2D     (cache-busted):  {bw_dram:.1f} GB/s")
    print(f"\n  => DRAM bandwidth = {bw_dram:.0f} GB/s  (used in bench_real.py)")
    print()

    print("=== Triton Kernel Launch Overhead ===")
    triton_kernel_latency_baseline()
