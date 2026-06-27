"""
CUDA-optimized implementations of Action-Flow kernels.

These are JIT-compiled via torch.utils.cpp_extension.load_inline.
Each kernel targets scenarios where the Triton implementation falls below
90% peak memory bandwidth utilization.

Version history:
  v1.0 (module v4) — initial CUDA kernels.
  v1.1 (module v9) — ShiftKV: seq-offset grid parallelization + float4
                     vectorized copy (was (H_kv,2) grid w/ serial stage
                     loop).  FusedRoPE: V path float4 vectorized copy.
                     NOTE: __syncthreads() after cos/sin load is kept —
                     removing it corrupts Q_out (nvcc register scheduling).
  v1.2 (module v14) — ShiftKV: coalesced multi-head block (grid (L_max,2),
                     256 threads cover all heads × D per seq → 8 KB coalesced
                     transactions).  v3 (chunked multi-seq) and v4 (streaming
                     ld.cs/st.cs) also implemented but give no gain at B>=16:
                     the floor there is DRAM row-switching on the varlen
                     multi-region access pattern, not merge/RFO/cache.
                     Small-shape (B=7) util 85.8%→88.8%.
"""

import torch
from torch.utils.cpp_extension import load_inline
import os

# ---------------------------------------------------------------------------
# CUDA Source Code (all kernels in one compilation unit)
# ---------------------------------------------------------------------------

CUDA_SOURCE = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// =============================================================================
// RMSNorm Forward – Prefill (M >= 64)
// One CTA per row.  Warp-shuffle reduction.  Scalar loads (bandwidth bound).
// 256 threads x 16 elements = 4096 coverage.
// =============================================================================

template<typename scalar_t, int BLOCK_DIM, int ELEMS_PER_THREAD>
__global__ void rmsnorm_fwd_prefill_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w,
    scalar_t* __restrict__ out,
    const int M,
    const int N,
    const float eps
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    float sum_sq = 0.0f;

    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int col = tid + e * BLOCK_DIM;
        if (col < N) {
            float x_val = static_cast<float>(x[row * N + col]);
            sum_sq += x_val * x_val;
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Full block reduction via shared memory
    __shared__ float warp_sums[32];
    int warp_id = tid / 32;
    int lane_id = tid & 31;
    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    float block_sum = 0.0f;
    int num_warps = (BLOCK_DIM + 31) / 32;
    if (tid < num_warps) {
        block_sum = warp_sums[tid];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
    }
    block_sum = __shfl_sync(0xffffffff, block_sum, 0);

    float rms = rsqrtf(block_sum / float(N) + eps);

    // Write back
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int col = tid + e * BLOCK_DIM;
        if (col < N) {
            float w_val = static_cast<float>(w[col]);
            float x_val = static_cast<float>(x[row * N + col]);
            out[row * N + col] = static_cast<scalar_t>(x_val * rms * w_val);
        }
    }
}


// =============================================================================
// RMSNorm Forward – Decode (M is small)
// Single-warp CTA.  Grid-stride loop over N.
// =============================================================================

template<typename scalar_t, int BLOCK_DIM>
__global__ void rmsnorm_fwd_decode_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w,
    scalar_t* __restrict__ out,
    const int M,
    const int N,
    const float eps
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    float sum_sq = 0.0f;

    for (int col = tid; col < N; col += BLOCK_DIM) {
        float val = static_cast<float>(x[row * N + col]);
        sum_sq += val * val;
    }

    // Warp reduction (single warp, BLOCK_DIM <= 32)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    sum_sq = __shfl_sync(0xffffffff, sum_sq, 0);

    float rms = rsqrtf(sum_sq / float(N) + eps);

    for (int col = tid; col < N; col += BLOCK_DIM) {
        float w_val = static_cast<float>(w[col]);
        float x_val = static_cast<float>(x[row * N + col]);
        out[row * N + col] = static_cast<scalar_t>(x_val * rms * w_val);
    }
}


// =============================================================================
// Fused RoPE + Write KV – OpenVLA
// Grid: (L_q, H_q + 2*H_kv).  64 threads per block (D=128, D_HALF=64).
// =============================================================================

template<typename scalar_t, int D, int D_HALF>
__global__ void fused_rope_write_kv_cuda_kernel(
    const scalar_t* __restrict__ Q_new,
    const scalar_t* __restrict__ K_new,
    const scalar_t* __restrict__ V_new,
    scalar_t* __restrict__ Q_out,
    scalar_t* __restrict__ kv_ring,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    const int total_L_q,
    const int total_max_L,
    const int prefill_len,
    const int stride_ql,
    const int stride_qh,
    const int stride_kl,
    const int stride_kh,
    const int stride_vl,
    const int stride_vh,
    const int stride_qo_l,
    const int stride_qo_h,
    const int stride_ring_k_dim,
    const int stride_ring_seq,
    const int stride_ring_h,
    const int stride_cos_l,
    const int stride_sin_l,
    const int H_q,
    const int H_kv
) {
    const int tok_idx = blockIdx.x;
    const int task_id = blockIdx.y;
    const int tid = threadIdx.x;

    if (tok_idx >= total_L_q) return;

    int pos, cu_seqlens_k_base;
    if (tok_idx < prefill_len) {
        pos = tok_idx;
        cu_seqlens_k_base = 0;
    } else {
        int decode_idx = tok_idx - prefill_len;
        int stage_idx = decode_idx + 1;
        pos = prefill_len + decode_idx;
        cu_seqlens_k_base = stage_idx * prefill_len + (stage_idx * (stage_idx - 1)) / 2;
    }
    int varlen_write_idx = cu_seqlens_k_base + pos;

    // Load cos/sin (64 threads, each loads one element from each half)
    float cos0, sin0, cos1, sin1;
    if (tid < D_HALF) {
        cos0 = static_cast<float>(cos[tok_idx * stride_cos_l + tid]);
        sin0 = static_cast<float>(sin[tok_idx * stride_sin_l + tid]);
        cos1 = static_cast<float>(cos[tok_idx * stride_cos_l + D_HALF + tid]);
        sin1 = static_cast<float>(sin[tok_idx * stride_sin_l + D_HALF + tid]);
    }
    __syncthreads();

    // Q processing
    if (task_id < H_q && tid < D_HALF) {
        int h = task_id;
        float q0 = static_cast<float>(Q_new[tok_idx * stride_ql + h * stride_qh + tid]);
        float q1 = static_cast<float>(Q_new[tok_idx * stride_ql + h * stride_qh + D_HALF + tid]);

        Q_out[tok_idx * stride_qo_l + h * stride_qo_h + tid] = static_cast<scalar_t>(q0 * cos0 - q1 * sin0);
        Q_out[tok_idx * stride_qo_l + h * stride_qo_h + D_HALF + tid] = static_cast<scalar_t>(q1 * cos1 + q0 * sin1);
    }

    // KV processing
    int kv_offset = task_id - H_q;
    if (kv_offset >= 0 && kv_offset < 2 * H_kv) {
        int is_v = kv_offset / H_kv;
        int h = kv_offset % H_kv;

        if (is_v == 0 && tid < D_HALF) {
            // K with RoPE: each thread handles 2 elements
            float k0 = static_cast<float>(K_new[tok_idx * stride_kl + h * stride_kh + tid]);
            float k1 = static_cast<float>(K_new[tok_idx * stride_kl + h * stride_kh + D_HALF + tid]);

            scalar_t* k_out = kv_ring + 0 * stride_ring_k_dim
                + varlen_write_idx * stride_ring_seq + h * stride_ring_h;
            k_out[tid] = static_cast<scalar_t>(k0 * cos0 - k1 * sin0);
            k_out[D_HALF + tid] = static_cast<scalar_t>(k1 * cos1 + k0 * sin1);
        } else if (is_v == 1) {
            // V without RoPE: 128-bit vectorized copy (16 float4 = 128 bf16)
            const scalar_t* v_ptr = V_new + tok_idx * stride_vl + h * stride_vh;
            scalar_t* v_out = kv_ring + 1 * stride_ring_k_dim
                + varlen_write_idx * stride_ring_seq + h * stride_ring_h;
            if (tid < 16) {
                const float4* v4 = reinterpret_cast<const float4*>(v_ptr);
                float4* o4 = reinterpret_cast<float4*>(v_out);
                o4[tid] = v4[tid];
            }
        }
    }
}


// =============================================================================
// Fused Qwen mRoPE + Write KV
// Grid: (L_q, H_q + 2*H_kv).  64 threads per block.
// =============================================================================

template<typename scalar_t, int D, int D_HALF>
__global__ void fused_qwen_mrope_write_kv_cuda_kernel(
    const scalar_t* __restrict__ Q_new,
    const scalar_t* __restrict__ K_new,
    const scalar_t* __restrict__ V_new,
    scalar_t* __restrict__ Q_out,
    scalar_t* __restrict__ kv_ring,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    const int* __restrict__ dim_source_map,
    const int total_L_q,
    const int total_max_L,
    const int prefill_len,
    const int stride_ql, const int stride_qh,
    const int stride_kl, const int stride_kh,
    const int stride_vl, const int stride_vh,
    const int stride_qo_l, const int stride_qo_h,
    const int stride_ring_k_dim,
    const int stride_ring_seq,
    const int stride_ring_h,
    const int stride_cos_m, const int stride_cos_l,
    const int stride_sin_m, const int stride_sin_l,
    const int H_q,
    const int H_kv
) {
    const int tok_idx = blockIdx.x;
    const int task_id = blockIdx.y;
    const int tid = threadIdx.x;

    if (tok_idx >= total_L_q) return;

    int pos, cu_seqlens_k_base;
    if (tok_idx < prefill_len) {
        pos = tok_idx;
        cu_seqlens_k_base = 0;
    } else {
        int decode_idx = tok_idx - prefill_len;
        int stage_idx = decode_idx + 1;
        pos = prefill_len + decode_idx;
        cu_seqlens_k_base = stage_idx * prefill_len + (stage_idx * (stage_idx - 1)) / 2;
    }
    int varlen_write_idx = cu_seqlens_k_base + pos;

    float cos0, sin0, cos1, sin1;
    if (tid < D_HALF) {
        int src0 = dim_source_map[tid];
        int src1 = dim_source_map[D_HALF + tid];
        cos0 = static_cast<float>(cos[src0 * stride_cos_m + pos * stride_cos_l + tid]);
        sin0 = static_cast<float>(sin[src0 * stride_sin_m + pos * stride_sin_l + tid]);
        cos1 = static_cast<float>(cos[src1 * stride_cos_m + pos * stride_cos_l + D_HALF + tid]);
        sin1 = static_cast<float>(sin[src1 * stride_sin_m + pos * stride_sin_l + D_HALF + tid]);
    }
    __syncthreads();

    // Q
    if (task_id < H_q && tid < D_HALF) {
        int h = task_id;
        float q0 = static_cast<float>(Q_new[tok_idx * stride_ql + h * stride_qh + tid]);
        float q1 = static_cast<float>(Q_new[tok_idx * stride_ql + h * stride_qh + D_HALF + tid]);

        Q_out[tok_idx * stride_qo_l + h * stride_qo_h + tid] = static_cast<scalar_t>(q0 * cos0 - q1 * sin0);
        Q_out[tok_idx * stride_qo_l + h * stride_qo_h + D_HALF + tid] = static_cast<scalar_t>(q1 * cos1 + q0 * sin1);
    }

    // K/V
    int kv_offset = task_id - H_q;
    if (kv_offset >= 0 && kv_offset < 2 * H_kv) {
        int is_v = kv_offset / H_kv;
        int h = kv_offset % H_kv;

        if (is_v == 0 && tid < D_HALF) {
            float k0 = static_cast<float>(K_new[tok_idx * stride_kl + h * stride_kh + tid]);
            float k1 = static_cast<float>(K_new[tok_idx * stride_kl + h * stride_kh + D_HALF + tid]);

            scalar_t* k_out = kv_ring + 0 * stride_ring_k_dim
                + varlen_write_idx * stride_ring_seq + h * stride_ring_h;
            k_out[tid] = static_cast<scalar_t>(k0 * cos0 - k1 * sin0);
            k_out[D_HALF + tid] = static_cast<scalar_t>(k1 * cos1 + k0 * sin1);
        } else if (is_v == 1 && tid < D_HALF) {
            const scalar_t* v_ptr = V_new + tok_idx * stride_vl + h * stride_vh;
            scalar_t* v_out = kv_ring + 1 * stride_ring_k_dim
                + varlen_write_idx * stride_ring_seq + h * stride_ring_h;
            v_out[tid] = v_ptr[tid];
            v_out[D_HALF + tid] = v_ptr[D_HALF + tid];
        }
    }
}


// =============================================================================
// Shift KV Cache – In-place shift (varlen ring buffer roll).
// Grid: (L_max, H_kv, 2).  One block per (seq_offset, head, K/V).
// Block: 16 threads, each copies one float4 (8 bf16) → D=128 covered.
// Stages traversed in REVERSE order inside each block to preserve the
// read-after-write dependency: dst region of stage s coincides with the
// src region of stage s+1, so forward-order parallel stages would race.
// Parallelism now comes from the seq-offset grid dimension (matching the
// Triton reference) instead of a single (H_kv,2) grid with a serial loop.
// =============================================================================

template<typename scalar_t, int D>
__global__ void shift_varlen_kv_cache_cuda_kernel(
    scalar_t* __restrict__ kv_cache,
    const int stride_kv_k_dim,
    const int stride_kv_seq,
    const int stride_kv_head,
    const int H_kv,
    const int prefill_len,
    const int B_stages
) {
    const int l_offset = blockIdx.x;
    const int h = blockIdx.y;
    const int kv_dim = blockIdx.z;
    const int tid = threadIdx.x;

    if (h >= H_kv) return;

    constexpr int N_VEC = D / 8;  // float4 = 8 bf16; D=128 → 16
    if (tid >= N_VEC) return;

    scalar_t* base = kv_cache + kv_dim * stride_kv_k_dim + h * stride_kv_head;

    // Reverse-order stage traversal preserves the RAW dependency.
    for (int stage = 0; stage < B_stages - 1; stage++) {
        int src_stage = B_stages - 2 - stage;
        int dst_stage = src_stage + 1;
        int src_len = prefill_len + src_stage;

        if (l_offset >= src_len) continue;  // this stage has no seq at this offset

        int src_base_seq = src_stage * prefill_len + (src_stage * (src_stage - 1)) / 2;
        int dst_base_seq = dst_stage * prefill_len + (dst_stage * (dst_stage - 1)) / 2;

        int src_l = src_base_seq + l_offset;
        int dst_l = dst_base_seq + l_offset;

        // 128-bit vectorized copy: D contiguous bf16 = 16 float4
        const float4* src4 = reinterpret_cast<const float4*>(
            base + src_l * stride_kv_seq);
        float4* dst4 = reinterpret_cast<float4*>(
            base + dst_l * stride_kv_seq);
        dst4[tid] = src4[tid];
    }
}


// =============================================================================
// Shift KV Cache v2 – coalesced multi-head block.
// Grid: (L_max, 2).  Block 256 threads handles (l_offset, kv_dim) across
// ALL heads × D in one shot.  For a fixed seq the heads and D are contiguous
// in memory → fully coalesced 8 KB read + 8 KB write per stage (vs v1.1's
// scattered 16-byte transactions whose stride = H_kv*D between seqs).
// Stages still traversed in REVERSE order (RAW dependency preserved).
// =============================================================================

template<typename scalar_t, int D>
__global__ void shift_varlen_kv_cache_v2_kernel(
    scalar_t* __restrict__ kv_cache,
    const int stride_kv_k_dim,
    const int stride_kv_seq,
    const int H_kv,
    const int prefill_len,
    const int B_stages
) {
    const int l_offset = blockIdx.x;
    const int kv_dim = blockIdx.y;
    const int tid = threadIdx.x;

    scalar_t* base = kv_cache + kv_dim * stride_kv_k_dim;
    const int n_vec = (H_kv * D) / 8;  // float4 = 8 bf16; 32*128/8 = 512

    // Reverse-order stage traversal preserves the RAW dependency.
    for (int stage = 0; stage < B_stages - 1; stage++) {
        int src_stage = B_stages - 2 - stage;
        int dst_stage = src_stage + 1;
        int src_len = prefill_len + src_stage;

        if (l_offset >= src_len) continue;

        int src_base_seq = src_stage * prefill_len + (src_stage * (src_stage - 1)) / 2;
        int dst_base_seq = dst_stage * prefill_len + (dst_stage * (dst_stage - 1)) / 2;

        int src_l = src_base_seq + l_offset;
        int dst_l = dst_base_seq + l_offset;

        // Coalesced copy: H_kv*D contiguous bf16 per seq.
        const float4* src4 = reinterpret_cast<const float4*>(
            base + src_l * stride_kv_seq);
        float4* dst4 = reinterpret_cast<float4*>(
            base + dst_l * stride_kv_seq);
        for (int v = tid; v < n_vec; v += blockDim.x) {
            dst4[v] = src4[v];
        }
    }
}


// =============================================================================
// Shift KV Cache v3 – chunked multi-seq block.
// Grid: (ceil(L_max/CHUNK), 2).  Block 256 threads handles CHUNK consecutive
// seqs × all heads × D per stage → larger contiguous transactions (CHUNK*8KB)
// than v2's single-seq 8KB, trading grid count for transaction size.
// Stages still reverse-order (RAW dependency preserved).
// =============================================================================

template<typename scalar_t, int D, int CHUNK>
__global__ void shift_varlen_kv_cache_v3_kernel(
    scalar_t* __restrict__ kv_cache,
    const int stride_kv_k_dim,
    const int stride_kv_seq,
    const int H_kv,
    const int prefill_len,
    const int B_stages
) {
    const int chunk = blockIdx.x;
    const int kv_dim = blockIdx.y;
    const int tid = threadIdx.x;

    scalar_t* base = kv_cache + kv_dim * stride_kv_k_dim;
    const int n_vec_per_seq = (H_kv * D) / 8;

    for (int stage = 0; stage < B_stages - 1; stage++) {
        int src_stage = B_stages - 2 - stage;
        int dst_stage = src_stage + 1;
        int src_len = prefill_len + src_stage;

        int seq_start = chunk * CHUNK;
        if (seq_start >= src_len) continue;
        int seq_end = min(seq_start + CHUNK, src_len);
        int n_vec_total = (seq_end - seq_start) * n_vec_per_seq;

        int src_base_seq = src_stage * prefill_len + (src_stage * (src_stage - 1)) / 2;
        int dst_base_seq = dst_stage * prefill_len + (dst_stage * (dst_stage - 1)) / 2;

        // CHUNK consecutive seqs form one contiguous run (stride == data size).
        const float4* src4 = reinterpret_cast<const float4*>(
            base + (src_base_seq + seq_start) * stride_kv_seq);
        float4* dst4 = reinterpret_cast<float4*>(
            base + (dst_base_seq + seq_start) * stride_kv_seq);
        for (int v = tid; v < n_vec_total; v += blockDim.x) {
            dst4[v] = src4[v];
        }
    }
}


// -----------------------------------------------------------------------------
// Streaming 128-bit load/store hints (ld.cs / st.cs): bypass L2 allocation
// so writes don't trigger read-for-ownership and reads don't evict useful
// lines.  ShiftKV never re-reads src or dst within the kernel, so caching
// is pure overhead.
// -----------------------------------------------------------------------------

__device__ __forceinline__ float4 ldcs_f4(const float4* __restrict__ p) {
    float4 r;
    asm volatile("ld.cs.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ void stcs_f4(float4* __restrict__ p, float4 v) {
    asm volatile("st.cs.v4.b32 [%0], {%1,%2,%3,%4};"
        :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w));
}


// =============================================================================
// Shift KV Cache v4 – streaming (cache-bypassing) multi-head block.
// Same grid/layout as v2 but uses ld.cs/st.cs to avoid L2 allocation and
// write RFO.  Best when data >> L2 (B>=16).
// =============================================================================

template<typename scalar_t, int D>
__global__ void shift_varlen_kv_cache_v4_kernel(
    scalar_t* __restrict__ kv_cache,
    const int stride_kv_k_dim,
    const int stride_kv_seq,
    const int H_kv,
    const int prefill_len,
    const int B_stages
) {
    const int l_offset = blockIdx.x;
    const int kv_dim = blockIdx.y;
    const int tid = threadIdx.x;

    scalar_t* base = kv_cache + kv_dim * stride_kv_k_dim;
    const int n_vec = (H_kv * D) / 8;

    for (int stage = 0; stage < B_stages - 1; stage++) {
        int src_stage = B_stages - 2 - stage;
        int dst_stage = src_stage + 1;
        int src_len = prefill_len + src_stage;

        if (l_offset >= src_len) continue;

        int src_base_seq = src_stage * prefill_len + (src_stage * (src_stage - 1)) / 2;
        int dst_base_seq = dst_stage * prefill_len + (dst_stage * (dst_stage - 1)) / 2;

        int src_l = src_base_seq + l_offset;
        int dst_l = dst_base_seq + l_offset;

        const float4* src4 = reinterpret_cast<const float4*>(
            base + src_l * stride_kv_seq);
        float4* dst4 = reinterpret_cast<float4*>(
            base + dst_l * stride_kv_seq);
        for (int v = tid; v < n_vec; v += blockDim.x) {
            stcs_f4(&dst4[v], ldcs_f4(&src4[v]));
        }
    }
}


// =============================================================================
// Kernel Launchers (C++ wrappers called from Python)
// =============================================================================

// --- RMSNorm Prefill ---
torch::Tensor rmsnorm_fwd_cuda_prefill(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor out
) {
    int M = x.size(0);
    int N = x.size(1);
    float eps = 1e-6;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16, at::ScalarType::Half,
        x.scalar_type(), "rmsnorm_fwd_prefill", ([&] {
            using scalar_t = scalar_t;
            constexpr int BLOCK_DIM = 256;
            constexpr int ELEMS_PER_THREAD = 16;  // 256 * 16 = 4096
            dim3 grid(M);
            dim3 block(BLOCK_DIM);
            rmsnorm_fwd_prefill_kernel<scalar_t, BLOCK_DIM, ELEMS_PER_THREAD>
                <<<grid, block>>>(
                    x.data_ptr<scalar_t>(),
                    w.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    M, N, eps
                );
        })
    );
    return out;
}


// --- RMSNorm Decode (M small, one-warp CTA for correct reduction) ---
torch::Tensor rmsnorm_fwd_cuda_decode(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor out
) {
    int M = x.size(0);
    int N = x.size(1);
    float eps = 1e-6;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16, at::ScalarType::Half,
        x.scalar_type(), "rmsnorm_fwd_decode", ([&] {
            using scalar_t = scalar_t;
            constexpr int BLOCK_DIM = 32;  // single warp = correct shuffle reduction
            dim3 grid(M);
            dim3 block(BLOCK_DIM);
            rmsnorm_fwd_decode_kernel<scalar_t, BLOCK_DIM>
                <<<grid, block>>>(
                    x.data_ptr<scalar_t>(),
                    w.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    M, N, eps
                );
        })
    );
    return out;
}


// --- Auto-select RMSNorm ---
torch::Tensor rmsnorm_fwd_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor out
) {
    int M = x.size(0);
    if (M >= 64) {
        return rmsnorm_fwd_cuda_prefill(x, w, out);
    } else {
        return rmsnorm_fwd_cuda_decode(x, w, out);
    }
}


// --- Fused RoPE + Write KV ---
void fused_rope_write_kv_cuda(
    torch::Tensor Q_new,       // (L_q, H_q, D)
    torch::Tensor K_new,       // (L_q, H_kv, D)
    torch::Tensor V_new,       // (L_q, H_kv, D)
    torch::Tensor Q_out,       // (L_q, H_q, D)
    torch::Tensor kv_ring,     // (2, total_L_kv, H_kv, D)
    torch::Tensor cos,         // (L_q, D)
    torch::Tensor sin,         // (L_q, D)
    int prefill_len
) {
    int L_q = Q_new.size(0);
    int H_q = Q_new.size(1);
    int H_kv = K_new.size(1);
    int D = Q_new.size(2);
    int total_L_kv = kv_ring.size(1);
    int D_half = D / 2;

    TORCH_CHECK(D == 128, "Only D=128 supported");

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16, at::ScalarType::Half,
        Q_new.scalar_type(), "fused_rope_write_kv_cuda", ([&] {
            using scalar_t = scalar_t;
            dim3 grid(L_q, H_q + 2 * H_kv);
            dim3 block(D_half);  // 64 threads
            fused_rope_write_kv_cuda_kernel<scalar_t, 128, 64>
                <<<grid, block>>>(
                    Q_new.data_ptr<scalar_t>(),
                    K_new.data_ptr<scalar_t>(),
                    V_new.data_ptr<scalar_t>(),
                    Q_out.data_ptr<scalar_t>(),
                    kv_ring.data_ptr<scalar_t>(),
                    cos.data_ptr<scalar_t>(),
                    sin.data_ptr<scalar_t>(),
                    L_q, total_L_kv, prefill_len,
                    Q_new.stride(0), Q_new.stride(1),
                    K_new.stride(0), K_new.stride(1),
                    V_new.stride(0), V_new.stride(1),
                    Q_out.stride(0), Q_out.stride(1),
                    kv_ring.stride(0), kv_ring.stride(1), kv_ring.stride(2),
                    cos.stride(0), sin.stride(0),
                    H_q, H_kv
                );
        })
    );
}


// --- Fused Qwen mRoPE + Write KV ---
void fused_qwen_mrope_write_kv_cuda(
    torch::Tensor Q_new,            // (L_q, H_q, D)
    torch::Tensor K_new,            // (L_q, H_kv, D)
    torch::Tensor V_new,            // (L_q, H_kv, D)
    torch::Tensor Q_out,            // (L_q, H_q, D)
    torch::Tensor kv_ring,          // (2, total_L_kv, H_kv, D)
    torch::Tensor cos,              // (3, L_q, D)
    torch::Tensor sin,              // (3, L_q, D)
    torch::Tensor dim_source_map,   // (D,)
    int prefill_len
) {
    int L_q = Q_new.size(0);
    int H_q = Q_new.size(1);
    int H_kv = K_new.size(1);
    int D = Q_new.size(2);
    int total_L_kv = kv_ring.size(1);
    int D_half = D / 2;

    TORCH_CHECK(D == 128, "Only D=128 supported");

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16, at::ScalarType::Half,
        Q_new.scalar_type(), "fused_qwen_mrope_write_kv_cuda", ([&] {
            using scalar_t = scalar_t;
            dim3 grid(L_q, H_q + 2 * H_kv);
            dim3 block(D_half);  // 64 threads per block
            fused_qwen_mrope_write_kv_cuda_kernel<scalar_t, 128, 64>
                <<<grid, block>>>(
                    Q_new.data_ptr<scalar_t>(),
                    K_new.data_ptr<scalar_t>(),
                    V_new.data_ptr<scalar_t>(),
                    Q_out.data_ptr<scalar_t>(),
                    kv_ring.data_ptr<scalar_t>(),
                    cos.data_ptr<scalar_t>(),
                    sin.data_ptr<scalar_t>(),
                    dim_source_map.data_ptr<int>(),
                    L_q, total_L_kv, prefill_len,
                    Q_new.stride(0), Q_new.stride(1),
                    K_new.stride(0), K_new.stride(1),
                    V_new.stride(0), V_new.stride(1),
                    Q_out.stride(0), Q_out.stride(1),
                    kv_ring.stride(0), kv_ring.stride(1), kv_ring.stride(2),
                    cos.stride(0), cos.stride(1),
                    sin.stride(0), sin.stride(1),
                    H_q, H_kv
                );
        })
    );
}


// --- Shift KV Cache ---
void shift_varlen_kv_cache_cuda(
    torch::Tensor kv_cache,  // (2, total_L_kv, H_kv, D)
    int B_stages,
    int prefill_len
) {
    int H_kv = kv_cache.size(2);
    int D = kv_cache.size(3);
    int L_max = prefill_len + B_stages - 1;

    TORCH_CHECK(D == 128, "Only D=128 supported");

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16, at::ScalarType::Half,
        kv_cache.scalar_type(), "shift_varlen_kv_cache_cuda", ([&] {
            using scalar_t = scalar_t;
            // v2 (selected): coalesced multi-head block.  v3 (chunked) and
            // v4 (streaming ld.cs/st.cs) were tried but gave no gain at
            // B>=16 — the floor there is DRAM row-switching on the varlen
            // multi-region access pattern, not cache/merge/RFO.
            dim3 grid(L_max, 2);
            dim3 block(256);
            shift_varlen_kv_cache_v2_kernel<scalar_t, 128>
                <<<grid, block>>>(
                    kv_cache.data_ptr<scalar_t>(),
                    kv_cache.stride(0),
                    kv_cache.stride(1),
                    H_kv, prefill_len, B_stages
                );
        })
    );
}
"""


# ---------------------------------------------------------------------------
# JIT Compilation
# ---------------------------------------------------------------------------

_cuda_module = None


def _get_cuda_module():
    """Lazy JIT compile the CUDA module on first use."""
    global _cuda_module
    if _cuda_module is not None:
        return _cuda_module

    cpp_source = r"""
    #include <torch/extension.h>

    torch::Tensor rmsnorm_fwd_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor out);
    torch::Tensor rmsnorm_fwd_cuda_prefill(torch::Tensor x, torch::Tensor w, torch::Tensor out);
    torch::Tensor rmsnorm_fwd_cuda_decode(torch::Tensor x, torch::Tensor w, torch::Tensor out);

    void fused_rope_write_kv_cuda(
        torch::Tensor Q_new, torch::Tensor K_new, torch::Tensor V_new,
        torch::Tensor Q_out, torch::Tensor kv_ring,
        torch::Tensor cos, torch::Tensor sin,
        int prefill_len);

    void fused_qwen_mrope_write_kv_cuda(
        torch::Tensor Q_new, torch::Tensor K_new, torch::Tensor V_new,
        torch::Tensor Q_out, torch::Tensor kv_ring,
        torch::Tensor cos, torch::Tensor sin,
        torch::Tensor dim_source_map,
        int prefill_len);

    void shift_varlen_kv_cache_cuda(
        torch::Tensor kv_cache, int B_stages, int prefill_len);
    """

    # Detect GPU architecture for optimal code generation
    try:
        major = torch.cuda.get_device_capability(0)[0]
        minor = torch.cuda.get_device_capability(0)[1]
        arch_flag = f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
    except Exception:
        arch_flag = "-gencode=arch=compute_110,code=sm_110"  # fallback: Thor (Blackwell)

    _cuda_module = load_inline(
        name="actionflow_cuda_ops_v14",
        cpp_sources=[cpp_source],
        cuda_sources=[CUDA_SOURCE],
        functions=[
            "rmsnorm_fwd_cuda",
            "rmsnorm_fwd_cuda_prefill",
            "rmsnorm_fwd_cuda_decode",
            "fused_rope_write_kv_cuda",
            "fused_qwen_mrope_write_kv_cuda",
            "shift_varlen_kv_cache_cuda",
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math", arch_flag],
        verbose=False,
    )
    return _cuda_module


# ---------------------------------------------------------------------------
# Python Wrappers (same API as Triton wrappers in ops.py)
# ---------------------------------------------------------------------------

def cuda_rmsnorm_fwd(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor):
    """CUDA RMSNorm forward (auto-selects prefill vs decode kernel)."""
    if not x.is_contiguous():
        x = x.contiguous()
    if not w.is_contiguous():
        w = w.contiguous()
    return _get_cuda_module().rmsnorm_fwd_cuda(x, w, out)


def cuda_rmsnorm_fwd_prefill(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor):
    """CUDA RMSNorm forward – prefill-optimized (M >= 64)."""
    if not x.is_contiguous():
        x = x.contiguous()
    if not w.is_contiguous():
        w = w.contiguous()
    return _get_cuda_module().rmsnorm_fwd_cuda_prefill(x, w, out)


def cuda_rmsnorm_fwd_decode(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor):
    """CUDA RMSNorm forward – decode-optimized (M < 64)."""
    if not x.is_contiguous():
        x = x.contiguous()
    if not w.is_contiguous():
        w = w.contiguous()
    return _get_cuda_module().rmsnorm_fwd_cuda_decode(x, w, out)


def cuda_fused_rope_write_kv(
    Q_new: torch.Tensor,
    K_new: torch.Tensor,
    V_new: torch.Tensor,
    Q_out: torch.Tensor,
    kv_ring: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    prefill_len: int,
):
    """CUDA Fused RoPE + Write KV (OpenVLA).
    
    Accepts Q_new/K_new/V_new as (B, L_q, H, D) or (L_q, H, D).
    Internal kernels always work on (L_q, H, D).
    """
    # Squeeze batch dim if present (matching Triton wrapper convention)
    if Q_new.dim() == 4:
        Q_new = Q_new.squeeze(0)
        K_new = K_new.squeeze(0)
        V_new = V_new.squeeze(0)
    return _get_cuda_module().fused_rope_write_kv_cuda(
        Q_new.contiguous(),
        K_new.contiguous(),
        V_new.contiguous(),
        Q_out.contiguous(),
        kv_ring.contiguous(),
        cos.contiguous(),
        sin.contiguous(),
        prefill_len,
    )


def cuda_fused_qwen_mrope_write_kv(
    Q_new: torch.Tensor,
    K_new: torch.Tensor,
    V_new: torch.Tensor,
    Q_out: torch.Tensor,
    kv_ring: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    dim_source_map: torch.Tensor,
    prefill_len: int,
):
    """CUDA Fused Qwen mRoPE + Write KV.
    
    Accepts Q_new/K_new/V_new as (B, L_q, H, D) or (L_q, H, D).
    """
    if Q_new.dim() == 4:
        Q_new = Q_new.squeeze(0)
        K_new = K_new.squeeze(0)
        V_new = V_new.squeeze(0)
    return _get_cuda_module().fused_qwen_mrope_write_kv_cuda(
        Q_new.contiguous(),
        K_new.contiguous(),
        V_new.contiguous(),
        Q_out.contiguous(),
        kv_ring.contiguous(),
        cos.contiguous(),
        sin.contiguous(),
        dim_source_map.contiguous(),
        prefill_len,
    )


def cuda_shift_varlen_kv_cache(
    kv_cache: torch.Tensor,
    B_stages: int,
    prefill_len: int,
):
    """CUDA in-place Shift VarLen KV Cache."""
    return _get_cuda_module().shift_varlen_kv_cache_cuda(
        kv_cache.contiguous(),
        B_stages,
        prefill_len,
    )
