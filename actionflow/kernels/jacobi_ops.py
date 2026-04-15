"""
Jacobi pipeline KV layout + fused RoPE/write + shift.

Used by ``jacobi_layers`` and :class:`actionflow.modeling.jacobi_pipeline.ActionFlowJacobiPipeline`.
``num_stages`` equals ``max_depth_K`` (full pipeline); KV ring length uses
``jacobi_total_kv_elements(num_stages, ...)``.

Aligned with memory notes: persistent prefix prefill_len; per-slot Jacobi K length prefill_len+jacobi_tokens.
"""

from __future__ import annotations

import triton
import triton.language as tl
import torch


def jacobi_stage_kv_lens(num_stages: int, prefill_len: int, jacobi_tokens: int) -> list[int]:
    if num_stages < 1:
        return []
    return [prefill_len] + [prefill_len + jacobi_tokens] * (num_stages - 1)


def jacobi_total_kv_elements(num_stages: int, prefill_len: int, jacobi_tokens: int) -> int:
    return sum(jacobi_stage_kv_lens(num_stages, prefill_len, jacobi_tokens))


def build_jacobi_cu_seqlens(
    num_stages: int,
    prefill_len: int,
    jacobi_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """cu_seqlens_q / cu_seqlens_k for FlashAttention varlen (same segment count as num_stages)."""
    q_lens = [prefill_len] + [jacobi_tokens] * (num_stages - 1)
    k_lens = jacobi_stage_kv_lens(num_stages, prefill_len, jacobi_tokens)

    # flash_attn 要求 cu_seqlens_* 为 int32；torch.cat/cumsum 在部分版本上会升到 int64
    q_lens_t = torch.tensor(q_lens, device=device, dtype=torch.int32)
    k_lens_t = torch.tensor(k_lens, device=device, dtype=torch.int32)
    z = torch.zeros(1, device=device, dtype=torch.int32)
    cu_q = torch.cat([z, torch.cumsum(q_lens_t, dim=0)], dim=0).to(torch.int32).contiguous()
    cu_k = torch.cat([z, torch.cumsum(k_lens_t, dim=0)], dim=0).to(torch.int32).contiguous()
    max_seqlen_q = max(q_lens)
    max_seqlen_k = max(k_lens)
    return cu_q, cu_k, int(max_seqlen_q), int(max_seqlen_k)


@triton.jit
def _jacobi_cu_seqlens_k_base(stage_idx, prefill_len, kv_chunk_len):
    """Base offset in the packed K buffer for stage_idx (0-based)."""
    # stage 0: 0
    # stage s>=1: prefill_len + (s-1)*kv_chunk_len
    return tl.where(stage_idx == 0, 0, prefill_len + (stage_idx - 1) * kv_chunk_len)


@triton.jit
def fused_rope_write_kv_jacobi_kernel(
    Q_new_ptr,
    K_new_ptr,
    V_new_ptr,
    Q_out_ptr,
    kv_ring_ptr,
    cos_ptr,
    sin_ptr,
    total_L_q: tl.int32,
    prefill_len: tl.int32,
    jacobi_tokens: tl.int32,
    num_stages: tl.int32,
    D: tl.int32,
    stride_ql: tl.int32,
    stride_qh: tl.int32,
    stride_kl: tl.int32,
    stride_kh: tl.int32,
    stride_vl: tl.int32,
    stride_vh: tl.int32,
    stride_qo_l: tl.int32,
    stride_qo_h: tl.int32,
    stride_ring_k_dim: tl.int32,
    stride_ring_seq: tl.int32,
    stride_ring_h: tl.int32,
    stride_cos_l: tl.int32,
    stride_sin_l: tl.int32,
    BLOCK_D: tl.constexpr,
    D_half: tl.constexpr,
    H_q: tl.constexpr,
    H_kv: tl.constexpr,
):
    tok_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    if h_idx >= H_q and h_idx >= H_kv:
        return

    kv_chunk_len = prefill_len + jacobi_tokens

    # RoPE position + ring write index
    pos = tok_idx.to(tl.int32)
    varlen_write_idx = tok_idx.to(tl.int32)

    if tok_idx < prefill_len:
        pos = tok_idx
        varlen_write_idx = tok_idx
    else:
        rel = tok_idx - prefill_len
        sub_stage = rel // jacobi_tokens
        local_tok = rel - sub_stage * jacobi_tokens
        stage_idx = sub_stage + 1
        pos = prefill_len + local_tok
        cu_base = _jacobi_cu_seqlens_k_base(stage_idx, prefill_len, kv_chunk_len)
        varlen_write_idx = cu_base + prefill_len + local_tok

    offsets_d_half = tl.arange(0, D_half)
    offsets_d = tl.arange(0, BLOCK_D)

    cos0_ptr = cos_ptr + pos * stride_cos_l + offsets_d_half
    sin0_ptr = sin_ptr + pos * stride_sin_l + offsets_d_half
    cos1_ptr = cos_ptr + pos * stride_cos_l + D_half + offsets_d_half
    sin1_ptr = sin_ptr + pos * stride_sin_l + D_half + offsets_d_half

    cos0 = tl.load(cos0_ptr, mask=offsets_d_half < D_half)
    sin0 = tl.load(sin0_ptr, mask=offsets_d_half < D_half)
    cos1 = tl.load(cos1_ptr, mask=offsets_d_half < D_half)
    sin1 = tl.load(sin1_ptr, mask=offsets_d_half < D_half)

    if h_idx < H_q:
        q_ptr = Q_new_ptr + tok_idx * stride_ql + h_idx * stride_qh
        q_out_ptr = Q_out_ptr + tok_idx * stride_qo_l + h_idx * stride_qo_h

        q0 = tl.load(q_ptr + offsets_d_half, mask=offsets_d_half < D_half)
        q1 = tl.load(q_ptr + D_half + offsets_d_half, mask=offsets_d_half < D_half)

        out0 = q0 * cos0 - q1 * sin0
        out1 = q1 * cos1 + q0 * sin0

        tl.store(q_out_ptr + offsets_d_half, out0, mask=offsets_d_half < D_half)
        tl.store(q_out_ptr + D_half + offsets_d_half, out1, mask=offsets_d_half < D_half)

    if h_idx < H_kv:
        k_ptr = K_new_ptr + tok_idx * stride_kl + h_idx * stride_kh
        k_out_ptr = (
            kv_ring_ptr
            + 0 * stride_ring_k_dim
            + varlen_write_idx * stride_ring_seq
            + h_idx * stride_ring_h
        )

        k0 = tl.load(k_ptr + offsets_d_half, mask=offsets_d_half < D_half)
        k1 = tl.load(k_ptr + D_half + offsets_d_half, mask=offsets_d_half < D_half)

        k_rot0 = k0 * cos0 - k1 * sin0
        k_rot1 = k1 * cos1 + k0 * sin0

        tl.store(k_out_ptr + offsets_d_half, k_rot0, mask=offsets_d_half < D_half)
        tl.store(k_out_ptr + D_half + offsets_d_half, k_rot1, mask=offsets_d_half < D_half)

        v_ptr = V_new_ptr + tok_idx * stride_vl + h_idx * stride_vh
        v_out_ptr = (
            kv_ring_ptr
            + 1 * stride_ring_k_dim
            + varlen_write_idx * stride_ring_seq
            + h_idx * stride_ring_h
        )

        v = tl.load(v_ptr + offsets_d, mask=offsets_d < D)
        tl.store(v_out_ptr + offsets_d, v, mask=offsets_d < D)


def fused_rope_write_kv_jacobi_wrapper(
    Q_new: torch.Tensor,
    K_new: torch.Tensor,
    V_new: torch.Tensor,
    kv_ring_buffer: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    prefill_len: int,
    jacobi_tokens: int,
    num_stages: int,
    q_varlen: torch.Tensor,
) -> None:
    B, L_q, H_q, D = Q_new.shape
    assert B == 1
    _, _, H_kv, D2 = K_new.shape
    assert D == D2 and D % 2 == 0

    total_L_kv = jacobi_total_kv_elements(num_stages, prefill_len, jacobi_tokens)
    _, total_L_buf, H_kv2, D3 = kv_ring_buffer.shape
    assert kv_ring_buffer.shape[0] == 2 and H_kv == H_kv2 and D == D3
    assert total_L_kv == total_L_buf, f"KV buffer length mismatch: need {total_L_kv}, got {total_L_buf}"

    D_half = D // 2
    Q_new_in = Q_new.squeeze(0).contiguous()
    K_new_in = K_new.squeeze(0).contiguous()
    V_new_in = V_new.squeeze(0).contiguous()

    assert q_varlen.is_contiguous() and kv_ring_buffer.is_contiguous()
    assert cos.is_contiguous() and sin.is_contiguous()

    grid = (L_q, max(H_q, H_kv))

    fused_rope_write_kv_jacobi_kernel[grid](
        Q_new_ptr=Q_new_in,
        K_new_ptr=K_new_in,
        V_new_ptr=V_new_in,
        Q_out_ptr=q_varlen,
        kv_ring_ptr=kv_ring_buffer,
        cos_ptr=cos,
        sin_ptr=sin,
        total_L_q=L_q,
        prefill_len=prefill_len,
        jacobi_tokens=jacobi_tokens,
        num_stages=num_stages,
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


def shift_jacobi_kv_cache_torch(kv_cache: torch.Tensor, num_stages: int, prefill_len: int, jacobi_tokens: int) -> torch.Tensor:
    """
    Roll pipeline slots: slot s <- slot s+1. Moving into stage 0 truncates to prefill_len.
    kv_cache: (2, total_L_kv, H_kv, D)
    """
    if num_stages <= 1:
        return kv_cache

    k_lens = jacobi_stage_kv_lens(num_stages, prefill_len, jacobi_tokens)
    bases: list[int] = []
    acc = 0
    for L in k_lens:
        bases.append(acc)
        acc += L
    assert acc == kv_cache.shape[1]

    # High index to low to avoid clobber
    for s in range(num_stages - 2, -1, -1):
        src_start = bases[s + 1]
        dst_start = bases[s]
        n = min(k_lens[s + 1], k_lens[s])
        kv_cache[:, dst_start : dst_start + n].copy_(kv_cache[:, src_start : src_start + n])
    return kv_cache
