import triton
import torch
import torch.nn as nn
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(
    X_ptr,
    W_ptr,
    Out_ptr,
    X_stride_m,
    X_stride_n,
    Out_stride_m,
    Out_stride_n,
    M,
    N,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Triton kernel for RMSNorm forward pass.
    Normalizes each row of the input using root mean square, then applies learnable weights.

    Args:
        X_ptr: Pointer to input tensor (M, N)
        W_ptr: Pointer to weight vector (N,)
        Out_ptr: Pointer to output tensor (M, N)
        X_stride_m, X_stride_n: Strides for input
        Out_stride_m, Out_stride_n: Strides for output
        M: Number of rows (batch_size * seq_len)
        N: Hidden dimension size
        eps: Small epsilon value for numerical stability
        BLOCK_SIZE_N: Block size along the hidden dimension (power of 2)
    """
    pid_m = tl.program_id(axis=0)  # Row index

    cols = tl.arange(0, BLOCK_SIZE_N)
    mask_n = cols < N

    # Load input row
    x_row_ptr = X_ptr + pid_m * X_stride_m + cols
    x_row = tl.load(x_row_ptr, mask=mask_n, other=0.0).to(tl.float32)

    # Compute variance (mean of squares)
    var = tl.sum(x_row * x_row, axis=0) / N

    # Compute reciprocal square root
    rsqrt_var = tl.rsqrt(var + eps)

    # Load weights
    w_ptr = W_ptr + cols
    w = tl.load(w_ptr, mask=mask_n, other=1.0)

    # Normalize and apply weight
    x_normed_weighted = (x_row * rsqrt_var) * w

    # Cast back to original dtype
    output = x_normed_weighted.to(tl.bfloat16) if X_ptr.dtype.element_ty == tl.bfloat16 else x_normed_weighted.to(tl.float16)

    # Store result
    out_ptr = Out_ptr + pid_m * Out_stride_m + cols
    tl.store(out_ptr, output, mask=mask_n)


class TritonLlamaRMSNorm(nn.Module):
    """
    Llama-style RMSNorm implemented with Triton for improved performance.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        input_shape = hidden_states.shape
        hidden_size = input_shape[-1]
        hidden_states_2d = hidden_states.view(-1, hidden_size)
        M, N = hidden_states_2d.shape

        output = torch.empty_like(hidden_states_2d)

        grid = (M,)
        BLOCK_SIZE_N = triton.next_power_of_2(N)

        rmsnorm_fwd_kernel[grid](
            hidden_states_2d,
            self.weight,
            output,
            hidden_states_2d.stride(0),
            hidden_states_2d.stride(1),
            output.stride(0),
            output.stride(1),
            M,
            N,
            self.variance_epsilon,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        return output.view(input_shape)

    def extra_repr(self) -> str:
        return f"weight_shape={tuple(self.weight.shape)}, eps={self.variance_epsilon} (Triton)"


# ==============================================================================
# Fused RoPE + Write KV into Varlen Ring Buffer
# ==============================================================================


@triton.jit
def fused_rope_write_kv_kernel(
    Q_new_ptr,
    K_new_ptr,
    V_new_ptr,
    Q_out_ptr,
    kv_ring_ptr,
    cos_ptr,
    sin_ptr,
    total_L_q: tl.int32,
    total_max_L: tl.int32,
    prefill_len: tl.int32,
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
    """
    Triton kernel that applies rotary embedding (RoPE) to Q/K and writes K/V into a ring buffer.
    Designed for variable-length sequence handling in streaming/autoregressive generation.

    Grid: (L_q, max(H_q, H_kv))
    Each program processes one token-head pair.
    """
    tok_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    if h_idx >= H_q and h_idx >= H_kv:
        return

    # Compute position and write index in ring buffer
    pos = 0
    cu_seqlens_k_base = 0

    if tok_idx < prefill_len:
        pos = tok_idx
        cu_seqlens_k_base = 0
    else:
        decode_idx = tok_idx - prefill_len
        stage_idx = decode_idx + 1
        pos = prefill_len + decode_idx
        cu_seqlens_k_base = stage_idx * prefill_len + (stage_idx * (stage_idx - 1) // 2)

    varlen_write_idx = cu_seqlens_k_base + pos

    offsets_d_half = tl.arange(0, D_half)
    offsets_d = tl.arange(0, BLOCK_D)

    # Load cos/sin for current position
    cos0_ptr = cos_ptr + pos * stride_cos_l + offsets_d_half
    sin0_ptr = sin_ptr + pos * stride_sin_l + offsets_d_half
    cos1_ptr = cos_ptr + pos * stride_cos_l + D_half + offsets_d_half
    sin1_ptr = sin_ptr + pos * stride_sin_l + D_half + offsets_d_half

    cos0 = tl.load(cos0_ptr, mask=offsets_d_half < D_half)
    sin0 = tl.load(sin0_ptr, mask=offsets_d_half < D_half)
    cos1 = tl.load(cos1_ptr, mask=offsets_d_half < D_half)
    sin1 = tl.load(sin1_ptr, mask=offsets_d_half < D_half)

    # Apply RoPE to Q if head exists
    if h_idx < H_q:
        q_ptr = Q_new_ptr + tok_idx * stride_ql + h_idx * stride_qh
        q_out_ptr = Q_out_ptr + tok_idx * stride_qo_l + h_idx * stride_qo_h

        q0 = tl.load(q_ptr + offsets_d_half, mask=offsets_d_half < D_half)
        q1 = tl.load(q_ptr + D_half + offsets_d_half, mask=offsets_d_half < D_half)

        out0 = q0 * cos0 - q1 * sin0
        out1 = q1 * cos1 + q0 * sin1

        tl.store(q_out_ptr + offsets_d_half, out0, mask=offsets_d_half < D_half)
        tl.store(q_out_ptr + D_half + offsets_d_half, out1, mask=offsets_d_half < D_half)

    # Process K and V
    if h_idx < H_kv:
        # K with RoPE
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
        k_rot1 = k1 * cos1 + k0 * sin1

        tl.store(k_out_ptr + offsets_d_half, k_rot0, mask=offsets_d_half < D_half)
        tl.store(k_out_ptr + D_half + offsets_d_half, k_rot1, mask=offsets_d_half < D_half)

        # V without RoPE
        v_ptr = V_new_ptr + tok_idx * stride_vl + h_idx * stride_vh
        v_out_ptr = (
            kv_ring_ptr
            + 1 * stride_ring_k_dim
            + varlen_write_idx * stride_ring_seq
            + h_idx * stride_ring_h
        )

        v = tl.load(v_ptr + offsets_d, mask=offsets_d < D)
        tl.store(v_out_ptr + offsets_d, v, mask=offsets_d < D)


def fused_rope_write_kv_wrapper(
    Q_new: torch.Tensor,
    K_new: torch.Tensor,
    V_new: torch.Tensor,
    kv_ring_buffer: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    prefill_len: int,
    q_varlen: torch.Tensor,
):
    """
    Python wrapper for fused RoPE and KV write kernel.

    Args:
        Q_new: New query states (1, L_q, H_q, D)
        K_new: New key states (1, L_q, H_kv, D)
        V_new: New value states (1, L_q, H_kv, D)
        kv_ring_buffer: Ring buffer storing historical K/V (2, total_L_kv, H_kv, D)
        cos, sin: Precomputed rotary embeddings (L or max_pos, D)
        prefill_len: Length of initial prefill sequence
        q_varlen: Output buffer for rotated queries (L_q, H_q, D), must be contiguous
    """
    B, L_q, H_q, D = Q_new.shape
    assert B == 1, "Batch size must be 1"
    _, _, H_kv, D2 = K_new.shape
    assert D == D2 and D % 2 == 0, "D must match and be even"

    D_half = D // 2
    decode_steps = L_q - prefill_len
    B_stages = decode_steps + 1
    total_L_kv_calc = B_stages * prefill_len + (B_stages - 1) * B_stages // 2

    _, total_L_kv_buffer, H_kv2, D3 = kv_ring_buffer.shape
    assert kv_ring_buffer.shape[0] == 2 and H_kv == H_kv2 and D == D3
    assert total_L_kv_calc == total_L_kv_buffer, f"KV buffer length mismatch: expected {total_L_kv_calc}, got {total_L_kv_buffer}"

    # Squeeze and ensure contiguous layout
    Q_new_in = Q_new.squeeze(0).contiguous()
    K_new_in = K_new.squeeze(0).contiguous()
    V_new_in = V_new.squeeze(0).contiguous()

    assert q_varlen.is_contiguous() and kv_ring_buffer.is_contiguous()
    assert cos.is_contiguous() and sin.is_contiguous()
    assert cos.shape[1] == D

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


# ==============================================================================
# In-place Shift of Varlen KV Cache (Ring Buffer Roll)
# ==============================================================================


@triton.jit
def get_cu_seqlens_k_base_kernel(stage_idx, PREFILL_LEN):
    """Helper function to compute base offset for cumulative sequence lengths."""
    return stage_idx * PREFILL_LEN + ((stage_idx * (stage_idx - 1)) >> 1)


@triton.jit
def shift_varlen_kv_cache_kernel(
    KV_Cache_ptr,
    stride_kv_k_dim,
    stride_kv_seq,
    stride_kv_head,
    H_kv,
    D,
    PREFILL_LEN,
    B_STAGES: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Triton kernel to perform in-place rolling of a varlen KV cache used as a ring buffer.
    Shifts all stages backward by one position (oldest dropped, newest retained).

    Grid: (L_max, H_kv, 2)
    """
    pid_l_offset = tl.program_id(0)  # 0 <= l < L_max
    pid_h = tl.program_id(1)          # Head index
    pid_k = tl.program_id(2)          # 0 for K, 1 for V

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    # Traverse from last valid stage to first
    for task_id in tl.static_range(B_STAGES - 1):
        src_stage_idx = (B_STAGES - 2) - task_id
        dst_stage_idx = (B_STAGES - 1) - task_id

        src_len = PREFILL_LEN + src_stage_idx

        if pid_l_offset < src_len:
            src_base = get_cu_seqlens_k_base_kernel(src_stage_idx, PREFILL_LEN)
            dst_base = get_cu_seqlens_k_base_kernel(dst_stage_idx, PREFILL_LEN)

            src_l = src_base + pid_l_offset
            dst_l = dst_base + pid_l_offset

            src_ptr = (
                KV_Cache_ptr
                + pid_k * stride_kv_k_dim
                + src_l * stride_kv_seq
                + pid_h * stride_kv_head
                + d_offsets
            )
            dst_ptr = (
                KV_Cache_ptr
                + pid_k * stride_kv_k_dim
                + dst_l * stride_kv_seq
                + pid_h * stride_kv_head
                + d_offsets
            )

            row_data = tl.load(src_ptr, mask=d_mask, other=0.0)
            tl.store(dst_ptr, row_data, mask=d_mask)


def shift_varlen_kv_cache_wrapper(kv_cache: torch.Tensor, B_stages: int, prefill_len: int):
    """
    Wrapper to shift varlen KV cache in-place during autoregressive generation.

    Args:
        kv_cache: Tensor of shape (2, total_L_kv, H_kv, D)
        B_stages: Number of stages currently stored (including prefill)
        prefill_len: Initial context length

    Returns:
        Modified kv_cache in-place.
    """
    if B_stages <= 1:
        return kv_cache

    assert kv_cache.dim() == 4
    K_V_dim, TOTAL_L_KV, H_kv, D = kv_cache.shape
    assert K_V_dim == 2

    kv_cache = kv_cache.contiguous()

    L_max = prefill_len + B_stages - 1
    grid = (L_max, H_kv, K_V_dim)

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
