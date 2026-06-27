from .ops import (
    TritonLlamaRMSNorm,
    fused_rope_write_kv_wrapper,
    fused_qwen_mrope_write_kv_wrapper,
    shift_varlen_kv_cache_wrapper
)

# CUDA-optimized ops (JIT-compiled on first import)
try:
    from .cuda_ops import (
        cuda_rmsnorm_fwd,
        cuda_rmsnorm_fwd_prefill,
        cuda_rmsnorm_fwd_decode,
        cuda_fused_rope_write_kv,
        cuda_fused_qwen_mrope_write_kv,
        cuda_shift_varlen_kv_cache,
    )
    HAS_CUDA_OPS = True
except Exception:
    HAS_CUDA_OPS = False

__all__ = [
    "TritonLlamaRMSNorm",
    "fused_rope_write_kv_wrapper",
    "fused_qwen_mrope_write_kv_wrapper",
    "shift_varlen_kv_cache_wrapper",
    "HAS_CUDA_OPS",
]