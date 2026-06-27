"""
Profile L2 read/write lookup hit/miss for the v1.2 ShiftKV kernel on 3 shapes.

Launched under ncu to capture, per kernel invocation:
  - lts__t_sectors_op_read_lookup_hit / _miss   (DRAM read proxy = read_miss * 32 B)
  - lts__t_sectors_op_write_lookup_hit / _miss  (write_hit = dirty-resident, no DRAM traffic;
                                                 write_miss = allocate; dirty-victim writeback proxy)

Each shape launches the kernel EXACTLY ONCE so ncu captures 3 launches total,
distinguishable by grid size (L_max = prefill + B - 1): 283 / 292 / 308.

Run:
  docker exec dyt_af_ops bash -lc "ncu --target-processes all \
    --kernel-name regex:shift_varlen --launch-skip 0 --launch-count 3 \
    --metrics lts__t_sectors_op_read_lookup_hit,lts__t_sectors_op_read_lookup_miss,\
lts__t_sectors_op_write_lookup_hit,lts__t_sectors_op_write_lookup_miss \
    --csv --units base \
    python actionflow/kernels/tests/profile_l2.py"
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
from actionflow.kernels.cuda_ops import cuda_shift_varlen_kv_cache

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
H_kv = 32
D = 128
PREFILL = 277
SHAPES = [7, 16, 32]

# buffer to keep L2 warm state realistic: one fresh buffer per shape
for B in SHAPES:
    total_L_kv = B * PREFILL + (B - 1) * B // 2
    kv = torch.randn(2, total_L_kv, H_kv, D, dtype=DTYPE, device=DEVICE)
    torch.cuda.synchronize()
    # marker so stdout can be correlated with ncu rows by order
    print(f"MARKER shape B={B} L_max={PREFILL + B - 1} grid_x={PREFILL + B - 1}", flush=True)
    cuda_shift_varlen_kv_cache(kv, B, PREFILL)
    torch.cuda.synchronize()
print("done", flush=True)
