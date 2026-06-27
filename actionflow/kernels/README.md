# Action-Flow Kernels — Benchmark & Findings

## 环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA Thor (Blackwell, CC 11.0) |
| DRAM 峰值带宽 | **238 GB/s**（cudaMemcpy D2D，cache-busted 实测） |
| L2 容量 | 32 MB（`cudaDeviceGetAttribute` 实测；注意 238 GB/s 是片外 DRAM 带宽，非 L2 片上带宽） |
| dtype | bfloat16 |
| 运行环境 | 容器 `local/actionflow:v0`（torch2.9-NV / triton3.4 / flash_attn2.7.4） |

---

## TL;DR — 最新结论（已用真实数据更正）

1. **计量口径是关键。** 旧基准（`record(); fn(); sync()` 逐次同步）会把每次 kernel 的**启动开销**算进 kernel 时间——Triton 的 Python launch ≈ 50 µs，CUDA C++ extension ≈ 8 µs。这只在**短 kernel** 上致命。
   - **FusedRoPE+KV**（~60 µs，和启动开销同量级）→ 旧口径严重失真，把 Triton 压低、显得 CUDA「60%→85%」大涨，**实为测量假象**。
   - **ShiftKV**（ms 级长 kernel）→ 启动开销 <5%，旧口径≈真实，README 旧数有效。

2. **FusedRoPE+KV 的真实情况（pipelined 口径）**：CUDA 恒定 ~233 GB/s（≈98% peak，稳）；Triton 随 shape 波动。CUDA 只在 **prefill 小、decode 长**（如 277/16–32，Triton 掉到 64–69%）处真正快 ~1.4–1.5×；在 **277/7 打平**；在 **prefill=513 处 Triton 反而更快**。**不是普涨。**

3. **ShiftKV 的收益是真的、稳健的**：B≥16 时 CUDA ≈44% vs Triton ≈32%（**1.38×**）；真实流水线里 B=7 也能到 **1.59×**（因相邻 GEMM 污染 L2，把 shift 逼到 DRAM，CUDA 的合并访存优势显现）。

4. **端到端**：把 FusedRoPE+KV 与 ShiftKV 换成 CUDA，整机 `predict_action` 提速 **1.01×–1.10×**（随 decode/prefill 增大而增大）。收益**几乎全部来自 ShiftKV**；瓶颈是占层时间 ~81% 的 GEMM（MLP/QKV/O），非这两个 memory-bound 小算子。

---

## 计量方法学：synced vs pipelined

`bench_real.py` 现在对每个 kernel 同时报告两种口径：

| 口径 | 做法 | 含义 |
|---|---|---|
| **synced**（旧） | 逐次 `record(); fn(); record(); synchronize()` | 每次 kernel 从 idle GPU 重启，**启动开销被计入**；对短 kernel 失真，且抖动大 |
| **pipelined**（真实） | 一对 event 包住 N 次背靠背 launch，中间不同步 | CPU 跑在前面、启动延迟被相邻 op 隐藏，≈**纯 kernel GPU 时间**，与真实 `packed_forward` 流水线一致 |

**判据**：当 kernel GPU 时间 ≫ 启动开销（ms 级）时 synced≈pipelined；当二者同量级（几十 µs）时，**只能信 pipelined**。

复跑：`python actionflow/kernels/bench_real.py`（Triton vs CUDA × 两口径）。

---

## 被测 Kernel 与 Shape

| Kernel | 文件 | 功能 |
|---|---|---|
| RMSNorm | `ops.py:rmsnorm_fwd_kernel` | Llama RMSNorm forward |
| FusedRoPE+KV | `ops.py:fused_rope_write_kv_kernel` | RoPE 旋转 Q/K + 写 varlen KV ring buffer |
| Shift KV Cache | `ops.py:shift_varlen_kv_cache_kernel` | 环形缓冲区原地移位 |

ActionFlow packed forward 把 prefill + 所有 decode step 拼成一次调用：`L_q = prefill + decode_len − 1`，`ShiftKV B_stages = decode_len`，`H_q=H_kv=32, D=128`。测试形状 prefill∈{277,385,513}、decode∈{7,16,24,32}（prefill=277 对应 text≈16 + 256 image tokens）。

---

## 结果 1：FusedRoPE+KV（Triton vs CUDA）

`us` 为每次调用耗时；`BW` 取 pipelined。`CU/Tri` = CUDA_pipe ÷ Triton_pipe（<1 = CUDA 快，>1 = Triton 快）。

| prefill | dec | MB | Tri synced | **Tri pipe** | Tri BW(util) | CUDA synced | **CUDA pipe** | CUDA BW(util) | CU/Tri |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 277 | 7 | 14.05 | 172.0 | **61.5** | 228 (96%) | 62.2 | **60.2** | 233 (98%) | 0.98× |
| 277 | 16 | 14.50 | 54.7 | **94.6** | 153 (64%) | 66.8 | **61.8** | 235 (99%) | **0.65×** |
| 277 | 24 | 14.90 | 152.8 | **92.9** | 160 (67%) | 77.4 | **63.7** | 234 (98%) | **0.68×** |
| 277 | 32 | 15.30 | 154.2 | **92.9** | 165 (69%) | 78.6 | **65.9** | 232 (97%) | **0.71×** |
| 385 | 7 | 19.42 | 167.6 | **91.5** | 212 (89%) | 94.4 | **82.1** | 236 (99%) | 0.90× |
| 385 | 16 | 19.87 | 170.8 | **94.2** | 211 (89%) | 97.4 | **84.4** | 235 (99%) | 0.90× |
| 385 | 24 | 20.26 | 171.8 | **93.5** | 217 (91%) | 98.8 | **86.2** | 235 (99%) | 0.92× |
| 385 | 32 | 20.66 | 172.2 | **95.0** | 218 (92%) | 100.1 | **87.8** | 235 (99%) | 0.92× |
| 513 | 7 | 25.78 | 190.0 | **96.2** | 268 (113%) | 121.4 | **108.5** | 238 (100%) | 1.13× |
| 513 | 16 | 26.22 | 191.4 | **96.1** | 273 (115%) | 123.1 | **110.4** | 237 (100%) | 1.15× |
| 513 | 24 | 26.62 | 192.9 | **96.2** | 277 (116%) | 124.9 | **111.7** | 238 (100%) | 1.16× |
| 513 | 32 | 27.02 | 196.6 | **99.5** | 272 (114%) | 126.2 | **113.1** | 239 (100%) | 1.14× |

**读法**：
- **synced 列不可信**：Triton synced 在 54.7–197 µs 间乱跳（277/16 甚至比 277/7 还低），纯启动开销 + 调度抖动。CUDA synced 稳定（启动开销小）。旧 README 的「Triton 65% / CUDA 85%」正源于此口径。
- **pipelined 是真相**：CUDA fused 恒在 **~233 GB/s（97–100% peak）**，无视 shape；Triton 波动大。
  - 277/decode≥16：Triton autotune 选了差配置，掉到 64–69% → **CUDA 真实快 1.4–1.5×**。
  - 277/7：**打平**（这就是端到端 per-op 里 fused 看似无提升的原因）。
  - 513：Triton 反超（>100% 说明数据部分 L2 命中）→ **CUDA 略慢**。
- >100% util = 工作集（14–27 MB < 32 MB L2）部分驻留 L2，已非纯 DRAM-bound。

**结论**：CUDA fused 的价值是**稳定贴 peak**（消除 Triton 的 shape 敏感），而非普涨；只优化了 V 拷贝（float4），Q/K RoPE 与 Triton 同为标量，故无 GPU 级带宽碾压。

---

## 结果 2：Shift KV Cache（Triton vs CUDA）

`CU/Tri` = Triton_pipe ÷ CUDA_pipe（>1 = CUDA 快）。

| prefill | B | MB | Tri synced | **Tri pipe** | Tri BW(util) | CUDA synced | **CUDA pipe** | CUDA BW(util) | CU/Tri |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 277 | 7 | 27.48 | 197.8 | **160.8** | 171 (72%) | 165.0 | **159.2** | 173 (73%) | 1.01× |
| 277 | 16 | 69.80 | 1014.3 | **976.4** | 71 (30%) | 814.3 | **684.9** | 102 (43%) | **1.43×** |
| 277 | 24 | 108.53 | 1646.9 | **1448.1** | 75 (31%) | 1052.8 | **1048.7** | 103 (43%) | **1.38×** |
| 277 | 32 | 148.31 | 1989.9 | **1953.3** | 76 (32%) | 1414.0 | **1414.4** | 105 (44%) | **1.38×** |
| 385 | 7 | 38.09 | 502.1 | **462.5** | 82 (34%) | 357.8 | **351.5** | 108 (45%) | **1.32×** |
| 385 | 16 | 96.34 | 1298.8 | **1261.2** | 76 (32%) | 933.6 | **933.2** | 103 (43%) | **1.35×** |
| 385 | 24 | 149.23 | 1965.0 | **1932.9** | 77 (32%) | 1416.6 | **1415.4** | 105 (44%) | **1.37×** |
| 385 | 32 | 203.16 | 2724.9 | **2687.9** | 76 (32%) | 1965.2 | **1966.9** | 103 (43%) | **1.37×** |
| 513 | 7 | 50.68 | 662.4 | **622.8** | 81 (34%) | 481.5 | **473.7** | 107 (45%) | **1.31×** |
| 513 | 16 | 127.80 | 1654.1 | **1618.7** | 79 (33%) | 1213.5 | **1211.2** | 106 (45%) | **1.34×** |
| 513 | 24 | 197.46 | 2591.9 | **2558.8** | 77 (32%) | 1899.7 | **1896.1** | 104 (44%) | **1.35×** |
| 513 | 32 | 268.17 | 3558.3 | **3520.4** | 76 (32%) | 2560.3 | **2556.1** | 105 (44%) | **1.38×** |

**读法**：
- synced≈pipelined（kernel 是 ms 级，启动开销可忽略）→ **README 旧 ShiftKV 数有效**。
- **B≥16：CUDA ≈44% vs Triton ≈32%（稳定 1.3–1.4×）**——真实 GPU 级收益（合并访存）。
- **B=7（27.48 MB < 32 MB L2）孤立测打平**（两者都 ~72%，L2 驻留）；但真实流水线里 B=7 也达 **1.59×**（见端到端），因相邻 MLP/proj GEMM 污染 L2、把 shift 逼到 DRAM，CUDA 的 8 KB 合并事务对 Triton 的分散 16 B 事务形成碾压。

### CUDA ShiftKV 设计要点

| | v1.0 | v1.1 | v1.2（当前） |
|---|---|---|---|
| ShiftKV | grid `(H_kv,2)`=64 block，block 内**串行**遍历所有 stage；标量 | grid `(L_max,H_kv,2)`（seq 维并行），stage **倒序串行**（保 RAW 依赖）；float4 | grid `(L_max,2)` block 256 线程，每 block 覆盖**所有 head×D**（固定 seq 下连续 → 8 KB 完全合并事务） |

> **RAW 依赖**：stage s 的 dst 区域 = stage s+1 的 src 区域，正序并行会数据竞争 → 采用 seq 维并行 + stage 倒序串行。
> **已排除的方向**（实现并实测无收益）：v3 chunked（事务变大但 grid 变小，抵消）、v4 streaming（`ld.cs/st.cs` 绕 L2，证明 RFO 非瓶颈）。

### 为什么 B≥16 只有 44%：2× 写回（不是 kernel 缺陷）

bench 的「MB」只计 **1× moved**（被搬数据逻辑大小）。B≥16 的 dst 工作集（70–148 MB）≫ 32 MB L2，写入的脏行在 kernel 执行期就被读流挤出、写回 DRAM → 真实物理流量 = **2× moved**。按 2× 计，44% × 2 ≈ **88% peak**——kernel 已在物理上跑满，无 kernel 层空间。

ncu 实测（`tests/profile_l2.py`，prefill=277）佐证：
- 读侧 `read_miss ≈ 1× moved`（27.5/69.8/148.4 MB），DRAM 读恒 1×，无放大。
- 写侧 `write_hit+write_miss ≈ 1× moved`；`write_miss` 恒低（4.6–5.1 MB）→ 写回由**读流推进**驱逐脏行造成，非 RFO/分配。
- B=7（dst 27.5 MB < L2）writeback≈0（证伪式：若 1× 写回则 2×=55 MB/130 µs=178% peak，物理不可能）→ 故 B=7 真实就是 ~88% 物理效率。

要再降只能算法级（缩小同时驻留 dst 工作集 ≤ L2）。

---

## 结果 3：RMSNorm（参考）

| L | MB | Tri synced | Tri pipe | Tri BW | CUDA synced | CUDA pipe | CUDA BW |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 283 | 6.96 | 64.3 | 45.9 | 152 | 28.3 | 13.0 | 536 |
| 391 | 9.62 | 64.2 | 40.4 | 238 | 24.6 | 16.9 | 569 |
| 519 | 12.76 | 49.2 | 32.4 | 394 | 27.9 | 20.5 | 622 |

CUDA RMSNorm 孤立测快 2–3×（BW 远超 238 → 数据 L2 驻留），**但 RMSNorm 仅占一层时间 ~0.5%**，端到端可忽略；且其 eps 硬编码 1e-6 与 Llama 不一定一致。**当前 ActionFlow 路径保持 Triton RMSNorm**。

---

## 端到端（真实 OpenVLA-7b，bf16）

`ops.py` 的 `fused_rope_write_kv_wrapper` / `shift_varlen_kv_cache_wrapper` 已 hardcode 调 CUDA kernel（Triton launch 注释保留）；RMSNorm 仍走 Triton。脚本：`vla-scripts/extern/bench_e2e_kernel.py`（单进程 monkeypatch 切换两种 kernel，背靠背对比）。

| text | decode | Triton | CUDA | 加速比 |
|--:|--:|--:|--:|--:|
| 16 | 7 | 138.94 ms | 137.48 ms | 1.01× |
| 256 | 7 | 170.29 ms | 163.46 ms | 1.04× |
| 16 | 32 | 336.28 ms | 317.09 ms | 1.06× |
| 256 | 32 | 366.59 ms | 334.20 ms | **1.10×** |

加速比随 decode（ShiftKV 工作量 ∝ B_stages）与 prefill 增大而升，峰值 1.10×。

### 每层 per-op 占比（prefill≈277 / decode=7，32 层累加 / 次）

脚本：`vla-scripts/extern/test_layer_op_breakdown.py`（CUDA event 计时，RMSNorm 两侧均 Triton）。

| op | Triton ms | 占比 | CUDA ms | 提升 |
|---|--:|--:|--:|--:|
| mlp | 52.25 | 53.5% | 51.53 | 1.01× |
| qkv_proj | 18.22 | 18.6% | 17.86 | 1.02× |
| o_proj | 8.47 | 8.7% | 8.51 | 1.00× |
| **shift_kv** | 8.17 | 8.4% | **5.13** | **1.59×** |
| flash_attn | 5.80 | 5.9% | 5.72 | 1.01× |
| **fused_rope_kv** | 2.04 | 2.1% | 2.12 | 0.96× |
| rmsnorm (×2) | 0.86 | 0.9% | 0.87 | 0.99× |
| other/resid | 1.90 | 1.9% | 1.89 | — |
| **层总计** | **97.70** | 100% | **93.63** | **1.04×** |

- **GEMM 主导**：mlp + qkv + o = **~81%**（compute-bound，未优化）。两个被优化 kernel 合计仅 **~10.5%**——这是端到端只有 1.04× 的根因。
- **收益全在 ShiftKV**（1.59×，省 3.04 ms = 层总 3.1%）；**fused_rope 在 277/7 打平**（与「结果 1」一致，此 shape CUDA≈Triton）。

---

## 总结：哪些是真的，能拿多少

| 优化 | synced 口径（旧） | pipelined 真相 | 端到端贡献 |
|---|---|---|---|
| **ShiftKV** | 32%→44% | **真收益**：B≥16 稳定 1.38×；真实流水线 B=7 达 1.59× | **几乎全部**（decode=32 时端到端 1.10×） |
| **FusedRoPE+KV** | 「65%→85%」 | **大部分是启动开销假象**：CUDA 恒 ~98% peak、稳；Triton shape 相关；277/7 打平、277/decode≥16 CUDA 快 1.4–1.5×、513 Triton 更快 | 小（277/7 打平） |
| **RMSNorm** | 快 2–3× | 真快但占比 0.5% | 可忽略（保持 Triton） |

**一句话**：两个 memory-bound 小算子合计只占一层 ~10.5%，CUDA 化的端到端收益 **1.01–1.10×**，且**主要靠 ShiftKV**（FusedRoPE 仅在 Triton autotune 翻车的 shape 上才有意义）。想要更大端到端提升，刀口必须转向占 81% 的 GEMM（量化 / 更优 matmul），memory-bound kernel 的天花板就在这里。

---

## 脚本与复现

所有脚本在容器 `local/actionflow:v0` 内运行（`docker exec dyt_af_ops bash -lc 'cd /home/daiyuntao/Action-Flow && python <脚本>'`）。

| 脚本 | 用途 |
|---|---|
| `actionflow/kernels/bench_real.py` | **核心基准**：Triton vs CUDA × synced/pipelined 双口径，覆盖 FusedRoPE+KV、ShiftKV、RMSNorm。本文所有 kernel 级数据来源。 |
| `vla-scripts/extern/bench_e2e_kernel.py` | **端到端 A/B**：加载真实 OpenVLA-7b，单进程内 monkeypatch 切换 Triton/CUDA，对比 `predict_action` 延迟。 |
| `vla-scripts/extern/test_layer_op_breakdown.py` | **Per-op 拆解**：用 CUDA Event 统计每层内各 op 耗时占比，定点 prefill=277/decode=7。 |
| `actionflow/kernels/tests/test_correctness.py` | 正确性：以 Triton 为 golden，验证 CUDA kernel 输出（atol=1e-2）。 |
| `actionflow/kernels/tests/profile_l2.py` | ncu L2 计数器：采集 `lts__t_sectors_op_*` 佐证 ShiftKV 2× 写回分析。 |
| `actionflow/kernels/cuda_ops.py` | CUDA kernel 实现与 JIT 编译（`torch.utils.cpp_extension.load_inline`）。 |
| `actionflow/kernels/ops.py` | Triton kernel 定义 + wrapper（FusedRoPE/ShiftKV 已 hardcode 调 CUDA，Triton launch 注释保留）。 |
| `actionflow/modeling/layers.py` | ActionFlow decoder layer（调用上述 wrapper，**未修改**）。 |
