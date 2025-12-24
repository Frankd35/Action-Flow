
# ActionFlow: Pipelined Action Acceleration for Vision Language Models on Edge

[![Paper](https://img.shields.io/badge/Paper-Arxiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/html/2512.20276v1)

**ActionFlow** is a system-level inference acceleration framework designed for Vision-Language-Action (VLA) models (e.g., OpenVLA). Tailored for resource-constrained edge platforms, it utilizes a novel **Cross-Request Pipelining** strategy and custom **Triton Kernels** to achieve significant inference speedups without retraining or compromising model accuracy.

> **Paper Title:** ActionFlow: A Pipelined Action Acceleration for Vision Language Models on Edge

## 🚀 Key Features

  * **⚡ High Performance:** Achieves **2.55x** FPS improvement on NVIDIA Jetson AGX Orin and RTX 5090.
  * **🧠 Lossless Accuracy:** System-level optimization maintains functional correctness identical to the original model (verified on LIBERO benchmarks).
  * **🛠️ Zero-Retraining:** Plug-and-play acceleration. No fine-tuning or model retraining required.
  * **🧩 Easy Integration:** Enables acceleration on existing OpenVLA checkpoints with a single line of code via non-intrusive monkey-patching.

## 📂 Project Structure

Based on the provided codebase, the project is organized into the core `actionflow` engine and external verification scripts.

```text
ActionFlow/
├── actionflow/                # Core Acceleration Engine
│   ├── integration.py         # [API] Injection entry point (contains `enable_actionflow`)
│   ├── kernels/               # [Backend] Custom Triton Kernels
│   │   └── ops.py             # Fused RoPE+WriteKV, RMSNorm, RingBuffer Shift kernels
│   └── modeling/              # [Core] Pipeline orchestration & Layer wrapping
│       ├── layers.py          # Zero-Copy Layer Wrappers (LlamaPIPEDecodeLayer)
│       └── pipeline.py        # Macro-Pipeline Scheduler (ActionFlowPipeline)
├── vla-scripts/
│   └── extern/                # Experiment & Validation Scripts
│       ├── verify_openvla.py  # Functional verification: Consistency check against baseline
│       └── benchmark.py       # Performance benchmarking: Latency & FPS profiling
└── ...
```

## 🛠️ Installation
### Prerequisites & Environment

We strongly recommend using **Docker** (or `jetson-containers` on Edge) to ensure a consistent environment.

**Jetson AGX Orin (Recommended for Edge)**
We utilized the `dustynv/openvla` container from [jetson-containers](https://github.com/dusty-nv/jetson-containers). You can launch the environment directly:

```bash
jetson-containers run --name openvla -it $(autotag openvla) bash
```
### Setup

Once inside the container or environment, install the ActionFlow engine:

```bash
cd ActionFlow
pip install -e .
```

## ⚡ Quick Start

ActionFlow is designed to be non-intrusive. simply load your OpenVLA model and call `enable_actionflow` to inject the acceleration engine.

```python
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from actionflow import enable_actionflow  # <--- Import ActionFlow

# 1. Load Standard OpenVLA
model_path = "openvla/openvla-7b"
model = AutoModelForVision2Seq.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2",
    trust_remote_code=True
).to("cuda")

# 2. Enable Acceleration 🚀
# max_new_tokens corresponds to the action chunk size (typically 7 for OpenVLA)
model = enable_actionflow(model, max_new_tokens=7) 

# 3. Inference as usual (Now Accelerated!)
# The predict_action method is automatically patched to use the Pipeline
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
# ... prepare inputs ...
action = model.predict_action(**inputs, unnorm_key="bridge_orig")
```

## 📊 Benchmarks & Scripts

We provide comprehensive scripts in `vla-scripts/extern/` for verification and benchmarking.

### 1\. Functional Verification (`verify_openvla.py`)

Verifies that the accelerated model produces identical action outputs compared to the original baseline, ensuring numerical correctness.

```bash
python vla-scripts/extern/verify_openvla.py
```

### 2\. Performance Benchmark (`benchmark.py`)

Measures Latency and Throughput (FPS) across varying Prefill and Decode lengths.

```bash
# Standard Autoregressive (Baseline)
python vla-scripts/extern/benchmark.py --use_pipe 0

# ActionFlow Accelerated (Ours)
python vla-scripts/extern/benchmark.py --use_pipe 1
```

**Performance Preview (AGX Orin):**

| Method | FPS (Speedup) | Latency (ms) |
| :--- | :--- | :--- |
| Baseline | 1.25 | 803.0 |
| **ActionFlow** | **3.20 (2.56x)** | **313.1** |

## 🔗 Citation
If you find ActionFlow useful for your research and applications, please cite our paper:


```bibtex
@misc{actionflow2025,
      title={ActionFlow: A Pipelined Action Acceleration for Vision Language Models on Edge}, 
      author={Yuntao Dai and Hang Gu and Teng Wang and Qianyu Cheng and Yifei Zheng and Zhiyong Qiu and Lei Gong and Wenqi Lou and Xuehai Zhou},
      year={2025},
      eprint={2512.20276},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.20276}
}
```
