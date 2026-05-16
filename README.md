
# ActionFlow: Pipelined Action Acceleration for Vision Language Models on Edge

[![Paper](https://img.shields.io/badge/Paper-Arxiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/html/2512.20276v1)

**ActionFlow** is a system-level inference acceleration framework designed for Vision-Language-Action (VLA) models (e.g., OpenVLA). Tailored for resource-constrained edge platforms, it utilizes a novel **Cross-Request Pipelining** strategy and custom **Triton Kernels** to achieve significant inference speedups without retraining or compromising model accuracy.

> **Paper Title:** ActionFlow: A Pipelined Action Acceleration for Vision Language Models on Edge

> 🎉 Accepted at **DAC 2026**!

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
│   ├── integration.py         # [API] OpenVLA injection entry point (contains `enable_actionflow`)
│   ├── integration_qwen.py    # [API] VLA-0 (Qwen2.5-VL) injection entry point (contains `enable_actionflow_qwen`)
│   ├── kernels/               # [Backend] Custom Triton Kernels
│   │   └── ops.py             # Fused RoPE+WriteKV, RMSNorm, RingBuffer Shift kernels
│   └── modeling/              # [Core] Pipeline orchestration & Layer wrapping
│       ├── layers.py          # Zero-Copy Layer Wrappers (LlamaPIPEDecodeLayer)
│       ├── pipeline.py        # Macro-Pipeline Scheduler (ActionFlowPipeline)
│       ├── qwen_layers.py     # Qwen2.5-VL Layer Wrappers (QwenPIPEDecodeLayer)
│       └── qwen_pipeline.py   # Qwen2.5-VL Pipeline (ActionFlowQwenPipeline)
├── vla-scripts/
│   └── extern/                # Experiment & Validation Scripts
│       ├── verify_openvla.py  # Functional verification: Consistency check against baseline
│       ├── benchmark.py       # Performance benchmarking: Latency & FPS profiling
│       └── perf_vla0.py       # VLA-0 performance testing (native vs ActionFlow)
├── experiments/
│   └── robot/
│       └── libero/            # LIBERO evaluation scripts
│           ├── run_libero_eval.py    # OpenVLA LIBERO eval
│           ├── run_vla0_libero_eval.py  # VLA-0 LIBERO eval with ActionFlow
│           └── libero_requirements.txt
└── pyproject.toml             # Includes `[project.optional-dependencies] vla0 = ["qwen-vl-utils"]`
```

## 🛠️ Installation
### Prerequisites & Environment

**1. Jetson AGX Orin (Recommended for Edge)**

We utilized the `dustynv/openvla` container from [jetson-containers](https://github.com/dusty-nv/jetson-containers). You can launch the environment directly:

```bash
jetson-containers run --name openvla -it $(autotag openvla) bash
```

**2. GPU Server / Desktop (General CUDA)**

For workstations or servers equipped with NVIDIA GPUs (e.g., RTX 3090/4090, A100, H100), please follow these steps:

**Step A: Base Environment**
First, follow the official [OpenVLA GitHub](https://github.com/openvla/openvla) instructions to install the base dependencies.

**Step B: Install Optimized Requirements**
Install the following specific versions of key libraries:

```bash
pip install transformers==4.49.0 
pip install triton==3.2.0
pip install flash_attn==2.8.3
pip install "accelerate>=0.26.0"

```

---

### Setup 

Once inside the container or environment, install the ActionFlow engine:

```bash
cd ActionFlow
pip install -e .
```

### VLA-0 (Qwen2.5-VL) Extra Setup

ActionFlow also supports VLA-0, a Qwen2.5-VL based policy. This is an additional option on top of **either** the Jetson or GPU Server environment above.

```bash
# Install VLA-0 extra dependency (qwen-vl-utils for vision processing)
pip install qwen-vl-utils

# Clone VLA-0 training/inference code
git clone <vla0-repo> /path/to/vla0
# Clone RoboVerse (LIBERO evaluation harness)
git clone <roboverse-repo> /path/to/vla0/libs/RoboVerse

# For LIBERO evaluation, install simulation dependencies
pip install imageio[ffmpeg] robosuite==1.4.1 bddl easydict cloudpickle gym
```

**Model Checkpoint**: Download a VLA-0 checkpoint (e.g., `ankgoyal/vla0-libero` from HuggingFace Hub) and set the path:

```bash
export VLA0_MODEL_PATH=/path/to/vla0-checkpoint
```

**Environment Variables**: The VLA-0 eval scripts need the following paths in `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/ActionFlow:/path/to/vla0:/path/to/vla0/libs/RoboVerse:$PYTHONPATH
```

For LIBERO simulation, also set:
```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
```

**Run Performance Test**:
```bash
python vla-scripts/extern/perf_vla0.py --mode actionflow
```

**Run LIBERO Evaluation**:
```bash
python experiments/robot/libero/run_vla0_libero_eval.py \
    --pretrained_checkpoint /path/to/vla0-checkpoint \
    --task_suite_name libero_spatial \
    --actionflow_K 35 \
    --seed 7
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

**Performance Preview (NVIDIA Jetson AGX Orin):**

| Mode | Quantization | FPS | Speedup | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- |
| Baseline (Autoregressive) | BF16 | 1.25 | 1.00x | 803.0 |
| ActionFlow | BF16 | **3.20** | **2.56x** | **313.1** |
| Baseline (Autoregressive) | INT4 | - | - | - |
| ActionFlow | INT4 | - | - | - |

**Performance Preview (NVIDIA THOR):**

| Mode | Quantization | FPS | Speedup | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- |
| Baseline (Autoregressive) | BF16 | 1.51 | 1.00x | 664.4 |
| ActionFlow | BF16 | **6.84** | **4.53x** | **146.1** |
| Baseline (Autoregressive) | INT4 | 2.09 | 1.38x | 477.5 |
| ActionFlow | INT4 | **8.78** | **5.81x** | **113.9** |

> **Note:** Speedup for ActionFlow is relative to the BF16 Baseline. Run `python vla-scripts/extern/benchmark.py --use_pipe 0 --quantize bf16` for BF16 Baseline, `--use_pipe 1 --quantize bf16` for BF16 ActionFlow, `--use_pipe 0 --quantize int4` for INT4 Baseline, and `--use_pipe 1 --quantize int4` for INT4 ActionFlow.

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
