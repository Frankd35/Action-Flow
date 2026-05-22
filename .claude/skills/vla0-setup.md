---
name: vla0-setup
description: >
  VLA-0 (Qwen2.5-VL-3B) + LIBERO 环境搭建完整指南。
  首选 Docker 容器安装（挂载项目路径），无合适镜像时用 conda。
  涵盖：镜像选择、PyTorch/Flash-Attention 安装、模型权重下载（仅 model_last/）、
  VLA-0 仓库安装、LIBERO 配置、ActionFlow 集成、仿真运行，以及 10+ 个常见踩坑的解决方案。
  路径均为占位符，拿到即可按步骤操作。
triggers:
  - vla0 setup
  - vla0 环境
  - vla0 libero
  - 搭建 vla0
  - vla0 安装
  - vla0 仿真
  - libero 环境
  - actionflow vla0
---

# VLA-0 + LIBERO 环境搭建与仿真指南

> 本指南记录从零搭建 VLA-0 (Qwen2.5-VL-3B) 在 LIBERO 上的评估环境，以及集成 ActionFlow 的完整流程与踩坑经验。
> **首选 Docker 容器安装**，将项目路径挂载进容器，保持宿主机环境干净。无合适镜像时再使用 conda。
> 环境目标：CUDA torch + Flash-Attention + LIBERO + VLA-0 + ActionFlow

---

## 前置条件

- NVIDIA GPU (CUDA >= 11.8)
- `docker` + `nvidia-docker` (`--runtime nvidia` 可用)
- 至少 30GB 磁盘空间（模型权重约 7GB + 依赖 + LIBERO 数据集）
- HuggingFace 网络访问（或配置 `HF_ENDPOINT=https://hf-mirror.com`）

---

## 1. 启动 Docker 容器（首选方案）

### 1.1 查找合适的 CUDA 镜像

先检查本地已有镜像，优先选择自带 CUDA torch 的镜像：

```bash
# 查看本地镜像
docker images | grep -E "cuda|pytorch|nvidia"

# 推荐镜像特征：
# - 基于 Ubuntu 22.04/24.04
# - 自带 CUDA 12.x
# - 自带 Python 3.10
# - 如已有 torch，需为 CUDA 版本（非 CPU-only）
```

**推荐镜像**（按优先级）：

| 镜像 | 说明 |
|------|------|
| `dustynv/openvla:r36.4.3-cu128-cp312-24.04` | Jetson 预配镜像，自带 CUDA torch 2.6.0 + flash_attn 2.7.4 |
| `nvcr.io/nvidia/pytorch:24.02-py3` | NVIDIA 官方 PyTorch 容器，CUDA 12.3 + torch 2.2 |
| `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` | PyTorch 官方容器 |
| 自定义 CUDA 基础镜像 | `nvidia/cuda:12.1.0-devel-ubuntu22.04` + 手动装 torch |

**踩坑**：不要选 PyPI CPU-only torch 的镜像。如果镜像自带 torch，运行 `python -c "import torch; print(torch.cuda.is_available())"` 验证必须为 `True`。

### 1.2 启动容器并挂载路径

```bash
# 设置路径（根据你的实际环境修改）
HOST_WORKSPACE=/home/yourname/workspace        # 宿主机工作目录
HOST_VLA0=/home/yourname/vla0                  # VLA-0 仓库路径
HOST_ACTIONFLOW=/home/yourname/Action-Flow     # ActionFlow 仓库路径
HOST_MODEL=/home/yourname/models               # 模型权重存放路径

# 启动容器
# 推荐使用 --ipc=host 避免共享内存不足
docker run -itd --rm --runtime nvidia --network host --ipc=host \
  --name vla0-eval \
  -v "$HOST_WORKSPACE":"/workspace" \
  -v "$HOST_VLA0":"/workspace/vla0" \
  -v "$HOST_ACTIONFLOW":"/workspace/Action-Flow" \
  -v "$HOST_MODEL":"/workspace/models" \
  YOUR_CUDA_IMAGE bash

# DNS 修复（国内环境）
docker exec vla0-eval bash -c 'echo "nameserver 223.5.5.5" > /etc/resolv.conf'

# 进入容器
docker exec -it vla0-eval bash
```

**注意**：容器启动后，后续所有操作（第 2-8 节）均在**容器内**执行。

### 1.3 容器内环境检查

```bash
# 验证 CUDA 可用
python -c "import torch; print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}')"

# 验证 Python 版本（需 3.10）
python --version

# 如无 torch 或版本不对，进入第 2 节安装
```

---

## 备选：1B. Conda 环境（无合适 Docker 镜像时使用）

如果本地没有合适的 CUDA 镜像，或不想用 Docker：

```bash
conda create -y -n vla0 python=3.10
conda activate vla0
```

然后按第 2 节开始安装 PyTorch 等依赖。

**注意**：conda 环境下**不要**用 pip 覆盖安装 PyPI 的 CPU-only torch。

---

## 2. 安装 PyTorch + Flash Attention

### 方式 A：容器已有 CUDA torch（推荐）

如果镜像已自带 CUDA torch，pin 住不升级：

```bash
pip install torch==2.6.0 torchvision==0.21.0 --no-index  # 使用本地/镜像自带版本
```

### 方式 B：从 PyPI 安装（x86_64 常规服务器 / conda 环境）

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
```

### Flash Attention

```bash
pip install flash-attn --no-build-isolation
```

**踩坑**：flash-attn 编译需要大量显存和内存，如果编译失败：
- 降低并行度：`MAX_JOBS=4 pip install flash-attn --no-build-isolation`
- 或直接使用预编译 wheel（如果有）
- Docker 内编译失败通常是因为内存限制，加 `--shm-size=8g` 重启容器

---

## 3. 下载 VLA-0 模型权重

只需要下载 `model_last/` 子目录，包含已合并的 HF 格式权重（约 7GB），不需要整个仓库：

```bash
pip install huggingface-hub
export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像

# 只下载 model_last/ 目录
huggingface-cli download ankgoyal/vla0-libero model_last/ \
    --local-dir ./vla0-libero-model \
    --local-dir-use-symlinks False
```

下载后目录结构：
```
vla0-libero-model/
  model_last/
    config.json
    model.safetensors
    ...
```

**踩坑**：
- `huggingface-cli download` 默认下载整个仓库，必须显式指定 `model_last/` 子目录。
- `model_last/` 是已合并的 HF 格式权重，可直接用 `AutoModel.from_pretrained` 加载。
- 不要下载 `.pth` 原始 checkpoint，体积大且加载方式不同。

**环境变量**：后续脚本依赖 `VLA0_CHECKPOINT` 或 `VLA0_MODEL_PATH`：

```bash
export VLA0_CHECKPOINT="/workspace/models/vla0-libero-model/model_last.pth"
export VLA0_MODEL_PATH="/workspace/models/vla0-libero-model/model_last"
```

---

## 4. 安装 VLA-0 仓库

```bash
cd /workspace
git clone --recurse-submodules https://github.com/NVlabs/vla0.git
cd vla0
```

### 4.1 安装 VLA-0 主包

```bash
PIP_REQ_EXTRAS=qwen pip install --no-build-isolation -e ".[qwen]"
```

这会安装：`peft`, `qwen-vl-utils`, `bitsandbytes` 等。

### 4.2 安装 RoboVerse 库

```bash
cd libs/RoboVerse
pip install -e .
cd ../..
```

**踩坑**：`RoboVerse` 的 setup.py 里如果触发 `PIP_REQ_EXTRAS=lerobot`，会安装 `torch==2.7.1` 覆盖你的 CUDA torch。**不要**设置 `PIP_REQ_EXTRAS=lerobot`。LIBERO eval 不需要 lerobot。

### 4.3 安装 LIBERO 及依赖（vla0 setup.py 自动处理）

```bash
PIP_REQ_EXTRAS=libero pip install --no-build-isolation -e ".[libero]"
```

这一步 `setup.py` 会自动完成以下操作：
- `git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git` 到 `libs/LIBERO`
- 创建缺失的 `libero/__init__.py`
- Patch `torch.load(..., weights_only=False)`（PyTorch 2.6+ 兼容）
- `git clone https://github.com/StanfordVL/egl_probe.git` 并安装
- 安装 LIBERO 的 requirements
- 强制 upgrade numpy 到 `1.26.4`（LIBERO 默认 downgrade 到 1.22.0 会不兼容）

**踩坑**：
- egl_probe 需要 `cmake >= 3.5`，如果系统 cmake 太旧：
  ```bash
  apt-get update && apt-get install -y cmake  # Debian/Ubuntu
  # 或 conda install -y -c conda-forge cmake
  ```
- 如果 LIBERO 安装卡住，检查是否卡在 `egl_probe` 编译，可能需要安装系统依赖：
  ```bash
  apt-get install -y libegl1-mesa-dev libgles2-mesa-dev ffmpeg
  ```

### 4.4 安装额外依赖

```bash
pip install qwen-vl-utils
pip install opencv-python-headless  # 替换 opencv-python，避免 GUI 依赖冲突
pip install imageio[ffmpeg]
```

**踩坑**：`opencv-python` 会与 LIBERO/robosuite 的 headless 渲染冲突，必须换成 `opencv-python-headless`：

```bash
pip uninstall -y opencv-python
pip install opencv-python-headless
```

---

## 5. 配置 LIBERO

创建 `~/.libero/config.yaml`（容器内 home 目录）：

```bash
mkdir -p ~/.libero
cat > ~/.libero/config.yaml << 'EOF'
benchmark_root: /workspace/vla0/libs/LIBERO/libero/libero
bddl_files: /workspace/vla0/libs/LIBERO/libero/libero/bddl_files
init_states: /workspace/vla0/libs/LIBERO/libero/libero/init_files
datasets: /workspace/vla0/libs/LIBERO/libero/datasets
assets: /workspace/vla0/libs/LIBERO/libero/libero/assets
EOF
```

---

## 6. 安装 ActionFlow

```bash
cd /workspace/Action-Flow
git checkout af_vla0_adapt  # 确保在 vla0 适配分支
pip install -e .
```

---

## 7. 环境验证

### 7.1 基础检查

```bash
python -c "
import torch
print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}')
from libero.libero.envs import OffScreenRenderEnv
print('LIBERO OK')
from actionflow import enable_actionflow_qwen
print('ActionFlow OK')
from rv_train.train import get_pretrained_model
print('VLA-0 OK')
"
```

### 7.2 性能基准测试

```bash
cd /workspace/Action-Flow

# VLA-0 Native
python vla-scripts/extern/perf_vla0.py --mode native

# VLA-0 + ActionFlow
python vla-scripts/extern/perf_vla0.py --mode actionflow
```

预期（A100/4090 级别）：
- Native: ~3.8s/it
- ActionFlow: ~0.43s/it，加速比约 8-9x

---

## 8. 跑 LIBERO 仿真

### 8.1 Native 模式（无 ActionFlow）

```bash
cd /workspace/Action-Flow
export VLA0_CHECKPOINT=/workspace/models/vla0-libero-model/model_last.pth

python experiments/robot/libero/run_vla0_libero_eval.py \
    --task_suite_name libero_spatial \
    --num_trials_per_task 50 \
    --use_pipe False \
    --seed 7
```

### 8.2 ActionFlow Pipe 模式

```bash
cd /workspace/Action-Flow
export VLA0_CHECKPOINT=/workspace/models/vla0-libero-model/model_last.pth

python experiments/robot/libero/run_vla0_libero_eval.py \
    --task_suite_name libero_spatial \
    --num_trials_per_task 50 \
    --use_pipe True \
    --seed 7
```

**预期成功率**：libero_spatial 约 80-95%（10 tasks x 50 trials = 500 episodes）。

### 8.3 后台运行 + 日志追踪

长时间仿真务必后台运行，通过日志跟踪进度：

```bash
nohup python experiments/robot/libero/run_vla0_libero_eval.py \
    --task_suite_name libero_spatial \
    --num_trials_per_task 50 \
    --use_pipe True \
    --seed 7 \
    > /workspace/vla0_pipe.log 2>&1 &

# 检查进度
tail -f /workspace/vla0_pipe.log
grep -E 'success=|successes:|# episodes completed' /workspace/vla0_pipe.log | tail -10
```

---

## 9. 踩坑总结

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Docker 内 `torch.cuda.is_available()` 为 False | nvidia-docker 未正确安装或未加 `--runtime nvidia` | 检查 `docker run --runtime nvidia`，宿主机执行 `nvidia-smi` 确认驱动正常 |
| `torch.load` 报错 `weights_only` | PyTorch 2.6+ 默认值改为 True | vla0 setup.py 已自动 patch LIBERO，或手动给 `torch.load` 加 `weights_only=False` |
| LIBERO 安装后 `numpy` 报错 | LIBERO downgrade numpy 到 1.22.0 | vla0 setup.py 已自动 upgrade 到 1.26.4 |
| `ModuleNotFoundError: libero` | LIBERO 缺少 `libero/__init__.py` | vla0 setup.py 已自动创建 |
| opencv GUI 冲突 / 段错误 | `opencv-python` 与 headless 渲染冲突 | `pip uninstall opencv-python && pip install opencv-python-headless` |
| torch 被覆盖成 CPU-only | `lerobot` 或 `roboverse` 依赖安装时升级了 torch | 使用 `--no-deps` 或 `PIP_REQ_EXTRAS=qwen`（不要加 lerobot） |
| egl_probe 编译失败 | cmake 版本太低或缺少 mesa dev | `apt-get install cmake libegl1-mesa-dev libgles2-mesa-dev` |
| flash-attn 编译失败 / 被杀掉 | Docker 默认 shared memory 太小 | 重启容器加 `--shm-size=8g` |
| VLA-0 模型加载报错 | `qwen_model_id` 指向了不存在的 HF 路径 | 确保 `VLA0_CHECKPOINT` 指向正确的本地路径，或修改 config.yaml |
| 仿真视频是倒的 | LIBERO 渲染方向与保存方向不一致 | eval 脚本中保存视频时加 `[::-1, ::-1]` 翻转 |
| 推理结果全错 / 动作异常 | `_CompatCfg` 参数不匹配训练配置 | `crop_img=0.875`, `return_ori_act=True` 必须匹配 `img_libero_aug.yaml` |
| `huggingface-hub` 版本冲突 | 与 transformers 不兼容 | 安装 `huggingface-hub>=0.26.0,<1.0` |

---

## 10. 关键参数速查

### _CompatCfg（eval 脚本中的配置适配）

必须与训练配置 `configs/img_libero_aug.yaml` 一致：

```python
class _CompatCfg:
    def __init__(self, vla_cfg):
        self.history = vla_cfg.MODEL.QWEN.history
        self.horizon = vla_cfg.MODEL.QWEN.horizon
        self.unifier = "image"
        self.LEROBOT = {
            "repo_id": "",
            "le_cam_list": None,
            "rv_cam_list": None,
            "action_key": "actions",
            "state_key": "state",
        }
        class _IMAGE:
            crop_img = 0.875          # 关键！必须匹配训练配置
            img_size = 224
            cam_list = ("3p1", "3p2")
            return_ee = False
            return_ori_act = True     # 关键！必须设为 True
            return_proprio = False
        self.IMAGE = _IMAGE()
```

### Pipe K 计算

```python
pipe_K = vla_cfg.MODEL.QWEN.horizon * \
         vla_cfg.MODEL.QWEN.original_action_dim * \
         (len(str(vla_cfg.MODEL.QWEN.num_bins_actions)) + 1)
```

典型值（horizon=1, act_dim=7, num_bins=1000）：`1 * 7 * (4 + 1) = 35`

### Gripper 动作裁剪

VLA-0 输出的 gripper 维度需要二值化：

```python
if act[-1] > 0:
    act[-1] = 1
elif act[-1] < 0:
    act[-1] = -1
```

---

## 参考链接

- VLA-0 论文 & 仓库：https://github.com/NVlabs/vla0
- 模型权重：https://huggingface.co/ankgoyal/vla0-libero
- LIBERO：https://github.com/Lifelong-Robot-Learning/LIBERO
- ActionFlow：https://github.com/your-org/Action-Flow (af_vla0_adapt 分支)
