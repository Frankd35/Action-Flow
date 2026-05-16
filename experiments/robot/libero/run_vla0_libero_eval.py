"""
run_vla0_libero_eval.py

Runs VLA-0 (Qwen2.5-VL) with ActionFlow pipeline in a LIBERO simulation environment.

Usage:
    python experiments/robot/libero/run_vla0_libero_eval.py \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name libero_spatial \
        --actionflow_K 35 \
        --seed 7
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import torch
import numpy as np

# Project root (Action-Flow)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# VLA-0 paths
VLA0_ROOT = "/home/daiyuntao/vla0"
if VLA0_ROOT not in sys.path:
    sys.path.append(VLA0_ROOT)
ROBOVERSE_ROOT = "/home/daiyuntao/vla0/libs/RoboVerse"
if ROBOVERSE_ROOT not in sys.path:
    sys.path.append(ROBOVERSE_ROOT)

from actionflow.integration_qwen import enable_actionflow_qwen
from rv_train.train import get_pretrained_model
from roboverse.evals.libero.eval import eval, get_evaluation_tasks


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path (e.g. model_final.pth)

    #################################################################################################################
    # ActionFlow parameters
    #################################################################################################################
    actionflow_K: int = 35                            # Pipeline depth / max new tokens

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"           # Task suite: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    task_name: str = ""                               # Specific task name (empty = all tasks in suite)

    #################################################################################################################
    # Eval parameters
    #################################################################################################################
    seed: int = 7                                     # Random seed
    action_horizon: int = 1                           # Action horizon for eval
    local_log_dir: str = "./experiments/logs"         # Local directory for eval logs
    save_video: bool = True                           # Save rollout videos

    # fmt: on


@torch.no_grad()
@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"

    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load VLA-0 model (QwenActor)
    print(f"[VLA0-Eval] Loading pretrained model from {cfg.pretrained_checkpoint}...")
    model, vla_cfg = get_pretrained_model(cfg.pretrained_checkpoint, 0, torch_compile=False)
    model.eval()

    # Patch model.generate with ActionFlow pipeline
    print(f"[VLA0-Eval] Enabling ActionFlow with K={cfg.actionflow_K}...")
    enable_actionflow_qwen(
        model.model,
        max_new_tokens=cfg.actionflow_K,
        enable_timing=False,
    )

    K = cfg.actionflow_K

    def model_act(**kwargs):
        """
        Pipeline fill wrapper: call model K times with the same input.
        First K-1 calls fill the pipeline; the Kth returns the mature action.
        """
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = None
                for _ in range(K):
                    out = model(**kwargs, get_loss=False, get_action=True)
                return out

    # Run roboverse LIBERO eval
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    eval(
        model=model_act,
        action_type=vla_cfg.MODEL.QWEN.action_type,
        cfg_path=vla_cfg.DATALOADER.ROBOVERSE.cfg_path,
        cfg_opts=vla_cfg.DATALOADER.ROBOVERSE.cfg_opts,
        task_name=cfg.task_name if cfg.task_name else None,
        task_suite_name=cfg.task_suite_name,
        log_dir=cfg.local_log_dir,
        save_video=cfg.save_video,
        seed=cfg.seed,
        action_horizon=cfg.action_horizon,
    )


if __name__ == "__main__":
    eval_libero()
