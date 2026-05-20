"""
VLA-0 LIBERO eval with proper model loading (uses get_pretrained_model + QwenActor).
Supports native and ActionFlow (pipe) modes.

Usage:
    # Native (3 episodes per task)
    python experiments/robot/libero/run_vla0_libero_eval_direct.py \
        --task_suite_name libero_spatial \
        --num_trials_per_task 3

    # ActionFlow pipe
    python experiments/robot/libero/run_vla0_libero_eval_direct.py \
        --task_suite_name libero_spatial \
        --num_trials_per_task 3 \
        --use_pipe True
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import torch
import tqdm
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

# VLA-0 paths
VLA0_REPO = "/home/daiyuntao/vla0_repo"
if VLA0_REPO not in sys.path:
    sys.path.insert(0, VLA0_REPO)
ROBOVERSE_PATH = f"{VLA0_REPO}/libs/RoboVerse"
if ROBOVERSE_PATH not in sys.path:
    sys.path.insert(0, ROBOVERSE_PATH)

from rv_train.train import get_pretrained_model

# ActionFlow
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from actionflow.integration_qwen import enable_actionflow_qwen

# Import libero observation conversion from roboverse
from roboverse.evals.libero.eval import libero_to_rv_obs
from roboverse.evals.libero.eval import (
    LIBERO_TASK_SUITE_MAX_STEPS, LIBERO_DUMMY_ACTION,
)

from experiments.robot.libero.libero_utils import (
    get_libero_env,
    save_rollout_video,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    set_seed_everywhere,
)

# VLA-0 checkpoint
VLA0_CHECKPOINT = os.environ.get(
    "VLA0_CHECKPOINT",
    "/home/daiyuntao/jetson-containers/data/models/huggingface/models--ankgoyal--vla0-libero/snapshots/6c62a511ad56042b2b3ea0a90c4def33e1ea3b96/model_last.pth"
)

NUM_STEPS_WAIT = 10


def load_vla0_model(device):
    """Load VLA-0 model using official get_pretrained_model. (No ActionFlow setup.)"""
    print(f"[VLA0] Loading checkpoint from {VLA0_CHECKPOINT}")
    model, cfg = get_pretrained_model(VLA0_CHECKPOINT, device, torch_compile=False)
    model.eval()
    return model, cfg


def run_vla0_pipe_inference(model, model_obs, pipe_K):
    """
    ActionFlow pipe inference via QwenActor.forward().

    Calls model(**model_obs) pipe_K times, each going through:
        QwenActor.forward() -> self.model.generate() -> patched_generate() -> pipe_forward()

    After pipe_K calls the ActionFlow pipeline is fully warm and returns
    pipe_K valid autoregressive tokens, which are decoded to actions.

    This mirrors the OpenVLA pattern in run_libero_eval.py:
        action = get_action(...)  # model call
        if use_pipe and wait_pipe < len(action):  # fill pipeline
            wait_pipe += 1
            continue
        else:  # pipeline full -> execute action
            ...

    Args:
        model: QwenActor instance
        model_obs: dict from libero_to_rv_obs
        pipe_K: pipeline depth (= max_new_tokens for ActionFlow)

    Returns:
        dict with 'out_ori_act' key from the last model call
    """
    out = None
    for i in range(pipe_K):
        out = model(**model_obs, get_loss=False, get_action=True)
        if (i + 1) % 20 == 0 or i == 0 or i == pipe_K - 1:
            print(f"[Pipe] model call {i+1}/{pipe_K}", flush=True)
    return out


@dataclass
class GenerateConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = False

    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = NUM_STEPS_WAIT
    num_trials_per_task: int = 3

    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_project: str = "YOUR_WANDB_PROJECT"
    wandb_entity: str = "YOUR_WANDB_ENTITY"

    seed: int = 7

    use_pipe: bool = True


class _CompatCfg:
    """Adapt VLA-0 training config to roboverse-style config for libero_to_rv_obs."""
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
            crop_img = 1.0
            img_size = 224
            cam_list = ("3p1", "3p2")
            return_ee = False
            return_ori_act = False
            return_proprio = False
        self.IMAGE = _IMAGE()


@torch.no_grad()
@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    set_seed_everywhere(cfg.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    use_af = cfg.use_pipe
    model, vla_cfg = load_vla0_model(device)
    pipe_K = vla_cfg.MODEL.QWEN.horizon * vla_cfg.MODEL.QWEN.original_action_dim * (len(str(vla_cfg.MODEL.QWEN.num_bins_actions)) + 1)
    print(f"[VLA0] Horizon={vla_cfg.MODEL.QWEN.horizon}, act_dim={vla_cfg.MODEL.QWEN.original_action_dim}, pipe_K={pipe_K}")

    if use_af:
        print(f"[VLA0] Enabling ActionFlow with K={pipe_K}...")
        enable_actionflow_qwen(
            model.model,
            max_new_tokens=pipe_K,
            enable_timing=False,
        )

    # Action horizon from config
    action_horizon = vla_cfg.MODEL.QWEN.horizon
    rollout_mode = "ActionFlow" if use_af else "Native"

    # Logging
    run_id = f"EVAL-{cfg.task_suite_name}-vla0-{DATE_TIME}"
    if use_af:
        run_id += "-ActionFlow"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(min(1, num_tasks_in_suite))):  # single task for verification
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            max_steps = LIBERO_TASK_SUITE_MAX_STEPS.get(cfg.task_suite_name, 220)

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")

            action_i = 0
            action_chunk = None

            while t < max_steps + cfg.num_steps_wait:
                try:
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get new action when: (a) first step, or (b) exhausted action horizon
                    if action_i >= action_horizon or t == cfg.num_steps_wait:
                        model_obs = libero_to_rv_obs(obs, task_description, _CompatCfg(vla_cfg))

                        if use_af:
                            out = run_vla0_pipe_inference(model, model_obs, pipe_K=pipe_K)
                        else:
                            out = model(**model_obs, get_loss=False, get_action=True)

                        action_chunk = out["out_ori_act"][0].cpu().numpy()
                        action_horizon_eff = min(action_horizon, len(action_chunk))
                        action_i = 0

                    act = action_chunk[action_i]
                    # Ensure gripper is ±1
                    if act[-1] > 0:
                        act[-1] = 1
                    elif act[-1] < 0:
                        act[-1] = -1

                    # Save image for replay (flip 180° to correct LIBERO render orientation)
                    replay_images.append(obs["agentview_image"][::-1, ::-1])

                    print(f"step: {t}, action: {act}")

                    obs, reward, done, info = env.step(act.tolist())
                    action_i += 1
                    t += 1

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    import traceback
                    traceback.print_exc()
                    break

            task_episodes += 1
            total_episodes += 1

            save_rollout_video(replay_images, total_episodes, success=done,
                               task_description=task_description, log_file=log_file,
                               mode=rollout_mode)

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        print(f"Task success rate: {float(task_successes) / float(task_episodes):.1f}")
        log_file.write(f"Task success rate: {float(task_successes) / float(task_episodes):.1f}\n")
        log_file.flush()

    log_file.close()


if __name__ == "__main__":
    eval_libero()
