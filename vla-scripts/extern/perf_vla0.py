"""
VLA-0 + ActionFlow Performance Test Script

Tests native and ActionFlow-accelerated inference on VLA-0 (Qwen2.5-VL).
Run inside Action-Flow Docker container:
    docker exec -w /home/daiyuntao/Action-Flow af-container \
        python vla-scripts/extern/perf_vla0.py [--mode native|actionflow|both]

Dependencies (install in Action-Flow env):
    pip install -e ".[vla0]"   # pulls in qwen-vl-utils

Environment:
    VLA0_MODEL_PATH: Path to VLA-0 model weights (default: HF cache)
    HORIZON: Action horizon (default: 1)
    ACT_DIM: Action dimension (default: 7)
    NUM_BINS: Discretization bins (default: 1000)
"""
import os
import sys
import time
import random
import argparse

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# === Path Setup ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from actionflow import enable_actionflow_qwen, print_timing_stats_qwen
    print("[ActionFlow] Package imported successfully.")
except ImportError as e:
    print(f"[ActionFlow] Import failed: {e}")
    sys.exit(1)

# === Configuration ===
MODEL_PATH = os.environ.get(
    "VLA0_MODEL_PATH",
    "/root/.cache/huggingface/models--ankgoyal--vla0-libero/model_last"
)

HORIZON = int(os.environ.get("HORIZON", "1"))
ACT_DIM = int(os.environ.get("ACT_DIM", "7"))
NUM_BINS = int(os.environ.get("NUM_BINS", "1000"))
MAX_NEW_TOKENS = ACT_DIM * HORIZON * (len(str(NUM_BINS)) + 1)

INSTRUCTION = "pick up the object"
NUM_RUNS = 5
WARMUP_RUNS = 3


def get_vla0_prompt(instruction: str, image: Image.Image) -> list:
    system_message = (
        f"Analyze the input image and predict robot actions for the next {HORIZON} timesteps. "
        f"Each action has {ACT_DIM} dimensions. Output a single sequence of {HORIZON * ACT_DIM} "
        f"integers (0-{NUM_BINS} each), representing the {HORIZON} timesteps sequentially. "
        "Provide only space separated numbers. Nothing else."
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]},
    ]


def parse_action_text(action_text: str) -> np.ndarray:
    parts = action_text.strip().split()
    numbers = [int(x) for x in parts if x.isdigit()]
    expected = ACT_DIM * HORIZON
    if len(numbers) >= expected:
        return np.array(numbers[:expected]) / NUM_BINS
    result = np.zeros(expected)
    if numbers:
        result[:len(numbers)] = np.array(numbers) / NUM_BINS
    return result


def prepare_inputs(processor, instruction, device):
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    messages = get_vla0_prompt(instruction, image)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in inputs.items()}


def load_model(device):
    """Load VLA-0 model to device."""
    print("[*] Loading model (BF16 + Flash Attention)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    return model


@torch.inference_mode()
def test_native(device):
    print("\n" + "=" * 60)
    print("Testing NATIVE VLA-0 Inference")
    print("=" * 60)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = load_model(device)

    # Warm-up
    for _ in range(WARMUP_RUNS):
        inputs = prepare_inputs(processor, INSTRUCTION, device)
        _ = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    torch.cuda.synchronize()

    times = []
    for i in range(NUM_RUNS):
        inputs = prepare_inputs(processor, INSTRUCTION, device)
        start = time.perf_counter()
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        action_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        action = parse_action_text(action_text)
        print(f"  Run {i + 1}: {elapsed:.3f}s, action={action[:7]}")

    avg = np.mean(times)
    print(f"\n  Native: Avg {avg:.3f}s ({1 / avg:.2f} FPS)")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    return times


@torch.inference_mode()
def test_actionflow(device):
    print("\n" + "=" * 60)
    print("Testing ActionFlow Accelerated VLA-0")
    print("=" * 60)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = load_model(device)
    model = enable_actionflow_qwen(model, max_new_tokens=MAX_NEW_TOKENS, enable_timing=True)

    # Prepare fixed input for warm-up and timing
    inputs = prepare_inputs(processor, INSTRUCTION, device)

    # Warm-up: K calls to fill all pipeline stages
    print(f"  Warming up pipeline (K={MAX_NEW_TOKENS} calls)...")
    last_output = None
    for _ in range(MAX_NEW_TOKENS):
        last_output = model.generate_accelerated(
            input_ids=inputs["input_ids"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            attention_mask=inputs.get("attention_mask"),
        )
    torch.cuda.synchronize()

    action_text = processor.decode(last_output[0], skip_special_tokens=True)
    action = parse_action_text(action_text)
    print(f"  Warm-up complete. action={action[:7]}")

    # Timing
    times = []
    for i in range(NUM_RUNS):
        start = time.perf_counter()
        output_ids = model.generate_accelerated(
            input_ids=inputs["input_ids"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            attention_mask=inputs.get("attention_mask"),
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        action_text = processor.decode(output_ids[0], skip_special_tokens=True)
        action = parse_action_text(action_text)
        print(f"  Run {i + 1}: {elapsed:.3f}s, action={action[:7]}")

    avg = np.mean(times)
    print(f"\n  ActionFlow: Avg {avg:.3f}s ({1 / avg:.2f} FPS)")
    print(f"   (Pipeline depth K={MAX_NEW_TOKENS}, per-step latency)")

    print_timing_stats_qwen(model)
    return times


def main():
    parser = argparse.ArgumentParser(description="VLA-0 ActionFlow Performance Test")
    parser.add_argument("--mode", choices=["native", "actionflow", "both"], default="both",
                        help="Test mode: native, actionflow, or both (default: both)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    print(f"[*] Model: {MODEL_PATH}")
    print(f"[*] Config: HORIZON={HORIZON}, ACT_DIM={ACT_DIM}, MAX_NEW_TOKENS={MAX_NEW_TOKENS}")

    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model not found at {MODEL_PATH}")
        print("[!] Set VLA0_MODEL_PATH env var or download model first")
        sys.exit(1)

    native_times = []
    af_times = []

    if args.mode in ("native", "both"):
        native_times = test_native(device)

    if args.mode in ("actionflow", "both"):
        af_times = test_actionflow(device)

    # Summary
    if args.mode == "both" and native_times and af_times:
        native_avg = np.mean(native_times)
        af_avg = np.mean(af_times)
        speedup = native_avg / af_avg

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  Native:     {native_avg:.3f}s ({1 / native_avg:.2f} FPS)")
        print(f"  ActionFlow: {af_avg:.3f}s ({1 / af_avg:.2f} FPS)")
        print(f"  Speedup:    {speedup:.1f}x")
        print("=" * 60)


if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    main()
