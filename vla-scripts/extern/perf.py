import time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from actionflow import enable_actionflow
    print("✅ [ActionFlow] Package imported successfully.")
except ImportError as e:
    print(f"❌ [ActionFlow] Import failed: {e}")
    sys.exit(1)


# === Verification Arguments
MODEL_PATH = "openvla/openvla-7b"
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
INSTRUCTION = "put spoon on towel"


def get_openvla_prompt(instruction: str) -> str:
    if "v01" in MODEL_PATH:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


@torch.inference_mode()
def verify_openvla() -> None:
    print(f"[*] Verifying OpenVLAForActionPrediction using Model `{MODEL_PATH}`")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[*] Using Device: {device}")

    # Load Processor & VLA
    print("[*] Instantiating Processor and Pretrained OpenVLA")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # === BFLOAT16 + FLASH-ATTN MODE ===
    print("[*] Loading in BF16 with Flash-Attention Enabled")
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    
    print("[*] Injecting ActionFlow Acceleration...")
    vla = enable_actionflow(vla, max_new_tokens=7)

    prompt = get_openvla_prompt(INSTRUCTION)

    # Warm-up run (optional but recommended for timing accuracy)
    print("[*] Running warm-up inference...")
    warmup_image = Image.fromarray(np.asarray(np.random.rand(256, 256, 3) * 255, dtype=np.uint8))
    warmup_inputs = processor(prompt, warmup_image).to(device, dtype=torch.bfloat16)
    # _ = vla.predict_action(**warmup_inputs, unnorm_key="bridge_orig", do_sample=False)
    _ = vla.predict_action(**warmup_inputs, unnorm_key="bridge_orig", do_sample=False)

    # Actual timing runs
    num_runs = 20
    times = []

    print(f"[*] Running {num_runs} inference iterations for timing...")
    for i in range(num_runs):
        image = Image.fromarray(np.asarray(np.random.rand(256, 256, 3) * 255, dtype=np.uint8))
        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

        start_time = time.perf_counter()
        # action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        elapsed = time.perf_counter() - start_time
        times.append(elapsed)
        print(f"\t=>> Iter {i+1:2d}: Time = {elapsed:.4f}s || Action = {action}")

    # Compute stats
    times = np.array(times)
    mean_time = times.mean()
    std_time = times.std()
    fps = 1.0 / times
    mean_fps = fps.mean()
    std_fps = fps.std()

    print("\n" + "="*60)
    print(f"📊 Inference Timing Results ({num_runs} runs):")
    print(f"   Avg Time: {mean_time:.4f} ± {std_time:.4f} seconds")
    print(f"   Avg FPS : {mean_fps:.2f} ± {std_fps:.2f}")
    print("="*60)


import random
if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    verify_openvla()
