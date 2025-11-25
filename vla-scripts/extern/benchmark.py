"""
benchmark.py

Benchmark OpenVLA inference latency under varying prefill and decode lengths.
Measures end-to-end generation time for different prompt (prefill) and output (decode) lengths.
Supports both standard Hugging Face autoregressive generation and accelerated ActionFlow pipeline.
"""

import argparse
import time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from typing import Dict, Tuple, Optional

import sys
import os


# === Path Setup ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from actionflow import enable_actionflow
    print("✅ [ActionFlow] Package imported successfully.")
except ImportError as e:
    print(f"❌ [ActionFlow] Import failed: {e}")
    sys.exit(1)


# === Configuration ===
MODEL_PATH: str = "openvla/openvla-7b"
SYSTEM_PROMPT: str = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
INSTRUCTION: str = "put spoon on towel"

PREFILL_LENGTHS: list = [16, 32, 64, 128, 256]   # Target text token count (image tokens ~256 fixed)
DECODE_LENGTHS: list = [7, 16, 24, 32]           # Number of generated action tokens
NUM_RUNS: int = 20                                # Number of timed runs per configuration (after warmup)


def get_openvla_prompt(instruction: str) -> str:
    """
    Generate the appropriate prompt template based on model version.
    v01 models use one format; others use a newer schema.

    Args:
        instruction: User task description (e.g., "pick up the cup")

    Returns:
        Formatted input prompt string
    """
    if "v01" in MODEL_PATH:
        return (
            f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
        )
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def construct_prompt_by_token_length(
    prompt_template: str,
    target_len: int,
    tokenizer
) -> str:
    """
    Constructs a prompt with approximately `target_len` tokens by padding with safe phrases.

    Args:
        prompt_template: Base prompt string
        target_len: Desired total number of tokens
        tokenizer: Hugging Face tokenizer

    Returns:
        Prompt string that decodes to close to `target_len` tokens
    """
    tokens = tokenizer(prompt_template, return_tensors="pt")["input_ids"][0]
    current_len = len(tokens)

    if current_len >= target_len:
        # Truncate if already too long
        truncated_ids = tokens[:target_len]
        return tokenizer.decode(truncated_ids, skip_special_tokens=True)
    else:
        # Pad using a neutral phrase until desired length
        pad_phrase = " continue."
        prompt = prompt_template
        while True:
            prompt += pad_phrase
            new_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
            if len(new_tokens) >= target_len:
                return tokenizer.decode(new_tokens[:target_len], skip_special_tokens=True)
            if len(prompt) > 2000:  # Prevent infinite loop
                break
        return prompt


@torch.inference_mode()
def benchmark_prefill_decode(use_pipe: bool = False):
    """
    Benchmark OpenVLA inference across various prefill and decode lengths.

    Args:
        use_pipe: If True, uses ActionFlow-accelerated pipeline instead of HF `.generate()`
    """
    print(f"[*] Loading OpenVLA model: `{MODEL_PATH}`")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("[*] Loading model in bfloat16 with FlashAttention-2...")
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    base_prompt = get_openvla_prompt(INSTRUCTION)
    print(f"[*] Base prompt template: {base_prompt}")

    mode_label = "ActionFlow Accelerated (PIPE)" if use_pipe else "Standard HuggingFace (Autoregressive)"
    print(f"\n🚀 BENCHMARK MODE: {mode_label}")

    results: Dict[Tuple[int, int], dict] = {}

    for prefill_len in PREFILL_LENGTHS:
        print(f"\n{'='*60}")
        print(f"Testing Prefill Length (text tokens): {prefill_len}")
        print(f"{'='*60}")

        # Construct prompt with target token count
        prompt_text = construct_prompt_by_token_length(base_prompt, prefill_len, processor.tokenizer)
        actual_prefill_tokens = len(processor.tokenizer(prompt_text, return_tensors="pt")["input_ids"][0])
        print(f" → Actual text tokens: {actual_prefill_tokens}")

        for decode_len in DECODE_LENGTHS:
            print(f"\n  Decode Length: {decode_len}")
            times = []

            # Configure ActionFlow if enabled
            if use_pipe:
                # Re-initialize or patch model with updated max_new_tokens
                vla = enable_actionflow(vla, max_new_tokens=decode_len)

            # Warm-up + timing loop
            for run_idx in range(NUM_RUNS + 2):  # 2 warmup runs
                # Simulate random image input
                image_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                image = Image.fromarray(image_array)

                # Prepare model inputs
                inputs = processor(prompt_text, image, return_tensors="pt").to(device, dtype=torch.bfloat16)

                # Define execution function
                if use_pipe:
                    run_func = lambda: vla.predict_action(**inputs, unnorm_key="bridge_orig")
                else:
                    run_func = lambda: vla.generate(
                        **inputs,
                        max_new_tokens=decode_len,
                        min_new_tokens=decode_len,
                        do_sample=False,
                        suppress_tokens=[processor.tokenizer.eos_token_id],
                    )

                # Skip timing for warmup
                if run_idx < 2:
                    _ = run_func()
                    continue

                # Synchronize before timing (for CUDA accuracy)
                if device.type == "cuda":
                    torch.cuda.synchronize()

                start_time = time.perf_counter()
                _ = run_func()
                if device.type == "cuda":
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start_time
                times.append(elapsed)

            # Compute performance metrics
            times_np = np.array(times)
            mean_time = times_np.mean()
            std_time = times_np.std()
            fps_vals = 1.0 / times_np
            mean_fps = fps_vals.mean()
            std_fps = fps_vals.std()

            results[(prefill_len, decode_len)] = {
                "mean_time": mean_time,
                "std_time": std_time,
                "mean_fps": mean_fps,
                "std_fps": std_fps,
                "actual_prefill_tokens": actual_prefill_tokens,
            }

            print(f"    → Avg Time: {mean_time:.4f} ± {std_time:.4f} s")
            print(f"    → Avg FPS : {mean_fps:.2f} ± {std_fps:.2f}")

    # ==============================
    # SUMMARY TABLE 1: Latency (seconds)
    # ==============================
    print("\n" + "=" * 80)
    print(f"SUMMARY 1/2: Inference Latency (seconds) | Mode: {'ActionFlow' if use_pipe else 'Standard'}")
    print("=" * 80)

    header = f"{'Total Tokens':<18}" + "".join([f"{dl:>14}" for dl in DECODE_LENGTHS])
    print(header)
    print("-" * len(header))

    for pl in PREFILL_LENGTHS:
        row = f"{(256 + pl):<18}"  # 256 image tokens + text tokens
        for dl in DECODE_LENGTHS:
            r = results[(pl, dl)]
            cell = f"{r['mean_time']:.4f}±{r['std_time']:.4f}"
            row += f"{cell:>14}"
        print(row)

    # ==============================
    # SUMMARY TABLE 2: Throughput (FPS)
    # ==============================
    print("\n" + "=" * 80)
    print(f"SUMMARY 2/2: Throughput (FPS) | Mode: {'ActionFlow' if use_pipe else 'Standard'}")
    print("=" * 80)

    header = f"{'Total Tokens':<18}" + "".join([f"{dl:>14}" for dl in DECODE_LENGTHS])
    print(header)
    print("-" * len(header))

    for pl in PREFILL_LENGTHS:
        row = f"{(256 + pl):<18}"
        for dl in DECODE_LENGTHS:
            r = results[(pl, dl)]
            cell = f"{r['mean_fps']:.2f}±{r['std_fps']:.2f}"
            row += f"{cell:>14}"
        print(row)

    print("\n💡 Note: 'Prefill' refers to variable-length text tokens. Image tokens are fixed at ~256.")
    print("✅ Benchmark completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenVLA Inference Latency Benchmark")
    parser.add_argument(
        "--use_pipe",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use ActionFlow acceleration pipeline (1 = True, 0 = False)"
    )
    args = parser.parse_args()

    # Set seed for reproducibility
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Run benchmark
    benchmark_prefill_decode(use_pipe=bool(args.use_pipe))
