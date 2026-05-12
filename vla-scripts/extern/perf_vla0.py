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
PIPELINE_DELAY = MAX_NEW_TOKENS - 1

INSTRUCTION = "pick up the object"
NUM_RUNS = 40
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


def normalize_eos_ids(eos_token_id):
    """Convert eos_token_id config to a Python set[int]."""
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, (list, tuple, set)):
        return {int(x) for x in eos_token_id}
    return {int(eos_token_id)}


def analyze_generated_tokens(token_ids: torch.Tensor, eos_ids: set[int]) -> dict:
    """
    Analyze generated token sequence:
    - length vs MAX_NEW_TOKENS
    - EOS existence and first index
    """
    ids = token_ids.detach().cpu().tolist()
    eos_positions = [idx for idx, tok in enumerate(ids) if tok in eos_ids]
    first_eos_idx = eos_positions[0] if eos_positions else None
    length = len(ids)
    return {
        "length": length,
        "over_max": length > MAX_NEW_TOKENS,
        "early_stop": length < MAX_NEW_TOKENS,
        "has_eos": len(eos_positions) > 0,
        "first_eos_idx": first_eos_idx,
    }


def summarize_alignment_records(alignment_records):
    if not alignment_records:
        return {
            "mean_max_diff": 0.0,
            "mean_mean_diff": 0.0,
            "mean_token_match": 0.0,
            "min_token_match": 0.0,
            "max_max_diff": 0.0,
            "max_mean_diff": 0.0,
        }
    return {
        "mean_max_diff": float(np.mean([x["max_diff"] for x in alignment_records])),
        "mean_mean_diff": float(np.mean([x["mean_diff"] for x in alignment_records])),
        "mean_token_match": float(np.mean([x["token_match"] for x in alignment_records])),
        "min_token_match": float(np.min([x["token_match"] for x in alignment_records])),
        "max_max_diff": float(np.max([x["max_diff"] for x in alignment_records])),
        "max_mean_diff": float(np.max([x["mean_diff"] for x in alignment_records])),
    }


def format_trace_digest(trace):
    if not trace:
        return "trace=NA"
    return (
        f"prefill={trace.get('prefill_len')} total={trace.get('total_seq_len')} "
        f"rope_mode={trace.get('rope_position_mode')} fallback={trace.get('rope_fallback_to_sequential')} "
        f"rope_head={trace.get('rope_indices_head')} rope_tail={trace.get('rope_indices_tail')} "
        f"stage_hs={trace.get('stage_hidden_lengths')} stage_ids={trace.get('stage_id_lengths')}"
    )


def analyze_token_mismatch(native_ids: torch.Tensor, af_ids: torch.Tensor) -> dict:
    native_list = native_ids.detach().cpu().tolist()
    af_list = af_ids.detach().cpu().tolist()
    min_len = min(len(native_list), len(af_list))
    mismatches = [i for i in range(min_len) if native_list[i] != af_list[i]]
    first_mismatch_idx = mismatches[0] if mismatches else None
    last_mismatch_idx = mismatches[-1] if mismatches else None
    mismatch_density = (len(mismatches) / min_len) if min_len > 0 else 0.0
    tail_window = min(8, min_len)
    tail_mismatch = 0
    if tail_window > 0:
        tail_start = min_len - tail_window
        tail_mismatch = sum(
            1 for i in range(tail_start, min_len) if native_list[i] != af_list[i]
        )
    return {
        "min_len": int(min_len),
        "first_mismatch_idx": first_mismatch_idx,
        "last_mismatch_idx": last_mismatch_idx,
        "mismatch_count": int(len(mismatches)),
        "mismatch_density": float(mismatch_density),
        "tail_mismatch_count": int(tail_mismatch),
    }


def prepare_inputs(processor, instruction, device):
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    return prepare_inputs_with_image(processor, instruction, device, image)


def prepare_inputs_with_image(processor, instruction, device, image):
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
def test_actionflow(device, rope_position_mode="model"):
    print("\n" + "=" * 60)
    print("Testing ActionFlow Accelerated VLA-0")
    print("=" * 60)
    print(f"  Pipeline delay: {PIPELINE_DELAY} rounds (native[i] ~ af[i+{PIPELINE_DELAY}])")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = load_model(device)
    model = enable_actionflow_qwen(
        model,
        max_new_tokens=MAX_NEW_TOKENS,
        enable_timing=True,
        rope_position_mode=rope_position_mode,
    )

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


@torch.inference_mode()
def debug_output_alignment(
    device,
    compare_rounds,
    debug_trace=False,
    verbose=True,
    rope_position_mode="sequential",
):
    """
    Compare native and ActionFlow outputs with pipeline-delay alignment.
    Native round i is aligned with AF round i + (K-1), where K=max_new_tokens.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Debug Native vs ActionFlow Round Alignment")
        print("=" * 60)
        print(f"  Compare rounds: {compare_rounds}")
        print(f"  Pipeline depth K={MAX_NEW_TOKENS}, delay={PIPELINE_DELAY}")
        print(f"  Alignment rule: native[i] <-> af[i + {PIPELINE_DELAY}]")

    # Build a long fixed image stream.
    # We keep all AF rounds unique to avoid repeated-tail inputs masking alignment.
    lag_sweep = 2
    total_af_rounds = compare_rounds + PIPELINE_DELAY + lag_sweep
    images = [
        Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        for _ in range(total_af_rounds)
    ]

    # --- Native pass ---
    processor_native = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model_native = load_model(device)
    native_eos_ids = normalize_eos_ids(model_native.generation_config.eos_token_id)
    if verbose:
        print(f"  Native eos_token_ids={sorted(native_eos_ids)}")
    native_actions = []
    native_tokens = []
    native_token_stats = []
    for i, image in enumerate(images[:compare_rounds]):
        inputs = prepare_inputs_with_image(processor_native, INSTRUCTION, device, image)
        output_ids = model_native.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        native_tokens.append(generated_ids[0].detach().cpu())
        stats = analyze_generated_tokens(generated_ids[0], native_eos_ids)
        native_token_stats.append(stats)
        action_text = processor_native.decode(generated_ids[0], skip_special_tokens=True)
        action = parse_action_text(action_text)
        native_actions.append(action)
        if verbose:
            print(
                f"  Native Round {i:2d}: len={stats['length']:2d}, "
                f"early_stop={stats['early_stop']}, has_eos={stats['has_eos']}, "
                f"first_eos_idx={stats['first_eos_idx']}, action={action[:7]}"
            )

    del model_native
    torch.cuda.empty_cache()

    # --- ActionFlow pass ---
    processor_af = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model_af = load_model(device)
    model_af = enable_actionflow_qwen(
        model_af,
        max_new_tokens=MAX_NEW_TOKENS,
        enable_timing=False,
        enable_debug_trace=debug_trace,
        rope_position_mode=rope_position_mode,
    )
    af_eos_ids = normalize_eos_ids(model_af.generation_config.eos_token_id)
    if verbose:
        print(f"  AF eos_token_ids={sorted(af_eos_ids)}")
        print(f"  AF warm-up {MAX_NEW_TOKENS} rounds before collection...")
    warm_inputs = prepare_inputs_with_image(processor_af, INSTRUCTION, device, images[0])
    for _ in range(MAX_NEW_TOKENS):
        _ = model_af.generate_accelerated(
            input_ids=warm_inputs["input_ids"],
            pixel_values=warm_inputs.get("pixel_values"),
            image_grid_thw=warm_inputs.get("image_grid_thw"),
            attention_mask=warm_inputs.get("attention_mask"),
            debug_trace=debug_trace,
        )

    af_actions_by_round = {}
    af_tokens_by_round = {}
    af_token_stats_by_round = {}
    af_trace_by_round = {}
    for r in range(total_af_rounds):
        image = images[r]
        inputs = prepare_inputs_with_image(processor_af, INSTRUCTION, device, image)
        output_ids = model_af.generate_accelerated(
            input_ids=inputs["input_ids"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            attention_mask=inputs.get("attention_mask"),
            debug_trace=debug_trace,
        )

        af_tokens_by_round[r] = output_ids[0].detach().cpu()
        af_token_stats_by_round[r] = analyze_generated_tokens(output_ids[0], af_eos_ids)
        af_trace_by_round[r] = getattr(model_af.actionflow_engine, "_last_debug_trace", {})
        action_text = processor_af.decode(output_ids[0], skip_special_tokens=True)
        action = parse_action_text(action_text)
        af_actions_by_round[r] = action

        if verbose and (r < 3 or r >= PIPELINE_DELAY):
            stats = af_token_stats_by_round[r]
            print(
                f"  AF Round {r:2d}: len={stats['length']:2d}, "
                f"early_stop={stats['early_stop']}, has_eos={stats['has_eos']}, "
                f"first_eos_idx={stats['first_eos_idx']}, action={action[:7]}"
            )
            if debug_trace:
                print(f"             {format_trace_digest(af_trace_by_round[r])}")

    # --- Aligned compare ---
    if verbose:
        print("\n  Aligned compare:")
    alignment_records = []
    for i in range(compare_rounds):
        af_round = i + PIPELINE_DELAY
        native_action = native_actions[i]
        af_action = af_actions_by_round[af_round]
        abs_diff = np.abs(native_action - af_action)
        native_ids = native_tokens[i]
        af_ids = af_tokens_by_round[af_round]
        n_stats = native_token_stats[i]
        a_stats = af_token_stats_by_round[af_round]
        min_len = min(native_ids.shape[0], af_ids.shape[0])
        token_match = (native_ids[:min_len] == af_ids[:min_len]).float().mean().item()
        token_mismatch = analyze_token_mismatch(native_ids, af_ids)
        record = {
            "native_round": int(i),
            "af_round": int(af_round),
            "max_diff": float(abs_diff.max()),
            "mean_diff": float(abs_diff.mean()),
            "token_match": float(token_match),
            "native_length": int(n_stats["length"]),
            "af_length": int(a_stats["length"]),
            "native_has_eos": bool(n_stats["has_eos"]),
            "af_has_eos": bool(a_stats["has_eos"]),
            "token_mismatch": token_mismatch,
        }
        alignment_records.append(record)
        if verbose:
            print(
                f"    native[{i:2d}] vs af[{af_round:2d}] | "
                f"max|diff|={record['max_diff']:.4f}, mean|diff|={record['mean_diff']:.4f}, "
                f"token_match={record['token_match']:.2%}, "
                f"len(n/a)=({record['native_length']}/{record['af_length']}), "
                f"eos(n/a)=({record['native_has_eos']}/{record['af_has_eos']})"
            )
            tm = record["token_mismatch"]
            print(
                f"      token_diff: first={tm['first_mismatch_idx']}, last={tm['last_mismatch_idx']}, "
                f"density={tm['mismatch_density']:.2%}, tail(8)={tm['tail_mismatch_count']}"
            )

    # --- Lag sweep around nominal delay ---
    if verbose:
        print("\n  Lag sweep (mean over compared rounds):")
    lag_metrics = {}
    for lag in range(PIPELINE_DELAY - lag_sweep, PIPELINE_DELAY + lag_sweep + 1):
        max_diffs = []
        mean_diffs = []
        token_matches = []
        for i in range(compare_rounds):
            af_action = af_actions_by_round[i + lag]
            abs_diff = np.abs(native_actions[i] - af_action)
            max_diffs.append(abs_diff.max())
            mean_diffs.append(abs_diff.mean())
            native_ids = native_tokens[i]
            af_ids = af_tokens_by_round[i + lag]
            min_len = min(native_ids.shape[0], af_ids.shape[0])
            token_match = (native_ids[:min_len] == af_ids[:min_len]).float().mean().item()
            token_matches.append(token_match)
        lag_metrics[lag] = {
            "mean_max_diff": float(np.mean(max_diffs)),
            "mean_mean_diff": float(np.mean(mean_diffs)),
            "mean_token_match": float(np.mean(token_matches)),
        }
        if verbose:
            print(
                f"    lag={lag:2d} | "
                f"mean(max|diff|)={lag_metrics[lag]['mean_max_diff']:.4f}, "
                f"mean(mean|diff|)={lag_metrics[lag]['mean_mean_diff']:.4f}, "
                f"mean(token_match)={lag_metrics[lag]['mean_token_match']:.2%}"
            )

    # --- Stop condition summary ---
    native_over_max = sum(1 for s in native_token_stats if s["over_max"])
    native_early_stop = sum(1 for s in native_token_stats if s["early_stop"])
    native_with_eos = sum(1 for s in native_token_stats if s["has_eos"])

    aligned_af_stats = [af_token_stats_by_round[i + PIPELINE_DELAY] for i in range(compare_rounds)]
    af_over_max = sum(1 for s in aligned_af_stats if s["over_max"])
    af_early_stop = sum(1 for s in aligned_af_stats if s["early_stop"])
    af_with_eos = sum(1 for s in aligned_af_stats if s["has_eos"])

    if verbose:
        print("\n  Token stop-condition summary (aligned rounds):")
        print(
            f"    Native: over_max={native_over_max}/{compare_rounds}, "
            f"early_stop={native_early_stop}/{compare_rounds}, "
            f"has_eos={native_with_eos}/{compare_rounds}"
        )
        print(
            f"    AF    : over_max={af_over_max}/{compare_rounds}, "
            f"early_stop={af_early_stop}/{compare_rounds}, "
            f"has_eos={af_with_eos}/{compare_rounds}"
        )

    aligned_summary = summarize_alignment_records(alignment_records)
    worst_rounds = sorted(
        alignment_records,
        key=lambda x: (x["token_match"], -x["max_diff"], -x["mean_diff"]),
    )[: min(3, len(alignment_records))]
    if verbose and worst_rounds:
        print("\n  Worst aligned rounds:")
        for wr in worst_rounds:
            tm = wr["token_mismatch"]
            print(
                f"    native[{wr['native_round']}] vs af[{wr['af_round']}]: "
                f"token_match={wr['token_match']:.2%}, max|diff|={wr['max_diff']:.4f}, "
                f"first_mismatch={tm['first_mismatch_idx']}, density={tm['mismatch_density']:.2%}"
            )
    nominal_lag_summary = lag_metrics.get(PIPELINE_DELAY, {})
    return {
        "compare_rounds": int(compare_rounds),
        "pipeline_delay": int(PIPELINE_DELAY),
        "aligned_records": alignment_records,
        "aligned_summary": aligned_summary,
        "worst_rounds": worst_rounds,
        "lag_metrics": lag_metrics,
        "nominal_lag_summary": nominal_lag_summary,
        "stop_summary": {
            "native": {
                "over_max": int(native_over_max),
                "early_stop": int(native_early_stop),
                "has_eos": int(native_with_eos),
                "total": int(compare_rounds),
            },
            "af_aligned": {
                "over_max": int(af_over_max),
                "early_stop": int(af_early_stop),
                "has_eos": int(af_with_eos),
                "total": int(compare_rounds),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="VLA-0 ActionFlow Performance Test")
    parser.add_argument("--mode", choices=["native", "actionflow", "both", "debug"], default="both",
                        help="Test mode: native, actionflow, both, or debug (default: both)")
    parser.add_argument("--compare-rounds", type=int, default=6,
                        help="Rounds to compare in --mode debug (default: 6)")
    parser.add_argument("--debug-trace", action="store_true",
                        help="Enable stage/rope trace dump in --mode debug")
    parser.add_argument("--rope-position-mode", choices=["sequential", "model"], default="model",
                        help="RoPE position mode for ActionFlow Qwen path")
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
        af_times = test_actionflow(device, rope_position_mode=args.rope_position_mode)

    if args.mode == "debug":
        debug_output_alignment(
            device,
            compare_rounds=args.compare_rounds,
            debug_trace=args.debug_trace,
            verbose=True,
            rope_position_mode=args.rope_position_mode,
        )

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
