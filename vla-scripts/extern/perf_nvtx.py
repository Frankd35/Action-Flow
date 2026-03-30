import time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import sys
import os
import argparse
import torch.cuda.nvtx as nvtx

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from actionflow import enable_actionflow, print_timing_stats
    from actionflow.modeling.layers import LlamaPIPEDecodeLayer
    print("✅ [ActionFlow] Package imported successfully.")
except ImportError as e:
    print(f"❌ [ActionFlow] Import failed: {e}")
    sys.exit(1)

# === Configuration
MODEL_PATH = "/home/daiyuntao/jetson-containers/data/models/huggingface/models--openvla--openvla-7b/snapshots/31f090d05236101ebfc381b61c674dd4746d4ce0"
INSTRUCTION = "put spoon on towel"

def monkey_patch_breakdown():
    """Add NVTX markers to internal components of LlamaPIPEDecodeLayer."""
    print("[*] Monkey-patching LlamaPIPEDecodeLayer for breakdown profiling...")
    
    original_packed_forward = LlamaPIPEDecodeLayer.packed_forward
    
    # We also need to import kernel wrappers to wrap them
    import actionflow.kernels.ops as ops
    from flash_attn import flash_attn_varlen_func
    
    def wrapped_packed_forward(self, *args, **kwargs):
        # We can't easily wrap inner parts without re-implementing, 
        # so we'll wrap the main blocks by replacing the calls in the method if possible,
        # or just wrap the whole thing with more context.
        # Actually, let's just wrap the key calls in the original method by patching the instance/class methods it calls.
        
        with nvtx.range("LlamaPIPEDecodeLayer.packed_forward"):
            return original_packed_forward(self, *args, **kwargs)

    LlamaPIPEDecodeLayer.packed_forward = wrapped_packed_forward

    # Patch projections in the original layer
    # These are nn.Linear (or bnb.nn.Linear4bit)
    def wrap_linear(layer, name):
        orig_forward = layer.forward
        def new_forward(*args, **kwargs):
            with nvtx.range(name):
                return orig_forward(*args, **kwargs)
        layer.forward = new_forward

    # This is tricky because we need to patch the weights/layers inside each decoder layer
    # We'll do this during model loading.

def get_openvla_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

@torch.inference_mode()
def run_perf():
    parser = argparse.ArgumentParser(description="Enhanced Performance Profiling for OpenVLA")
    parser.add_argument("--dtype", type=str, default="int4", choices=["bf16", "int4"], help="Data type")
    parser.add_argument("--use_af", type=int, default=1, choices=[0, 1], help="Use ActionFlow")
    parser.add_argument("--runs", type=int, default=20, help="Number of iterations")
    parser.add_argument("--profile", type=int, default=0, choices=[0, 1], help="Enable torch.profiler")
    cli_args = parser.parse_args()

    device = torch.device("cuda")
    print(f"[*] Config: dtype={cli_args.dtype}, use_af={cli_args.use_af}, runs={cli_args.runs}, profile={cli_args.profile}")

    # Load Processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Load Model
    if cli_args.dtype == "bf16":
        print("[*] Loading in BF16")
        vla = AutoModelForVision2Seq.from_pretrained(
            MODEL_PATH, attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        ).to(device)
    else:
        print("[*] Loading in INT4-BNB")
        vla = AutoModelForVision2Seq.from_pretrained(
            MODEL_PATH, attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
            ),
            low_cpu_mem_usage=True, trust_remote_code=True
        )

    # Apply Layer Breakdown NVTX Markers (Always Enabled)
    monkey_patch_breakdown()
    # Patch linear layers in all decoder blocks
    for i, layer in enumerate(vla.language_model.model.layers):
        # Projections
        def wrap_module(m, name):
            orig_f = m.forward
            def new_f(*args, **kwargs):
                input_tensor = args[0] if len(args) > 0 else kwargs.get("input", None)
                if input_tensor is not None and torch.is_tensor(input_tensor):
                    # Attach metadata to the tensor itself
                    try:
                        input_tensor._af_name = name
                    except:
                        pass
                with nvtx.range(name): return orig_f(*args, **kwargs)
            m.forward = new_f

        wrap_module(layer.self_attn.q_proj, f"L{i}.Q_proj")
        wrap_module(layer.self_attn.k_proj, f"L{i}.K_proj")
        wrap_module(layer.self_attn.v_proj, f"L{i}.V_proj")
        wrap_module(layer.self_attn.o_proj, f"L{i}.O_proj")
        wrap_module(layer.mlp.gate_proj, f"L{i}.Gate_proj")
        wrap_module(layer.mlp.up_proj, f"L{i}.Up_proj")
        wrap_module(layer.mlp.down_proj, f"L{i}.Down_proj")

    if cli_args.use_af:
        print("[*] Enabling ActionFlow...")
        vla = enable_actionflow(vla, max_new_tokens=7, enable_timing=True)

    prompt = get_openvla_prompt(INSTRUCTION)
    image = Image.fromarray(np.asarray(np.random.rand(256, 256, 3) * 255, dtype=np.uint8))
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

    # Warm-up
    print("[*] Warm-up...")
    with nvtx.range("warmup"):
        if cli_args.use_af:
            _ = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        else:
            _ = vla.generate(**inputs, max_new_tokens=7, do_sample=False)

    # Profiler setup
    prof = None
    if cli_args.profile:
        log_dir = f"/home/daiyuntao/profile_results/tb_logs/exp_{cli_args.dtype}_{'af' if cli_args.use_af else 'noaf'}"
        os.makedirs(log_dir, exist_ok=True)
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=True, with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
        )
        prof.start()

    # Timing runs
    times = []
    print(f"[*] Running {cli_args.runs} iterations...")
    for i in range(cli_args.runs):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with nvtx.range(f"iter_{i}"):
            if cli_args.use_af:
                with nvtx.range("predict_action"):
                    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            else:
                with nvtx.range("generate"):
                    action = vla.generate(**inputs, max_new_tokens=7, do_sample=False)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        times.append(elapsed)
        
        if prof: prof.step()
        
        if (i+1) % 5 == 0 or i == 0:
            print(f"\t=>> Iter {i + 1:2d}: Time = {elapsed:.4f}s")

    if prof: prof.stop()

    # Stats
    times = np.array(times)
    print("\n" + "=" * 60)
    print(f"📊 Results ({cli_args.dtype}, AF={cli_args.use_af}):")
    print(f"   Avg Time: {times.mean():.4f} ± {times.std():.4f} s")
    print(f"   Avg FPS : {1.0/times.mean():.2f}")
    print("=" * 60)

    if cli_args.use_af:
        print_timing_stats(vla)

if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    run_perf()
