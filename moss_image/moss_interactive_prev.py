# %% [markdown]
# # MOSS-Image, PRE-SPLIT version - for A/B against moss_interactive.py
#
# Same single-call generation as before the three-stage split, with the DiT precision
# exposed as the one variable. Use it to compare dtypes on identical code:
#
#   "fp32"  the reference - the image you confirmed correct
#   "fp16"  2x faster, half the memory, 5.4/255 mean pixel difference from fp32
#   "bf16"  equally fast but wrong here: 41.4/255 and 70% of pixels changed at a fixed seed
#
# The three-stage file is bit-identical to this one at equal dtype (measured 0.000), so
# whatever differs between runs below is precision, not the code path.

# %% 1. configuration
DIT_DTYPE = "fp32"              # "fp32" (reference) | "fp16" (fast, close) | "bf16" (broken)
LORA_PATH = "TURBO"             # "TURBO", a path to a .safetensors, or "" for no LoRA
PROMPT = "beautiful girl"
HEIGHT = WIDTH = 1024
STEPS, CFG, SEED = 8, 1.0, 42   # upstream non-distilled defaults: 50 steps, CFG 3
GPU = "0"

# %% 2. load
import os
import pathlib
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", GPU)
# Anchored to this file, not to the kernel's working directory and not to the old
# az_samples copy: sys.path and the output folder both follow this repo.
# (__file__ is missing in some interactive kernels, hence the cwd fallback.)
BASE = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path.cwd()
HERE = BASE                                   # import path for moss_image_gen
OUT_DIR = BASE / "outputs"                    # generated images land here (git-ignored)
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

import moss_image_gen as moss

moss.DIT_DTYPE = DIT_DTYPE      # must be set before the weights are built
pipe, _ = moss.build_pipeline(dtype=DIT_DTYPE)
moss.set_memory_mode(pipe, "model", str(moss.DISK_CACHE))

# %% 3. LoRA
lora = str(moss.TURBO_LORA) if LORA_PATH == "TURBO" else LORA_PATH
moss.apply_lora(pipe.transformer, lora, 1.0)

# %% 4. generate, in one call
SKIP_LAYERS = [] if CFG <= 1.0 else [22]
tag = f"{DIT_DTYPE}_{'lora' if lora else 'nolora'}_{SEED}"
out = OUT_DIR / f"prev_{tag}.png"

image = moss.generate_image(pipe, PROMPT, height=HEIGHT, width=WIDTH, steps=STEPS,
                            cfg=CFG, seed=SEED, skip_layers=SKIP_LAYERS, out=out)
moss.log(f"{out.name}: DiT {next(pipe.transformer.parameters()).dtype}, "
         f"{'LoRA on' if lora else 'no LoRA'}, {STEPS} steps, CFG {CFG}, seed {SEED}")
image
