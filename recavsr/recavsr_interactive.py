#!/usr/bin/env python3
"""recavsr_interactive - upscale a video with ReCaVSR, from Python, no workflow JSON.

Five cells, in the order you run them:

    0. imports            everything up front - torch, ComfyUI and the node pack all load here
    1. weights + load     which weight files, check them, load transformer + FlashDecoder
    2. parameters + run   the clip, the settings, and the upscale itself
    3. shrink             the 4x file is bigger than you want to keep - shrink it on disk
    4. free               the card back

Re-run cell 2 for every further clip; the model stays loaded from cell 1. Cell 3 needs no GPU at
all, so it also works on a render from last week. Cell 4 only when you are done with the session.

Measured on this box (RTX 5090, int8, fast kernels): 1248x720 at 3.24 fps, 2496x1440 at
3.50 fps - see README.md.

KERNEL: the ComfyUI venv, selected as the Jupyter kernel (Ctrl+Enter follows the kernel, not
the status bar). The repo's .venv works too but its comfy-kitchen has no window_attention, so
the driver drops to `exact` and decode is several times slower.
Pylance resolves the recavsr_* imports through python.analysis.extraPaths -> recavsr/lib in
.vscode/settings.json; it never runs cell 0's sys.path.insert.
"""

# %% 0. imports - all of it, up front
import os
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "lib"))

GPU = os.environ.get("CUDA_GPU", "0")          # 1 and 2 are held by the strata pair on this box
if GPU != "":
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU   # before torch loads - ComfyUI reads the device once

import recavsr_load as L                       # paths, weights, load(), fit(), headroom(), free()
from recavsr_engine import upscale             # the streaming loop
from recavsr_media import list_inputs, probe, shrink   # what the clip is, and the shrink pass

## %% 1. the weight files, and load them
WEIGHTS = "int8"            # "int8" 6.0 GB and ~1.8x faster, or "bf16" 10.1 GB
DECODER = "flashdecoder"    # "vae" needs vae.safetensors and is ~30x slower to decode

print("weights dir:", L.MODELS)
print("missing:", L.check(WEIGHTS) or "none")

ENGINE = L.engine()
L.load(ENGINE, WEIGHTS)
if DECODER == "vae":
    L.load_vae(ENGINE)
print(ENGINE["model"].precision, "|", L.headroom())

# %% 2. upscale parameters, then the run
# INPUT       = BASE / "inputs" / "demo_input.mp4"   # a Path or a plain string, either is fine
INPUT       = "/home/andrewzhu/storage_1t_1/az_git_folder/az-comfy-code/minimax_h3/outputs/h3_fast_flex/interactive_fast_832x1184_73.mp4"
OUT_DIR     = BASE / "outputs"
MAX_FRAMES  = 96                                 # 0 = whole clip
START_FRAME = 0

SCALE       = 4.0            # output = input x scale. Trained for 4; 2 and 1.5 are the others
KERNELS     = "fast"         # "exact" reproduces the reference bit for bit, and is slower
KV_OFFLOAD  = False          # True: the attention history lives in system RAM - big frames, small card
COLOR_FIX   = "none"         # "adain" or "wavelet" pull the colour back to the input
SEED        = 42             # the single denoising step's noise
POST_SIZE   = None           # (height, width) to shrink to after the model, e.g. render x4 -> 1248x720

INPUT = pathlib.Path(INPUT)          # cell 2 is the one you edit, so accept a string too
if not INPUT.is_file():
    raise SystemExit(f"no such file: {INPUT}\ninputs/ has: {[p.name for p in list_inputs(BASE / 'inputs')]}")

INFO    = probe(INPUT)
MODEL_H = int(INFO["height"] * SCALE + 0.5)
MODEL_W = int(INFO["width"] * SCALE + 0.5)
OUT_HW  = POST_SIZE or (MODEL_H, MODEL_W)
OUT     = OUT_DIR / f"{INPUT.stem}_x{SCALE:g}_{OUT_HW[1]}x{OUT_HW[0]}.mp4"
OUT.parent.mkdir(parents=True, exist_ok=True)
print(INFO, "->", f"{OUT_HW[1]}x{OUT_HW[0]}")

# does this size fit the card, and what has to be switched on to make it fit
CHUNK, KV_OFFLOAD, NEED = L.fit(MODEL_H, MODEL_W, KV_OFFLOAD)
print(f"step needs {NEED / 2**30:.1f} GiB; input projection "
      f"{L.LQ_CHUNK_FRAMES if CHUNK else 'whole block'} frames at a time; "
      f"attention history {'in RAM' if KV_OFFLOAD else 'in VRAM'}")

STATS = upscale(ENGINE, str(INPUT), str(OUT), scale=SCALE, color_fix=COLOR_FIX, seed=SEED,
                decoder=DECODER, kernels=KERNELS, kv_offload=KV_OFFLOAD, max_frames=MAX_FRAMES,
                start_frame=START_FRAME, post_size=POST_SIZE)
print(STATS["frames"], "frames", STATS["in_size"], "->", STATS["out_size"],
      f"in {STATS['seconds']:.1f} s = {STATS['measured_fps']:.2f} fps",
      f"(transformer {STATS['transformer_s']:.1f} s, decode {STATS['decode_s']:.1f} s) -> {OUT}")

# %% 3. shrink before saving - a 4x file is bigger than you want to keep
SHRINK   = "0.5"           # "0.5" half each way, "720" a target height, "1248x720" exact, "" = skip
CRF      = 20              # hevc quality, lower = bigger file; 20 is generous for a shrunk copy
KEEP_BIG = True            # True keeps the big render too; False replaces it with the shrunk copy

# Re-runnable on its own: if this kernel was started before this cell existed, `shrink` is not
# in the namespace and OUT may not either - so pick both up rather than failing with a NameError.
try:
    shrink
except NameError:
    from recavsr_media import shrink
try:
    OUT
except NameError:
    OUT = max((BASE / "outputs").glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    print("cell 2 has not run in this kernel; using the newest file in outputs/:", OUT.name)

# This re-encodes the file that cell 2 just wrote - no model, no GPU render, so it costs seconds
# and can be re-run over and over. For a 4x render destined for a smaller size, POST_SIZE in
# cell 2 is the better path (it shrinks the decoded frames before the encode, so one pass and no
# re-encode of a huge file); this cell is for shrinking something that is already on disk.
if SHRINK:
    SMALL = shrink(OUT, size=SHRINK, crf=CRF, keep_original=KEEP_BIG)

# %% 4. clean up VRAM
L.free(ENGINE)
