#!/usr/bin/env python3
"""recavsr_interactive - upscale a video with ReCaVSR, from Python, no workflow JSON.

Run the cells top to bottom once, then re-run cell 6 for every clip you want to do. ReCaVSR is
one model plus its own decoder, loaded in one cell; the streaming loop runs per block, so a long
clip costs time and disk rather than memory.

Measured on this box: see README.md.

KERNEL: the ComfyUI venv -
    /mnt/data_2t_3/github_repos/ComfyUI/venv_py312_comfyui/bin/python
Pylance needs `recavsr/lib` in python.analysis.extraPaths (.vscode/settings.json, git-ignored)
or every recavsr_* import in a cell shows unresolved - it never runs cell 0's sys.path.insert.
"""

# %% 0. setup
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "lib"))

import recavsr_load as L

## %% 1. what to upscale
INPUT      = BASE / "inputs" / "demo_input.mp4"      # list_inputs() below shows what is there
OUT_DIR    = BASE / "outputs"
MAX_FRAMES = 96                                      # 0 = whole clip
START_FRAME = 0

## %% 2. the settings that matter
SCALE       = 4.0            # output = input x scale. Trained for 4; 2 and 1.5 are the others
WEIGHTS     = "int8"         # "int8" 6.0 GB and ~1.8x faster, or "bf16" 10.1 GB
DECODER     = "flashdecoder" # "vae" needs vae.safetensors and is ~30x slower to decode
KERNELS     = "fast"         # "exact" reproduces the reference bit for bit, and is slower
KV_OFFLOAD  = False          # True: the attention history lives in system RAM - big frames, small card
COLOR_FIX   = "none"         # "adain" or "wavelet" pull the colour back to the input
SEED        = 42             # the single denoising step's noise
POST_SIZE   = None           # e.g. (720, 1248) as (height, width): render x4, then Lanczos down -
                             # that is how you compare against a 2x finish at the same size

## %% 3. what the clip is, and whether the weights are there
from recavsr_media import list_inputs, probe

print(list_inputs(BASE / "inputs"))
INFO = probe(INPUT)
MODEL_H = int(INFO["height"] * SCALE + 0.5)
MODEL_W = int(INFO["width"] * SCALE + 0.5)
print(INFO, "->", (MODEL_W, MODEL_H), "padded to", L.STRIDE, "px grid")
print("missing weights:", L.check(WEIGHTS) or "none")

## %% 4. load: transformer + FlashDecoder + prompt (into RAM; they go to the GPU on the first block)
ENGINE = L.engine()          # L.engine("cuda:0"); pick the card with CUDA_GPU / --gpu in the CLI
L.load(ENGINE, WEIGHTS)
if DECODER == "vae":
    L.load_vae(ENGINE)
print(ENGINE["model"].precision, "|", L.headroom())

## %% 5. will this size run, and what has to be switched on
CHUNK, KV_OFFLOAD, NEED = L.fit(MODEL_H, MODEL_W, KV_OFFLOAD)
print(f"step needs {NEED / 2**30:.1f} GiB; input projection {CHUNK and L.LQ_CHUNK_FRAMES or 'whole block'} "
      f"frames at a time; attention history {'in RAM' if KV_OFFLOAD else 'in VRAM'}")

## %% 6. run it
from recavsr_engine import upscale

OUT_HW = POST_SIZE or (int(INFO["height"] * SCALE + 0.5), int(INFO["width"] * SCALE + 0.5))
OUT = OUT_DIR / f"{pathlib.Path(INPUT).stem}_x{SCALE:g}_{OUT_HW[1]}x{OUT_HW[0]}.mp4"
OUT.parent.mkdir(parents=True, exist_ok=True)
STATS = upscale(ENGINE, str(INPUT), str(OUT), scale=SCALE, color_fix=COLOR_FIX, seed=SEED,
                decoder=DECODER, kernels=KERNELS, kv_offload=KV_OFFLOAD, max_frames=MAX_FRAMES,
                start_frame=START_FRAME, post_size=POST_SIZE)

## %% 7. the numbers, and the card back
print(STATS["frames"], "frames", STATS["in_size"], "->", STATS["out_size"],
      f"in {STATS['seconds']:.1f} s = {STATS['measured_fps']:.2f} fps",
      f"(transformer {STATS['transformer_s']:.1f} s, decode {STATS['decode_s']:.1f} s)")
print(L.headroom())

## %% 8. done with this clip: free the card
L.free(ENGINE)
