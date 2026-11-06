#!/usr/bin/env python3
"""relayvsr_interactive - upscale a video with RelayVSR, from Python, no workflow JSON.

Run the cells top to bottom once; re-run cell 7 + 8 per clip. RelayVSR is a two-model
system: the **Wan backbone** makes reference latents for sparse keyframes and the
**Flash decoder** super-resolves every frame, so the two load in separate cells and you
can hold either one on its own.

Measured on the RTX 5090 (32 GB) in this box:  see README.md.

KERNEL: the project .venv (the workspace default) - verified with the INT8 conv-rot path;
        the ComfyUI venv works too. Pylance resolves the relayvsr_* imports through
        python.analysis.extraPaths -> relayvsr/lib in .vscode/settings.json.
"""

# %% 0. setup
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "lib"))

import relayvsr_load as L

## %% 1. what to upscale
OUT_DIR     = BASE / "outputs"
MAX_FRAMES  = 0               # 0 = whole clip
START_FRAME = 0

## %% 2. the settings that matter
SCALE       = 2                     # which weights to use: 2 or 4
SR_RATIO    = 2                     # output ratio: x2 weights take 2; x4 weights take 2 or 4
CHUNK_LEN   = 16                    # frames per streaming window (reference interval + 1)
SEED        = 42                    # only the reference-frame noise; the model is 1-step distilled
DTYPE       = "bf16"
OFFLOAD     = True               # True parks the Wan blocks in CPU RAM: needed on 24 GB cards
CODEC, CRF, PRESET = "h264", 18, "medium"

## %% 3. weights (relayvsr/models, git-ignored)
L.check_weights(SCALE)

## %% 4. engine: ComfyUI core bootstrap + device choice
engine = L.engine(scale=SCALE, dtype=DTYPE, offload=OFFLOAD)

## %% 5. load the diffusion model  (the Wan backbone, INT8 conv-rot via comfy.quant_ops)
L.load_wan(engine)

## %% 6. load the VAE-side model  (the Flash decoder; it replaces the video VAE)
L.load_flash(engine)

# %% 7. run  - builds the frame stream; the actual compute happens in cell 8
INPUT = str(BASE / "inputs" / "demo_input.mp4")   # the official demo clip; point this at any mp4

frames, plan = L.run(engine, INPUT, sr_ratio=SR_RATIO, chunk_len=CHUNK_LEN, seed=SEED,
                     max_frames=MAX_FRAMES, start_frame=START_FRAME)
plan

## %% 8. save the output video
out = OUT_DIR / (pathlib.Path(INPUT).stem + f"_x{SR_RATIO}.mp4")
path, written, seconds = L.save(frames, out, fps=plan["fps"], codec=CODEC, crf=CRF,
                                preset=PRESET, audio_from=INPUT if plan["has_audio"] else None)
print(f"{written} frames  {plan['in_size']} -> {plan['out_size']}  {seconds:.1f} s  "
      f"{written / seconds:.2f} fps\n{path}")

# %% 9. done with it? hand the VRAM back
# L.free(engine)
