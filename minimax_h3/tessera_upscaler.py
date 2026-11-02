#!/usr/bin/env python3
"""tessera_upscaler - upscale any video with MiniMax H3 ref2va + ComfyUI-Tessera.

Set INPUT, run the cells top to bottom once. Re-run the last cell for the next clip - the
model stays loaded.

lib/h3_tessera.py is the general machinery and deliberately knows nothing about this
machine: every model path, the prompt string and all the sampling/tiling settings are
declared here and passed in as arguments. Copy this file, point the paths at your own
checkpoints, and the lib needs no edits.

What it does is not a resize: the clip is re-generated at the larger size with the source
itself wired in as the <Video 1> reference, so faces, edges and texture come back
reconstructed rather than interpolated. That is also the cost - real sampling over a bigger
token grid, tiled and chunked. A 5-second 416x576 clip goes to 832x1152 in about 150 s on
this box (3 temporal chunks x 4 tiles x 4 steps).

KERNEL: the project .venv.
"""

# %% 0. setup
import os
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "lib"))

import h3_load as L
import h3_tessera as T

## %% 1. model files - passed to the lib, which has no opinion about where they live
COMFY_ROOT      = "/mnt/data_2t_3/github_repos/ComfyUI"
TESSERA_ROOT    = f"{COMFY_ROOT}/custom_nodes/ComfyUI-Tessera"
REF2VA          = f"{COMFY_ROOT}/models/diffusion_models/minimax_h3/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER    = f"{COMFY_ROOT}/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE       = f"{COMFY_ROOT}/models/vae/minimax_h3/minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE       = f"{COMFY_ROOT}/models/vae/minimax_h3_audio_vae_fp32.safetensors"

# The pack's house recipe names a rank-averaged taomate conversion that is not published
# under that name; these are the upstream originals it was made from, both verified to
# attach fully to the ref2va base (208/208 and 258/258 target modules).
LORAS = [(f"{COMFY_ROOT}/models/loras/minimax_h3/taomate_h3_3step_comfy.safetensors", 1.0),
         (f"{COMFY_ROOT}/models/loras/minimax_h3/minimax_h3_lms_v1.0_r64.safetensors", 0.75)]

## %% 2. what to upscale
INPUT       = str(BASE / "outputs" / "h3_fast_4step_tessera" / "draft_416x576_75.mp4")
OUT_DIR     = BASE / "outputs" / "tessera_upscaler"    # file name is built from the source
UPSCALE_BY  = 2        # 2 is the pack's validated ratio; 4 works but the plate gets huge
SEED        = 0
MAX_FRAMES  = 0        # 0 = whole clip; set e.g. 240 to bound the time on a long input

## %% 3. the prompt - <Video 1> is the source clip; declare it and demand it be reproduced
# one to one, or the refine will "improve" your subject into a stranger. This string is the
# single most important setting in the file.
REFINE_PROMPT = (
    "subject_definitions:\n"
    "<Video 1>: the reference video. Its picture, motion, framing, timing and content are "
    "the target video's content.\n\n"
    "retention_analysis:\n"
    "<Video 1>: fully_preserved - <Video 1> is reproduced one to one as the target video. "
    "Every frame matches the source.\n\n"
    "Enhance this video with sharp, crisp details while preserving a natural photorealistic "
    "appearance."
)

## %% 4. knobs - defaults are the pack's house recipe
STEPS, SCHEDULER, SAMPLER = 4, "simple", "er_sde"   # 4 assumes the taomate 3-step adapter;
CFG = 1.0                                           # without one, push steps toward 20
TAU, DENSE_BLOCKS, EXTRA_TOKENS = 1.5, "0,1", 256   # sparse attention over the whole
SHIFT_VIDEO, SHIFT_AUDIO = 12.0, 3.0                # trajectory; blocks 0 and 1 stay dense
TILE_BUDGET = "low_vram"                            # spatial and temporal tiles scale together to fit this
TEMPORAL_SIZE, TEMPORAL_OVERLAP = 32, 8             # frames per chunk, and the hand-off
TILE_W, TILE_H, CONTEXT_BORDER = 1024, 1024, 64     # the lib gives Plan and Refine the same
                                                    # numbers, so they cannot disagree
UNET_GPU, SUPPORT_GPU = 0, 2                        # DiT on the 5090, text encoder + VAEs on the 3090 Ti

## %% 5. engine + models - run once per session (ref2va 21 GB, text encoder 15 GB)
UNET_GPU, SUPPORT_GPU = (int(x) for x in
                         os.environ.get("H3_GPUS", f"{UNET_GPU}, {SUPPORT_GPU}").split(","))

L.init_engine(verbose=True, devices=[UNET_GPU, SUPPORT_GPU])
T.init(COMFY_ROOT, TESSERA_ROOT)
clip, vae, audio_vae = T.load_support(comfy_root=COMFY_ROOT, text_encoder=TEXT_ENCODER,
                                      video_vae=VIDEO_VAE, audio_vae=AUDIO_VAE,
                                      device=SUPPORT_GPU)
model = T.load_ref2va(comfy_root=COMFY_ROOT, model_path=REF2VA, loras=LORAS, device=UNET_GPU,
                      tau=TAU, dense_blocks=DENSE_BLOCKS, extra_tokens=EXTRA_TOKENS,
                      shift_video=SHIFT_VIDEO, shift_audio=SHIFT_AUDIO)

## %% 6. go - re-run this cell for the next clip
images, audio, fps = T.read_video(INPUT, comfy_root=COMFY_ROOT, max_frames=MAX_FRAMES)
T.upscale(comfy_root=COMFY_ROOT, video_path=INPUT, clip=clip, vae=vae, audio_vae=audio_vae,
          model=model, prompt=REFINE_PROMPT, out_dir=OUT_DIR, upscale_by=UPSCALE_BY,
          steps=STEPS, scheduler=SCHEDULER, sampler=SAMPLER, seed=SEED, cfg=CFG,
          tile_budget=TILE_BUDGET, temporal_size=TEMPORAL_SIZE,
          temporal_overlap=TEMPORAL_OVERLAP, tile_width=TILE_W, tile_height=TILE_H,
          context_border=CONTEXT_BORDER, images=images, audio=audio, fps=fps, release=False)

## %% 7. optional - a whole folder, one model load. Skip cell 6 when using this.
IN_DIR = None         # e.g. pathlib.Path("/home/andrewzhu/videos")
if IN_DIR:
    for clip_path in sorted(pathlib.Path(IN_DIR).glob("*.mp4")):
        print(f"\n=== {clip_path.name} ===")
        T.upscale(comfy_root=COMFY_ROOT, video_path=clip_path, clip=clip, vae=vae,
                  audio_vae=audio_vae, model=model, prompt=REFINE_PROMPT, out_dir=OUT_DIR,
                  upscale_by=UPSCALE_BY, steps=STEPS, sampler=SAMPLER, seed=SEED,
                  tile_budget=TILE_BUDGET, temporal_size=TEMPORAL_SIZE,
                  temporal_overlap=TEMPORAL_OVERLAP, tile_width=TILE_W, tile_height=TILE_H,
                  context_border=CONTEXT_BORDER, max_frames=MAX_FRAMES, release=False)
    L.free(model)
