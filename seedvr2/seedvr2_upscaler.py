#!/usr/bin/env python3
"""seedvr2_upscaler - upscale any video with SeedVR2, from Python, no workflow JSON.

Set INPUT, run the cells top to bottom once, then re-run the last cell per clip.

Two implementations live in lib/seedvr2_core.py and both are reachable from here:

  * **pack path** (default) - the SeedVR2 pack's own nodes (ussoewwin fork, same node ids as
    numz's). This is the one that works on a 32 GB card, because it has BlockSwap:
    transformer blocks stream between GPU and CPU RAM. 124 frames at 416x576 -> 832x1152 in
    58.7 s, peak 19.8 GB.
  * **core path** (`upscale()`) - ComfyUI's built-in SeedVR2 nodes, no custom node pack at
    all. Cleaner, and it loads the same W4A8 checkpoint, but on this build the VAE pins
    ~25 GiB and never lets go, so long clips die partway. Kept for reference and for short
    clips.

The technique that made the difference is the one in the GUI workflow
user/default/workflows/upscale/SeedVR2_Video_Upscaling_az.json: BlockSwap on top of an
offload device, plus tiled VAE encode/decode.

KERNEL: the project .venv.
"""

# %% 0. setup
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "lib"))

import seedvr2_core as S

## %% 1. what to upscale
INPUT = str(BASE.parent / "minimax_h3" / "outputs" / "h3_fast_4step_tessera" / "draft_416x576_75.mp4")
OUT_DIR = BASE / "outputs" / "pack"
MAX_FRAMES = 0                 # 0 = whole clip

## %% 2. models
COMFY_ROOT = "/mnt/data_2t_3/github_repos/ComfyUI"
PACK_ROOT = ""                 # "" -> <COMFY_ROOT>/custom_nodes/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT
DIT_MODEL = "seedvr2_7b_convrot_w4a8.safetensors"   # 4.44 GB W4A8, in models/SEEDVR2/
VAE_MODEL = "ema_vae_fp16.safetensors"              # 478 MB

## %% 3. the settings that matter
DEVICE = "cuda:0"
OFFLOAD_DEVICE = "cpu"         # required for BlockSwap; "cuda:2" would be faster if a card is free
BLOCKS_TO_SWAP = 16            # of 36 blocks; saved 6.9 GB here, 80 swaps
SWAP_IO_COMPONENTS = True      # embeddings + norms off the GPU too
RESOLUTION = 832               # SHORTEST edge of the output - not longest, not a multiplier
BATCH_SIZE = 41                # frames per temporal window, 4n+1; 41 fit at 832x1152
TEMPORAL_OVERLAP = 4
COLOR_CORRECTION = "wavelet"   # wavelet keeps the upscaled high frequencies, lab is the most faithful tint
VAE_TILE, VAE_TILE_OVERLAP = 512, 64
SEED = 0

## %% 4. load - once per session
dit = S.load_dit_pack(comfy_root=COMFY_ROOT, pack_root=PACK_ROOT, model=DIT_MODEL,
                      device=DEVICE, offload_device=OFFLOAD_DEVICE,
                      blocks_to_swap=BLOCKS_TO_SWAP, swap_io_components=SWAP_IO_COMPONENTS,
                      cache_model=True)
vae_encode, vae_decode = S.load_vae_pack(comfy_root=COMFY_ROOT, pack_root=PACK_ROOT,
                                         model=VAE_MODEL, device=DEVICE,
                                         offload_device=OFFLOAD_DEVICE, encode_tiled=True,
                                         decode_tiled=True, tile_size=VAE_TILE,
                                         tile_overlap=VAE_TILE_OVERLAP)

## %% 5. go - re-run for the next clip
images, audio, fps = S.read_video(comfy_root=COMFY_ROOT, video_path=INPUT, max_frames=MAX_FRAMES)
S.upscale_pack(comfy_root=COMFY_ROOT, pack_root=PACK_ROOT, images=images, fps=fps, dit=dit,
               vae_encode=vae_encode, vae_decode=vae_decode, resolution=RESOLUTION,
               batch_size=BATCH_SIZE, temporal_overlap=TEMPORAL_OVERLAP,
               color_correction=COLOR_CORRECTION, offload_device=OFFLOAD_DEVICE, seed=SEED,
               audio=audio, out_dir=OUT_DIR, source_name=pathlib.Path(INPUT).name)

## %% 6. optional - a whole folder, one model load
IN_DIR = None                  # e.g. pathlib.Path("/home/andrewzhu/videos")
if IN_DIR:
    for clip_path in sorted(pathlib.Path(IN_DIR).glob("*.mp4")):
        print(f"\n=== {clip_path.name} ===")
        images, audio, fps = S.read_video(comfy_root=COMFY_ROOT, video_path=clip_path,
                                          max_frames=MAX_FRAMES)
        S.upscale_pack(comfy_root=COMFY_ROOT, pack_root=PACK_ROOT, images=images, fps=fps,
                       dit=dit, vae_encode=vae_encode, vae_decode=vae_decode,
                       resolution=RESOLUTION, batch_size=BATCH_SIZE,
                       temporal_overlap=TEMPORAL_OVERLAP, color_correction=COLOR_CORRECTION,
                       offload_device=OFFLOAD_DEVICE, seed=SEED, audio=audio, out_dir=OUT_DIR,
                       source_name=clip_path.name)

## %% 7. the core-node path, for short clips or for running without the pack
# Same checkpoint, no custom nodes at all - but see README: this ComfyUI build retains the
# VAE working set, so it tops out around a few dozen frames.
if False:
    model = S.load_model(comfy_root=COMFY_ROOT, unet_name=DIT_MODEL)
    vae = S.load_vae(comfy_root=COMFY_ROOT, vae_name="seedvr2_ema_vae_fp16.safetensors")
    images, audio, fps = S.read_video(comfy_root=COMFY_ROOT, video_path=INPUT, max_frames=24)
    S.upscale(comfy_root=COMFY_ROOT, images=images, fps=fps, model=model, vae=vae,
              out_dir=BASE / "outputs" / "core", scale_multiplier=2.0, color_correction="lab",
              vae_tile=0, audio=audio, source_name=pathlib.Path(INPUT).name)
