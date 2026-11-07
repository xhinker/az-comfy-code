#!/usr/bin/env python3
"""HyperFlow H3 Flash, one component per cell - the interactive version.

HyperFlow H3 Flash is Video Rebirth's compressed MiniMax-H3: 13.7B parameters instead of ~31B
(3 of 50 blocks dropped, 102 linear layers made low-rank), sampled in 4/6/8 steps at CFG 1.0.
Same text encoder, same video and audio VAEs as the full H3 - only the transformer is new, so
this file is the ordinary H3 interactive script with the HyperFlow loader and its sigma grid
swapped in.

KERNEL: run with the ComfyUI interpreter - comfy-kitchen, comfy-aimdo and the attention
kernels are installed only there:

    /mnt/data_2t_3/github_repos/ComfyUI/venv_py312_comfyui/bin/python

  VS Code:  Ctrl+Shift+P -> "Notebook: Select Notebook Kernel" -> pick the path above.
  Terminal: that path, followed by this file.
  Jupyter:  <that path> -m ipykernel install --user --name comfyui

Cells run top to bottom once; after that re-run any single cell you like. No ComfyUI server is
involved - it imports ComfyUI's own `comfy` package, registers the two custom node packs the
loader needs, and calls the node classes directly.
"""

# %% 0. setup - put minimax_h3/ on sys.path so h3_load, h3_core and h3_hyperflow import
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path.cwd()
sys.path.insert(0, str(BASE / "lib"))      # h3_load.py, h3_core.py, h3_hyperflow.py


# %% 1. model files - the GGUF is the only new download
COMFY = pathlib.Path("/mnt/data_2t_3/github_repos/ComfyUI")

# fl2va serves t2v, i2v and first+last-frame. Q4_K_M (9.1 GB) is the recommended consumer
# default; Q6_K (11.6 GB) and Q8_0 (14.9 GB) are closer to full precision but leave less room
# next to the 15.7 GB text encoder on a 32 GB card.
GGUF = f"{COMFY}/models/diffusion_models/minimax_h3/minimax_h3_fl2va_hyperflow_flash_14b-Q4_K_M.gguf"

# companions are the ordinary MiniMax H3 files - nothing extra to fetch
CLIP = f"{COMFY}/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE = f"{COMFY}/models/vae/minimax_h3/minimax_h3_video_vae_int8_convrot.safetensors"
# VAE = f"{COMFY}/models/vae/minimax_h3_video_vae_fp16.safetensors"   # sharper, tighter fit
AUDIO_VAE = f"{COMFY}/models/vae/minimax_h3_audio_vae_fp32.safetensors"

for _p in (GGUF, CLIP, VAE, AUDIO_VAE):
    print("OK      " if pathlib.Path(_p).is_file() else "MISSING ", _p)


# %% 2. engine + the two custom node packs (ComfyUI-GGUF, ComfyUI-HyperFlow-Flash)
# init_engine switches on DynamicVRAM (weights staged as file slices, streamed layer by
# layer); load_node_packs registers the nodes the HyperFlow loader depends on.
import h3_load as L
import h3_hyperflow as HF

L.init_engine(verbose=True)
HF.load_node_packs()
L.status()          # want dynamic_vram=True aimdo=True


# %% 3. the transformer  (HyperFlowGGUFLoader)
# weights_in="ram" pins the part that misses VRAM in host memory; "disk" leaves it in the file's
# page cache (what --fast-disk does). With 125 GB of RAM, ram is the faster choice here.
model = HF.load_gguf(GGUF, weights_in="ram")
L.status()


# %% 4. the text encoder  (CLIPLoader)  ~2s
clip = L.load_text_encoder(CLIP, type="minimax", device="default")
L.status()


# %% 5. the VAEs  (VAELoader)
vae = L.load_vae(VAE)
audio_vae = L.load_vae(AUDIO_VAE)


# %% 6. sampling contract - this is not a "pick a sampler" model
# STEPS: 4 (speed) / 6 (default) / 8 (quality). The grid is fixed by the checkpoint; anything
# else is off-distribution. CFG is fixed at 1.0: there is no negative prompt at all.
# The schedule is unshifted knots mapped through video_shift 12; the model also derives the
# audio stream's noise from its own shift (12 video / 3 audio = ComfyUI's H3 defaults, which is
# what HyperFlow was distilled with - leave them alone).
STEPS, SEED = 6, 42
sigmas = HF.sigmas(model, STEPS)

# No LoRA and no sparse attention here unless you re-measure: the two-time conditioning and the
# low-rank linears are what make 6 steps work, and both the turbo LoRA and sol-attn were built
# for the full-size transformer.


# %% 7. prompt and geometry
# Native short edge is 768: 1344x768 (16:9), 1024x1024, 768x1344. Below that it goes soft.
# length is frames at 24 fps on the 17k+5 grid: 22 ~ 1 s, 73 ~ 3 s, 124 ~ 5.2 s, 362 ~ 15 s.
# Describe subject, action, setting, camera move AND sound in one prompt - audio is generated
# with the video.
PROMPT = ("A chef slides a steaming bowl of ramen across a wooden counter, rain taps on the "
          "window behind him, camera pushes in slowly, ambient kitchen noise and soft rain")
MODE, WIDTH, HEIGHT, LENGTH = "t2v", 1344, 768, 124

# keyframe mode (fl2va covers i2v and first+last):
# MODE, IMAGE = "i2v", f"{BASE}/ref/first_frame.png"

import h3_core as H

cond, latent = H.encode(clip, vae, PROMPT, mode=MODE, width=WIDTH, height=HEIGHT, length=LENGTH,
                        image=globals().get("IMAGE"))
L.status()          # conditioner releases here; the transformer takes the room


# %% 8. generate
# euler + BasicGuider + the fixed sigma grid. Passing sigmas= overrides steps/scheduler.
# Measured on this box (RTX 5090, Q4_K_M, int8 video VAE), 1344x768, 6 steps:
#   22 frames  (0.9 s)  denoise 10.4 s  + decode/mux 2.9 s
#   73 frames  (3.0 s)  denoise 46.3 s  (i2v, first keyframe)
#   124 frames (5.2 s)  denoise 88.4 s  + decode/mux 8.0 s  = 96 s end to end
# Video Rebirth's own 5090 number for this quant is 99 s, so this is at spec. The DiT holds
# ~9.1 GB resident; the text encoder is released before sampling, which is what keeps a 32 GB
# card out of trouble.
import time

t0 = time.perf_counter()
out = H.denoise(model, cond, latent, sampler="euler", sigmas=sigmas, seed=SEED)
print(f"denoise: {time.perf_counter() - t0:.1f} s for {len(sigmas) - 1} steps")

mp4 = H.decode(vae, audio_vae, out, H.out_dir_for(__file__) / f"hyperflow_{WIDTH}x{HEIGHT}_{LENGTH}f_{SEED}.mp4")
print(f"-> {mp4}")


# %% 9. VRAM - what is resident now, and how to hand it back
L.status()
# L.free(clip, vae, audio_vae)   # one component
# L.free_all()                   # everything; each re-stages from disk on next use
