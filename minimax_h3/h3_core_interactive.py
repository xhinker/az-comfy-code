#!/usr/bin/env python3
"""MiniMax H3, one component per cell - the interactive version.

KERNEL: run this with the ComfyUI interpreter, not the repo venv - comfy-kitchen,
comfy-aimdo and the attention kernels are installed only there:

    /mnt/data_2t_3/github_repos/ComfyUI/venv_py312_comfyui/bin/python

  VS Code:  Ctrl+Shift+P -> "Notebook: Select Notebook Kernel" -> pick the path above.
  Terminal: that path, followed by this file.
  Jupyter:  <that path> -m ipykernel install --user --name comfyui

Cells are numbered in run order; run them top to bottom once, then re-run any single cell
you like. Nothing talks to a ComfyUI server - it imports ComfyUI's own `comfy` package and
calls the H3 node classes directly. Weights are staged, not loaded: the 32 GB transformer
returns in 0.2 s because it stays as file slices and streams to the GPU layer by layer.
"""

# %% 0. setup - put minimax_h3/ on sys.path so h3_load and h3_core import
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path.cwd()
sys.path.insert(0, str(BASE))


# %% 1. model files - absolute paths, edit these to point anywhere
COMFY = pathlib.Path("/mnt/data_2t_3/github_repos/ComfyUI")

UNET = f"{COMFY}/models/diffusion_models/minimax_h3/MiniMax-H3-Ref2VA_int8_convrot.safetensors"
CLIP = f"{COMFY}/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
# the fp16 video decoder is sharper but does not fit next to the DynamicVRAM cache on 32 GB:
# VAE = f"{COMFY}/models/vae/minimax_h3_video_vae_fp16.safetensors"
VAE = f"{COMFY}/models/vae/minimax_h3/minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = f"{COMFY}/models/vae/minimax_h3_audio_vae_fp32.safetensors"

for _p in (UNET, CLIP, VAE, AUDIO_VAE):
    print("OK      " if pathlib.Path(_p).is_file() else "MISSING ", _p)


# %% 2. engine - must run before any model touches CUDA; switches on DynamicVRAM
import h3_load as L

L.init_engine(verbose=True)
L.status()          # want dynamic_vram=True aimdo=True; if False, nothing downstream fits


# %% 3. the transformer  (UNETLoader)  ~0.2s
model = L.load_diffusion_model(UNET, weight_dtype="default")
L.status()


# %% 4. the text encoder  (CLIPLoader)  ~2s
# device="cpu" keeps the 15 GB conditioner off the GPU - slower prompts, more headroom.
clip = L.load_text_encoder(CLIP, type="minimax", device="default")
L.status()


# %% 5. the VAEs  (VAELoader)  ~0.1s each
vae = L.load_vae(VAE)
audio_vae = L.load_vae(AUDIO_VAE)


# %% 6. optional patches - each returns a new patcher; the old one stays valid
LORA = f"{COMFY}/models/loras/minimax_h3/minimax_h3_turbo_4step_ckpt850_pruned_comfyui.safetensors"
model = L.load_lora(model, LORA, 1.0)                                # then 4-8 steps is enough
# The shift is a training-consistency setting, not a taste dial: it has to match what the
# weights were trained or distilled with. It moves where your steps land on the noise line
# (sigma = s*t / (1 + (s-1)*t)) AND it is what the DiT uses to derive the audio stream's
# noise level and per-stream timestep labels. Wrong audio shift = distorted or dead audio.
#   released checkpoint, no LoRA .... 12.0 / 3.0   (ComfyUI's own default, applied anyway)
#   this turbo LoRA (pruned conv.) .. 12.0 / 4-6   (drbaph card: "Video sigma shift 12, Audio sigma shift 4-6")
#   FastH3 8-step template .......... 10.0 / 3.0
SHIFT_VIDEO, SHIFT_AUDIO = 12.0, 5.0
model = L.sigma_shift(model, shift_video=SHIFT_VIDEO, shift_audio=SHIFT_AUDIO)
_to = model.model_options["transformer_options"]
print("sigma shift in effect:", _to["minimax_h3_sigma_shift_video"], "/", _to["minimax_h3_sigma_shift_audio"])
model = L.sparse_attention(model, "sol-attn", tau=1.3, start_percent=0.2)   # ~1.7x, any weights


# %% 7. prompt and geometry
# Native is a 768 px short edge: 1344x768 (16:9), 1024x1024, 768x1344. Below that the
# result looks soft no matter what the sampler does.
# length is frames at 24 fps on the 17k+5 grid: 22 ~ 1s, 73 ~ 3s, 124 ~ 5s, 362 ~ 15s.
# Off-grid values snap up - asking for 62 gives 73.
PROMPT = ("A red fox trotting through a snowy pine forest, snow crunching underfoot, "
          "wind through the branches, shallow depth of field")
MODE, WIDTH, HEIGHT, LENGTH = "t2v", 1344, 768, 226

# other modes (uncomment one; references are tagged <Picture 1>, <Picture 2>, ... in the
# prompt, in the same order as the files):
# MODE, IMAGE = "i2v", f"{BASE}/ref/first_frame.png"
# MODE, REF_IMAGES = "ref2va", [f"{BASE}/ref/person.png", f"{BASE}/ref/room.png"]

import h3_core as H

cond, latent = H.encode(clip, vae, PROMPT, mode=MODE, width=WIDTH, height=HEIGHT, length=LENGTH,
                        image=globals().get("IMAGE"), ref_images=globals().get("REF_IMAGES", ()))
L.status()          # the conditioner releases here; the transformer gets the room


# %% 8. generate
# STEPS MATTER. The released checkpoint is guidance-distilled but still wants ~20 steps;
# 4 steps gives a soft, smeared frame that looks like a bad VAE. With the turbo LoRA from
# cell 6, 4-8 steps is enough. ~3 it/s at 832x480 on a 5090, ~5.2 s/it at 1344x768.
STEPS, SEED = 8, 10

out = H.denoise(model, cond, latent, steps=STEPS, seed=SEED)
mp4 = H.decode(vae, audio_vae, out, H.out_dir_for(__file__) / f"interactive_{WIDTH}x{HEIGHT}_{SEED}.mp4")
print(f"-> {mp4}")


# %% 9. VRAM - what is resident now, and how to hand it back
L.status()
# L.free(clip, vae, audio_vae)   # one component
# L.free_all()                   # everything; each re-stages from disk on next use
