#!/usr/bin/env python3
"""h3_core_interactive, but pointed at the fast workflow's image and prompt - a speed A/B.

Same components, same LoRA, same shift, same sol-attn as `h3_core_interactive.py`; the
prompt and first frame come from `minimax_h3_5s_fast.json` so the two runs do the same work.

Compare with `minimax_h3_fast_test.py` (same geometry, same LoRAs, same seed):

  | stage    | this file (core path)          | fast_test (workflow path)      |
  |----------|--------------------------------|--------------------------------|
  | sampler  | res_multistep + simple         | dual_clock_euler + native_flow |
  | base     | Ref2VA int8 convrot (full)     | fl2va pruned int8 convrot      |
  | sol-attn | ComfyUI BlockSparseAttention   | T8 SolAttnMiniMax              |

Each stage is timed and printed, so any gap is attributable rather than guessed. Measured
2026-10-30 at 768x1344 x 226 frames, 8 steps: every sparse variant sampled in 141-154 s,
dense took 325 s. Sparse attention is the lever; the base is worth ~8%; the solver sits
inside run-to-run noise.

KERNEL: the project .venv (it carries comfy-aimdo, comfy-kitchen, torchaudio) or the
ComfyUI venv - both work. Run cells top to bottom once.
"""

# %% 0. setup - put minimax_h3/ on sys.path so h3_load and h3_core import
import pathlib
import sys
import time

BASE = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path.cwd()
sys.path.insert(0, str(BASE))


## %% 1. model files - identical to h3_core_interactive.py
COMFY = pathlib.Path("/mnt/data_2t_3/github_repos/ComfyUI")

UNET        = f"{COMFY}/models/diffusion_models/minimax_h3/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP        = f"{COMFY}/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE         = f"{COMFY}/models/vae/minimax_h3/minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE   = f"{COMFY}/models/vae/minimax_h3_audio_vae_fp32.safetensors"

# UNET_RESIDENT picks the layout:
#   True  - the transformer sits fully in GPU 0's VRAM (~21 GB for the pruned checkpoint),
#           no streaming per step and nothing to re-stage, which means the conditioner and
#           the VAEs have to live on another card.
#   False - DynamicVRAM streams the transformer through GPU 0's cache, so the support
#           components are better off on GPU 0 as well.
# UNET_RESIDENT = True was tried and does not fit this box: with the weights non-dynamic,
# applying the 4-LoRA chain materializes them on GPU 0 and OOMs at ~30 GB (19 LoRA
# applications failed before the fatal one). Fully resident weights need a card big enough
# for weights + LoRA materialization + activations. DynamicVRAM streaming is what makes
# 20 GB of weights plus a 15 GB conditioner plus VAEs fit in 32 GB at all.
UNET_RESIDENT = False

# Where the support components live: an index (2, "cuda:2"), "cpu", or None for ComfyUI.
# GPU 2 is a 3090 Ti - about half the 5090's memory bandwidth and none of the fp8/int8 fast
# paths these nvfp4 / int8-convrot weights use, so it is a place to put things when GPU 0 is
# full, not a speed upgrade: measured there load 5.6 s / encode 3.9 s / decode 26.8 s
# against 2.3 s / 1.9 s / 15.3 s on GPU 0. GPU 1 runs tqserve - keep it out of
# init_engine's device list.
TE_GPU, VAE_GPU = (2, 2)

# keep=True sets offload_device = load_device, i.e. the component never leaves its card.
# That is affordable on the 32 GB card, not on the 24 GB one: the nvfp4 conditioner peaks
# near 20 GB during its quantized load, so pinned on GPU 2 it OOMs. When the transformer is
# resident on GPU 0 there is no room there for the conditioner either, so it stages through
# RAM on GPU 2 (keep=False) instead.
SUPPORT_KEEP = True

# the reference and the prompt this test borrows from the saved fast workflow
WORKFLOW = COMFY / "user/default/workflows/minimax_h3/minimax_h3_5s_fast.json"

for _p in (UNET, CLIP, VAE, AUDIO_VAE, FIRST_FRAME, WORKFLOW):
    print("OK      " if pathlib.Path(_p).is_file() else "MISSING ", _p)


# (name, strength) in the order the workflow chains them - strength matters as much as order
LORAS = [
    (f"{COMFY}/models/loras/minimax_h3/minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors", 0.75),
    (f"{COMFY}/models/loras/minimax_h3/minimax_h3_fl2v_turbo_8step_v1.0_768p_FeiHouRemix_v0.6_compat_v001_T8.safetensors", 0.75),
    (f"{COMFY}/models/loras/minimax_h3/h3_musubi_v4-000040.safetensors", 0.35),
    (f"{COMFY}/models/loras/minimax_h3/SoftBreastMotion.safetensors", 0.5),
]

for _p in [UNET, CLIP, VAE, AUDIO_VAE, FIRST_FRAME, WORKFLOW, *[p for p, _ in LORAS]]:
    print("OK      " if pathlib.Path(_p).is_file() else "MISSING ", _p)

# %% 2. engine - must run before any model touches CUDA; switches on DynamicVRAM
import h3_load as L

L.init_engine(verbose=True, devices=[0, TE_GPU])   # arm only the cards we use
L.status()          # want dynamic_vram=True aimdo=True; if False, nothing downstream fits


## %% 3. the transformer  (UNETLoader)  ~0.2s
model = L.load_diffusion_model(UNET, weight_dtype="default", resident=UNET_RESIDENT)
L.status()


## %% 4. the text encoder  (CLIPLoader)  ~2s
clip = L.load_text_encoder(CLIP, type="minimax", device=TE_GPU, keep=SUPPORT_KEEP)
L.status()


## %% 5. the VAEs  (VAELoader)  ~0.1s each
vae = L.load_vae(VAE, device=VAE_GPU, keep=SUPPORT_KEEP)
audio_vae = L.load_vae(AUDIO_VAE, device=VAE_GPU, keep=SUPPORT_KEEP)
L.resident()      # what is pinned, on which device
L.per_gpu()       # and where the bytes actually landed

## %% 6. the LoRA chain, same order and strengths as the graph
for path, strength in LORAS:
    model = L.load_lora(model, path, strength)
    print(f"lora x{strength}: {pathlib.Path(path).name}")
L.status()


# %% shift
SHIFT_VIDEO, SHIFT_AUDIO = 12.0, 5.0     # drbaph card for this pruned turbo conversion: 12 / 4-6
model = L.sigma_shift(model, shift_video=SHIFT_VIDEO, shift_audio=SHIFT_AUDIO)

## %% memory-efficient attention (KJ nodes) - int8 QK / fp8 V, lowers peak VRAM.
# Needs sageattention >= 2.2, which is not on PyPI; 2.2.0 is built from source into this
# venv (recipe in requirements.txt). Order matters: the workflow patches sage attention
# first and sol-attn chains onto the result, so keep this above the sparse-attention cell.
model = L.sage_attention_patch(model)

## %% sparse attention - two interchangeable implementations, pick one
# T8 SolAttnMiniMax: Triton kernels built for SM120, same tau/start the workflow uses.
model = L.sol_attn_t8(model, tau=1.3, start_percent=0.2, end_percent=0.9, min_tokens=12288)
# ComfyUI's own BlockSparseAttention: measured 10.15 s/it here vs 10.06 for T8 at the same
# base and LoRA chain, so this swap is not where speed comes from. end_percent is the dial
# that is: 0.9 leaves the last 10% of steps dense, 1.0 covers every step.
# model = L.sparse_attention(model, "sol-attn", tau=1.3, start_percent=0.2)


# %% 7. prompt and geometry - taken from the fast workflow, not rewritten here
import json

def workflow_prompt(path=WORKFLOW, node_type="MiniMaxH3ImageToVideo"):
    wf = json.loads(pathlib.Path(path).read_text())
    for n in wf["nodes"]:
        if n["type"] == node_type:
            return n["widgets_values"][0]
    raise KeyError(node_type)

STEPS = 8
PROMPT = workflow_prompt()

FIRST_FRAME = "/home/andrewzhu/storage_8t_4/comfyui_output/output/WanVideo_SCAIL_00016.png"
MODE, IMAGE = "i2v", FIRST_FRAME
# matched to minimax_h3_fast_t8.py so the two are directly comparable
LENGTH = 126        # frame count - the fallback, used only when SECONDS is None
SECONDS = 7.0       # duration in seconds; when this is set it overrides LENGTH
SAMPLER, SCHEDULER = "res_multistep", "simple"      # what h3_core_interactive.py uses
# SAMPLER, SCHEDULER = "euler", "beta"              # one evaluation per step, like the workflow

import h3_core as H

# seconds wins over the frame count, and snaps up to H3's 17k+5 grid (5.0s -> 124 frames)
if SECONDS is not None:
    LENGTH = H.frames_for(SECONDS)

# The long side is the only size setting; the short side follows the first frame's aspect
# ratio. H3 needs both axes to be multiples of 32 (16x VAE compression x the 2x2 patchify)
# inside a 768*1344 area cap - the previous hardcoded 512x860 had a height of 860, which is
# not a multiple of 32 (860 // 16 = 53, an odd latent row count the patchify cannot tile).
# LONG_SIDE = None defers to ComfyUI's own adapt_canvas, which fixes the SHORT edge at 768.
LONG_SIDE = 1280
WIDTH, HEIGHT = H.fit_size(IMAGE, LONG_SIDE, length=LENGTH)
print(f"prompt {len(PROMPT)} chars | {WIDTH}x{HEIGHT} x {LENGTH} frames "
      f"({H.duration_of(LENGTH):.2f}s) | {STEPS} steps | {SAMPLER}+{SCHEDULER}")

_t = time.perf_counter()
cond, latent = H.encode(clip, vae, PROMPT, mode=MODE, width=WIDTH, height=HEIGHT,
                        length=LENGTH, image=IMAGE)
ENCODE_S = time.perf_counter() - _t
print(f"encode  {ENCODE_S:6.1f}s")
L.status()


# %% 8. generate - timed per stage

for i in range(3):
    SEED = 68+i
    _t = time.perf_counter()
    out = H.denoise(model, cond, latent, steps=STEPS, sampler=SAMPLER, scheduler=SCHEDULER, seed=SEED)
    SAMPLE_S = time.perf_counter() - _t

    _t = time.perf_counter()
    mp4 = H.decode(vae, audio_vae, out,
                H.out_dir_for(__file__) / f"interactive_fast_{WIDTH}x{HEIGHT}_{SEED}.mp4")
    DECODE_S = time.perf_counter() - _t

    print(f"sample  {SAMPLE_S:6.1f}s  ({SAMPLE_S / STEPS:.2f}s per step)")
    print(f"decode  {DECODE_S:6.1f}s")
    print(f"total   {ENCODE_S + SAMPLE_S + DECODE_S:6.1f}s")
    print(f"-> {mp4}")
    # Measured: switching res_multistep+simple for euler+beta changed nothing outside noise, and
    # four LoRAs cost nothing measurable at load or per step. Sparse attention is the ~2.3x.
    #
    # Keeping things loaded does not remove the warm-up either. h3_core's stages call
    # free(clip)/free_all() between themselves (release=True); passing release=False keeps them
    # loaded and a second generation in the same process still pays it: 106.6 s then 103.6 s,
    # first two steps ~20 s against 10.4 s steady. The working set (20 GB weights + 15 GB
    # conditioner + VAEs) is larger than the 32 GB card, so VBAR re-faults weights each pass.
    # That ~40 s is fixed per generation - it amortizes with more steps or a longer clip, and
    # only disappears if the weights actually fit, which resident=True cannot manage here.


# %% 9. VRAM - what is resident now, and how to hand it back
L.per_gpu()
L.resident()      # still there after a full run? that is the point of keep=True
L.status()
# L.free_all()
