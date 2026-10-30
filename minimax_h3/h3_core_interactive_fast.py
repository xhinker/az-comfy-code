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

for _p in (UNET, CLIP, VAE, AUDIO_VAE):
    print("OK      " if pathlib.Path(_p).is_file() else "MISSING ", _p)


# (name, strength) in the order the workflow chains them - strength matters as much as order
PIKA_FLEX = f"{COMFY}/models/loras/minimax_h3/pika_h3_flex_fl2va_comfy_bf16.safetensors"
# pika_h3_flex_ref2va_comfy_bf16.safetensors is the same adapter for the Ref2VA base - this
# UNET is fl2va, so the fl2va file is the one that matches.

# FLEX swaps the step-distillation adapter, it does not add a third one: the two turbo
# adapters and Pika Flex are all doing the same job (teaching the model to take few steps),
# so stacking them fights. Pika Flex is BF16 rank 128 (fused QKV rank 384), 416 tensors,
# the same 12 module key patterns as the turbo pruned LoRA and no adaln pairs, so it lands
# on the pruned fl2va layout. Card: use model strength 1.0 - the conversion preserves unit
# scaling and ships no alpha tensors.
FLEX = True

DISTILL = ([(PIKA_FLEX, 1.0)] if FLEX else
           [(f"{COMFY}/models/loras/minimax_h3/minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors", 0.75),
            (f"{COMFY}/models/loras/minimax_h3/minimax_h3_fl2v_turbo_8step_v1.0_768p_FeiHouRemix_v0.6_compat_v001_T8.safetensors", 0.75)])
# Style/motion adapters are a different category from acceleration adapters, so they are not
# forbidden the way a second turbo is. Their own metadata says they were trained on the plain
# H3 base at the same schedule Flex uses - h3_musubi_v4 carries ss_h3_shift_video 12.0,
# ss_h3_shift_audio 3.0, uniform timestep sampling, rank 16/alpha 16; SoftBreastMotion is
# rank 32 on MiniMax-H3. So nothing conflicts. The one thing to expect: 0.35 and 0.5 were
# tuned for the 8-step turbo chain, and at six steps content LoRAs read stronger.
# Set False for the single-adapter config the official template actually verifies.
CONTENT_ON_FLEX = True
CONTENT = [] if (FLEX and not CONTENT_ON_FLEX) else [
    (f"{COMFY}/models/loras/minimax_h3/h3_musubi_v4-000040.safetensors", 0.35),
    (f"{COMFY}/models/loras/minimax_h3/SoftBreastMotion.safetensors", 0.5),
]
LORAS = DISTILL + CONTENT

for _p in [UNET, CLIP, VAE, AUDIO_VAE, *[p for p, _ in LORAS]]:
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
# Flex card: video shift 12, audio shift 3. The turbo chain card says 12 / 4-6, which is
# what 5.0 is for - keep the pair with the adapter, not as a taste dial.
SHIFT_VIDEO, SHIFT_AUDIO = (12.0, 3.0) if FLEX else (12.0, 5.0)
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


# %% 7. prompt and geometry
STEPS = 6 if FLEX else 8      # Flex is a six-NFE adapter

FIRST_FRAME = "/home/andrewzhu/storage_8t_4/comfyui_output/output/WanVideo_SCAIL_00016.png"
MODE, IMAGE = "i2v", FIRST_FRAME
# matched to minimax_h3_fast_t8.py so the two are directly comparable
LENGTH = 126        # frame count - the fallback, used only when SECONDS is None
SECONDS = 7.0       # duration in seconds; when this is set it overrides LENGTH
# The official Flex template (video_pika_h3_flex_fl2va.json) uses KSamplerSelect=lcm with
# ManualSigmas and BasicGuider (CFG 1.0). lcm is the sampler that injects the contract's
# "fresh intermediate noise"; SCHEDULER is ignored once SIGMAS is set below.
SAMPLER, SCHEDULER = ("lcm", "simple") if FLEX else ("res_multistep", "simple")
# SAMPLER, SCHEDULER = "euler", "beta"              # one evaluation per step, like the workflow

import h3_core as H

# The prompt lives in a markdown file. prompts/ is gitignored - prompt drafts are working
# content, not source. None resolves to minimax_h3/prompts/prompt.md; point PROMPT_FILE at
# another file to keep several prompts side by side. There is no fallback to a saved graph
# any more: an empty file stops the run instead of quietly generating old graph text.
PROMPT_FILE = None
PROMPT = H.read_prompt(PROMPT_FILE)
if not PROMPT:
    raise ValueError(f"no prompt - write one in {BASE / 'prompts' / 'prompt.md'}")

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
    SEED = 73+i
    _t = time.perf_counter()
    # The six-NFE schedule. flex_sigmas reproduces the template's ManualSigmas string
    # exactly: "1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0". Choosing 6 steps in
    # a generic scheduler gives an even grid, not this one.
    sigmas = H.flex_sigmas(shift_video=SHIFT_VIDEO) if FLEX else None
    out = H.denoise(model, cond, latent, steps=STEPS, sampler=SAMPLER, scheduler=SCHEDULER,
                    seed=SEED, sigmas=sigmas)
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
