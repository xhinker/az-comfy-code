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

UNET = f"{COMFY}/models/diffusion_models/minimax_h3/MiniMax-H3-Ref2VA_int8_convrot.safetensors"
CLIP = f"{COMFY}/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE = f"{COMFY}/models/vae/minimax_h3/minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = f"{COMFY}/models/vae/minimax_h3_audio_vae_fp32.safetensors"

# the reference and the prompt this test borrows from the saved fast workflow
FIRST_FRAME = f"{COMFY}/input/output_seed_1422.png"
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

L.init_engine(verbose=True)
L.status()          # want dynamic_vram=True aimdo=True; if False, nothing downstream fits


## %% 3. the transformer  (UNETLoader)  ~0.2s
model = L.load_diffusion_model(UNET, weight_dtype="default")
L.status()


## %% 4. the text encoder  (CLIPLoader)  ~2s
clip = L.load_text_encoder(CLIP, type="minimax", device="default")
L.status()


## %% 5. the VAEs  (VAELoader)  ~0.1s each
vae = L.load_vae(VAE)
audio_vae = L.load_vae(AUDIO_VAE)

# %% 6. the LoRA chain, same order and strengths as the graph
for path, strength in LORAS:
    model = L.load_lora(model, path, strength)
    print(f"lora x{strength}: {pathlib.Path(path).name}")
L.status()


#%% shift
SHIFT_VIDEO, SHIFT_AUDIO = 12.0, 5.0     # drbaph card for this pruned turbo conversion: 12 / 4-6
model = L.sigma_shift(model, shift_video=SHIFT_VIDEO, shift_audio=SHIFT_AUDIO)

#%% memory-efficient attention (KJ nodes) - int8 QK / fp8 V, lowers peak VRAM.
# Needs sageattention >= 2.2, which is not on PyPI; 2.2.0 is built from source into this
# venv (recipe in requirements.txt). Order matters: the workflow patches sage attention
# first and sol-attn chains onto the result, so keep this above the sparse-attention cell.
model = L.sage_attention_patch(model)

#%%
model = L.sparse_attention(model, "sol-attn", tau=1.3, start_percent=0.2)


# %% 7. prompt and geometry - taken from the fast workflow, not rewritten here
import json

def workflow_prompt(path=WORKFLOW, node_type="MiniMaxH3ImageToVideo"):
    wf = json.loads(pathlib.Path(path).read_text())
    for n in wf["nodes"]:
        if n["type"] == node_type:
            return n["widgets_values"][0]
    raise KeyError(node_type)


PROMPT = workflow_prompt()
MODE, IMAGE = "i2v", FIRST_FRAME
# matched to minimax_h3_fast_test.py so the two are directly comparable
WIDTH, HEIGHT, LENGTH = 768, 1344, 226
STEPS, SEED = 8, 63
SAMPLER, SCHEDULER = "res_multistep", "simple"      # what h3_core_interactive.py uses
# SAMPLER, SCHEDULER = "euler", "beta"              # one evaluation per step, like the workflow
print(f"prompt {len(PROMPT)} chars | {WIDTH}x{HEIGHT} x {LENGTH} | {STEPS} steps | {SAMPLER}+{SCHEDULER}")

import h3_core as H

_t = time.perf_counter()
cond, latent = H.encode(clip, vae, PROMPT, mode=MODE, width=WIDTH, height=HEIGHT,
                        length=LENGTH, image=IMAGE)
ENCODE_S = time.perf_counter() - _t
print(f"encode  {ENCODE_S:6.1f}s")
L.status()


# %% 8. generate - timed per stage
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


# %% 9. VRAM - what is resident now, and how to hand it back
L.status()
# L.free_all()
