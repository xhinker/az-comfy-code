#!/usr/bin/env python3
"""minimax_h3_5s_fast.json, as Python cells.

Source workflow: ComfyUI/user/default/workflows/minimax_h3/minimax_h3_5s_fast.json
Graph it reproduces (from the workflow's own note):

  UNET -> LoRA x4 -> MemoryEfficientSageAttentionPatch -> ModelAttentionBackend
       -> SolAttnMiniMax -> DualClockSamplerT8 -> BasicGuider -> SamplerCustomAdvanced
       -> AVDecodeT8 -> CreateVideo -> SaveVideo

Nothing goes through the ComfyUI server: the custom-node classes are imported and called
directly, so this is the same code path the server runs, minus the HTTP layer and the graph.

KERNEL: the project .venv (it now carries comfy-aimdo, comfy-kitchen and torchaudio).
"""

# %% 0. setup - minimax_h3/, ComfyUI, and the T8 custom-node pack on sys.path
import importlib.util
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path.cwd()
COMFY = pathlib.Path("/mnt/data_2t_3/github_repos/ComfyUI")
T8 = COMFY / "custom_nodes/comfyui-minimax-h3-audio-T8"

for p in (BASE, COMFY, T8):
    sys.path.insert(0, str(p))

# Do NOT import ComfyUI's `server` here: it drags in comfy_aimdo.host_buffer, which snapshots
# the native library at import time, and right now that library has not been loaded yet.
# The server stub belongs in cell 7, after init_engine().


# %% 1. the files this workflow uses - absolute paths, edit to point anywhere
UNET = f"{COMFY}/models/diffusion_models/minimax_h3/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP = f"{COMFY}/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE = f"{COMFY}/models/vae/minimax_h3/minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = f"{COMFY}/models/vae/minimax_h3_audio_vae_fp32.safetensors"
FIRST_FRAME = f"{COMFY}/input/output_seed_1422.png"
WORKFLOW = COMFY / "user/default/workflows/minimax_h3/minimax_h3_5s_fast.json"

# (name, strength) in the order the workflow chains them - strength matters as much as order
LORAS = [
    (f"{COMFY}/models/loras/minimax_h3/minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors", 0.75),
    (f"{COMFY}/models/loras/minimax_h3/minimax_h3_fl2v_turbo_8step_v1.0_768p_FeiHouRemix_v0.6_compat_v001_T8.safetensors", 0.75),
    (f"{COMFY}/models/loras/minimax_h3/h3_musubi_v4-000040.safetensors", 0.35),
    (f"{COMFY}/models/loras/minimax_h3/SoftBreastMotion.safetensors", 0.5),
]

for _p in [UNET, CLIP, VAE, AUDIO_VAE, FIRST_FRAME, WORKFLOW, *[p for p, _ in LORAS]]:
    print("OK      " if pathlib.Path(_p).is_file() else "MISSING ", _p)


## %% 2. engine - before any model touches CUDA; switches on DynamicVRAM
import h3_load as L

L.init_engine(verbose=True)
L.status()          # want dynamic_vram=True aimdo=True


## %% 3. the transformer - note this is the PRUNED int8 convrot base, not the Ref2VA one
model = L.load_diffusion_model(UNET, weight_dtype="default")


## %% 4. the text encoder
clip = L.load_text_encoder(CLIP, type="minimax", device="default")


## %% 5. the VAEs
vae = L.load_vae(VAE)
audio_vae = L.load_vae(AUDIO_VAE)


## %% 6. the LoRA chain, same order and strengths as the graph
for path, strength in LORAS:
    model = L.load_lora(model, path, strength)
    print(f"lora x{strength}: {pathlib.Path(path).name}")
L.status()


# %% 7. the attention chain - sage memory patch, then pytorch backend, then sol-attn
# KJ's node module touches PromptServer.instance at import time; in-process there is no
# server, so give it the attribute it expects. Safe here because init_engine() has run.
import server
if not hasattr(server.PromptServer, "instance"):
    server.PromptServer.instance = None

def _out(res):
    """Node classes return io.NodeOutput; some return a bare tuple."""
    return res.result[0] if hasattr(res, "result") else res[0]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# sageattention >= 2.2 is what this needs, and 2.2.0 is not on PyPI - it has to be built
# from source (recipe in minimax_h3/README.md and requirements.txt). It is installed in this
# venv now, so this is on; set False to compare against the dense path.
SAGE_PATCH = True
if SAGE_PATCH:
    kj = _load_module("kj_ltxv_nodes", f"{COMFY}/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py")
    model = _out(kj.MiniMaxH3MemoryEfficientSageAttentionPatch.execute(model))
else:
    print("skipping MiniMaxH3MemoryEfficientSageAttentionPatch (needs sageattention>=2.2)")

model = L.attention_backend(model, "pytorch attention")     # the workflow sets this explicitly

import sol_attn_minimax_v2
model = _out(sol_attn_minimax_v2.SolAttnMiniMax.execute(
    model, tau=1.3, start_percent=0.2, end_percent=0.9, min_tokens=12288,
    sink_conditioning="exact_kv_and_rows", morton=False, morton_curve="2d_frame",
    dense_blocks="", verbose=False, tau_profile=None, routed_cap_percent=0,
    centroid_tail=True, reuse_qkv_memory=False))
# min_tokens=12288 is a gate, not a size: below that token count sol-attn does not engage,
# so a short or small clip silently runs dense.


# %% 8. prompt, geometry, seed
# The prompt this workflow uses is a long structured one (subject_definitions, summary,
# retention_analysis, detailed_description, soundscape). Rather than duplicate it here,
# read it straight out of the saved workflow so the two never drift apart.
import json

def workflow_prompt(path=WORKFLOW, node_type="MiniMaxH3ImageToVideo"):
    wf = json.loads(pathlib.Path(path).read_text())
    for n in wf["nodes"]:
        if n["type"] == node_type:
            return n["widgets_values"][0]
    raise KeyError(node_type)


PROMPT = workflow_prompt()
# The saved widgets are portrait 768x1344 at 226 frames (~9.4 s). The workflow's own note
# suggests 864x480 x 124 (5.0 s) as the cheap canvas - ~50k tokens, still above the
# sol-attn gate. steps must match the turbo LoRA: 8 here, 4 with a 4-step LoRA.
WIDTH, HEIGHT, LENGTH = 768, 1344, 226
STEPS, SHIFT_VIDEO, SHIFT_AUDIO = 8, 12.0, 3.0
SAMPLER, SCHEDULER = "dual_clock_euler", "native_flow"
SEED = 62
print(f"prompt {len(PROMPT)} chars | {WIDTH}x{HEIGHT} x {LENGTH} frames | {STEPS} steps")


# %% 9. encode - MiniMaxH3ImageToVideo with the first frame
import h3_core as H

cond, latent = H.encode(clip, vae, PROMPT, mode="i2v", width=WIDTH, height=HEIGHT,
                        length=LENGTH, image=FIRST_FRAME)
L.status()


# %% 10. sample - the T8 dual-clock setup, then the same guider/sampler the graph uses
from h3_t8.sampling import setup_dual_clock_sampling
import comfy.sample
from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise

L.free_all()        # the DiT needs the whole card to stream its layers
model_s, sampler, sigmas = setup_dual_clock_sampling(
    model, latent, STEPS, SHIFT_VIDEO, SHIFT_AUDIO, SAMPLER, SCHEDULER)
print("sigmas:", [round(float(s), 3) for s in sigmas])

samples = comfy.sample.fix_empty_latent_channels(model_s, latent["samples"])
noise = Noise_RandomNoise(SEED).generate_noise(latent)
guider = Guider_Basic(model_s)
guider.set_conds(cond)
out = dict(latent)
out["samples"] = guider.sample(noise, samples, sampler, sigmas, seed=SEED, disable_pbar=False)


## %% 11. decode and mux - AVDecodeT8's own decode, then the same CreateVideo/SaveVideo path
from h3_t8.audio_ops import decode_av_latent
from comfy_api.latest import InputImpl, Types
from fractions import Fraction

L.free(model_s)
frames, audio, _vlat, _alat = decode_av_latent(out, vae, audio_vae)
wave = audio.get("waveform")
audio = {**audio, "waveform": wave.detach()} if hasattr(wave, "detach") else audio

video = InputImpl.VideoFromComponents(Types.VideoComponents(
    images=frames.detach(), audio=audio, frame_rate=Fraction(24)),
    bit_depth=8, color_space="sRGB")
mp4 = H.out_dir_for(__file__) / f"fast_5s_{WIDTH}x{HEIGHT}_{SEED}.mp4"
video.save_to(mp4)
print(f"-> {mp4}")


# %% 12. hand it back
L.status()
# L.free_all()
