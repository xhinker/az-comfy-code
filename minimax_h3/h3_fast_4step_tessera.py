#!/usr/bin/env python3
"""h3_fast_4step_tessera - draft a clip cheap, then re-render it sharp with ComfyUI-Tessera.

Stage 1 is the four-step draft from h3_fast_4step.py: LightX2V adapter, core sol-attn,
deliberately small (LONG_SIDE 576, 5 seconds) because its only job here is to produce a
source clip fast.

Stage 2 is ComfyUI-Tessera (hashu786/ComfyUI-Tessera), which is not a sampler-speed tool -
it is a MiniMax-H3 *ref2va* re-renderer. It takes the draft, builds an upscaled "plate"
latent plus guide conditioning from it in temporal windows (TesseraEncode), plans a tile
grid (TesseraPlan), and resamples tile-by-tile / chunk-by-chunk (TesseraRefine) with the
draft itself as <Video 1> reference. The point, per the pack's own notes and the DeepForest
test clip, is that detail and faces come back *reconstructed* rather than interpolated:
a squished face at 480x864 is regenerated properly at 1440x2592.

So the two stages are complementary: 4 NFE to find the shot, ref2va refine to finish it.
Cost note - the refine is a real generation, not a resize: at UPSCALE_BY=2 it samples
REFINE_STEPS forwards over a 2x-larger token grid, tiled. It is the expensive half.

Video-only for this test: the refine runs without the audio half of the AV latent, and the
draft's soundtrack is muxed back in. The pack's full recipe also wires an audio latent
through (LTXVConcatAVLatent), a Qwen-Image first-frame anchor, SAM3 hero masks and a
separate fine model - all optional, all skipped here to keep the first run legible.

KERNEL: the project .venv. Run cells top to bottom once.
"""

# %% 0. setup - put minimax_h3/ on sys.path so h3_load and h3_core import
import os
import pathlib
import sys
import time

BASE = pathlib.Path(__file__).resolve().parent
COMFY = "/mnt/data_2t_3/github_repos/ComfyUI"
sys.path.insert(0, str(BASE / "lib"))
sys.path.insert(0, COMFY)
sys.path.insert(0, f"{COMFY}/custom_nodes/ComfyUI-Tessera")   # tessera/nodes.py

import h3_load as L

## %% 1. components - same three files the Flex tutorial lists
UNET = f"{COMFY}/models/diffusion_models/minimax_h3/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REF2VA = f"{COMFY}/models/diffusion_models/minimax_h3/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
TE = f"{COMFY}/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = f"{COMFY}/models/vae/minimax_h3/minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = f"{COMFY}/models/vae/minimax_h3_audio_vae_fp32.safetensors"

# The pack's refine adapters. The workflow names a rank-averaged conversion
# (minimax_h3_taomate_3step_lora_avg_rank_19_bf16) that is not published under that name;
# these are the upstream originals it was made from, both verified to attach 258/258 to the
# ref2va base:
#   taomate_h3_3step_comfy.safetensors   Robert1212star/TaoMate-H3-3Step-ComfyUI (2.48 GB)
#   minimax_h3_lms_v1.0_r64.safetensors  RunningHubAI (metadata: minimax_h3_ref2va_sharpness)
# The loader skips anything missing and says so, so the file still runs without them.
LORAS = [(f"{COMFY}/models/loras/minimax_h3/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors", 1.0),
         (f"{COMFY}/models/loras/minimax_h3/h3_musubi_v4-000040.safetensors", 0.35),
         (f"{COMFY}/models/loras/minimax_h3/SoftBreastMotion.safetensors", 0.5)]
REFINE_LORAS = [(f"{COMFY}/models/loras/minimax_h3/taomate_h3_3step_comfy.safetensors", 1.0),
                (f"{COMFY}/models/loras/minimax_h3/minimax_h3_lms_v1.0_r64.safetensors", 0.75)]

## %% 2. engine - DynamicVRAM needs every device named before anything allocates
# H3_GPUS overrides the cards without editing the file - useful when the notebook kernel is
# already holding the support card (it did: an ipykernel with 18 GB on GPU 2, OOM incoming).
UNET_GPU, TE_GPU, VAE_GPU = (int(x) for x in os.environ.get("H3_GPUS", "0, 2, 2").split(","))
L.init_engine(verbose=True, devices=[UNET_GPU, TE_GPU])

## %% 3. keep budgets - what stays resident vs what streams
SUPPORT_KEEP = 0.0
UNET_RESIDENT = False

## %% 4. draft geometry - small on purpose, this is the cheap half
SEED                        = 76
LONG_SIDE                   = 768
SECONDS                     = 5.0
STEPS                       = 4
SAMPLER, SCHEDULER          = "res_multistep", "simple"
SHIFT_VIDEO, SHIFT_AUDIO    = 12.0, 3.0

### %% 4b. refine settings - the expensive half
UPSCALE_BY          = 2                 # plate = draft x 2; the pack validates 2x with an anchor, 4x without
REFINE_STEPS        = 4                 # the pack samples this LoRA at 3; 4 leaves a little margin
REFINE_SAMPLER      = "er_sde"          # the pack's choice for the main pass
TILE_BUDGET         = "low_vram"        # spatial and temporal tiles scale together to fit this
TEMPORAL_SIZE       = 32                # frames per temporal chunk (the example uses 124 on a bigger card)
TEMPORAL_OVERLAP    = 8
TILE_W, TILE_H      = 1024, 1024        # the pack's validated tile size; Plan and Refine must agree
CONTEXT_BORDER      = 64                # context carried into every tile edge
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

## %% 5. prompt for the draft - same markdown file the other generators read
FIRST_FRAME = "/home/andrewzhu/storage_8t_4/comfyui_output/output/WanVideo_SCAIL_00016.png"
MODE, IMAGE = "i2v", FIRST_FRAME

import h3_core as H

PROMPT_FILE = None
PROMPT = H.read_prompt(PROMPT_FILE)
if not PROMPT:
    raise ValueError(f"no prompt - write one in {BASE / 'prompts' / 'prompt.md'}")
LENGTH = H.frames_for(SECONDS)

## %% 6. load - text encoder and VAEs are shared by both stages
clip        = L.load_text_encoder(TE, device=TE_GPU, keep=SUPPORT_KEEP)
vae         = L.load_vae(VIDEO_VAE, device=VAE_GPU, keep=SUPPORT_KEEP)
audio_vae   = L.load_vae(AUDIO_VAE, device=VAE_GPU, keep=SUPPORT_KEEP)
L.resident()

## %% 7. stage 1 - the four-step draft
model = L.load_diffusion_model(UNET, weight_dtype="default", device=UNET_GPU, resident=UNET_RESIDENT)
model = L.sigma_shift(model, shift_video=SHIFT_VIDEO, shift_audio=SHIFT_AUDIO)
for path, strength in LORAS:
    if not pathlib.Path(path).is_file():
        print(f"skip missing lora: {pathlib.Path(path).name}")
        continue
    model = L.load_lora(model, path, strength)
    print(f"lora x{strength}: {pathlib.Path(path).name}")
model = L.sparse_attention(model, "sol-attn", tau=1.3, start_percent=0.2, end_percent=0.9,
                           min_tokens=12288, extra_tokens=256, verbose=True)
L.status()

W, H_ = H.fit_size(IMAGE, long_side=LONG_SIDE, length=LENGTH)
print(f"draft {W}x{H_} x {LENGTH} frames ({H.duration_of(LENGTH):.2f}s) | {STEPS} steps")

#%%
cond, latent    = H.encode(clip, vae, PROMPT, mode=MODE, width=W, height=H_, length=LENGTH, image=IMAGE)

#%%
_t = time.perf_counter()
out             = H.denoise(model, cond, latent, steps=STEPS, sampler=SAMPLER, scheduler=SCHEDULER, seed=SEED)
draft           = H.decode(vae, audio_vae, out, H.out_dir_for(__file__) / f"draft_{W}x{H_}_75.mp4")
print(f"draft done in {time.perf_counter() - _t:.1f}s -> {draft}")

# %% 8. hand the draft to Tessera - free the draft model first, the refiner needs the room
L.free(model)
del model

from comfy_extras.nodes_video import GetVideoComponents, CreateVideo
from comfy_api.latest import InputImpl, Types
from comfy_extras.nodes_audio import VAEEncodeAudio, VAEDecodeAudio
from comfy_extras.nodes_lt import LTXVConcatAVLatent
from comfy_extras.nodes_mask import SolidMask
from nodes import SetLatentNoiseMask              # VideoFromFile takes any absolute path; LoadVideo insists on the ComfyUI input dir
from nodes import ConditioningZeroOut          # core nodes.py, not comfy_extras
import comfy.samplers                           # same sampler/sigma calls h3_core.denoise uses
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
from tessera.nodes import TesseraEncode, TesseraPlan, TesseraRefine

def result(node_output):
    """NodeOutput / dict-with-result / tuple / list -> the tuple of values.

    TesseraPlan returns a plain dict {"ui": ..., "result": (...)}, not a NodeOutput, so the
    dict case has to be checked before the attribute case - a dict has no .result attribute
    and would otherwise come back as a single-element tuple."""
    if isinstance(node_output, dict) and "result" in node_output:
        return tuple(node_output["result"])
    if hasattr(node_output, "result"):
        return node_output.result
    if isinstance(node_output, (tuple, list)):
        return tuple(node_output)
    return (node_output,)

vid                         = InputImpl.VideoFromFile(str(draft))                         # core VIDEO container
comps                       = result(GetVideoComponents.execute(vid))   # images, audio, fps, bit_depth, color_space
images, draft_audio, fps    = comps[0], comps[1], comps[2]
print(f"source clip: {tuple(images.shape)} @ {float(fps):.1f} fps "
      f"{'with' if draft_audio is not None else 'without'} audio")

## %% 9. the ref2va model, patched the way the pack's example workflow patches it
# UNETLoader -> ModelAttentionBackend -> MiniMaxChunkFeedForward -> BlockSparseAttention
# (sol-attn tau 1.5, dense_blocks '0,1', extra_tokens 256) -> 3-step distill -> LMS detail
model = L.load_diffusion_model(REF2VA, weight_dtype="default", device=UNET_GPU, resident=UNET_RESIDENT)
model = L.sigma_shift(model, shift_video=SHIFT_VIDEO, shift_audio=SHIFT_AUDIO)
model = L.sparse_attention(model, "sol-attn", tau=1.5, start_percent=0.0, end_percent=1.0,
                           dense_blocks="0,1", min_tokens=12288, extra_tokens=256, verbose=True)
applied = 0
for path, strength in REFINE_LORAS:
    if not pathlib.Path(path).is_file():
        print(f"skip missing refine lora: {pathlib.Path(path).name}")
        continue
    model = L.load_lora(model, path, strength)
    applied += 1
    print(f"refine lora x{strength}: {pathlib.Path(path).name}")
if not applied:
    print("no distill/detail adapter on the refiner - the refine will look softer and "
          "REFINE_STEPS=4 is likely too few for a bare ref2va model")
L.status()

# %% 10. guide conditioning: the draft is <Video 1>, fully preserved
target_w, target_h = int(images[0].shape[1]) * UPSCALE_BY, int(images[0].shape[0]) * UPSCALE_BY
pos_ref, _ = result(
    MiniMaxH3ReferenceToVideo.execute(
        clip=clip, prompt=REFINE_PROMPT, width=target_w, height=target_h, length=LENGTH,
        vae=vae, audio_vae=audio_vae, ref_videos={"ref_video_0": images}
    )
)

enc = result(TesseraEncode().run(images=images, vae=vae, positive=pos_ref,
                               upscale_by=UPSCALE_BY, upscale_method="lanczos", segment_frames=56))
positive, plate, plate_w, plate_h, plate_frames = enc[0], enc[1], enc[2], enc[3], enc[4]
print(f"plate {plate_w}x{plate_h} x {plate_frames} frames; latent keys {sorted(plate['samples'])}")

# The H3 DiT takes an AV latent - model.forward reads x[1] as the audio stream - so a
# video-only plate crashes inside the model. Same construction as the pack's workflow:
# encode the draft's soundtrack, mask it to full noise so the model re-renders the audio
# instead of copying it, then concat onto the plate.
audio_latent    = result(VAEEncodeAudio.execute(vae=audio_vae, audio=draft_audio))[0]
noise_mask      = result(SolidMask.execute(1.0, 32, 32))[0]
audio_latent    = SetLatentNoiseMask().set_mask(audio_latent, noise_mask)[0]
av_latent       = result(LTXVConcatAVLatent.execute(video_latent=plate, audio_latent=audio_latent))[0]
print(f"AV latent: {sorted(av_latent['samples'])} shape {tuple(av_latent['samples'].shape)}")

plan_out = result(TesseraPlan().run(width=int(plate_w), height=int(plate_h), frames=int(plate_frames),
                                  positive=positive, vae=vae, spatial_tile=TILE_BUDGET,
                                  temporal_tile=TILE_BUDGET, temporal_size=TEMPORAL_SIZE,
                                  temporal_overlap=TEMPORAL_OVERLAP,
                                  tile_width=TILE_W, tile_height=TILE_H,
                                  context_border=CONTEXT_BORDER))
print(f"tile plan:\n{plan_out[1]}")

# %% 11. refine - tile by tile, chunk by chunk
_t = time.perf_counter()
sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"),
                                         SCHEDULER, REFINE_STEPS)
refined = result(TesseraRefine().run(
    model=model, positive=positive, negative=ConditioningZeroOut().zero_out(pos_ref)[0],   # classic nodes return (output,)
    latent=av_latent, sampler=comfy.samplers.sampler_object(REFINE_SAMPLER), sigmas=sigmas,
    vae=vae, clip=clip, prompt=REFINE_PROMPT, plan=plan_out[3],
    pixel_out=True, spatial_tile=TILE_BUDGET, temporal_tile=TILE_BUDGET,
    temporal_size=TEMPORAL_SIZE, temporal_overlap=TEMPORAL_OVERLAP,
    tile_width=TILE_W, tile_height=TILE_H, context_border=CONTEXT_BORDER,
    cfg=1.0, seed=SEED, seed_mode="fixed", console_progress="overall"))
refined_latent, report, refined_images = refined[0], refined[1], refined[2]
print(f"refine {time.perf_counter() - _t:.1f}s for {tuple(refined_images.shape)}")
print(f"report:\n{report}")

## %% 12. write it out - the Video object's own writer.
# SaveVideo is an io.ComfyNode that reads cls.hidden.extra_pnginfo / .prompt, which only
# exist under the graph executor; called from a script they are None and it throws. save_to
# is the same encode path with an explicit destination, and it skips the node's
# write-into-output-dir-then-move dance.
# The refine hands back the refined *video* stream; feeding that to the audio VAE crashes
# inside tiled_scale (it wants the nested AV latent's audio stream). The refined clip keeps
# the draft's exact frame count and timing, so the draft's soundtrack muxes back 1:1 -
# audio is carried through, not regenerated. The pack's full recipe regenerates it by
# keeping the AV latent nested end to end.
print(f"refined latent: shape={tuple(refined_latent['samples'].shape)} "
      f"nested={getattr(refined_latent['samples'], 'is_nested', False)}")
refined_audio = draft_audio
video = result(CreateVideo.execute(refined_images, float(fps), audio=refined_audio))[0]
dest = H.out_dir_for(__file__) / f"tessera_x{UPSCALE_BY}_{plate_w}x{plate_h}_75.mp4"
video.save_to(str(dest), format=Types.VideoContainer("mp4"), codec=Types.VideoCodec("h264"))
print(f"draft  {W}x{H_} -> {draft}")
print(f"refine {plate_w}x{plate_h} -> {dest}")
L.status()
