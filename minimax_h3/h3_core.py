#!/usr/bin/env python3
"""MiniMax H3 stages on ComfyUI's core code - the part that runs after the loaders.

Loaders, DynamicVRAM setup and VRAM helpers live in h3_load.py; this file only wires
the four stages, each one a plain function over the components you loaded:

    cond, latent = encode(clip, vae, "a red fox ...", length=124)
    latent = denoise(model, cond, latent, steps=20, seed=7)
    mp4 = decode(vae, audio_vae, latent, OUT_DIR / "fox.mp4")

Latent is ComfyUI's joint AV dict: samples is a NestedTensor pair, video
[B,24,T,H/16,W/16] and audio [B,32,2,T40]. One denoising loop produces both streams,
which is why there is no negative prompt and no guidance scale anywhere: the released
checkpoints are guidance-distilled, so every step is exactly one forward pass.

Run with the ComfyUI venv:
    /mnt/data_2t_3/github_repos/ComfyUI/venv_py312_comfyui/bin/python h3_core.py ...
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from fractions import Fraction

BASE = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path.cwd()
sys.path.insert(0, str(BASE))

import h3_load as L                                   # noqa: E402
from h3_load import (UNET, CLIP, VAE_VIDEO, VAE_AUDIO, free, free_all, load_diffusion_model,  # noqa: E402
                     load_lora, load_text_encoder, load_vae, sparse_attention, status,
                     sigma_shift, attention_backend, vram_gb)

OUT_DIR = BASE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHIFT_DEFAULT = (12.0, 3.0)


def _result(node_output):
    """io.NodeOutput is a tuple of the node's outputs; older builds return a bare tuple."""
    return node_output.result if hasattr(node_output, "result") else tuple(node_output)


def image_tensor(path):
    """ComfyUI IMAGE tensors are [B, H, W, C] float in 0..1."""
    import numpy as np
    import torch
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return torch.from_numpy(np.asarray(img, dtype="float32") / 255.0)[None]


# --------------------------------------------------------------------------- stage 2
def encode(clip, vae, prompt: str, *, mode: str = "t2v", width: int = 1344, height: int = 768,
           length: int = 124, image: str | None = None, last_image: str | None = None,
           ref_images=(), ref_image_size: str = "match", audio_vae=None, release: bool = True):
    """MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo -> (conditioning, AV latent).

    length is frames at 24 fps on the 17k+5 grid: 22 ~ 1 s, 124 ~ 5 s, 362 ~ 15 s.
    For ref2va, tag the references in the prompt as <Picture 1>, <Picture 2>, ... in the
    same order you pass them; that ordering is what labels them.
    """
    L.init_engine()
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo

    if mode == "ref2va":
        refs = {f"ref_image_{i}": image_tensor(p) for i, p in enumerate(ref_images, start=1)} or None
        out = MiniMaxH3ReferenceToVideo.execute(clip=clip, vae=vae, audio_vae=audio_vae, prompt=prompt,
                                                width=width, height=height, length=length,
                                                ref_image_size=ref_image_size, ref_images=refs)
    else:
        extra = {}
        if image:
            extra["first_frame"] = image_tensor(image)
        if last_image:
            extra["last_frame"] = image_tensor(last_image)
        out = MiniMaxH3ImageToVideo.execute(clip=clip, vae=vae, prompt=prompt,
                                            width=width, height=height, length=length, **extra)
    cond, latent = _result(out)
    if release:
        free(clip)          # the 15 GB conditioner has done its work; the DiT gets the room
    return cond, latent


# --------------------------------------------------------------------------- stage 3
def denoise(model, cond, latent, *, steps: int = 20, sampler: str = "res_multistep",
            scheduler: str = "simple", seed: int = 7, release: bool = True):
    """BasicGuider + BasicScheduler + RandomNoise + SamplerCustomAdvanced, in one call.

    cfg is fixed at 1.0 by construction: comfy.samplers.sample() insists on a negative
    conditioning and H3 has none, so this uses Guider_Basic like BasicGuider does.
    """
    L.init_engine()
    import comfy.samplers
    import comfy.sample
    from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise

    if release:
        free_all()          # drop the conditioner and VAEs before the DiT streams in
    latent = dict(latent)
    latent["samples"] = comfy.sample.fix_empty_latent_channels(model, latent["samples"])
    noise = Noise_RandomNoise(seed).generate_noise(latent)
    sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps)
    guider = Guider_Basic(model)
    guider.set_conds(cond)
    out = dict(latent)
    out["samples"] = guider.sample(noise, latent["samples"], comfy.samplers.sampler_object(sampler),
                                   sigmas, seed=seed, disable_pbar=False)
    return out


# --------------------------------------------------------------------------- stage 4
def decode(vae, audio_vae, latent, out_path, *, fps: float = 24.0, tile_size: int | None = None,
           release: bool = True):
    """Decode both streams and mux them. tile_size is in pixels (VAEDecodeTiled units):
    None = plain decode with an automatic tiled retry, 512 if you want tiles up front."""
    L.init_engine()
    from comfy_api.latest import InputImpl, Types
    from comfy_extras.nodes_audio import vae_decode_audio

    if release:
        free_all()          # the DiT is done; the video VAE needs several GB of activations
    video_latent = latent["samples"].unbind()[0]      # last stream is the audio one
    try:
        frames = vae.decode(video_latent) if not tile_size else _decode_tiled(vae, video_latent, tile_size)
    except Exception as e:
        if "out of memory" not in str(e).lower():
            raise
        print(f"plain VAE decode OOMed, retrying tiled at 512 px: {e}")
        L.init_engine()["comfy"].model_management.soft_empty_cache()
        frames = _decode_tiled(vae, video_latent, 512)
    if frames.ndim == 5:                 # H3's decoder keeps a time axis: [B,T,H,W,C]
        frames = frames.reshape(-1, *frames.shape[-3:])
    audio = vae_decode_audio(audio_vae, latent)
    # ComfyUI's AUDIO is a dict {"waveform", "sample_rate"}; the latents carry grad and
    # the PyAV muxer wants plain arrays, so detach inside the dict.
    wave = audio.get("waveform") if isinstance(audio, dict) else None
    if wave is not None and hasattr(wave, "detach"):
        audio = {**audio, "waveform": wave.detach()}
    frames = frames.detach()
    video = InputImpl.VideoFromComponents(
        Types.VideoComponents(images=frames, audio=audio, frame_rate=Fraction(fps)),
        bit_depth=8, color_space="sRGB")
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    video.save_to(str(out_path))
    return out_path


def _decode_tiled(vae, video_latent, tile_size: int):
    """The maths from VAEDecodeTiled: pixel tiles converted to latent units, temporal
    tile derived from the VAE's temporal compression."""
    c = vae.spacial_compression_decode()
    tc = vae.temporal_compression_decode()
    if tc:
        tile_t = max(2, 64 // tc)
        overlap_t = max(1, min(tile_t // 2, 8 // tc))
    else:
        tile_t = overlap_t = None
    return vae.decode_tiled(video_latent, tile_x=tile_size // c, tile_y=tile_size // c,
                            overlap=max(1, 64 // c), tile_t=tile_t, overlap_t=overlap_t)


# ------------------------------------------------------------------------------ CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="MiniMax-H3 stages on ComfyUI's core modules")
    ap.add_argument("prompt")
    ap.add_argument("--mode", choices=["t2v", "i2v", "ref2va"], default="t2v")
    ap.add_argument("--image")
    ap.add_argument("--last-image")
    ap.add_argument("--ref-image", action="append", default=[])
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--length", type=int, default=124)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--sampler", default="res_multistep")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--unet", default=UNET)
    ap.add_argument("--clip", default=CLIP)
    ap.add_argument("--vae", default=VAE_VIDEO)
    ap.add_argument("--audio-vae", default=VAE_AUDIO)
    ap.add_argument("--lora", action="append", default=[], metavar="NAME:STRENGTH")
    ap.add_argument("--sparse", choices=["none", "sol-attn", "vsa", "sla"], default="none")
    ap.add_argument("--tau", type=float, default=1.3)
    ap.add_argument("--keep-percent", type=float, default=10.0)
    ap.add_argument("--backend", choices=[None, "pytorch attention", "comfy kitchen attention"], default=None)
    ap.add_argument("--shift-video", type=float, default=SHIFT_DEFAULT[0])
    ap.add_argument("--shift-audio", type=float, default=SHIFT_DEFAULT[1])
    ap.add_argument("--tile-vae", type=int, default=0, help="decode the video VAE in N-pixel tiles")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    t = time.time()
    model = load_diffusion_model(a.unet)
    clip = load_text_encoder(a.clip)
    vae = load_vae(a.vae)
    audio_vae = load_vae(a.audio_vae)
    for spec in a.lora:
        name, _, strength = spec.partition(":")
        model = load_lora(model, name, float(strength) if strength else 1.0)
    if (a.shift_video, a.shift_audio) != SHIFT_DEFAULT:
        model = sigma_shift(model, a.shift_video, a.shift_audio)
    if a.backend:
        model = attention_backend(model, a.backend)
    if a.sparse != "none":
        model = sparse_attention(model, a.sparse, tau=a.tau, keep_percent=a.keep_percent)
    print(f"components ready in {time.time() - t:.1f}s")
    status()

    t = time.time()
    cond, latent = encode(clip, vae, a.prompt, mode=a.mode, width=a.width, height=a.height,
                          length=a.length, image=a.image, last_image=a.last_image,
                          ref_images=a.ref_image, audio_vae=audio_vae)
    print(f"encode {time.time() - t:5.1f}s")

    for i in range(a.seeds):
        seed = a.seed + i
        t0 = time.time()
        out = denoise(model, cond, latent, steps=a.steps, sampler=a.sampler,
                      scheduler=a.scheduler, seed=seed)
        t_sample = time.time() - t0
        t1 = time.time()
        dst = decode(vae, audio_vae, out, OUT_DIR / f"{a.name or 'h3_' + str(seed)}.mp4",
                     tile_size=(a.tile_vae or None))
        print(f"seed {seed}: sample {t_sample:6.1f}s + decode {time.time() - t1:5.1f}s -> {dst}")


if __name__ == "__main__":
    main()
