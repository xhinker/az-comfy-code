#!/usr/bin/env python3
"""h3_tessera - general-purpose video upscaler built on ComfyUI-Tessera + MiniMax H3 ref2va.

Nothing in this module knows where your models live, which adapters to hang off the model,
or what the prompt should say: every path, string and setting is passed in by the caller.
`minimax_h3/tessera_upscaler.py` is the worked example.

This is not a resize. The clip is re-generated at the larger size with the source itself
wired in as the <Video 1> reference, so faces, edges and texture come back reconstructed
rather than interpolated - which is why it costs real sampling time, not milliseconds.

Contracts the ComfyUI graph hides when you call the nodes from a script, handled once here:

  * the Tessera nodes are *instance* methods - TesseraPlan().run(...), not TesseraPlan.run()
  * TesseraPlan.run returns a plain {"ui": ..., "result": (...)} dict, not a NodeOutput, and
    declares tile_width / tile_height / context_border with no defaults, so Plan and Refine
    must be handed the same numbers or they silently disagree
  * ConditioningZeroOut returns a tuple of outputs - index [0] is the conditioning
  * SaveVideo is an io.ComfyNode that reads cls.hidden.extra_pnginfo, which only exists under
    the graph executor; headless, write with the Video object's own save_to instead
  * the H3 DiT reads x[1] as the audio stream, so a video-only plate crashes inside
    model.forward - the plate is always concatenated with an audio latent, synthesised as
    silence when the source has no soundtrack

Requires the project loader `h3_load` on sys.path (it wraps UNETLoader / LoRA / attention
backend with DynamicVRAM already wired).
"""

import pathlib
import sys
import time


def init(comfy_root: str, tessera_root: str | None = None) -> None:
    """Put ComfyUI and the Tessera pack on sys.path. Idempotent.

    tessera_root defaults to <comfy_root>/custom_nodes/ComfyUI-Tessera."""
    tessera_root = tessera_root or f"{comfy_root}/custom_nodes/ComfyUI-Tessera"
    for path in (str(comfy_root), str(tessera_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


def result(node_output):
    """NodeOutput / dict-with-result / tuple / list -> tuple of values.

    The dict case has to come first: TesseraPlan hands back a plain dict, a dict has no
    .result attribute, and the fall-through would wrap the whole thing as one element."""
    if isinstance(node_output, dict) and "result" in node_output:
        return tuple(node_output["result"])
    if hasattr(node_output, "result"):
        return node_output.result
    if isinstance(node_output, (tuple, list)):
        return tuple(node_output)
    return (node_output,)


def load_support(*, comfy_root: str, text_encoder: str, video_vae: str, audio_vae: str,
                 device: int = 0, keep: float = 0.0):
    """Text encoder + video VAE + audio VAE. Shared by every refine; load once per session.

    `keep` is the DynamicVRAM resident budget for these three (0.0 = stream them)."""
    init(comfy_root)
    import h3_load as L
    clip = L.load_text_encoder(text_encoder, device=device, keep=keep)
    vae = L.load_vae(video_vae, device=device, keep=keep)
    avae = L.load_vae(audio_vae, device=device, keep=keep)
    L.resident()
    return clip, vae, avae


def load_ref2va(*, comfy_root: str, model_path: str, loras=(), device: int = 0,
                resident: bool = False, tau: float = 1.5, dense_blocks: str = "0,1",
                min_tokens: int = 12288, extra_tokens: int = 256, start_percent: float = 0.0,
                end_percent: float = 1.0, shift_video: float = 12.0, shift_audio: float = 3.0,
                verbose: bool = True):
    """Load the ref2va checkpoint and patch it: attention backend first, then the adapters.

    tau 1.5 across the whole trajectory with blocks 0 and 1 dense is the pack's own
    setting. extra_tokens re-admits the highest-scoring tokens per query block so an
    individually important token survives a low-scoring block; set it to 0 only if you
    switch to VSA, whose weights are trained against their own sparsity pattern.
    Missing LoRA files are skipped with a warning rather than raising."""
    init(comfy_root)
    import h3_load as L
    model = L.load_diffusion_model(model_path, weight_dtype="default", device=device,
                                   resident=resident)
    model = L.sigma_shift(model, shift_video=shift_video, shift_audio=shift_audio)
    model = L.sparse_attention(model, "sol-attn", tau=tau, start_percent=start_percent,
                               end_percent=end_percent, dense_blocks=dense_blocks,
                               min_tokens=min_tokens, extra_tokens=extra_tokens,
                               verbose=verbose)
    applied = 0
    for path, strength in loras:
        if not pathlib.Path(path).is_file():
            print(f"skip missing refine lora: {pathlib.Path(path).name}")
            continue
        model = L.load_lora(model, path, strength)
        applied += 1
        print(f"refine lora x{strength}: {pathlib.Path(path).name}")
    if not applied and verbose:
        print("no distill/detail adapter on the refiner - expect a softer refine, and a "
              "low step count is likely too few for a bare ref2va model")
    L.status()
    return model


def round_to(n: float, multiple: int = 16) -> int:
    """H3's patchify wants clean multiples; an arbitrary source size x2 is not one by luck."""
    return max(multiple, int(round(n / multiple)) * multiple)


def read_video(video_path, *, comfy_root: str, max_frames: int = 0):
    """Any path on disk -> (images, audio, fps). LoadVideo insists on the ComfyUI input dir;
    VideoFromFile takes anything. max_frames trims the clip and slices the audio to match."""
    init(comfy_root)
    from comfy_api.latest import InputImpl
    from comfy_extras.nodes_video import GetVideoComponents
    comps = result(GetVideoComponents.execute(InputImpl.VideoFromFile(str(video_path))))
    images, audio, fps = comps[0], comps[1], comps[2]
    if max_frames and images.shape[0] > max_frames:
        seconds = max_frames / float(fps)
        images = images[:max_frames]
        if audio is not None:
            import torch
            wave = audio["waveform"]
            audio = {**audio, "waveform": wave[..., :int(seconds * audio["sample_rate"])]}
        print(f"trimmed to {max_frames} frames ({seconds:.1f}s)")
    print(f"source: {tuple(images.shape)} @ {float(fps):.1f} fps "
          f"{'with' if audio is not None else 'without'} audio")
    return images, audio, fps


def upscale(*, comfy_root: str, video_path, clip, vae, audio_vae, model, prompt: str,
            out_path=None, out_dir=None, upscale_by: float = 2.0, steps: int = 4,
            sampler: str = "er_sde", scheduler: str = "simple", seed: int = 0, cfg: float = 1.0,
            tile_budget: str = "low_vram", temporal_size: int = 32, temporal_overlap: int = 8,
            tile_width: int = 1024, tile_height: int = 1024, context_border: int = 64,
            segment_frames: int = 56, audio_sample_rate: int = 32000,
            images=None, audio=None, fps=None, max_frames: int = 0, release: bool = True,
            verbose: bool = True):
    """Re-render `video_path` at `upscale_by` x. Returns the written mp4.

    Everything that varies between users - the prompt, the step count and sampler, the tile
    geometry, where the file lands - is an argument. `steps=4 / er_sde` are the pack's
    numbers *for a distill adapter*; without one, raise steps toward 20.

    Pass `images` / `audio` / `fps` from read_video() to reuse an already-decoded (and
    trimmed) clip; otherwise the file is read here. `out_path` wins over `out_dir`; with
    neither, the file lands next to the source as <stem>_x<N>_<W>x<H>.mp4."""
    init(comfy_root)
    import torch
    import h3_load as L
    import comfy.samplers
    from comfy_api.latest import Types
    from comfy_extras.nodes_audio import VAEEncodeAudio
    from comfy_extras.nodes_lt import LTXVConcatAVLatent
    from comfy_extras.nodes_mask import SolidMask
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
    from comfy_extras.nodes_video import CreateVideo
    from nodes import ConditioningZeroOut, SetLatentNoiseMask
    from tessera.nodes import TesseraEncode, TesseraPlan, TesseraRefine

    if images is None:
        images, audio, fps = read_video(video_path, comfy_root=comfy_root, max_frames=max_frames)
    frames = int(images.shape[0])
    src_h, src_w = int(images[0].shape[0]), int(images[0].shape[1])
    target_w, target_h = round_to(src_w * upscale_by), round_to(src_h * upscale_by)

    # 1. the source clip is the reference. The prompt has to declare it and demand it be
    #    reproduced one to one - that is how you avoid a refine that "improves" your
    #    subject into a stranger.
    pos_ref, _ = result(MiniMaxH3ReferenceToVideo.execute(
        clip=clip, prompt=prompt, width=target_w, height=target_h, length=frames,
        vae=vae, audio_vae=audio_vae, ref_videos={"ref_video_0": images}))

    # 2. plate + guide conditioning, built in temporal windows
    enc = result(TesseraEncode().run(images=images, vae=vae, positive=pos_ref,
                                     upscale_by=upscale_by, upscale_method="lanczos",
                                     segment_frames=segment_frames))
    positive, plate, plate_w, plate_h, plate_frames = enc[0], enc[1], enc[2], enc[3], enc[4]
    if verbose:
        print(f"plate {plate_w}x{plate_h} x {plate_frames} frames")

    # 3. the DiT needs the audio half of the AV latent; a silent source gets synthesised
    #    silence rather than a crash inside model.forward
    if audio is None:
        seconds = plate_frames / float(fps or 24.0)
        audio = {"waveform": torch.zeros(1, 2, int(seconds * audio_sample_rate),
                                         dtype=torch.float32),
                 "sample_rate": audio_sample_rate}
        if verbose:
            print(f"no audio track - synthesised {seconds:.1f}s of silence at "
                  f"{audio_sample_rate} Hz for the AV latent")
    audio_latent = result(VAEEncodeAudio.execute(vae=audio_vae, audio=audio))[0]
    audio_latent = SetLatentNoiseMask().set_mask(
        audio_latent, result(SolidMask.execute(1.0, 32, 32))[0])[0]
    av_latent = result(LTXVConcatAVLatent.execute(video_latent=plate,
                                                  audio_latent=audio_latent))[0]

    # 4. plan, then refine - both get the same tile numbers, so they cannot disagree
    plan = result(TesseraPlan().run(width=int(plate_w), height=int(plate_h),
                                    frames=int(plate_frames), positive=positive, vae=vae,
                                    spatial_tile=tile_budget, temporal_tile=tile_budget,
                                    temporal_size=temporal_size,
                                    temporal_overlap=temporal_overlap, tile_width=tile_width,
                                    tile_height=tile_height, context_border=context_border))
    if verbose:
        print(plan[1])

    sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"),
                                             scheduler, steps)
    _t = time.perf_counter()
    refined = result(TesseraRefine().run(
        model=model, positive=positive, negative=ConditioningZeroOut().zero_out(pos_ref)[0],
        latent=av_latent, sampler=comfy.samplers.sampler_object(sampler), sigmas=sigmas,
        vae=vae, clip=clip, prompt=prompt, plan=plan[3], pixel_out=True,
        spatial_tile=tile_budget, temporal_tile=tile_budget, temporal_size=temporal_size,
        temporal_overlap=temporal_overlap, tile_width=tile_width, tile_height=tile_height,
        context_border=context_border, cfg=cfg, seed=seed, seed_mode="fixed",
        console_progress="overall"))
    refined_images = refined[2]
    print(f"refine {time.perf_counter() - _t:.1f}s -> {tuple(refined_images.shape)}")

    # 5. write. Audio is carried from the source, not regenerated: the returned latent is
    #    nested but decoding its audio stream with the audio VAE dies in tiled_scale, and
    #    the refine preserves frame count and timing, so the source track muxes back 1:1.
    stem = pathlib.Path(str(video_path)).stem
    if out_path is None:
        name = f"{stem}_x{upscale_by:g}_{plate_w}x{plate_h}.mp4"
        out_path = pathlib.Path(out_dir) / name if out_dir else \
            pathlib.Path(str(video_path)).resolve().parent / name
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    video = result(CreateVideo.execute(refined_images, float(fps or 24.0), audio=audio))[0]
    video.save_to(str(out_path), format=Types.VideoContainer("mp4"), codec=Types.VideoCodec("h264"))
    if release:          # batch callers pass release=False and free the model once at the end
        L.free(model)
    print(f"wrote {out_path}")
    return out_path
