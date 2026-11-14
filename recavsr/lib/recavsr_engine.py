#!/usr/bin/env python3
"""One streaming ReCaVSR pass: a clip in, an mp4 out, block by block.

Same loop the ReCaVSR Upscale Video node runs, driven from here so it can be called from a
notebook cell or a CLI without a workflow JSON. The transformer and the decoder come from the
node pack (see recavsr_load.py); everything below is the driving: the block schedule, the
memory switches, the colour match and the encode.

    upscale(engine, "in.mp4", "out.mp4", scale=4)

Per block: read `valid` frames, bilinear them up to the model size, one denoising step for the
block, decode its latent frames as they land, colour-match each decoded frame against its own
input frame, and hand the uint8 frames to an encoder thread. Nothing but the current block is
ever resident, so clip length costs disk.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

import comfy.model_management

import recavsr_load as L
from recavsr_media import BlockFrames, read_audio
from recavsr.session import (DecoderStream, VideoStream, adain_color_fix, block_schedule,
                             padded_size, prepare_block, scaled_size, upscale_frames, wavelet_color_fix)
from recavsr.video_io import Writer, pick_encoder


def to_uint8(rgb: torch.Tensor, frames, color_fix: str) -> torch.Tensor:
    """[T, 3, H, W] in [-1, 1] on the device -> [T, H, W, 3] uint8, colour matched to `frames`
    (the block's uint8 input). AdaIN matches per-frame mean and variance; wavelet keeps the
    output's high frequencies over the input's low ones."""
    if color_fix == "wavelet":
        out = wavelet_color_fix(rgb, frames)
    else:
        if color_fix == "adain":
            rgb = adain_color_fix(rgb, upscale_frames(frames, (rgb.shape[-2], rgb.shape[-1])).to(rgb.device))
        out = (rgb.float().clamp(-1, 1) + 1) * 0.5
    return out.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def upscale(e: dict, src, out, scale: float = 4.0, color_fix: str = "none", seed: int = 42,
            decoder: str = "flashdecoder", kernels: str = "fast", kv_offload: bool = False,
            encoder: str = "auto", max_frames: int = 0, start_frame: int = 0, post_size=None,
            keep_latents: str = "", verbose: bool = True) -> dict:
    """Run the clip through ReCaVSR and write the result. Returns a stats dict.

    post_size: (w, h) to shrink the model's output to afterwards - a Lanczos pass on the decoded
    frames, so x4 rendered big and shrunk keeps more detail than asking for a smaller ratio.
    keep_latents: a .pt path to also dump the generated latents (for experiments).
    """
    model = e.get("model")
    if model is None:
        raise SystemExit("no model loaded - call recavsr_load.load(engine) first")
    device = torch.device(e["device"])
    if kernels == "fast" and not L.kernels_ok():
        print("kernels: this interpreter's comfy-kitchen has no window_attention -> using exact "
              "(the ComfyUI venv has the fused ones: %s)" % L.PYTHON)
        kernels = "exact"

    source = BlockFrames(src)
    total = source.total - start_frame
    if max_frames:
        total = min(total, max_frames)
    in_h, in_w = source.get(start_frame, start_frame + 1).shape[1:3]
    size = scaled_size(in_h, in_w, scale)                      # (height, width), rounded half up
    height, width = padded_size(*size)                         # up to the 32-pixel token grid
    blocks = list(block_schedule(total))
    out_size = tuple(post_size) if post_size else size

    chunk, kv_offload, need = L.fit(height, width, kv_offload)
    card = comfy.model_management.get_total_memory(device) * 0.85
    if need > card:
        fits = scale * (card / need) ** 0.5
        raise SystemExit(f"{size[1]}x{size[0]} needs about {need / 2**30:.0f} GiB of VRAM for the step "
                         f"alone; this card has {card / 0.85 / 2**30:.0f} GiB. Lower --scale to about "
                         f"{fits:.2f}, or feed a smaller clip.")

    chosen = pick_encoder(encoder, out_size[1], out_size[0], source.fps)
    audio = read_audio(src, start_frame, total, float(source.fps)) if source.info["has_audio"] else None
    model.transformer.model.lq_proj.chunk = L.LQ_CHUNK_FRAMES if chunk else 0

    if verbose:
        print(f"{source.total} frames {in_w}x{in_h} -> {out_size[1]}x{out_size[0]} (x{scale}; model "
              f"{width}x{height}), {len(blocks)} blocks, {model.precision} transformer, {kernels} kernels, "
              f"{decoder}, colour {color_fix}, encoder {chosen}"
              + (", attention history in RAM" if kv_offload else "")
              + (f", input projection {L.LQ_CHUNK_FRAMES} frames at a time" if chunk else ""))

    comfy.model_management.load_models_gpu(model.patchers() if decoder == "flashdecoder" else [model.transformer],
                                          memory_required=need)
    started = time.perf_counter()
    stream = VideoStream(model.transformer.model, model.prompt, height, width, seed, device, kernels,
                         torch.device("cpu") if kv_offload else None)
    decode = DecoderStream(model.decoder.model, model.latent_format, device, torch.bfloat16, kernels) \
        if decoder == "flashdecoder" else None
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda d=None: None)
    sync(device)
    if verbose:
        print(f"models on {device} and conditioning ready in {time.perf_counter() - started:.1f} s")

    latents, written, transform_s, decode_s = [], 0, 0.0, 0.0
    writer = Writer(str(out), out_size[1], out_size[0], source.fps, chosen, device, audio)
    # Inference mode for the whole loop. The node pack never turns autograd off itself - in the
    # ComfyUI server the sampler's exec_mode does - so driven from here it would build a
    # gradient graph over every intermediate and roughly double the VRAM of a step.
    grad = torch.inference_mode()
    grad.__enter__()
    try:
        for number, (start, valid, block) in enumerate(blocks, 1):
            raw = source.get(start_frame + start, start_frame + start + valid)
            lq = prepare_block(raw, 0, valid, block, size, device)
            t0 = time.perf_counter()
            latent = stream.step(lq)
            sync(device)
            t1 = time.perf_counter()
            if decode is None:                                   # the stock VAE decodes at the end
                latents.append(latent.cpu())
            else:
                offset, remaining = 0, valid
                for block_rgb in decode.step(latent, lq):      # [1, 3, frames, H, W]
                    take = min(remaining, block_rgb.shape[2])
                    if take:
                        # one batch of decoded frames as [T, 3, H, W], cropped to the asked-for size
                        rgb = block_rgb[0, :, :take, :out_size[0], :out_size[1]].transpose(0, 1)
                        if post_size:
                            rgb = F.interpolate(rgb.float(), size=out_size, mode="lanczos3",
                                               antialias=True).clamp(-1, 1)
                        writer.put(to_uint8(rgb, raw[offset:offset + take], color_fix))
                        offset += take
                        remaining -= take
                sync(device)
            t2 = time.perf_counter()
            written += valid
            transform_s += t1 - t0
            decode_s += t2 - t1
            if verbose:
                print(f"  block {number}/{len(blocks)}  frames {start + 1}-{start + valid}  "
                      f"transformer {(t1 - t0) * 1000:.0f} ms  decode {(t2 - t1) * 1000:.0f} ms  "
                      f"{written / (time.perf_counter() - started):.2f} fps")
        if decode is None:
            t0 = time.perf_counter()
            rgb = e["vae"].decode(model.latent_format.process_out(torch.cat(latents, 2)))
            rgb = rgb.reshape(-1, *rgb.shape[-3:])[:total, :out_size[0], :out_size[1]].permute(0, 3, 1, 2)
            for start in range(0, total, 4):
                frames = rgb[start:start + 4].to(device) * 2 - 1
                if post_size:
                    frames = F.interpolate(frames.float(), size=out_size, mode="lanczos3", antialias=True)
                writer.put(to_uint8(frames, source.get(start_frame + start, start_frame + start + frames.shape[0]),
                                    color_fix))
            decode_s += time.perf_counter() - t0
    finally:
        grad.__exit__(None, None, None)
        writer.close()
        source.close()

    if keep_latents:
        torch.save(torch.cat(latents, 2) if latents else None, keep_latents)

    elapsed = time.perf_counter() - started
    stats = {"frames": written, "in_size": (in_w, in_h), "out_size": (out_size[1], out_size[0]),
             "model_size": (width, height), "scale": scale, "fps": float(source.fps), "blocks": len(blocks),
             "seconds": elapsed, "measured_fps": written / elapsed, "transformer_s": transform_s,
             "decode_s": decode_s, "kernels": kernels, "decoder": decoder, "weights": e.get("kind"),
             "kv_offload": kv_offload, "lq_chunk": chunk, "color_fix": color_fix, "output": str(out)}
    if verbose:
        print(f"{written} frames in {elapsed:.1f} s ({written / elapsed:.2f} fps): transformer "
              f"{transform_s:.1f} s, decode {decode_s:.1f} s -> {out}")
    return stats
