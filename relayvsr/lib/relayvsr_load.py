#!/usr/bin/env python3
"""Loaders and machine plumbing for RelayVSR, on ComfyUI's core code.

RelayVSR is two models, and they load separately so you can hold each one in a
variable from an interactive session:

    load_wan(engine)     the large Wan backbone that makes keyframe reference latents
    load_flash(engine)   the small Flash decoder that super-resolves every frame

Unlike the H3 / MOSS folders in this repo, RelayVSR has no released ComfyUI node to call:
PR Comfy-Org/ComfyUI#17511 merged RelayVSR into master on 2026-11-05 (RelayVSRConditioning +
VSRDecode), but the newest release, 0.44.0, predates it and needs a comfy-kitchen newer than
the 0.2.37 here for its window_attention. So the networks here are the reference
implementation ported into lib/. What ComfyUI core still gives us is
the part that matters on a consumer card - `comfy.quant_ops` INT8 conv-rotation kernels
for the Wan linears (5.2 GB of weights instead of 9.4 GB in bf16) and
`comfy.model_management` for device and memory reporting.

Weights live in `relayvsr/models/` (git-ignored). They are the Comfy-Org repackage of
kopper/RelayVSR, so the same files the ComfyUI workflow uses:

    models/diffusion_models/relayvsr_x4_int8_convrot.safetensors      5.16 GB
    models/diffusion_models/relayvsr_x2_lightx2v_int8_convrot.safetensors 5.16 GB
    models/vae/relayvsr_flash_vae_x4_fp32.safetensors                 128 MB
    models/vae/relayvsr_flash_vae_x2_fp32.safetensors                 128 MB

KERNEL: either venv works - both have torch + comfy-kitchen, and `comfy` is reached through
the sys.path insert below rather than an installed package. Verified on this box:

    .venv/bin/python                                              (py3.13, torch 2.15.0+cu132)
    /mnt/data_2t_3/github_repos/ComfyUI/venv_py312_comfyui/bin/python   (py3.12, torch 2.12.1)

Pylance needs `relayvsr/lib` in python.analysis.extraPaths (.vscode/settings.json) or every
`import relayvsr_*` in the scripts shows as unresolved - it never runs the cell-0 sys.path.insert.
"""
from __future__ import annotations

import pathlib
import sys

COMFY = pathlib.Path("/mnt/data_2t_3/github_repos/ComfyUI")          # the engine install
PYTHON = str(COMFY / "venv_py312_comfyui" / "bin" / "python")   # one working interpreter; the repo .venv works too

# Put ComfyUI on sys.path up front: relayvsr_weights imports comfy.quant_ops, and that
# only resolves from inside the ComfyUI checkout.
if str(COMFY) not in sys.path:
    sys.path.insert(0, str(COMFY))

HERE = pathlib.Path(__file__).resolve().parent.parent                # .../relayvsr
MODELS = HERE / "models"

WEIGHTS = {
    4: {
        "wan": MODELS / "diffusion_models" / "relayvsr_x4_int8_convrot.safetensors",
        "flash": MODELS / "vae" / "relayvsr_flash_vae_x4_fp32.safetensors",
    },
    2: {
        "wan": MODELS / "diffusion_models" / "relayvsr_x2_lightx2v_int8_convrot.safetensors",
        "flash": MODELS / "vae" / "relayvsr_flash_vae_x2_fp32.safetensors",
    },
}

_M = {}      # comfy modules, filled by init_comfy()


def init_comfy(verbose: bool = True, device: str | None = None):
    """Put ComfyUI on sys.path and take its device / memory helpers. Returns the module map."""
    if _M:
        return _M
    if str(COMFY) not in sys.path:
        sys.path.insert(0, str(COMFY))
    import comfy.model_management as mm          # noqa: F401
    _M["mm"] = mm
    if hasattr(mm, "init_mm"):                   # newer builds need the explicit init
        mm.init_mm()
    if verbose:
        print(f"comfy {COMFY}  device={mm.get_torch_device()}  "
              f"vram={mm.get_total_memory(mm.get_torch_device()) / 1e9:.1f} GB")
    return _M


def check_weights(scale: int = 4) -> dict:
    """Resolve the two files for a scale and report their safetensors format tags."""
    from safetensors import safe_open
    from relayvsr_metadata import inspect_safetensors

    if scale not in WEIGHTS:
        raise ValueError(f"RelayVSR ships x2 and x4 weights; asked for x{scale}")
    out = {}
    for part, path in WEIGHTS[scale].items():
        if not path.is_file():
            raise FileNotFoundError(
                f"missing RelayVSR {part} weights: {path}\n"
                "download them with:  ./get_weights.sh   (see relayvsr/README.md)")
        info = inspect_safetensors(path)
        if info is None:
            raise ValueError(f"{path} is not a RelayVSR checkpoint (no format tag)")
        component, tag_scale = info
        if component != part or tag_scale != scale:
            raise ValueError(f"{path} is a {component} x{tag_scale} weight, expected {part} x{scale}")
        out[part] = {"path": str(path), "size_gb": path.stat().st_size / 1e9, "format": info}
    return out


def engine(scale: int = 4, dtype: str = "bf16", offload: bool = False, device=None, verbose: bool = True):
    """Build the engine for x2 or x4 weights. Nothing is read off disk until load_wan()."""
    init_comfy(verbose=verbose)                # comfy on sys.path + model_management
    from relayvsr_engine import RelayVSREngine
    from relayvsr_metadata import BundleInfo

    info = check_weights(scale)
    mm = _M.get("mm")
    dev = device or (mm.get_torch_device() if mm else __import__("torch").device("cuda"))
    bundle = BundleInfo(
        directory=MODELS,
        wan_path=pathlib.Path(info["wan"]["path"]),
        flash_path=pathlib.Path(info["flash"]["path"]),
        scale=scale,
    )
    eng = RelayVSREngine(bundle, dtype=dtype, offload=offload, device=dev)
    if verbose:
        print(f"x{scale} bundle  dtype={dtype}  offload={offload}  device={dev}")
        for part in ("wan", "flash"):
            print(f"  {part:6s} {info[part]['size_gb']:.2f} GB  {info[part]['path']}")
    return eng


def load_wan(eng, verbose: bool = True):
    """Stage 1: the Wan backbone (INT8 conv-rot linears go through comfy.quant_ops)."""
    import time
    t0 = time.perf_counter()
    eng.load_wan()
    if verbose:
        print(f"Wan backbone loaded in {time.perf_counter() - t0:.1f} s  "
              f"layers={len(eng.wan.blocks)}  dim={eng.wan_pre.dim if hasattr(eng.wan_pre, 'dim') else '?'}  "
              f"{vram_line()}")
    return eng


def load_flash(eng, verbose: bool = True):
    """Stage 2: the Flash decoder (fp32 file, ~128 MB)."""
    import time
    t0 = time.perf_counter()
    eng.load_flash()
    if verbose:
        print(f"Flash decoder loaded in {time.perf_counter() - t0:.1f} s  "
              f"depth={eng.flash_depth}  {vram_line()}")
    return eng


def vram_line() -> str:
    import torch
    if not torch.cuda.is_available():
        return "vram=n/a"
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    return f"vram alloc={alloc:.1f} GB reserved={reserved:.1f} GB"


def free(eng, verbose: bool = True):
    """Drop both networks and hand the memory back."""
    import torch
    eng.free()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if verbose:
        print(f"freed  {vram_line()}")


def parse_downscale(spec, out_w: int, out_h: int):
    """Resolve a downscale spec to a final (width, height), or None for "leave it alone".

    Three forms, chosen by how they are written:
        "0.5"       a factor of the model output - must carry the decimal point
        "720"       a target height; the width follows the aspect ratio
        "1248x720"  an exact size
    Dimensions are rounded down to even numbers because the encoder is YUV420.
    """
    s = "" if spec is None else str(spec).strip().lower().replace(" ", "")
    if not s:
        return None
    if "x" in s:
        try:
            w, h = (int(v) for v in s.split("x", 1))
        except ValueError:
            raise SystemExit(f"downscale: expected WxH, got {spec!r}")
    elif "." in s:
        f = float(s)
        if not 0.0 < f < 1.0:
            raise SystemExit(f"downscale: a factor has to be between 0 and 1 (got {spec!r}); "
                             "write a bare number for a target height, or WxH for an exact size")
        w, h = int(out_w * f), int(out_h * f)
    else:
        h = int(s)
        w = int(round(out_w * h / out_h))
    if w < 2 or h < 2:
        raise SystemExit(f"downscale {spec!r} asks for a {w}x{h} picture; a factor needs to be "
                         "above 0, a target height at least 2 pixels")
    w, h = w - w % 2, h - h % 2
    if (w, h) == (out_w, out_h):
        return None
    if w > out_w or h > out_h:
        raise SystemExit(f"downscale {spec!r} -> {w}x{h} is bigger than the model output "
                         f"{out_w}x{out_h}; this option only shrinks")
    return (w, h)


def downscale_frames(frames, size):
    """Lanczos shrink of the model's frames, applied as they stream (no extra file)."""
    import numpy as np
    from PIL import Image
    width, height = size
    for frame in frames:
        yield np.asarray(Image.fromarray(frame).resize((width, height), Image.LANCZOS))


def run(eng, input_path, sr_ratio: int = 2, chunk_len: int = 16, seed: int = 42,
        detail: float = 1.0, downscale=None,
        max_frames: int = 0, start_frame: int = 0, check=None):
    """Start an upscale. Returns (frames, plan) - the work happens as `frames` is consumed.

    `frames` yields HWC uint8 numpy arrays, one per input frame, at the output size.
    """
    import relayvsr_media as media

    info = media.video_info(input_path)
    target = eng.geometry.output_size(sr_ratio, info["height"], info["width"])
    plan = {
        "input": str(input_path), "in_size": (info["width"], info["height"]),
        "out_size": (target[1], target[0]), "fps": info["fps"], "frames_in_file": info["frames"],
        "sr_ratio": sr_ratio, "native_scale": eng.geometry.scale, "chunk_len": chunk_len,
        "seed": seed, "detail": detail, "max_frames": max_frames, "start_frame": start_frame,
        "has_audio": info["has_audio"],
    }
    frames = eng.restore(
        relayv_iter(input_path, max_frames, start_frame),
        seed=seed, sr_ratio=sr_ratio, chunk_len=chunk_len, detail=detail, check=check,
    )
    # The shrink happens after the model, on the way to the encoder: the x4 weights rendered at
    # 4x and then downscaled keep measurably more detail than the x2 weights at 2x.
    size = parse_downscale(downscale, *plan["out_size"]) if downscale else None
    if size:
        frames = downscale_frames(frames, size)
        plan["model_size"], plan["out_size"], plan["downscale"] = plan["out_size"], size, str(downscale)
    return frames, plan


def relayv_iter(input_path, max_frames: int = 0, start_frame: int = 0):
    import relayvsr_media as media
    return media.iter_frames(input_path, max_frames=max_frames, start_frame=start_frame)


def save(frames, out_path, fps: float, codec: str = "h264", crf: int = 18,
         preset: str = "medium", audio_from=None, report_every: int = 16):
    """Consume the frames from run() into a video file. Returns (path, frames_written, seconds)."""
    import time
    import torch
    import relayvsr_media as media

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with media.VideoWriter(out_path, fps=fps, codec=codec, crf=crf, preset=preset,
                           audio_from=audio_from) as writer:
        for i, frame in enumerate(frames):
            writer.write(frame)
            if report_every and (i + 1) % report_every == 0:
                dt = time.perf_counter() - t0
                print(f"  {i + 1} frames  {dt:6.1f} s  {(i + 1) / dt:5.2f} fps", flush=True)
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"peak CUDA memory: {peak:.1f} GB")
    return str(out_path), writer.frames, time.perf_counter() - t0
