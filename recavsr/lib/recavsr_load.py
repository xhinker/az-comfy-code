#!/usr/bin/env python3
"""Load ReCaVSR in-process, on ComfyUI's own model and memory management.

ReCaVSR (kopperx/ReCaVSR, arXiv 2609.37831) is a one-step streaming video super-resolution
model on Wan2.2 TI2V-5B: 21 frames, then 8 at a time, one denoising step per block, the
previous block's latents recycled back in, and every transformer layer keeping only the
slice of the past K/V its learned router picked. It decodes with an LR-conditioned
FlashDecoder instead of the Wan VAE.

Unlike the relayvsr/ folder in this repo, this one does NOT reimplement the network.
ComfyUI-ReCaVSR (PedroMarinhoDev/ComfyUI-ReCaVSR, in ComfyUI/custom_nodes) is a native node
pack that runs the released weights on ComfyUI's ops, and it is GPL-3.0 - so its code stays
in its own repo and is imported here, never copied. What lives in this folder is the driver:
the clip in, the streaming loop, the mp4 out, and the knobs.

    load(engine, "int8")     transformer + FlashDecoder + prompt (into RAM; they go to the
                             GPU on the first run, through ComfyUI's own streaming)
    load_vae(engine)         the stock Wan2.2 VAE, only for decoder="vae"
    fit(h, w)                will this output size run, and what has to be switched on

Weights live in ComfyUI's models/recavsr - the same folder the nodes read - and get_weights.sh
fetches them:

    transformer_int8_convrot.safetensors  5.98 GB   int8 ConvRot, ~1.8x the bf16 speed
    transformer.safetensors              10.1 GB   the released bf16
    prompt.safetensors                   4.0 MB    the baked-in prompt embedding
    flashdecoder.safetensors             218 MB    the LR-conditioned decoder
    vae.safetensors                     2.69 GB    optional, for --decoder vae

KERNEL: run with the ComfyUI venv -
    /mnt/data_2t_3/github_repos/ComfyUI/venv_py312_comfyui/bin/python   (py3.12, torch 2.12.1+cu130)
Both `comfy` and the `recavsr` package resolve through the sys.path inserts below, not from
installed packages. triton and opencv must be in that venv (they are); without triton the
attention kernels fall back to PyTorch and get several times slower.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

COMFY = pathlib.Path("/mnt/data_2t_3/github_repos/ComfyUI")                    # the engine install
PACK = COMFY / "custom_nodes" / "ComfyUI-ReCaVSR"                              # the node pack we drive
PYTHON = str(COMFY / "venv_py312_comfyui" / "bin" / "python")                  # the interpreter to use
MODELS = pathlib.Path(os.environ.get("RECAVSR_MODELS", str(COMFY / "models" / "recavsr")))

for _p in (str(COMFY), str(PACK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

WEIGHTS = {"int8": "transformer_int8_convrot.safetensors", "bf16": "transformer.safetensors"}
REQUIRED = ("prompt.safetensors", "flashdecoder.safetensors")
VAE_FILE = "vae.safetensors"

# One output token covers a 32x32 pixel patch: the VAE's 16x and the model's 2x2 patch.
STRIDE = 32
# What a streaming step costs per output token, measured on this box as peak-minus-weights.
ACT_BYTES_PER_TOKEN = int(2.6 * 1024 * 1024)
# The router keeps 66 planes of bf16 keys and values over the 30 transformer blocks; this is
# the part that fits in system RAM with kv_offload.
KV_BYTES_PER_TOKEN = 66 * 2 * 3072 * 2
# Feeding the LR input projection 4 frames at a time instead of a whole block.
LQ_CHUNK_SAVING_PER_TOKEN = int(0.66 * 1024 * 1024)
LQ_CHUNK_FRAMES = 4


def pick_gpu(index) -> None:
    """Pin the process to one card. Call BEFORE anything imports torch - the only reliable
    way to choose a card here, since ComfyUI reads the device once, globally.

        CUDA_GPU=2 ./recavsr_upscale.py ...          (the CLI does this for you)
    """
    if index is None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)


def engine(device: str = "cuda:0") -> dict:
    """The handle the cells and the CLI pass around: model set, optional VAE, device."""
    return {"model": None, "vae": None, "device": device, "kind": "int8"}


def _complete(path: pathlib.Path) -> bool:
    """Is this safetensors file actually finished on disk?

    A plain size check lies: aria2 writes the full length immediately and fills it in out of
    order, so a half-downloaded file looks whole. A safetensors file is an 8-byte header length,
    a JSON header, then the tensor bytes - so the header says where the data should end.
    """
    if path.with_suffix(path.suffix + ".aria2").exists():
        return False                                          # aria2's own "not finished" marker
    try:
        with path.open("rb") as handle:
            length = int.from_bytes(handle.read(8), "little")
            if not 0 < length < 100_000_000:
                return False
            header = json.loads(handle.read(length))
        end = max((v["data_offsets"][1] for key, v in header.items()
                   if key != "__metadata__" and isinstance(v, dict) and "data_offsets" in v), default=0)
        return path.stat().st_size >= 8 + length + end
    except Exception:
        return False


def check(kind: str = "int8") -> list[str]:
    """Weight files that are missing or still downloading, [] when everything needed is ready."""
    missing = []
    for name in REQUIRED + (WEIGHTS[kind],):
        path = MODELS / name
        if not path.is_file():
            missing.append(f"{path} (absent)")
        elif not _complete(path):
            missing.append(f"{path} (truncated or still downloading)")
    return missing


def load(e: dict, kind: str = "int8"):
    """Load the transformer, the FlashDecoder and the prompt. They land in RAM; ComfyUI moves
    them to the GPU at the first run and streams them back out when the card is tight."""
    missing = check(kind)
    if missing:
        raise SystemExit("missing weights, run get_weights.sh:\n  " + "\n  ".join(missing))
    from recavsr import loader

    e["kind"] = kind
    e["model"] = loader.load(str(MODELS / WEIGHTS[kind]), str(MODELS / "flashdecoder.safetensors"),
                             str(MODELS / "prompt.safetensors"))
    return e["model"]


def load_vae(e: dict):
    """The stock Wan2.2 VAE, only needed for decoder="vae" (finer texture, ~30x slower decode)."""
    path = MODELS / VAE_FILE
    if not path.is_file():
        raise SystemExit(f"no {VAE_FILE} in {MODELS}; run get_weights.sh bf16")
    import comfy.sd
    import comfy.utils

    e["vae"] = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(path)))
    return e["vae"]


def kernels_ok() -> bool:
    """Can the fused `fast` kernels run in THIS interpreter?

    The pack reaches them through `comfy.quant_ops.ck.window_attention`, which needs a
    comfy-kitchen new enough to carry it. `comfy` comes from the ComfyUI checkout either way,
    but comfy-kitchen comes from the venv - the repo's .venv has one without it, the ComfyUI
    venv's has it. Without it the FlashDecoder's window op raises AttributeError.
    """
    try:
        import comfy.quant_ops as q

        return hasattr(getattr(q, "ck", None), "window_attention")
    except Exception:
        return False


def activation_bytes(height: int, width: int, kv_offload: bool = False, lq_chunk: bool = False) -> int:
    """VRAM the step itself needs at this padded output size - the weights can stream, this cannot."""
    per_token = (ACT_BYTES_PER_TOKEN
                 - (KV_BYTES_PER_TOKEN if kv_offload else 0)
                 - (LQ_CHUNK_SAVING_PER_TOKEN if lq_chunk else 0))
    return (height // STRIDE) * (width // STRIDE) * per_token


def fit(height: int, width: int, kv_offload: bool = False) -> tuple[bool, bool, int]:
    """(lq_chunk, kv_offload, bytes) - the cheapest switch that makes this size run on the card.

    Whole blocks first; if the step does not fit, run the LR input projection a few frames at a
    time (one bf16 rounding step from the whole-block result); if it still does not fit, put the
    attention history in system RAM, which costs a transfer at every step.
    """
    import comfy.model_management

    card = comfy.model_management.get_total_memory(comfy.model_management.get_torch_device()) * 0.85
    chunk = activation_bytes(height, width, kv_offload) > card
    if chunk and not kv_offload and activation_bytes(height, width, False, True) > card:
        kv_offload = True
    return chunk, kv_offload, activation_bytes(height, width, kv_offload, chunk)


def headroom(device: str = "cuda:0") -> str:
    """What the card shows right now, straight from CUDA - no ComfyUI bookkeeping involved."""
    import torch

    if not torch.cuda.is_available():
        return "no CUDA"
    index = torch.device(device).index or 0
    free_b, total_b = torch.cuda.mem_get_info(index)
    return f"{(total_b - free_b) / 2**30:.1f} / {total_b / 2**30:.1f} GiB used on cuda:{index}"


def free(e: dict, verbose: bool = True) -> None:
    """Drop the model set and let the allocator take the card back."""
    import gc

    import torch

    e["model"] = None
    e["vae"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if verbose:
        print("freed;", headroom(e["device"]))
