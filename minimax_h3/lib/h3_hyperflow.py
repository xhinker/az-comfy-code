#!/usr/bin/env python3
"""HyperFlow H3 Flash on top of the existing MiniMax H3 loaders.

HyperFlow H3 Flash (Video Rebirth, Nov 2026) is a compressed, few-step MiniMax-H3: the fl2va
partition drops 3 of 50 transformer blocks and low-rank-factorizes 102 linear layers, 13.7B
parameters against the ~31B original. It samples at CFG 1.0 in 4, 6 or 8 steps on a fixed sigma
grid, and every token is conditioned on two times - the current noise level t and the step
endpoint r - which is what makes 6 steps enough. Two-time needs the sampler's sigma schedule
inside the model call; ComfyUI core already passes it as transformer_options["sample_sigmas"]
(comfy/samplers.py, Guider.inner_sample), so nothing here has to wire it up.

What this module adds over h3_load / h3_core:

    load_node_packs()   import ComfyUI-GGUF + ComfyUI-HyperFlow-Flash and register their nodes
    load_gguf()         HyperFlowGGUFLoader in one call -> a patched MODEL
    sigmas()            HyperFlowSigmas -> the 4/6/8-step schedule to hand h3_core.denoise()
    info()              what the loader actually patched

Everything else - text encoder, VAEs, conditioning, sampling, decode, mux - is the ordinary H3
path in h3_load.py / h3_core.py, unchanged. The GGUF is the consumer weight format: Q4_K_M is
9.1 GB on disk and dequantizes layer by layer as it streams through DynamicVRAM.

Weights: huggingface.co/VideoRebirthAI/hyperflow-h3-flash (MiniMax H3 Community License - note
the territorial clause: not licensed in the EU, UK, South Korea or the US without authorization).
Node pack: github.com/Video-Rebirth/ComfyUI-HyperFlow-Flash (GPL-3.0), cloned into
ComfyUI/custom_nodes/; it is imported, never copied into this repo.
"""
from __future__ import annotations

import importlib
import pathlib
import sys

import h3_load as L

COMFY = L.COMFY
GGUF_DIR = COMFY / "models" / "diffusion_models" / "minimax_h3"

# the recommended consumer default: 9.1 GB, benchmarked at 99 s for a 5.2 s clip on an RTX 5090
DEFAULT_GGUF = GGUF_DIR / "minimax_h3_fl2va_hyperflow_flash_14b-Q4_K_M.gguf"

# companion files are the ordinary MiniMax H3 ones - nothing new to download
CLIP = L.CLIP                                   # qwen3vl_32b_minimax_h3_nvfp4_awq (15.7 GB)
VAE = L.VAE_VIDEO                               # int8 convrot video decoder (fits on 32 GB)
VAE_FP16 = L.VAE_VIDEO_FP16
AUDIO_VAE = L.VAE_AUDIO

PACKS = ("ComfyUI-GGUF", "ComfyUI-HyperFlow-Flash")
STEP_CHOICES = (4, 6, 8)
WEIGHTS_IN = ("auto", "ram", "disk")

_packs: dict[str, object] = {}


def load_node_packs(verbose: bool = True) -> dict[str, object]:
    """Register the two custom node packs inside this process, the way main.py would.

    The HyperFlow loader finds ComfyUI-GGUF by looking up UnetLoaderGGUF in ComfyUI's node
    registry, so the packs have to be registered, not merely importable.
    """
    if _packs:
        return _packs

    L.init_engine(verbose=False)          # comfy must be importable before the packs are
    import nodes

    custom = COMFY / "custom_nodes"
    if str(custom) not in sys.path:
        sys.path.insert(0, str(custom))

    for name in PACKS:
        path = custom / name
        if not path.is_dir():
            raise ImportError(f"missing custom node pack: {path}\n"
                              f"  git clone https://github.com/Video-Rebirth/ComfyUI-HyperFlow-Flash.git "
                              f"(and city96/ComfyUI-GGUF) into {custom}")
        try:
            mod = importlib.import_module(name)
        except ImportError as exc:
            raise ImportError(f"{name} failed to import: {exc}") from exc
        if hasattr(mod, "register"):
            mod.register()
        nodes.NODE_CLASS_MAPPINGS.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
        nodes.NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
        _packs[name] = mod

    if "UnetLoaderGGUF" not in nodes.NODE_CLASS_MAPPINGS:
        raise RuntimeError("ComfyUI-GGUF registered no UnetLoaderGGUF - the HyperFlow loader "
                           "cannot find its GGUF reader. Update ComfyUI-GGUF.")
    if verbose:
        print(f"node packs registered: {', '.join(PACKS)}")
    return _packs


def load_gguf(path: str | None = None, weights_in: str = "auto", verbose: bool = True):
    """HyperFlowGGUFLoader -> a MODEL whose low-rank linears and two-time embedder are installed.

    weights_in: ram keeps the weights that do not fit on the GPU in pinned host memory, disk
    leaves them in the file's page cache and streams them (what --fast-disk does in the UI).
    auto asks ComfyUI's storage policy. On this box (32 GB VRAM, 125 GB RAM) ram is fine and
    faster than disk.
    """
    packs = load_node_packs(verbose=False)
    if weights_in not in WEIGHTS_IN:
        raise ValueError(f"weights_in must be one of {WEIGHTS_IN}")
    target = pathlib.Path(path or DEFAULT_GGUF)
    if not target.is_file():
        raise FileNotFoundError(
            f"no HyperFlow GGUF at {target}\n"
            f"  hf download VideoRebirthAI/hyperflow-h3-flash fl2va_gguf/"
            f"minimax_h3_fl2va_hyperflow_flash_14b-Q4_K_M.gguf --local-dir <dir>")

    import time
    t0 = time.perf_counter()
    patcher = packs["ComfyUI-HyperFlow-Flash"].loader_nodes.load_hyperflow_gguf(str(target), weights_in)

    # The pack builds the two-time endpoint embedder and the low-rank wrappers as fresh
    # nn.Modules, and fresh modules default to requires_grad=True. ComfyUI freezes what it
    # loads itself, but not what a patch adds afterwards, so the sampler's output tensor
    # comes back attached to a graph and the video VAE then dies inside comfy-kitchen's
    # int8_linear with "Can't export tensors that require gradient". Freeze the whole model.
    trainable = sum(1 for prm in patcher.model.parameters() if prm.requires_grad)
    patcher.model.diffusion_model.requires_grad_(False)
    if verbose and trainable:
        print(f"HyperFlow: froze {trainable} patch-added parameters that wanted gradients")

    if verbose:
        print(f"HyperFlow transformer loaded in {time.perf_counter() - t0:.1f} s")
        info(patcher)
    return patcher


def sigmas(model, steps: int = 6):
    """The schedule this checkpoint samples: 4 speed / 6 default / 8 quality.

    Raw knots are unshifted and mapped through video_shift 12 (see the pack's step_sigmas).
    Pass the result as h3_core.denoise(..., sigmas=) - it sets steps from the list length.
    """
    packs = load_node_packs(verbose=False)
    if int(steps) not in STEP_CHOICES:
        raise ValueError(f"HyperFlow runs {STEP_CHOICES} steps, got {steps}")
    tensor = packs["ComfyUI-HyperFlow-Flash"].loader_nodes.HyperFlowSigmas().sigmas(model, int(steps))[0]
    print(f"{steps}-step sigmas: " + ", ".join(f"{float(x):.4f}" for x in tensor))
    return tensor


def info(patcher) -> None:
    """Report what the loader patched, so a silent fallback is visible instead of invisible."""
    model = patcher.model.diffusion_model
    schedule = getattr(model, "hyperflow_sigmas", None)
    low_rank = sum(1 for m in model.modules() if type(m).__name__ == "LowRankLinear")
    split_qkv = sum(1 for m in model.modules() if type(m).__name__ == "SplitQKV")
    two_time = hasattr(model, "endpoint_time_embedder")
    print(f"blocks={len(model.blocks)}  params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B "
          f"(low-rank linears={low_rank}, split-qkv={split_qkv})")
    print(f"two-time (t,r) conditioning: {'on' if two_time else 'OFF - this GGUF has no endpoint embedder'}")
    if schedule is not None:
        print(f"built-in schedule: {len(schedule) - 1} steps")


def main() -> None:
    """Smoke check: register the packs and load the transformer, nothing else."""
    load_node_packs()
    patcher = load_gguf()
    L.status()
    L.free(patcher)


if __name__ == "__main__":
    main()
