#!/usr/bin/env python3
"""Component loaders for MiniMax H3, on ComfyUI's core code.

One function per component, each exposing the parameters of the matching ComfyUI
node, so you can load exactly one thing at a time from an interactive session:

    load_diffusion_model()   -> UNETLoader        (unet_name, weight_dtype)
    load_text_encoder()      -> CLIPLoader        (clip_name, type, device)
    load_vae()               -> VAELoader         (vae_name)
    load_lora(model, ...)    -> LoraLoaderModelOnly (lora_name, strength_model)

init_engine() has to run first. It does the two assignments main.py does and an
importing script has to repeat: comfy.memory_management.aimdo_enabled = True and
CoreModelPatcher = ModelPatcherDynamic. Those are what switch on DynamicVRAM -
weights memory-mapped from disk as file slices and streamed into VRAM layer by
layer, which is both why ComfyUI loads a 32 GB model in seconds and why it can
free VRAM automatically when something else needs it. Without them the whole
transformer is loaded resident and the video VAE has nowhere to decode.

Run with the ComfyUI venv (comfy-kitchen, comfy-aimdo, sageattention live there):

    /mnt/data_2t_3/github_repos/ComfyUI/venv_py312_comfyui/bin/python
"""
from __future__ import annotations

import logging
import pathlib
import sys

COMFY = pathlib.Path("/mnt/data_2t_3/github_repos/ComfyUI")   # the engine install

# The interpreter this needs: comfy-kitchen, comfy-aimdo and the attention kernels are
# installed there and nowhere else. In VS Code, select it as the notebook kernel.
PYTHON = "/mnt/data_2t_3/github_repos/ComfyUI/venv_py312_comfyui/bin/python"

# Full paths, so there is nothing to guess. Every loader also accepts a ComfyUI-relative
# name (looked up in the model folders) or an absolute path like these.
# int8 is the default video VAE because it is the one that fits: the fp16 decoder's
# activations do not leave room next to the DynamicVRAM weight cache on a 32 GB card.
UNET = f"{COMFY}/models/diffusion_models/minimax_h3/MiniMax-H3-Ref2VA_int8_convrot.safetensors"
CLIP = f"{COMFY}/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE_VIDEO = f"{COMFY}/models/vae/minimax_h3/minimax_h3_video_vae_int8_convrot.safetensors"
VAE_VIDEO_FP16 = f"{COMFY}/models/vae/minimax_h3_video_vae_fp16.safetensors"
VAE_AUDIO = f"{COMFY}/models/vae/minimax_h3_audio_vae_fp32.safetensors"
LORA_TURBO = f"{COMFY}/models/loras/minimax_h3/minimax_h3_turbo_4step_ckpt850_pruned_comfyui.safetensors"

_M = None       # the imported comfy modules, filled by init_engine()


def resolve(folder: str, name: str) -> str:
    """An absolute path is used as-is; anything else is looked up in ComfyUI's folders."""
    if name.startswith("/"):
        if not pathlib.Path(name).is_file():
            raise FileNotFoundError(f"no model file at {name}")
        return name
    return _M["folder_paths"].get_full_path_or_raise(folder, name)


def init_engine(verbose: bool = True, vram_headroom_gb: float = 0.0, devices=None):
    """Bootstrap the comfy package and switch on DynamicVRAM. Returns the module map.

    Every entry point in this file calls it, so an interactive session can just start
    calling loaders - but calling it yourself first is how you control logging.

    devices: which GPUs aimdo arms. Any device you later hand to a loader's device=
    has to appear here, otherwise the staged load dies with
    'comfy-aimdo device N is not initialized'.
    """
    global _M
    if _M is not None:
        return _M

    if str(COMFY) not in sys.path:
        sys.path.insert(0, str(COMFY))

    # comfy.cli_args parses sys.argv at import time; park our own flags so argparse
    # doesn't die on the first unrecognised option.
    # Order matters twice over. comfy_aimdo.host_buffer snapshots control.lib at import
    # time (lib = control.lib), so the native library has to be loaded BEFORE any comfy
    # module that imports it - otherwise HostBuffer dies on a None lib. And comfy's own
    # cli_args parses sys.argv on import, so our flags have to be parked meanwhile.
    try:
        import comfy_aimdo.control as aimdo
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"comfy_aimdo is not installed in this interpreter ({sys.executable}).\n"
            f"  command line:  {PYTHON} your_script.py\n"
            f"  VS Code:       Ctrl+Shift+P -> 'Notebook: Select Notebook Kernel' ->\n"
            f"                 'Python Environments' or 'Existing' and pick\n"
            f"                 {PYTHON}\n"
            f"  Jupyter:       {PYTHON} -m ipykernel install --user --name comfyui"
        ) from exc

    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        from comfy.cli_args import args, enables_dynamic_vram
        aimdo_headroom = None if args.reserve_vram is None else int(args.reserve_vram * 1024 ** 3)
        aimdo_ready = aimdo.init(simple_vram_headroom=aimdo_headroom,
                                 nvml_pressure=not args.disable_nvml_pressure)
        # host_buffer / vram_buffer snapshot `lib = control.lib` at import time. If anything
        # pulled them in before init() ran - importing ComfyUI's `server` module does exactly
        # that - they keep lib=None and every staged load dies later inside
        # register_load_device with 'NoneType' has no attribute 'hostbuf_allocate'. Re-bind.
        for _name in ("host_buffer", "vram_buffer"):
            _mod = sys.modules.get(f"comfy_aimdo.{_name}")
            if _mod is not None and getattr(_mod, "lib", None) is None:
                _mod.lib = aimdo.lib
                if verbose:
                    logging.info(f"re-bound comfy_aimdo.{_name}.lib (it was imported before init)")
        import comfy.memory_management
        import comfy.model_management
        import comfy.model_patcher
        import comfy.sd
        import comfy.utils
        import folder_paths
    finally:
        sys.argv = argv

    if verbose:
        # ComfyUI's INFO lines are what tell you whether staging engaged:
        # "Model MiniMaxH3 prepared for dynamic VRAM loading. 32427MB Staged."
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        logging.getLogger().setLevel(logging.INFO)   # comfy already installed handlers

    mm = comfy.model_management
    dynamic, ok, gpu_ids, aimdo_ready = False, False, [], locals().get("aimdo_ready", False)
    if aimdo_ready and (args.enable_dynamic_vram or (enables_dynamic_vram() and mm.is_nvidia())):
        # init() (above, before the comfy imports) loaded the native library;
        # init_devices() arms each GPU. Calling it on its own silently returns False.
        gpu_ids = list(devices) if devices is not None else [d.index for d in mm.get_all_torch_devices()]
        # Two protocols across aimdo versions: bare device indices (0.4.9+) or
        # (index, headroom_bytes) pairs. One form returns False on the other's build
        # rather than raising, so try both instead of trusting a TypeError.
        ok = False
        for attempt in (gpu_ids,
                        [(i, int((args.vram_headroom + vram_headroom_gb) * 1024 ** 3)) for i in gpu_ids]):
            try:
                ok = bool(aimdo.init_devices(attempt))
            except TypeError:
                continue
            if ok:
                break
        if ok:
            aimdo.set_log_info()
            if vram_headroom_gb:
                # The VRAM the dynamic budget keeps free, in bytes. VBAR treats the whole
                # card as its weight cache; plain (non-VBAR) allocations - the video VAE's
                # activations - have to fit in what's left, so leaving 1-2 GB off the table
                # is what makes decode possible on a 32 GB card.
                aimdo.set_simple_vram_headroom(int(vram_headroom_gb * 1024 ** 3))
            # The two lines that make it real. Everything downstream - the mmap'd
            # safetensors reader, ModelPatcherDynamic, the prefetcher - keys off these.
            comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
            comfy.memory_management.aimdo_enabled = True
            dynamic = True
    if verbose:
        logging.info(f"aimdo.init={aimdo_ready} armed={ok} gpu_ids={gpu_ids} "
                     f"visible={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    if not dynamic:
        logging.warning("DynamicVRAM is OFF: models will load fully resident, and a 32 GB "
                        "transformer plus a video VAE will not fit on a 32 GB card. Usually "
                        "aimdo failed to arm one of the visible GPUs - on a multi-GPU box run "
                        "with CUDA_VISIBLE_DEVICES=<gpu>, or pass devices=[0].")

    _M = dict(comfy=comfy, folder_paths=folder_paths, args=args, dynamic_vram=dynamic)
    return _M


# ------------------------------------------------------------------ device & residency
# Components loaded with keep=True are tracked here so free_all() can leave them alone.
_LOADED, _PINNED = [], []


def _torch_device(device):
    """Coerce a device spec to a torch.device, or None for 'let ComfyUI decide'.

    Accepts None / "default" / "" (ComfyUI's choice), "cpu", an int index (2), ":2",
    "cuda:2", or an existing torch.device.
    """
    if device is None or isinstance(device, torch.device):
        return device
    s = str(device).strip()
    if s in ("", "default", "auto"):
        return None
    if s == "cpu":
        return torch.device("cpu")
    if s.isdigit():
        return torch.device(f"cuda:{s}")
    if s.startswith(":"):
        return torch.device(f"cuda{s}")
    return torch.device(s)


def _pin(patcher):
    """Remember a patcher so free_all() skips it.

    Deliberately does NOT touch offload_device. For a non-dynamic patcher ComfyUI's
    patch_weight_to_device() copies every LoRA-patched weight into self.backup on the
    OFFLOAD device; point that at the GPU and the whole model is duplicated in VRAM
    (19.5 GiB weights + 19.5 GiB backups) and the LoRA stage OOMs around block 42 of 50.
    Keeping the offload device on the CPU is what leaves the weights room to be resident.
    """
    if patcher is None or any(patcher is p for p in _PINNED):
        return
    _PINNED.append(patcher)


def resident():
    """What is pinned where - the answer to 'did the keep actually take effect?'."""
    for patcher in _PINNED:
        try:
            n = patcher.model_size() / 1024 ** 3        # staged size, what actually occupies memory
        except Exception:
            n = float("nan")                            # summing lazy params reports logical dtype, not staged bytes
        print(f"pinned: {type(patcher.model).__name__:<24} load={patcher.load_device} "
              f"offload={patcher.offload_device}  {n:.1f} GB")


def per_gpu():
    """Allocated GB per visible CUDA device - how you confirm a component landed on GPU 2."""
    if not torch.cuda.is_available():
        return
    print("  ".join(f"gpu{i}: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB"
                    for i in range(torch.cuda.device_count())))


# --------------------------------------------------------------------------- loaders
def load_diffusion_model(unet_name: str = UNET, weight_dtype: str = "default",
                         disable_dynamic: bool = False, device=None, keep: bool = False,
                         resident: bool = False):
    """UNETLoader. weight_dtype: default | fp8_e4m3fn | fp8_e4m3fn_fast | fp8_e5m2.

    Returns a ModelPatcher whose weights are staged on disk/RAM and streamed to the
    GPU on demand - the same path ComfyUI uses, so the load time is the same.

    resident=True turns that off: the transformer is loaded fully onto `device` once
    (disable_dynamic, plain CoreModelPatcher) and pinned there, so nothing streams per
    step and nothing re-stages after a free. It costs VRAM outright - the pruned fl2va
    checkpoint is ~21 GB of weights, so on a 32 GB card the conditioner and the video VAE
    have to live somewhere else. resident=False keeps the streaming path.
    """
    m = init_engine()
    options = {}
    if weight_dtype == "fp8_e4m3fn":
        options["dtype"] = torch_dtype("float8_e4m3fn")
    elif weight_dtype == "fp8_e4m3fn_fast":
        options["dtype"] = torch_dtype("float8_e4m3fn")
        options["fp8_optimizations"] = True
    elif weight_dtype == "fp8_e5m2":
        options["dtype"] = torch_dtype("float8_e5m2")
    d = _torch_device(device)
    if d is not None:
        options["load_device"] = options["offload_device"] = d
    if resident:
        disable_dynamic = True          # no ModelPatcherDynamic: weights stay put
        keep = True                     # free_all() must not drop them - but see _pin():
                                        # offload_device stays on the CPU, that is not a bug
    path = resolve("diffusion_models", unet_name)
    model = m["comfy"].sd.load_diffusion_model(path, model_options=options,
                                               disable_dynamic=disable_dynamic)
    print(f"unet weights: {'resident in VRAM' if resident else 'streamed (DynamicVRAM)'}")
    _LOADED.append(model)
    if keep:
        _pin(model)
    print(f"unet: {unet_name}  vram {vram_gb():.1f} GB allocated")
    return model


def load_text_encoder(clip_name: str = CLIP, type: str = "minimax", device=None, keep: bool = False):
    """CLIPLoader. type is the CLIPType name (minimax for H3's Qwen3-VL conditioner).

    device: None/"default" (ComfyUI's pick), "cpu", or a specific GPU - 2, ":2", "cuda:2".
    keep=True pins it: offload_device is set equal to load_device and free_all() skips it.
    Measured on this box it saves nothing on its own - within one process ComfyUI keeps the
    conditioner staged, and after a real free_all() the re-stage from the OS page cache costs
    about 1 s (encode 3.0 s either way). Use it when you want a component to survive an
    explicit free, not as a speed dial.
    """
    m = init_engine()
    clip_type = getattr(m["comfy"].sd.CLIPType, type.upper(), m["comfy"].sd.CLIPType.STABLE_DIFFUSION)
    options = {}
    d = _torch_device(device)
    if d is not None:
        options["load_device"] = options["offload_device"] = d
    path = resolve("text_encoders", clip_name)
    clip = m["comfy"].sd.load_clip(ckpt_paths=[path],
                                   embedding_directory=m["folder_paths"].get_folder_paths("embeddings"),
                                   clip_type=clip_type, model_options=options)
    _LOADED.append(getattr(clip, "patcher", clip))
    if keep:
        _pin(getattr(clip, "patcher", clip))
    print(f"clip: {clip_name}  vram {vram_gb():.1f} GB allocated")
    return clip


def load_vae(vae_name: str = VAE_VIDEO, device=None, keep: bool = False):
    """VAELoader - the same call for the video VAE and the audio VAE.

    device goes straight into comfy.sd.VAE(device=...), which sets vae.device (every decode
    moves its samples there) and the patcher's load_device. keep=True pins it resident.
    """
    m = init_engine()
    path = resolve("vae", vae_name)
    sd, metadata = m["comfy"].utils.load_torch_file(path, return_metadata=True)
    vae = m["comfy"].sd.VAE(sd=sd, device=_torch_device(device), metadata=metadata)
    vae.throw_exception_if_invalid()
    _LOADED.append(getattr(vae, "patcher", None))
    if keep:
        _pin(getattr(vae, "patcher", None))
    print(f"vae: {vae_name}  vram {vram_gb():.1f} GB allocated")
    return vae


def load_lora(model, lora_name: str, strength_model: float = 1.0):
    """LoraLoaderModelOnly. Returns a new patcher; the old one is still valid."""
    m = init_engine()
    path = resolve("loras", lora_name)
    sd, metadata = m["comfy"].utils.load_torch_file(path, safe_load=True, return_metadata=True)
    patched, _ = m["comfy"].sd.load_lora_for_models(model, None, sd, strength_model, 0,
                                                    lora_metadata=metadata)
    print(f"lora: {lora_name} x{strength_model}")
    return patched


# ---------------------------------------------------------------- model patches
def sigma_shift(model, shift_video: float = 12.0, shift_audio: float = 3.0):
    """MiniMaxH3SigmaShift - retarget the flow schedule the sampler walks.

    12.0 / 3.0 are what the released checkpoint was trained with; the FastH3 template
    patches 10.0 / 3.0 for its 8-step distilled run. shift_video drives the sigma grid
    (sigma = s*t / (1 + (s-1)*t)); both values also reach the DiT, which derives the
    audio schedule and the per-stream timestep labels from them. Not a quality dial -
    only set it when a LoRA or template says the shift differs.
    """
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
    out = MiniMaxH3SigmaShift.execute(model=model, shift_video=shift_video, shift_audio=shift_audio)
    return out.result[0] if hasattr(out, "result") else out[0]


def attention_backend(model, backend: str = "comfy kitchen attention"):
    """ModelAttentionBackend: 'comfy kitchen attention' (INT8) or 'pytorch attention'."""
    from comfy_extras.nodes_model_advanced import ModelAttentionBackend
    out = ModelAttentionBackend.execute(model=model, attention=backend)
    return out.result[0] if hasattr(out, "result") else out[0]


def sage_attention_patch(model, kj_module: str = ""):
    """MiniMaxH3MemoryEfficientSageAttentionPatch (KJ nodes): int8-QK / fp8-V attention,
    which lowers peak VRAM on long sequences. Needs sageattention >= 2.2, which PyPI does
    not carry - build it from source (recipe in requirements.txt). Raises ImportError with
    the reason so a caller falls back deliberately rather than silently."""
    import importlib.util
    try:
        from sageattention.core import get_cuda_arch_versions
        archs = get_cuda_arch_versions()
    except Exception as exc:
        raise ImportError("sageattention >= 2.2 is required for the KJ memory-efficient "
                          "attention patch; build it from source (see requirements.txt)") from exc
    path = pathlib.Path(kj_module or f"{COMFY}/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py")
    if not path.is_file():
        raise FileNotFoundError(f"KJ node module not found at {path}")
    if str(COMFY) not in sys.path:
        sys.path.insert(0, str(COMFY))            # the module does `from server import PromptServer`
    import server
    if not hasattr(server.PromptServer, "instance"):
        server.PromptServer.instance = None       # in-process: there is no prompt server
    spec = importlib.util.spec_from_file_location("kj_ltxv_nodes", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    out = mod.MiniMaxH3MemoryEfficientSageAttentionPatch.execute(model=model)
    print("sage attention patch:", "/".join(archs))
    return out.result[0] if hasattr(out, "result") else out[0]


def sparse_attention(model, method: str = "sol-attn", tau: float = 1.3, keep_percent: float = 10.0,
                     start_percent: float = 0.2, end_percent: float = 1.0, dense_blocks: str = "",
                     min_tokens: int = 12288, sink_conditioning: str = "exact_kv_and_rows"):
    """BlockSparseAttention. sol-attn works on any weights (tau trades speed for detail);
    vsa needs FastH3 weights; sla needs an SLA-style LoRA."""
    from comfy_extras.nodes_sparse_attention import apply_block_sparse_attention, parse_block_list
    vsa = method == "vsa"
    return apply_block_sparse_attention(
        model, tau=float(tau),
        topk_ratio=0.0 if method == "sol-attn" else float(keep_percent) / 100.0,
        vsa=vsa, start_percent=start_percent, end_percent=end_percent, min_tokens=min_tokens,
        dense_blocks=parse_block_list(dense_blocks), sink_conditioning=sink_conditioning,
        extra_tokens=0 if vsa else 256, verbose=False)


# ---------------------------------------------------------------------- memory
def sol_attn_t8(model, tau: float = 1.3, start_percent: float = 0.2, end_percent: float = 0.9,
                min_tokens: int = 12288, sink_conditioning: str = "exact_kv_and_rows",
                morton: bool = False, morton_curve: str = "2d_frame", dense_blocks: str = "",
                centroid_tail: bool = True, reuse_qkv_memory: bool = False,
                routed_cap_percent: int = 0, tau_profile=None, verbose: bool = False,
                t8_dir: str = ""):
    """SolAttnMiniMax from the comfyui-minimax-h3-audio-T8 pack - Triton kernels built for
    SM120. Same job as sparse_attention()'s ComfyUI-native BlockSparseAttention: pick one or
    the other, never both. Measured on this box at 768x1344 x 226: 10.06 s/it here vs 10.15
    s/it native, so treat it as a knob for tau/end_percent coverage, not as a speed win.
    Needs the T8 pack installed and comfy_kitchen exposing sol_attn."""
    import importlib.util
    root = pathlib.Path(t8_dir or f"{COMFY}/custom_nodes/comfyui-minimax-h3-audio-T8")
    path = root / "sol_attn_minimax_v2.py"
    if not path.is_file():
        raise FileNotFoundError(f"T8 sol-attn module not found at {path}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))            # the module lives at the pack root
    spec = importlib.util.spec_from_file_location("sol_attn_minimax_v2", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    out = mod.SolAttnMiniMax.execute(
        model, tau=float(tau), start_percent=float(start_percent), end_percent=float(end_percent),
        min_tokens=int(min_tokens), sink_conditioning=sink_conditioning, morton=morton,
        morton_curve=morton_curve, dense_blocks=dense_blocks, verbose=verbose,
        tau_profile=tau_profile, routed_cap_percent=routed_cap_percent,
        centroid_tail=centroid_tail, reuse_qkv_memory=reuse_qkv_memory)
    print(f"sol-attn (T8 Triton): tau={tau} steps {start_percent}-{end_percent} min_tokens={min_tokens}")
    return out.result[0] if hasattr(out, "result") else out[0]


def vram_gb() -> float:
    import torch
    return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def free(*objs, empty_cache: bool = True) -> None:
    """Release specific components (pass the patcher / clip / vae you loaded)."""
    init_engine(verbose=False)
    mm = _M["comfy"].model_management
    for obj in objs:
        if obj is None:
            continue
        patcher = obj.patcher if hasattr(obj, "patcher") else obj
        try:
            mm.unload_model_and_clones(patcher)
        except Exception:
            pass
    if empty_cache:
        mm.soft_empty_cache()
    print(f"freed {len(objs)} component(s)  vram {vram_gb():.1f} GB allocated")


def free_all(empty_cache: bool = True, keep_pinned: bool = True) -> None:
    """Release loaded models. Each one re-stages from disk on next use.

    keep_pinned=True (default) frees only what this module loaded and did not pin, so a
    text encoder or VAE loaded with keep=True survives. Pass False to get ComfyUI's own
    unload-everything behaviour.
    """
    init_engine(verbose=False)
    mm = _M["comfy"].model_management
    if not keep_pinned or not _LOADED:
        mm.unload_all_models()
    else:
        for patcher in list(_LOADED):
            if any(patcher is p for p in _PINNED):
                continue
            try:
                mm.unload_model_and_clones(patcher)
            except Exception:
                pass
    if empty_cache:
        mm.soft_empty_cache()
    print(f"freed  vram {vram_gb():.1f} GB allocated  pinned kept: {len(_PINNED)}")


def status() -> None:
    init_engine(verbose=False)
    mm = _M["comfy"].model_management
    print(f"dynamic_vram={_M['dynamic_vram']}  aimdo={_M['comfy'].memory_management.aimdo_enabled}  "
          f"allocated={vram_gb():.2f} GB  loaded={[type(m.model).__name__ for m in mm.current_loaded_models]}")


def torch_dtype(name: str):
    import torch
    return getattr(torch, name)


import torch  # noqa: E402  (after init_engine's own import; only used by helpers)
