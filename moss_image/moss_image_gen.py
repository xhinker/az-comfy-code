#!/usr/bin/env python3
"""MOSS-Image-SFT in diffusers, built entirely from local files.

diffusers has no MOSS pipeline. Checked against the installed 0.41.0: no
MossImagePipeline, no MOSS transformer, no module - and the model card itself
loads it with custom_pipeline=, which is Hub-hosted code, not core. So this
script builds the pipeline itself from the official model definition, vendored
into ./moss_def/ (code and config only - no weights).

Everything heavy comes off local disk:

  transformer   ComfyUI models/diffusion_models/moss_image_sft/moss_image_sft_bf16.safetensors
  text encoder  ComfyUI models/text_encoders/moss_image_sft/qwen3.5_9b_bf16.safetensors
                (converted from the int8_convrot original by convert_text_encoder.py,
                 which uses ComfyUI's own dequantisation kernels)
  vae           ComfyUI models/vae/moss_image_sft/flux2-vae.safetensors
  tokenizer     ComfyUI comfy/text_encoders/qwen35_tokenizer (vocab + merges)
                with the official chat template from ./moss_def/tokenizer/

Nothing in here touches the network.

Memory modes: see README.md. Upstream sampling defaults are used unless
overridden - euler flow matching, 50 steps, CFG 3, shift 5, negative-branch
layer skip 22, empty negative, dimensions multiples of 16.
"""

import argparse
import contextlib
import gc
import importlib.util
import inspect
import json
import logging
import shutil
import time
from pathlib import Path

import torch
from torch import nn

import logging


class _DropTorchaoProbeWarning(logging.Filter):
    """diffusers 0.41 probes torchao for tensor classes that torchao 0.18 moved out of
    `torchao.dtypes` (NF4Tensor, torchao.dtypes.uintx), logs a warning about it, and
    buries the actual ImportError at debug level - which is why the message appears
    with no explanation even though torchao is installed. Nothing here loads a
    torchao-serialised checkpoint: the weights are plain bf16 safetensors, and the
    int8 text encoder is dequantised by ComfyUI's kernels before diffusers sees it.
    Scoped to that one message, so a genuine torchao problem still gets through."""

    def filter(self, record):
        return "Unable to import `torchao` Tensor objects" not in record.getMessage()


# create the logger before diffusers does, so the filter is already attached when it warns
logging.getLogger("diffusers.quantizers.torchao.torchao_quantizer").addFilter(
    _DropTorchaoProbeWarning())

HERE = Path(__file__).resolve().parent
DEF_DIR = HERE / "moss_def"

COMFY = Path("/mnt/data_2t_3/github_repos/ComfyUI")
DIT_PATH = COMFY / "models/diffusion_models/moss_image_sft/moss_image_sft_bf16.safetensors"
TE_PATH = COMFY / "models/text_encoders/moss_image_sft/qwen3.5_9b_bf16.safetensors"
VAE_PATH = COMFY / "models/vae/moss_image_sft/flux2-vae.safetensors"
TOKENIZER_ASSETS = COMFY / "comfy/text_encoders/qwen35_tokenizer"
TOKENIZER_DIR = DEF_DIR / "tokenizer"
DISK_CACHE = HERE / ".offload_cache"
TURBO_LORA = COMFY / "models/loras/moss_image_sft/lora.safetensors"

UPSTREAM_FREE_VRAM_GIB = 40.0  # OpenMOSS inference_runtime.py: below this, offload
RESIDENT_MIN_GIB = 16.0     # enough for the 10 GiB DiT to stay resident plus activations

# DiT weight precision. fp32 is the reference. fp16 is 2x faster and half the memory and
# stays close to it (5.4/255 mean pixel difference at a fixed seed). bf16 is equally fast
# but wrong for this model: 41.4/255 mean difference and 70% of pixels changed - a
# different picture, faces included. It is a mantissa-width problem (fp16 has 11 bits,
# bf16 has 8), not a half-vs-full-precision one. The authors' loader and ComfyUI both ask
# for bf16; ComfyUI gets away with it because its own layer implementations cast
# activations to fp32 where this implementation does not.
DIT_DTYPE = "fp16"   # confirmed by eye against the fp32 reference; "fp32" if in doubt


def log(msg):
    print(f"[moss] {msg}", flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def free_vram_gib(device="cuda"):
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info(device)
    return free / 1024 ** 3


@contextlib.contextmanager
def skipping_random_init():
    """Construct 14.6B parameters only to overwrite them, and initialising them is
    the whole cost: measured 70.5 s of the 76 s build, against 2.4 s to actually
    load the weights and 0.0 s to read the files (safetensors maps them lazily).
    ComfyUI is fast for the same reason - it never fills weights it is about to
    replace. Deterministic fills are left alone, so things like the RMSNorm ones_
    still happen; anything the checkpoint does not supply is zeroed by load_into."""
    saved = {}
    for name in ("uniform_", "normal_", "kaiming_uniform_", "kaiming_normal_",
                 "xavier_uniform_", "xavier_normal_", "trunc_normal_", "dirac_",
                 "orthogonal_", "sparse_", "calculate_gain"):
        saved[name] = getattr(torch.nn.init, name)
        setattr(torch.nn.init, name, lambda tensor, *a, **k: tensor)
    tensor_uniform, tensor_normal = torch.Tensor.uniform_, torch.Tensor.normal_
    torch.Tensor.uniform_ = lambda self, *a, **k: self
    torch.Tensor.normal_ = lambda self, *a, **k: self
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(torch.nn.init, name, fn)
        torch.Tensor.uniform_, torch.Tensor.normal_ = tensor_uniform, tensor_normal


class _LoRAPath(nn.Module):
    """Low-rank term, kept unmerged and computed in fp32. The 8-step DMD card is
    explicit that the distillation delta is smaller than bf16's rounding step, so
    merging it into bf16 weights discards most of the distillation."""

    def __init__(self, down, up, scale):
        super().__init__()
        self.register_buffer("down", down.to(torch.float32))
        self.register_buffer("up", up.to(torch.float32))
        self.scale = float(scale)

    def forward(self, x):
        y = torch.nn.functional.linear(x.float(), self.down)
        return torch.nn.functional.linear(y, self.up) * self.scale


def _lora_hook(holder):
    def hook(module, inputs, output):
        return output + holder(inputs[0]).to(output.dtype)
    return hook


def remove_lora(model):
    """Undo apply_lora. Adapters are forward hooks rather than module swaps, so the
    model's structure and any offload hooks on it are untouched and this is safe to
    call repeatedly."""
    for handle in getattr(model, "_moss_lora_hooks", ()):
        handle.remove()
    model._moss_lora_hooks = []
    if hasattr(model, "moss_lora"):
        delattr(model, "moss_lora")
    for name, orig in getattr(model, "_moss_loha_orig", {}).items():
        try:
            module = model.get_submodule(name)
            with torch.no_grad():
                module.weight.copy_(orig.to(module.weight.device, module.weight.dtype))
        except Exception:
            pass
    model._moss_loha_orig = {}
    return model


def dtype_of(module):
    return next(module.parameters()).dtype


LORA_DOWNS = (".lora_down.weight", ".lora_A.weight", ".lora_A.default.weight")
LORA_UPS = (".lora_up.weight", ".lora_B.weight", ".lora_B.default.weight")


def split_lora_key(key):
    """Return (base, role) for the naming dialects in the wild, or (None, None).

        kohya / ComfyUI   <base>.lora_down.weight   <base>.lora_up.weight   <base>.alpha
        PEFT / ai-toolkit <base>.lora_A.weight      <base>.lora_B.weight
        diffusers PEFT    <base>.lora_A.default.weight (same tensors, extra segment)

    A maps to down (rank x in) and B to up (out x rank), which is what the shapes say:
    a rank-1 adapter on a 8192->3072 Linear ships lora_A [1, 8192] and lora_B [3072, 1]."""
    for suffix in LORA_DOWNS:
        if key.endswith(suffix):
            return key[:-len(suffix)], "down"
    for suffix in LORA_UPS:
        if key.endswith(suffix):
            return key[:-len(suffix)], "up"
    if key.endswith(".alpha"):
        return key[:-len(".alpha")], "alpha"
    return None, None


def _read_lora(path):
    """Load one file into (downs, ups, alphas) regardless of naming dialect."""
    from safetensors.torch import load_file
    state = load_file(str(path))
    downs, ups, alphas = {}, {}, {}
    for key, tensor in state.items():
        base, role = split_lora_key(key)
        if base is None:
            continue
        if role == "alpha":
            alphas[base] = float(tensor)
        elif role == "down":
            downs[base] = tensor
        else:
            ups[base] = tensor
    return downs, ups, alphas, state



LOHA_MARKER = ".hada_w1_a"


def _read_loha(path):
    """Load a LoHA (Hadamard LoRA) file into {base: (w1a, w1b, w2a, w2b, alpha)}.

    LoHA delta is the element-wise (Hadamard) product of two low-rank rebuilds, NOT their
    sum:  delta = (alpha/rank) * ((w1a@w1b) * (w2a@w2b)).  Reference: ComfyUI
    comfy/weight_adapter/loha.py HadaWeight.forward.  Tucker/CP (hada_t1/hada_t2) not supported."""
    from safetensors.torch import load_file
    state = load_file(str(path))
    out = {}
    for key in state:
        if key.endswith(LOHA_MARKER):
            base = key[:-len(LOHA_MARKER)]
            w1a = state[base + ".hada_w1_a"]; w1b = state[base + ".hada_w1_b"]
            w2a = state[base + ".hada_w2_a"]; w2b = state[base + ".hada_w2_b"]
            alpha = float(state[base + ".alpha"]) if (base + ".alpha") in state else float(w1a.shape[1])
            out[base] = (w1a, w1b, w2a, w2b, alpha)
    return out


def _loha_delta(w1a, w1b, w2a, w2b, alpha, strength):
    """Full-rank LoHA delta for one module, fp32: (strength*alpha/rank)*((w1a@w1b)*(w2a@w2b))."""
    rank = w1b.shape[0]
    m1 = w1a.to(torch.float32) @ w1b.to(torch.float32)
    m2 = w2a.to(torch.float32) @ w2b.to(torch.float32)
    return (strength * alpha / rank) * (m1 * m2)


def _loha_targets(model, name, delta):
    """Map a LoHA key (possibly fused qkv) to [(submodule_name, delta_slice), ...]."""
    if name.endswith(".qkv"):
        stem = name[:-len(".qkv")]
        names = [stem + ".q", stem + ".k", stem + ".v"]
        try:
            for n in names:
                model.get_submodule(n)
        except AttributeError:
            return None
        t = delta.shape[0] // 3
        return [(names[0], delta[:t]), (names[1], delta[t:2*t]), (names[2], delta[2*t:])]
    try:
        model.get_submodule(name)
    except AttributeError:
        return None
    return [(name, delta)]


def _apply_loha(model, path, strength, slot):
    """Add a LoHA adapter full-rank delta to the weights in place (as ComfyUI loha.py does).
    Fused qkv delta is split across the model separate q/k/v Linears. Originals stashed on
    CPU so remove_lora restores them exactly."""
    loha = _read_loha(path)
    if not hasattr(model, "_moss_loha_orig"):
        model._moss_loha_orig = {}
    matched, skipped = 0, []
    for base, (w1a, w1b, w2a, w2b, alpha) in loha.items():
        name = base[len("diffusion_model."):] if base.startswith("diffusion_model.") else base
        delta = _loha_delta(w1a, w1b, w2a, w2b, alpha, strength)
        targets = _loha_targets(model, name, delta)
        if targets is None:
            skipped.append(name); continue
        for tname, tdelta in targets:
            module = model.get_submodule(tname)
            if not hasattr(module, "weight") or tuple(module.weight.shape) != tuple(tdelta.shape):
                skipped.append(tname); continue
            if tname not in model._moss_loha_orig:
                model._moss_loha_orig[tname] = module.weight.detach().cpu().clone()
            with torch.no_grad():
                module.weight.add_(tdelta.to(module.weight.device, module.weight.dtype))
            matched += 1
    if matched == 0:
        raise SystemExit(f"{path}: no LoHA adapter matched a module in {type(model).__name__}. "
                         f"Found {len(loha)} factors; first {list(loha)[:3]}.")
    log(f"lora[{slot}]: LoHA {matched} modules at strength {strength} ({Path(path).name})"
        + (f"; {len(skipped)} unmatched e.g. {skipped[:2]}" if skipped else ""))


def apply_lora(model, path, strength=1.0):
    """Attach one adapter, or several at once. Each contributes
    y = Wx + (alpha / rank) * up(down(x)); files with no .alpha tensors - which is every
    PEFT/ai-toolkit export - fall back to alpha = rank, i.e. scale 1.0, the convention
    ComfyUI uses, so a LoRA that works there works here.

    Stacking: pass a list of paths with a matching list of strengths, or one strength for
    all of them.

        apply_lora(pipe.transformer, [moss.TURBO_LORA, slider], [1.0, 0.4])

    Every adapter adds its own term to the same module output, so the combined effect is
    the sum of the individual deltas and the order in the list does not matter. Each file
    still needs to hit at least one module of this network, and each is reported
    separately so a LoRA that silently matches nothing stays visible.

    Calling this replaces everything attached by the previous call; path=None or "" (or a
    list of them) removes all adapters."""
    remove_lora(model)
    paths = list(path) if isinstance(path, (list, tuple)) else [path]
    strengths = ([float(x) for x in strength] if isinstance(strength, (list, tuple))
                 else [float(strength)] * len(paths))
    if len(paths) != len(strengths):
        raise SystemExit(f"lora: got {len(paths)} path(s) but {len(strengths)} strength(s)")

    holders, hooks = nn.ModuleDict(), []
    for slot, (one_path, one_strength) in enumerate(zip(paths, strengths)):
        if one_path is None or str(one_path).strip() == "":
            log(f"lora[{slot}]: skipped (empty path)")
            continue
        if not Path(one_path).is_file():
            raise SystemExit(f"lora: not a file: {one_path}")
        from safetensors.torch import load_file as _lf
        if any(k.endswith(LOHA_MARKER) for k in _lf(str(one_path))):
            _apply_loha(model, one_path, one_strength, slot)
            continue
        downs, ups, alphas, state = _read_lora(one_path)
        matched, skipped = 0, []
        for base, down in sorted(downs.items()):
            up = ups.get(base)
            if up is None:
                skipped.append(base); continue
            name = base[len("diffusion_model."):] if base.startswith("diffusion_model.") else base
            try:
                target = model.get_submodule(name)
            except AttributeError:
                skipped.append(name); continue
            rank = down.shape[0]
            scale = one_strength * alphas.get(base, float(rank)) / rank
            holder = _LoRAPath(down, up, scale)
            # Two adapters can target the same Linear: forward hooks chain, each seeing
            # what the previous one returned, so their deltas simply add.
            holders.add_module(f"l{slot}_{name.replace('.', '__')}", holder)
            hooks.append(target.register_forward_hook(_lora_hook(holder)))
            matched += 1
        if matched == 0:
            raise SystemExit(
                f"{one_path}: no adapter matched a module in {type(model).__name__}. "
                f"Found {len(downs)} down / {len(ups)} up factors; first keys were "
                f"{list(state)[:3]}. If the file targets a different network there is "
                f"nothing here to attach it to.")
        log(f"lora[{slot}]: {matched} adapters at strength {one_strength} "
            f"({Path(one_path).name})"
            + (f"; {len(skipped)} unmatched e.g. {skipped[:2]}" if skipped else ""))

    loha_n = len(getattr(model, "_moss_loha_orig", {}))
    if not hooks and loha_n == 0:
        log("lora: none loaded")
        return model
    # registered on the model so offload moves the fp32 factors along with the weights
    model.add_module("moss_lora", holders)
    model._moss_lora_hooks = hooks
    if hooks:
        if dtype_of(model) == torch.bfloat16:
            log("lora: WARNING the model is bf16. Measured on this adapter, the median "
                "injection is 0.10% of the activation it edits while bf16 spacing is 0.195%, "
                "so 145 of 252 adapters round to zero every pass and the distillation is "
                "largely lost - visible as badly wrong faces at 8 steps. Use fp32 or fp16.")
        shared = len(hooks) - len(holders)
        log(f"lora: {len(hooks)} injection points live from {len(holders)} factors"
            + (f" ({shared} modules carry more than one adapter, their deltas sum)" if shared else ""))
    return model


def load_definition():
    """Import the vendored official model definition (transformer, scheduler, pipeline)."""
    path = DEF_DIR / "pipeline_moss_image.py"
    spec = importlib.util.spec_from_file_location("pipeline_moss_image", path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["pipeline_moss_image"] = module
    spec.loader.exec_module(module)
    return module


def ensure_tokenizer():
    """ComfyUI ships the Qwen3.5 vocabulary *and* its added-token registry; the
    official MOSS repo ships the chat template. Neither alone works: those chat and
    thinking tokens are added tokens, not vocab entries, so with the official
    tokenizer_config.json they have no ids and the template gets BPE-split into
    literal characters - 24 junk tokens where ComfyUI gets 12 real ones. That feeds
    the text encoder noise instead of a prompt, which yields sharp, well-exposed
    images of nothing in particular at any step count."""
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("vocab.json", "merges.txt", "tokenizer_config.json"):
        # ComfyUI's config is the authoritative one: it carries added_tokens_decoder
        shutil.copy2(TOKENIZER_ASSETS / name, TOKENIZER_DIR / name)
    template = (TOKENIZER_DIR / "chat_template.jinja").read_text()

    import re
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    tokenizer.chat_template = template

    probe = tokenizer.apply_chat_template([{"role": "user", "content": "beautiful girl"}],
                                          tokenize=False, add_generation_prompt=True,
                                          enable_thinking=True)
    if not isinstance(probe, str) or "beautiful girl" not in probe:
        raise SystemExit("tokenizer chat template did not render the prompt")
    # build the token strings without writing them out, and require each to encode
    # to exactly one id - an unregistered one fragments into characters
    left, right = "<", ">"
    control = sorted(set(re.findall(left + r"\|[a-zA-Z0-9_]+\|" + right, probe)))
    unregistered = [t for t in control
                    if len(tokenizer(t, add_special_tokens=False)["input_ids"]) != 1]
    if unregistered:
        raise SystemExit(f"tokenizer: {len(unregistered)} of {len(control)} chat tokens have no id "
                         f"in {TOKENIZER_DIR}/tokenizer_config.json; copy the one ComfyUI ships at "
                         f"{TOKENIZER_ASSETS} (it carries added_tokens_decoder)")
    log(f"tokenizer ready: {len(tokenizer)} vocab, {len(control)} chat tokens registered, "
        f"chat template with enable_thinking OK")
    return tokenizer


def split_fused_qkv(state):
    """ComfyUI stores each block's q/k/v as one fused `proj.qkv` matrix; the
    official class keeps three separate Linear layers. Splitting the rows in
    order is exactly what `qkv(x).chunk(3, -1)` does at runtime."""
    out = {}
    for key, value in state.items():
        head, sep, tail = key.rpartition(".")
        if sep and head.endswith(".proj.qkv") and tail in ("weight", "bias"):
            base = head[: -len("qkv")]                      # keep the trailing dot
            for name, part in zip(("q", "k", "v"), value.chunk(3, dim=0)):
                out[f"{base}{name}.{tail}"] = part.contiguous()
        else:
            out[key] = value
    return out


def to_base_model_keys(state):
    """The checkpoint is the full Qwen3_5 conditional-generation model; the
    pipeline only calls the base model for `last_hidden_state`. Drop the `model.`
    prefix that wrapper adds, and drop the two heads it owns: `lm_head` produces
    logits and `mtp` is a multi-token-prediction head, neither is reachable from
    encode_prompt."""
    out = {}
    for key, value in state.items():
        if key.startswith("lm_head.") or key.startswith("mtp."):
            continue
        out[key[len("model."):] if key.startswith("model.") else key] = value
    return out


def load_into(model, path, allow_missing_prefixes=(), transform=None):
    """Load a ComfyUI safetensors file into a freshly built model and prove the
    key names line up before going any further."""
    from safetensors.torch import load_file
    state = load_file(str(path))
    if transform is not None:
        state = transform(state)
    result = model.load_state_dict(state, strict=False)
    del state
    gc.collect()
    model_keys = set(model.state_dict())
    loaded = len(model_keys) - len(result.missing_keys)
    unexplained_missing = [k for k in result.missing_keys
                           if not any(k.startswith(p) for p in allow_missing_prefixes)]
    log(f"{type(model).__name__}: {loaded}/{len(model_keys)} tensors loaded from {Path(path).name}"
        + (f", {len(result.missing_keys)} left at init ({allow_missing_prefixes})" if result.missing_keys else ""))
    if result.unexpected_keys:
        raise SystemExit(f"{path}: {len(result.unexpected_keys)} tensors in the file are not in the "
                         f"model, e.g. {sorted(result.unexpected_keys)[:4]}")
    if unexplained_missing:
        raise SystemExit(f"{path}: model tensors with no weights, e.g. {sorted(unexplained_missing)[:4]}")
    if result.missing_keys:
        # random init is skipped during construction, so anything left over would
        # otherwise hold uninitialised memory rather than zeros
        tensors = model.state_dict()
        for key in result.missing_keys:
            if key in tensors:
                tensors[key].zero_()
        log(f"{type(model).__name__}: zeroed {len(result.missing_keys)} tensors the file does not carry")
    return model.eval()


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
DTYPE_NAMES = {v: k for k, v in DTYPES.items()}


def resolve_dtype(dtype, what="dtype"):
    """Accept "bf16" / "fp16" / "fp32", or a torch.dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return DTYPES[str(dtype).lower()]
    except KeyError:
        raise SystemExit(f"{what} must be one of {sorted(DTYPES)}, got {dtype!r}")


def cast_params(model, dtype):
    """Cast weights only. Buffers stay fp32 - RoPE bands are derived constants and the
    model rebuilds them in fp32 for exactly that reason."""
    if dtype != torch.float32:
        for param in model.parameters():
            param.data = param.data.to(dtype)
    return model


def build_transformer(module, dtype=None):
    cfg = {k: v for k, v in read_json(DEF_DIR / "transformer/config.json").items() if not k.startswith("_")}
    accepted = set(inspect.signature(module.MOSSImageTransformer.__init__).parameters)
    kwargs = {k: v for k, v in cfg.items() if k in accepted}
    with skipping_random_init():
        model = module.MOSSImageTransformer(**kwargs)
    dtype = resolve_dtype(DIT_DTYPE if dtype is None else dtype, "transformer dtype")
    model = cast_params(load_into(model, DIT_PATH, transform=split_fused_qkv), dtype)
    log(f"transformer weights: {DTYPE_NAMES[dtype]}"
        + (" - the DMD LoRA cannot survive bf16, see apply_lora; use fp16 with adapters"
           if dtype == torch.bfloat16 else ""))
    return model


def build_text_encoder(dtype=torch.bfloat16):
    dtype = resolve_dtype(dtype, "text encoder dtype")
    if dtype == torch.float32:
        # 9B parameters in fp32 is 36.6 GiB, more than the card can stage even with the
        # DiT offloaded, so fp32 here means "broken" rather than "accurate".
        log("text encoder: fp32 would need 36.6 GiB and cannot be staged on this card; "
            "using fp16")
        dtype = torch.float16
    from transformers import AutoConfig, AutoModel
    config = AutoConfig.from_pretrained(str(DEF_DIR / "text_encoder"))
    with torch.device("cpu"), skipping_random_init():
        model = AutoModel.from_config(config, dtype=dtype)
    # the checkpoint is text-only; the vision tower of Qwen3_5Model stays at init
    # and is never called for a text prompt.
    return cast_params(load_into(model, TE_PATH, transform=to_base_model_keys,
                                 allow_missing_prefixes=("visual",)), dtype)


def safetensors_shape(path, key):
    """Read one tensor's shape out of a safetensors header without loading it."""
    import struct
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    return tuple(header[key]["shape"])


def ldm_to_diffusers_vae_keys(state, num_blocks=4):
    """ComfyUI ships the FLUX.2 VAE in original LDM/CompVis naming while
    AutoencoderKLFlux2 uses diffusers module names. Both sides carry 251 tensors,
    so the mapping is 1:1 and load_into() proves it. Three things differ:

      * module names (down.N.block.M -> down_blocks.N.resnets.M, and so on)
      * the decoder counts its up-blocks from the image end, diffusers from the
        latent end, so those indices are reversed (verified against the
        nin_shortcut shapes, which only line up that way round)
      * LDM attention projections are 1x1 convolutions; diffusers uses Linear
    """
    import re

    def up_block(match):
        return f"decoder.up_blocks.{num_blocks - 1 - int(match.group(1))}."

    out = {}
    for key, value in state.items():
        key = re.sub(r"^encoder\.down\.(\d+)\.block\.(\d+)\.",
                     r"encoder.down_blocks.\1.resnets.\2.", key)
        key = re.sub(r"^encoder\.down\.(\d+)\.downsample\.",
                     r"encoder.down_blocks.\1.downsamplers.0.", key)
        key = re.sub(r"^decoder\.up\.(\d+)\.block\.(\d+)\.",
                     lambda m: up_block(m) + f"resnets.{m.group(2)}.", key)
        key = re.sub(r"^decoder\.up\.(\d+)\.upsample\.",
                     lambda m: up_block(m) + "upsamplers.0.", key)
        key = (key.replace(".nin_shortcut", ".conv_shortcut")
                  .replace(".mid.block_1.", ".mid_block.resnets.0.")
                  .replace(".mid.block_2.", ".mid_block.resnets.1.")
                  .replace(".mid.attn_1.norm", ".mid_block.attentions.0.group_norm")
                  .replace(".mid.attn_1.q", ".mid_block.attentions.0.to_q")
                  .replace(".mid.attn_1.k", ".mid_block.attentions.0.to_k")
                  .replace(".mid.attn_1.v", ".mid_block.attentions.0.to_v")
                  .replace(".mid.attn_1.proj_out", ".mid_block.attentions.0.to_out.0")
                  .replace(".norm_out.", ".conv_norm_out.")
                  .replace("encoder.quant_conv", "quant_conv")
                  .replace("decoder.post_quant_conv", "post_quant_conv"))
        if ".attentions.0." in key and value.ndim == 4:
            value = value.squeeze(-1).squeeze(-1).contiguous()   # 1x1 conv -> Linear
        out[key] = value
    return out


def build_vae(dtype=torch.float32):
    from diffusers import AutoencoderKLFlux2
    cfg = {k: v for k, v in read_json(DEF_DIR / "vae/config.json").items() if not k.startswith("_")}
    accepted = set(inspect.signature(AutoencoderKLFlux2.__init__).parameters)
    kwargs = {k: v for k, v in cfg.items() if k in accepted}
    # The repo config describes the reduced "small decoder" that ships with the
    # model; ComfyUI carries the full FLUX.2 decoder instead. Take the width from
    # the file, which is the better decoder and the one we actually have.
    declared = kwargs.get("decoder_block_out_channels") or kwargs["block_out_channels"]
    actual = safetensors_shape(VAE_PATH, "decoder.conv_in.weight")[0]
    if declared[-1] != actual:
        log(f"vae: config declares a {declared[-1]}-channel decoder, file has {actual}; using the full decoder")
        kwargs.pop("decoder_block_out_channels", None)
    with skipping_random_init():
        model = AutoencoderKLFlux2(**kwargs)
    model = load_into(model, VAE_PATH, transform=ldm_to_diffusers_vae_keys)
    return model.to(resolve_dtype(dtype, "vae dtype"))


def build_scheduler(module):
    """Only pass keys the scheduler actually declares; the shared FlowMatchEuler
    config carries several it does not."""
    cfg = {k: v for k, v in read_json(DEF_DIR / "scheduler/scheduler_config.json").items()
           if not k.startswith("_")}
    accepted = set(inspect.signature(module.MOSSImageScheduler.__init__).parameters)
    dropped = sorted(set(cfg) - accepted)
    if dropped:
        log(f"scheduler: ignoring unused config keys {dropped}")
    return module.MOSSImageScheduler(**{k: v for k, v in cfg.items() if k in accepted})


def build_pipeline(dtype="bf16", vae_dtype=torch.float32, te_dtype=None):
    """Assemble the pipeline from the local ComfyUI files.

    dtype - precision for the transformer and the text encoder. Activations inside both
            follow the weights, so this also sets the intermediate precision:
        "bf16"  default: the checkpoint's native precision and what the upstream loader
                asks for. Accurate on its own (3.04/255 vs fp32 at 8 steps) but it erases
                the DMD LoRA - apply_lora warns if you combine them.
        "fp16"  same speed and memory as bf16 with 4x finer spacing. Use this with any
                LoRA; visually indistinguishable from fp32 here.
        "fp32"  reference. ~2x slower per step, 18.9 GiB of DiT weights, and the text
                encoder is clamped to fp16 because fp32 will not fit the card.
    vae_dtype - fp32 by default, matching the authors' from_components_pretrained. Pass
                dtype to match everything; decode fidelity is what is at risk.
    te_dtype - overrides dtype for the text encoder alone. Worth having because encoder
                precision lands directly in the conditioning, so it is a visible knob
                rather than an implementation detail: bf16 is what upstream uses.

    The denoising state itself stays fp32 whatever you pick: the vendored pipeline pins
    it with .float(), and accumulating the sampler over steps in fp32 is worth having.
    """
    dtype = resolve_dtype(dtype)
    for path, why in ((DIT_PATH, "transformer"), (TE_PATH, "text encoder (run convert_text_encoder.py)"),
                      (VAE_PATH, "vae"), (TOKENIZER_ASSETS, "tokenizer assets in ComfyUI")):
        if not Path(path).exists():
            raise SystemExit(f"missing {why}: {path}")
    t0 = time.time()
    module = load_definition()
    pipe = module.MOSSImagePipeline(
        transformer=build_transformer(module, dtype),
        scheduler=build_scheduler(module),
        text_encoder=build_text_encoder(dtype if te_dtype is None else te_dtype),
        tokenizer=ensure_tokenizer(),
        vae=build_vae(vae_dtype),
    )
    log(f"pipeline assembled from local files in {time.time() - t0:.1f}s")
    return pipe, module


def set_memory_mode(pipe, mode, disk_path, device="cuda"):
    """Install the offload strategy. Returns the mode actually used."""
    if mode == "auto":
        free = free_vram_gib(device)
        mode = ("full" if free >= UPSTREAM_FREE_VRAM_GIB
                else "resident" if free >= RESIDENT_MIN_GIB else "model")
        log(f"auto -> {mode} ({free:.1f} GiB free VRAM)")

    if mode == "full":
        pipe.to(device)
    elif mode == "model":
        pipe.enable_model_cpu_offload(device=device)
    elif mode == "resident":
        # The text encoder still stages in and out, but the DiT and VAE stay on the GPU
        # once loaded. accelerate's default hook moves every module back to CPU after
        # each forward, so for a 10 GiB transformer each denoising step pays ~20 GiB of
        # PCIe traffic plus an empty_cache() - roughly as much time as the compute.
        pipe.enable_model_cpu_offload(device=device)
        pinned = []
        for name in ("transformer", "vae"):
            hook = getattr(getattr(pipe, name, None), "_hf_hook", None)
            if hook is not None and getattr(hook, "offload", False):
                hook.offload = False
                pinned.append(name)
        log(f"resident: keeping {' and '.join(pinned)} on the GPU between steps")
    elif mode == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
    elif mode in ("stream", "disk"):
        from diffusers.hooks import apply_group_offloading
        pipe.enable_model_cpu_offload(device=device)
        kwargs = dict(onload_device=device, offload_device="cpu", offload_type="block_level",
                      num_blocks_per_group=1, use_stream=True)
        if mode == "disk":
            Path(disk_path).mkdir(parents=True, exist_ok=True)
            kwargs["offload_to_disk_path"] = str(disk_path)
            log(f"disk tier: {disk_path}")
        try:
            apply_group_offloading(pipe.transformer, **kwargs)
        except Exception as exc:
            raise SystemExit(f"could not combine model offload with block streaming ({exc}); "
                             "use --memory model instead")
    else:
        raise SystemExit(f"unknown memory mode: {mode}")
    return mode


def compile_dit(pipe, dynamic=True):
    """Compile the DiT with torch.compile. Fuses the pointwise ops (RMSNorm, modulation,
    SiLU, RoPE) and strips Python overhead. Measured on the 5090: 1.66x on the 8-step
    denoise (9.8s -> 5.9s), no graph breaks. One-time warmup (~16s on the first image) -
    a win for multi-image sessions, a loss for a single image.
    """
    t0 = time.time()
    pipe.transformer.compile(fullgraph=False, dynamic=dynamic)
    log(f"DiT compiled (setup {time.time() - t0:.1f}s; the first image pays the warmup)")


def cuda_allocated_gib():
    return torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0


def offload_module(pipe, name):
    """Put one pipeline component back on the CPU and return its VRAM."""
    module = getattr(pipe, name)
    hook = getattr(module, "_hf_hook", None)
    if hook is not None:
        hook.offload = True
    module.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    log(f"{name} -> CPU ({cuda_allocated_gib():.1f} GiB allocated)")


def set_dit_resident(pipe, keep):
    """keep=True: leave the DiT on the GPU between runs instead of staging the 9.5 GiB
    back out every time, so the next generation starts straight at step 1. keep=False:
    hand the VRAM back to whatever else wants the card.

    Must be applied *after* the pipeline call: diffusers' maybe_free_model_hooks() runs
    at the end of __call__ and reinstalls accelerate hooks with offload enabled, which
    silently undoes any flag set beforehand. generate_latent() does this for you."""
    hook = getattr(pipe.transformer, "_hf_hook", None)
    if hook is not None:
        hook.offload = not keep
    if keep and torch.cuda.is_available():
        pipe.transformer.to("cuda")   # stage it in once, here, rather than at step 1


@torch.no_grad()
def encode_text(pipe, prompt, *, negative=None, max_sequence_length=4096, offload_encoder=True):
    """Text to conditioning tensors. Pass negative (even as "") when you intend to run
    with CFG, so the negative branch is encoded now too and the text encoder never has
    to come back. Returns a dict to hand straight to generate_latent()."""
    log(f"encoding {len(prompt)} chars of prompt")
    embeds, mask = pipe.encode_prompt(prompt, max_sequence_length=max_sequence_length)
    conditioning = {"prompt_embeds": embeds, "prompt_attention_mask": mask}
    if negative is not None:
        conditioning["negative_prompt_embeds"], conditioning["negative_prompt_attention_mask"] = \
            pipe.encode_prompt([negative] * embeds.shape[0], max_sequence_length=max_sequence_length)
    if offload_encoder:
        offload_module(pipe, "text_encoder")
    log(f"conditioning ready: {tuple(embeds.shape)}")
    return conditioning


@torch.no_grad()
def generate_latent(pipe, conditioning, *, height=1024, width=1024, steps=50, cfg=3.0,
                    seed=42, skip_layers=(22,), keep_dit=True, out=None):
    """Conditioning to latents - the denoising loop on its own, no VAE. Re-run it with a
    different seed or step count without re-encoding the prompt."""
    for name, value in (("height", height), ("width", width)):
        if value <= 0 or value % 16:
            raise SystemExit(f"{name} must be a positive multiple of 16, got {value}")
    set_dit_resident(pipe, keep_dit)
    log(f"denoising {width}x{height} steps={steps} cfg={cfg} skip_layers={list(skip_layers)}")
    t0 = time.time()
    # fp32 weights, no autocast. Autocast is what this model's comments ask for and it is
    # faster, but it caches bf16 copies of every weight next to the fp32 originals - 28.3
    # GiB for this DiT, which does not fit a 32 GiB card next to anything else. Revisit if
    # the weights ever get smaller.
    latents = pipe(
        None,
        prompt_embeds=conditioning["prompt_embeds"],
        prompt_attention_mask=conditioning["prompt_attention_mask"],
        negative_prompt_embeds=conditioning.get("negative_prompt_embeds"),
        negative_prompt_attention_mask=conditioning.get("negative_prompt_attention_mask"),
        height=height, width=width,
        num_inference_steps=steps, guidance_scale=cfg,
        uncond_skip_layers=list(skip_layers),
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="latent",
    ).images
    log(f"latent {tuple(latents.shape)} in {time.time() - t0:.1f}s")
    # after the call, because the pipeline reinstalls its hooks on the way out
    set_dit_resident(pipe, keep_dit)
    log(f"{cuda_allocated_gib():.1f} GiB allocated, DiT {'kept on GPU' if keep_dit else 'released'}")
    if not keep_dit:
        offload_module(pipe, "transformer")
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latents.detach().cpu(), out)
        log(f"saved latents {out}")
    return latents


@torch.no_grad()
def decode_latent(pipe, latents, *, out=None):
    """Latents to pixels. Mirrors the pipeline's own decode: undo the batch-norm
    regulariser, 2x2 unpatchify back to the VAE's 32 channels, decode, clamp."""
    # accelerate chains the offload hooks: the VAE's pre_forward offloads the module
    # before it in the chain, i.e. the DiT. Remember whether the caller asked for the DiT
    # to stay resident so the decode does not quietly undo it.
    tf_hook = getattr(pipe.transformer, "_hf_hook", None)
    wanted_resident = tf_hook is not None and tf_hook.offload is False

    device = next(pipe.vae.parameters()).device
    if device.type == "cpu" and torch.cuda.is_available():
        pipe.vae.to("cuda")          # the VAE is only ~0.3 GiB; keep the decode on GPU
        device = torch.device("cuda")
    z = latents.to(device=device, dtype=pipe.vae.dtype)
    bn, eps = pipe.vae.bn, pipe.vae.config.batch_norm_eps
    z = z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(z).view(1, -1, 1, 1)
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
    t0 = time.time()
    images = pipe.vae.decode(z, return_dict=False)[0].clamp(-1, 1)
    image = pipe.image_processor.postprocess(images, output_type="pil")[0]
    if wanted_resident:
        set_dit_resident(pipe, True)
    log(f"decoded in {time.time() - t0:.1f}s "
        f"({cuda_allocated_gib():.1f} GiB allocated, DiT on "
        f"{next(pipe.transformer.parameters()).device})")
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out)
        log(f"saved {out}")
    return image


def generate_image(pipe, prompt, *, height=1024, width=1024, steps=50, cfg=3.0, seed=42,
                   negative="", skip_layers=(22,), out=None):
    """One generation. Everything expensive lives in `pipe`, so repeating this only
    pays for the denoising - which is what makes it usable interactively."""
    for name, value in (("height", height), ("width", width)):
        if value <= 0 or value % 16:
            raise SystemExit(f"{name} must be a positive multiple of 16, got {value}")
    t0 = time.time()
    conditioning = encode_text(pipe, prompt, negative=negative if cfg != 1.0 else None)
    latents = generate_latent(pipe, conditioning, height=height, width=width, steps=steps,
                              cfg=cfg, seed=seed, skip_layers=skip_layers, keep_dit=False)
    image = decode_latent(pipe, latents)
    log(f"generated in {time.time() - t0:.1f}s")
    if torch.cuda.is_available():
        log(f"peak VRAM {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GiB")
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out)
        log(f"saved {out}")
    return image


def build_prompt(args):
    if args.layout:
        return json.dumps(json.loads(Path(args.layout).read_text(encoding="utf-8")), ensure_ascii=False)
    if args.prompt:
        return args.prompt
    raise SystemExit("give a prompt, or --layout FILE.json")


def main():
    global DIT_DTYPE
    ap = argparse.ArgumentParser(description="MOSS-Image-SFT via diffusers, from local ComfyUI files",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("prompt", nargs="?", help="natural language, tags, or a JSON layout string")
    ap.add_argument("--layout", type=Path, help="JSON layout file, used as the prompt")
    ap.add_argument("--memory", default="auto",
                    choices=["auto", "full", "resident", "model", "sequential", "stream", "disk"])
    ap.add_argument("--disk-path", default=str(DISK_CACHE))
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--negative", default="", help="upstream default is empty")
    ap.add_argument("--skip-layer", default="22", help="negative-branch layers to skip, 'none' to disable")
    ap.add_argument("--shift", type=float, help="override the scheduler shift (config default 5)")
    ap.add_argument("--lora", type=Path, help=f"kohya/ComfyUI LoRA, applied unmerged in fp32 (default with --turbo: {TURBO_LORA})")
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--turbo", action="store_true",
                    help="8-step DMD preset: forces 8 steps, CFG 1, no layer skip, and the turbo LoRA")
    ap.add_argument("--dit-dtype", default=DIT_DTYPE, choices=["fp32", "fp16", "bf16"],
                    help="DiT weight precision; bf16 is fast but distorts this model")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the DiT (~1.6x denoise; one-time ~16s warmup)")
    ap.add_argument("--vae-bf16", action="store_true",
                    help="run the VAE decode in bf16 (~1.3x decode)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    DIT_DTYPE = args.dit_dtype

    for name, value in (("height", args.height), ("width", args.width)):
        if value <= 0 or value % 16:
            raise SystemExit(f"{name} must be a positive multiple of 16, got {value}")
    if args.turbo:
        args.lora = args.lora or TURBO_LORA
        args.steps, args.cfg, args.skip_layer = 8, 1.0, "none"
        if not Path(args.lora).is_file():
            raise SystemExit(f"--turbo wants the distillation LoRA, not found: {args.lora}")
    skip_layers = [] if args.skip_layer.lower() == "none" else [int(x) for x in args.skip_layer.split(",")]

    pipe, _ = build_pipeline(dtype=args.dit_dtype)

    if args.lora:
        apply_lora(pipe.transformer, args.lora, args.lora_strength)

    if args.shift:
        cfg = dict(pipe.scheduler.config)
        cfg["shift"] = args.shift
        from pipeline_moss_image import MOSSImageScheduler
        pipe.scheduler = MOSSImageScheduler.from_config(cfg)
        log(f"scheduler shift -> {args.shift}")

    mode = set_memory_mode(pipe, args.memory, args.disk_path)

    if args.vae_bf16:
        pipe.vae.to(torch.bfloat16)
        log("VAE decode -> bf16")
    if args.compile:
        compile_dit(pipe)

    out = args.out or HERE / "out" / f"moss_{time.strftime('%Y%m%d_%H%M%S')}_{args.seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    generate_image(pipe, build_prompt(args), height=args.height, width=args.width,
                   steps=args.steps, cfg=args.cfg, seed=args.seed, negative=args.negative,
                   skip_layers=skip_layers, out=out)


if __name__ == "__main__":
    main()
