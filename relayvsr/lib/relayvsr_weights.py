#!/usr/bin/env python3
"""RelayVSR weight loading and the low-level ops the two networks share.

No LightX2V dependency: safetensors plus torch. Key names match the official checkpoints
exactly, and Wan / Flash weights come back as plain dicts of tensors so a whole block can
be parked on CPU when VRAM is short.

Wan linears may be stored in ComfyUI native int8 conv-rotation format: a module carrying
``<module>.comfy_quant`` is rebuilt as a ``QuantizedTensor`` so ``F.linear`` goes straight
to ComfyUI int8 kernel; modules without that marker load as plain floats.
"""

# Ported from the RelayVSR reference implementation (github.com/kopperx/RelayVSR) and the
# ComfyUI node port (github.com/bbaudio-2025/Comfyui-Relay-VSR), both Apache-2.0; those in
# turn derive from LightX2V (github.com/ModelTC/LightX2V). Modified here: module layout,
# English comments, and wiring to ComfyUI core through relayvsr_load.py.

import json

import torch
import torch.nn.functional as F

from comfy.quant_ops import QuantizedTensor, get_layout_class

INT8_LAYOUT = "TensorWiseINT8Layout"

def fp32_layer_norm(x, eps, weight=None, bias=None, cast=False):
    """LayerNorm in float32; with cast=True the result returns to the input dtype."""
    output = F.layer_norm(
        x.float(),
        (x.shape[-1],),
        None if weight is None else weight.float(),
        None if bias is None else bias.float(),
        eps,
    )
    return output.to(x.dtype) if cast else output

def channel_rms_norm(x, weight, eps=1e-6):
    """Normalise over channels (dim=1), used by the 4D lq_proj features.

    The official checkpoints store the gain with trailing singleton dims (C, 1, 1);
    the Comfy-Org repackage flattens it to (C,), so reshape it to broadcast over
    channels instead of over the last spatial axis.
    """
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(dim=1, keepdim=True) + eps)
    if weight.dim() == 1 and value.dim() > 2:
        weight = weight.view(-1, *([1] * (value.dim() - 2)))
    return (value * weight).to(x.dtype)

def rms_norm(x, weight, eps):
    return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)

# The Comfy-Org repackage of the RelayVSR Wan backbone carries Wan-native parameter
# names where the official checkpoints use diffusers ones - same tensors, different
# labels. Resolve once per key rather than duplicating every loader.
WAN_KEY_ALIASES = [
    ("lq_proj.conv1", "lq_proj_in.conv1"),
    ("lq_proj.conv2", "lq_proj_in.conv2"),
    ("lq_proj.norm1", "lq_proj_in.norm1"),
    ("lq_proj.norm2", "lq_proj_in.norm2"),
    ("lq_proj.out_proj", "lq_proj_in.linear"),
    ("condition_embedder.text_embedder.linear_1", "text_embedding.0"),
    ("condition_embedder.text_embedder.linear_2", "text_embedding.2"),
    ("condition_embedder.time_embedder.linear_1", "time_embedding.0"),
    ("condition_embedder.time_embedder.linear_2", "time_embedding.2"),
    ("condition_embedder.time_proj", "time_projection.1"),
    ("prompt_embeds", "positive_conditioning"),
    ("attn1.to_q", "self_attn.q"),
    ("attn1.to_k", "self_attn.k"),
    ("attn1.to_v", "self_attn.v"),
    ("attn1.to_out.0", "self_attn.o"),
    ("attn1.norm_q", "self_attn.norm_q"),
    ("attn1.norm_k", "self_attn.norm_k"),
    ("attn2.to_q", "cross_attn.q"),
    ("attn2.to_k", "cross_attn.k"),
    ("attn2.to_v", "cross_attn.v"),
    ("attn2.to_out.0", "cross_attn.o"),
    ("attn2.norm_q", "cross_attn.norm_q"),
    ("attn2.norm_k", "cross_attn.norm_k"),
    ("norm2", "norm3"),
    ("scale_shift_table", "head.modulation"),
    ("scale_shift_table", "modulation"),
    ("ffn.net.0.proj", "ffn.0"),
    ("ffn.net.2", "ffn.2"),
    ("proj_out", "head.head"),
]


def resolve_key(keys, name):
    """The name this checkpoint actually uses for a canonical key, or KeyError."""
    if name in keys:
        return name
    for a, b in WAN_KEY_ALIASES:
        if a in name and name.replace(a, b) in keys:
            return name.replace(a, b)
    raise KeyError(f"{name} (or an alias of it) is not in this checkpoint")


class Checkpoint:
    """Thin wrapper over a safetensors handle; caches the key set to spot quantized modules."""

    def __init__(self, handle):
        self.handle = handle
        self.keys = frozenset(handle.keys())
        self._name = {}

    def key(self, name):
        """Canonical name -> whatever this checkpoint calls it."""
        hit = self._name.get(name)
        if hit is None:
            hit = resolve_key(self.keys, name)
            self._name[name] = hit
        return hit

    def tensor(self, key):
        return self.handle.get_tensor(self.key(key))

    def quantized(self, key):
        real = self.key(key)
        return real[: -len(".weight")] + ".comfy_quant" in self.keys

def _get(handle, key, device, dtype):
    key = handle.key(key)
    if key.endswith(".weight") and handle.quantized(key):
        return _quantized_weight(handle, key, device, dtype)
    return handle.tensor(key).to(device=device, dtype=dtype)

def _quantized_weight(handle, key, device, dtype):
    module = key[: -len(".weight")]
    conf = json.loads(handle.tensor(module + ".comfy_quant").numpy().tobytes())
    if conf.get("format") != "int8_tensorwise":
        raise ValueError(f"unsupported quantization format {conf.get('format')!r}: {module}")
    qdata = handle.tensor(key).to(device=device)
    scale = handle.tensor(module + ".weight_scale").to(device=device, dtype=torch.float32)
    params = get_layout_class(INT8_LAYOUT).Params(
        scale=scale,
        orig_dtype=dtype,
        orig_shape=tuple(qdata.shape),
        is_weight=True,
        convrot=bool(conf.get("convrot", False)),
        convrot_groupsize=int(conf.get("convrot_groupsize", 256)),
    )
    return QuantizedTensor(qdata, INT8_LAYOUT, params)

def _linear(handle, key, device, dtype, bias=True):
    weight = _get(handle, key + ".weight", device, dtype)
    if not bias:
        return weight, None
    bias_tensor = _get(handle, key + ".bias", device, dtype)
    return weight, bias_tensor

def load_wan_pre(handle, device, dtype):
    return {
        "patch_w": _get(handle, "patch_embedding.weight", device, dtype),
        "patch_b": _get(handle, "patch_embedding.bias", device, dtype),
        "lq1_w": _get(handle, "lq_proj.conv1.weight", device, dtype),
        "lq1_b": _get(handle, "lq_proj.conv1.bias", device, dtype),
        "lq2_w": _get(handle, "lq_proj.conv2.weight", device, dtype),
        "lq2_b": _get(handle, "lq_proj.conv2.bias", device, dtype),
        "lq_n1": _get(handle, "lq_proj.norm1.weight", device, dtype),
        "lq_n2": _get(handle, "lq_proj.norm2.weight", device, dtype),
        "lqo_w": _get(handle, "lq_proj.out_proj.weight", device, dtype),
        "lqo_b": _get(handle, "lq_proj.out_proj.bias", device, dtype),
        "text1_w": _get(handle, "condition_embedder.text_embedder.linear_1.weight", device, dtype),
        "text1_b": _get(handle, "condition_embedder.text_embedder.linear_1.bias", device, dtype),
        "text2_w": _get(handle, "condition_embedder.text_embedder.linear_2.weight", device, dtype),
        "text2_b": _get(handle, "condition_embedder.text_embedder.linear_2.bias", device, dtype),
        "time1_w": _get(handle, "condition_embedder.time_embedder.linear_1.weight", device, dtype),
        "time1_b": _get(handle, "condition_embedder.time_embedder.linear_1.bias", device, dtype),
        "time2_w": _get(handle, "condition_embedder.time_embedder.linear_2.weight", device, dtype),
        "time2_b": _get(handle, "condition_embedder.time_embedder.linear_2.bias", device, dtype),
        "timep_w": _get(handle, "condition_embedder.time_proj.weight", device, dtype),
        "timep_b": _get(handle, "condition_embedder.time_proj.bias", device, dtype),
        "prompt": _get(handle, "prompt_embeds", device, dtype),
    }

def _table(handle, name, device, dtype):
    # The official checkpoints store per-block modulation as (1, 6, dim); flatten to (6, dim).
    tensor = _get(handle, name, device, dtype)
    return tensor[0] if tensor.dim() == 3 and tensor.shape[0] == 1 else tensor

def load_wan_block(handle, index, device, dtype):
    prefix = f"blocks.{index}."
    return {
        "sq_w": _get(handle, prefix + "attn1.to_q.weight", device, dtype),
        "sq_b": _get(handle, prefix + "attn1.to_q.bias", device, dtype),
        "sk_w": _get(handle, prefix + "attn1.to_k.weight", device, dtype),
        "sk_b": _get(handle, prefix + "attn1.to_k.bias", device, dtype),
        "sv_w": _get(handle, prefix + "attn1.to_v.weight", device, dtype),
        "sv_b": _get(handle, prefix + "attn1.to_v.bias", device, dtype),
        "so_w": _get(handle, prefix + "attn1.to_out.0.weight", device, dtype),
        "so_b": _get(handle, prefix + "attn1.to_out.0.bias", device, dtype),
        "sqn": _get(handle, prefix + "attn1.norm_q.weight", device, dtype),
        "skn": _get(handle, prefix + "attn1.norm_k.weight", device, dtype),
        "cq_w": _get(handle, prefix + "attn2.to_q.weight", device, dtype),
        "cq_b": _get(handle, prefix + "attn2.to_q.bias", device, dtype),
        "ck_w": _get(handle, prefix + "attn2.to_k.weight", device, dtype),
        "ck_b": _get(handle, prefix + "attn2.to_k.bias", device, dtype),
        "cv_w": _get(handle, prefix + "attn2.to_v.weight", device, dtype),
        "cv_b": _get(handle, prefix + "attn2.to_v.bias", device, dtype),
        "co_w": _get(handle, prefix + "attn2.to_out.0.weight", device, dtype),
        "co_b": _get(handle, prefix + "attn2.to_out.0.bias", device, dtype),
        "cqn": _get(handle, prefix + "attn2.norm_q.weight", device, dtype),
        "ckn": _get(handle, prefix + "attn2.norm_k.weight", device, dtype),
        "norm2_w": _get(handle, prefix + "norm2.weight", device, dtype),
        "norm2_b": _get(handle, prefix + "norm2.bias", device, dtype),
        "sst": _table(handle, prefix + "scale_shift_table", device, dtype),
        "f0_w": _get(handle, prefix + "ffn.net.0.proj.weight", device, dtype),
        "f0_b": _get(handle, prefix + "ffn.net.0.proj.bias", device, dtype),
        "f2_w": _get(handle, prefix + "ffn.net.2.weight", device, dtype),
        "f2_b": _get(handle, prefix + "ffn.net.2.bias", device, dtype),
    }

def load_wan_post(handle, device, dtype):
    return {
        "sst": _get(handle, "scale_shift_table", device, dtype),
        "proj_w": _get(handle, "proj_out.weight", device, dtype),
        "proj_b": _get(handle, "proj_out.bias", device, dtype),
    }

def load_flash_pre(handle, device, dtype):
    return {
        "c0": _linear(handle, "lr_encoder.layers.0", device, dtype),
        "c2": _linear(handle, "lr_encoder.layers.2", device, dtype),
        "c4": _linear(handle, "lr_encoder.layers.4", device, dtype),
        "c6": _linear(handle, "lr_encoder.layers.6", device, dtype),
        "cond": _linear(handle, "condition_encoder.projection", device, dtype),
        "lrn": _get(handle, "lr_encoder.norm.weight", device, dtype),
        "condn": _get(handle, "condition_encoder.norm.weight", device, dtype),
        "vtype": _get(handle, "video_type_embedding", device, dtype),
        "ctype": _get(handle, "condition_type_embedding", device, dtype),
    }

def load_flash_block(handle, index, device, dtype):
    prefix = f"backbone.blocks.{index}.attention."
    ff = f"backbone.blocks.{index}.feed_forward."
    return {
        "q": _linear(handle, prefix + "query", device, dtype),
        "k": _linear(handle, prefix + "key", device, dtype),
        "v": _linear(handle, prefix + "value", device, dtype),
        "ck": _linear(handle, prefix + "condition_key", device, dtype),
        "cv": _linear(handle, prefix + "condition_value", device, dtype),
        "o": _linear(handle, prefix + "output", device, dtype),
        "kn": _get(handle, prefix + "key_norm.weight", device, dtype),
        "vn": _get(handle, prefix + "value_norm.weight", device, dtype),
        "an": _get(handle, f"backbone.blocks.{index}.attention_norm.weight", device, dtype),
        "fn": _get(handle, f"backbone.blocks.{index}.feed_forward_norm.weight", device, dtype),
        "gu_w": _get(handle, ff + "gate_up_proj.weight", device, dtype),
        "dn_w": _get(handle, ff + "down.weight", device, dtype),
    }

def load_flash_post(handle, device, dtype):
    return {
        "head": _linear(handle, "spatial_head.0", device, dtype),
        "proj": _linear(handle, "residual_projection", device, dtype),
    }
