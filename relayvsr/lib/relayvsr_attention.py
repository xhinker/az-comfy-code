#!/usr/bin/env python3
"""Rectangular-window local attention for RelayVSR, on plain torch SDPA.

Two window policies, one per network:

  shifted=True  (Wan backbone)  the window is clamped inside the image:
                                base = clamp(qy - wh/2, 0, H - wh), keys are the
                                wh x ww block starting at base
  shifted=False (Flash decoder) the window is centred on each query

The official code gathers the K/V neighbourhood of each 16x16 query block into a
contiguous tensor and runs masked SDPA in Triton. Same geometry here, torch SDPA only.
"""

# Ported from the RelayVSR reference implementation (github.com/kopperx/RelayVSR) and the
# ComfyUI node port (github.com/bbaudio-2025/Comfyui-Relay-VSR), both Apache-2.0; those in
# turn derive from LightX2V (github.com/ModelTC/LightX2V). Modified here: module layout,
# English comments, and wiring to ComfyUI core through relayvsr_load.py.

from collections import OrderedDict

import torch
import torch.nn.functional as F

_TILE = 16
_CACHE_BYTES = 1 << 30
_GATHER_BYTES = 1 << 30

_layouts = OrderedDict()
_layout_bytes = 0

def clear_layout_cache():
    global _layout_bytes
    _layouts.clear()
    _layout_bytes = 0

def _build_layout(height, width, windows, shifted, chunk, device):
    tile_y, tile_x = torch.meshgrid(
        torch.arange(0, height, _TILE, device=device),
        torch.arange(0, width, _TILE, device=device),
        indexing="ij",
    )
    tile_y, tile_x = tile_y.flatten(), tile_x.flatten()
    qr = torch.arange(_TILE * _TILE, device=device)
    parts = []
    for start in range(0, tile_y.numel(), chunk):
        y, x = tile_y[start : start + chunk, None], tile_x[start : start + chunk, None]
        qy, qx = y + qr[None] // _TILE, x + qr[None] % _TILE
        qi = (qy * width + qx).clamp(0, height * width - 1).flatten()
        keys, masks = [], []
        for frame, (wh, ww) in enumerate(windows):
            rows, cols = wh + _TILE - 1, ww + _TILE - 1
            if shifted:
                rows, cols = min(height, rows), min(width, cols)
            kr = torch.arange(rows * cols, device=device)
            top, left = y - wh // 2, x - ww // 2
            qtop, qleft = qy - wh // 2, qx - ww // 2
            if shifted:
                top = top.clamp(0, max(0, height - wh))
                left = left.clamp(0, max(0, width - ww))
                qtop = qtop.clamp(0, max(0, height - wh))
                qleft = qleft.clamp(0, max(0, width - ww))
            ky, kx = top + kr[None] // cols, left + kr[None] % cols
            valid = (ky >= 0) & (ky < height) & (kx >= 0) & (kx < width)
            keys.append(frame * height * width + (ky * width + kx).clamp(0, height * width - 1))
            masks.append(
                valid[:, None]
                & (ky[:, None] >= qtop[:, :, None])
                & (ky[:, None] < qtop[:, :, None] + wh)
                & (kx[:, None] >= qleft[:, :, None])
                & (kx[:, None] < qleft[:, :, None] + ww)
            )
        ki = torch.cat(keys, -1)
        mask = torch.cat(masks, -1)[:, None].contiguous()
        owner = destination = None
        if height % _TILE or width % _TILE:
            # Queries that fall outside the image in edge blocks are not written back.
            owner = ((qy < height) & (qx < width)).flatten().nonzero().flatten()
            destination = qi.index_select(0, owner)
        parts.append((qi, ki.flatten(), mask, owner, destination, qy.shape[0], ki.shape[1]))
    return parts

def _layout(height, width, windows, shifted, chunk, device):
    global _layout_bytes
    cache_key = (height, width, windows, shifted, chunk, device.type, device.index)
    cached = _layouts.get(cache_key)
    if cached is not None:
        _layouts.move_to_end(cache_key)
        return cached[0]
    parts = _build_layout(height, width, windows, shifted, chunk, device)
    storage_bytes = 0
    for part in parts:
        for tensor in part:
            if isinstance(tensor, torch.Tensor):
                storage_bytes += tensor.numel() * tensor.element_size()
    while _layouts and _layout_bytes + storage_bytes > _CACHE_BYTES:
        _, (_, old_bytes) = _layouts.popitem(last=False)
        _layout_bytes -= old_bytes
    if storage_bytes <= _CACHE_BYTES:
        _layouts[cache_key] = (parts, storage_bytes)
        _layout_bytes += storage_bytes
    return parts

def local_attention(query, key, value, height, width, windows, shifted):
    """query: (B, heads, H*W, dim)；key/value: (B, kv_heads, len(windows)*H*W, dim)。"""
    b, heads, tokens, dim = query.shape
    kv_heads = key.shape[1]
    n = height * width
    if (
        tokens != n
        or key.shape != value.shape
        or key.shape[0] != b
        or key.shape[2] != len(windows) * n
        or key.shape[3] != dim
        or heads % kv_heads
    ):
        raise ValueError("RelayVSR local attention got mismatched Q/K/V shapes")
    if query.dtype != key.dtype or key.dtype != value.dtype:
        raise ValueError("RelayVSR local attention needs Q/K/V with the same dtype")

    lengths = [
        (min(height, wh + _TILE - 1) * min(width, ww + _TILE - 1) if shifted else (wh + _TILE - 1) * (ww + _TILE - 1))
        for wh, ww in windows
    ]
    bytes_per_tile = sum(lengths) * kv_heads * dim * key.element_size() * 2
    limit = max(1, min(128, _GATHER_BYTES // max(1, bytes_per_tile)))
    chunk = 1 << (limit.bit_length() - 1)
    parts = _layout(height, width, windows, shifted, chunk, query.device)

    output = query.new_empty((b, tokens, heads, dim))
    for batch in range(b):
        qs, ks, vs = (t[batch].transpose(0, 1).contiguous() for t in (query, key, value))
        for qi, ki, mask, owner, destination, count, key_length in parts:
            q = qs.index_select(0, qi).view(count, _TILE * _TILE, heads, dim).transpose(1, 2)
            k = ks.index_select(0, ki).view(count, key_length, kv_heads, dim).transpose(1, 2)
            v = vs.index_select(0, ki).view(count, key_length, kv_heads, dim).transpose(1, 2)
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, enable_gqa=heads != kv_heads
            )
            attended = attended.transpose(1, 2).reshape(-1, heads, dim)
            if owner is not None:
                attended = attended.index_select(0, owner)
            output[batch].index_copy_(0, qi if destination is None else destination, attended)
    return output.transpose(1, 2)

def full_attention(query, key, value):
    """Unmasked whole-sequence attention: query (tokens, heads, dim), key/value (len, kv_heads, dim)."""
    q = query.transpose(0, 1).unsqueeze(0)
    k = key.transpose(0, 1).unsqueeze(0)
    v = value.transpose(0, 1).unsqueeze(0)
    out = F.scaled_dot_product_attention(q, k, v, enable_gqa=query.shape[1] != key.shape[1])
    return out[0].transpose(0, 1).flatten(1)

def wan_local_attention(query, key, value, height, width, window, key_frames):
    """Wan: the current frame queries the last key_frames cached frames with clamped windows."""
    windows = (tuple(window),) * key_frames
    attended = local_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        height,
        width,
        windows,
        shifted=True,
    )
    return attended[0].transpose(0, 1).flatten(1)

def flash_local_attention(query, key, value, height, width, windows):
    """Flash: per-frame queries attend the condition/history frames with centred windows."""
    attended = local_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        height,
        width,
        tuple(tuple(w) for w in windows),
        shifted=False,
    )
    return attended[0].transpose(0, 1).flatten(1)
