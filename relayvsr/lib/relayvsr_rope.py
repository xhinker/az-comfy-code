#!/usr/bin/env python3
"""3D RoPE for the Wan backbone, per-frame relative RoPE for the Flash decoder.

Kept identical to the official implementation:

  Wan    three axes (t/h/w) split across head_dim; cos/sin stored in float32, so
         apply_rope multiplies and adds in float32, then writes back to the compute dtype
  Flash  axis dims fixed at (16, 24, 24); cos/sin use the compute dtype directly
"""

# Ported from the RelayVSR reference implementation (github.com/kopperx/RelayVSR) and the
# ComfyUI node port (github.com/bbaudio-2025/Comfyui-Relay-VSR), both Apache-2.0; those in
# turn derive from LightX2V (github.com/ModelTC/LightX2V). Modified here: module layout,
# English comments, and wiring to ComfyUI core through relayvsr_load.py.

import torch

def apply_rope(x, cosine, sine):
    """x: (tokens, heads, dim)，cosine/sine: (tokens, dim)。"""
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cosine[:, None, 0::2] - odd * sine[:, None, 1::2]
    out[..., 1::2] = even * sine[:, None, 1::2] + odd * cosine[:, None, 0::2]
    return out

class RotaryEmbedding:
    def __init__(self, *, wan, device, dtype, head_dim=128, max_length=1024, theta=10000.0):
        self.wan = wan
        self.device, self.dtype = device, dtype
        self.max_length = max_length
        if wan:
            # As in the official code: time takes head_dim - 4*(head_dim//6), the others 2*(head_dim//6).
            self.dims = (
                head_dim - 4 * (head_dim // 6),
                2 * (head_dim // 6),
                2 * (head_dim // 6),
            )
        else:
            self.dims = (16, 24, 24)
        self.cache = {}
        self.tables = []
        for dim in self.dims:
            if wan:
                # The official code builds the tables on CPU in float64, then casts to float32 on device.
                inverse = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
                angles = torch.outer(torch.arange(max_length, dtype=torch.float64), inverse)
                self.tables.append(
                    (
                        angles.cos().repeat_interleave(2, -1).float().to(device),
                        angles.sin().repeat_interleave(2, -1).float().to(device),
                    )
                )
            else:
                self.tables.append(
                    1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
                )

    def for_positions(self, height, width, positions):
        key = (height, width, positions)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if self.wan and (
            max(height, width) > self.max_length or min(positions) < 0 or max(positions) >= self.max_length
        ):
            raise ValueError("RelayVSR position is outside the RoPE range of these weights")
        frames = len(positions)
        coordinates = (
            torch.tensor(positions, device=self.device).view(frames, 1, 1).expand(frames, height, width).reshape(-1),
            torch.arange(height, device=self.device).view(1, height, 1).expand(frames, height, width).reshape(-1),
            torch.arange(width, device=self.device).view(1, 1, width).expand(frames, height, width).reshape(-1),
        )
        cosines, sines = [], []
        for coordinate, table in zip(coordinates, self.tables):
            if self.wan:
                cosine, sine = table[0][coordinate], table[1][coordinate]
            else:
                angles = coordinate.float().unsqueeze(-1) * table
                cosine = angles.cos().repeat_interleave(2, -1)
                sine = angles.sin().repeat_interleave(2, -1)
            cosines.append(cosine)
            sines.append(sine)
        result = (torch.cat(cosines, -1), torch.cat(sines, -1))
        if not self.wan:
            result = tuple(value.to(self.dtype) for value in result)
        if len(self.cache) >= 32:
            # Keep only the recent position tables so a long clip cannot grow this without bound.
            self.cache.clear()
        self.cache[key] = result
        return result

    def clear(self):
        self.cache.clear()
