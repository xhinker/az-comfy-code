#!/usr/bin/env python3
"""Spatial tiling: split a frame into overlapping rectangles, render each as if it were a
whole frame, and stitch the results back with linear cross-fades.

Why this exists: the Wan backbone keeps a keyframe K/V cache proportional to the frame's
token count, (W/wan_stride) x (H/wan_stride) x 30 layers x 5 keyframes. A 912x1280 frame
under the x4 weights needs 33.6 GB of that cache on a 32 GB card and cannot run at all,
while a 2x2 grid of ~512x704 tiles needs ~10 GB per tile - and tiles run one at a time.

Tiles are aligned to the geometry's stride so no tile gets padded, and the overlap region
is cross-faded linearly so the seams stay invisible. Bigger overlap = fewer boundary
artifacts, more recomputation: the Wan attention window spans 22x40 patches, i.e. 176x320
input pixels under the x4 weights, so an overlap of half that is the theoretically clean
cut. 64 px is the community default and looks fine in practice.

Ported from the ComfyUI node port (github.com/bbaudio-2025/Comfyui-Relay-VSR, Apache-2.0);
comments translated to English, tile alignment taken from RelayVSRGeometry, and the grid
helper added so the caller only passes "2x2".
"""
import math

import numpy as np


def axis_tiles(size, count, overlap, stride):
    """Cut one axis into `count` equal-length (start, end) intervals.

    Starts stay aligned to `stride`; the last interval snaps to the edge so no tile is a
    leftover sliver.
    """
    if count <= 1 or size < stride * 2:
        return [(0, size)]
    count = min(count, size // stride)
    if count <= 1:
        return [(0, size)]
    tile = math.ceil((size + (count - 1) * overlap) / count)
    tile = min(size, math.ceil(tile / stride) * stride)
    tiles = []
    for index in range(count):
        if index == count - 1:
            start = size - tile
        else:
            start = round(index * (size - tile) / (count - 1))
            start -= start % stride
            start = min(max(start, 0), size - tile)
        if tiles and start <= tiles[-1][0]:
            continue
        tiles.append((start, start + tile))
    if tiles[-1][1] < size:
        tiles[-1] = (size - tile, size)
    return tiles


def _ramp(length, fade_in, fade_out):
    alpha = np.ones(length, dtype=np.float32)
    if fade_in > 0:
        count = min(int(fade_in), length)
        alpha[:count] = (np.arange(count, dtype=np.float32) + 0.5) / count
    if fade_out > 0:
        count = min(int(fade_out), length)
        alpha[length - count:] = (np.arange(count, dtype=np.float32)[::-1] + 0.5) / count
    return alpha


def axis_alphas(tiles, scale):
    """Per-tile linear weight along one axis, sampled at output pixels."""
    alphas = []
    for index, (start, end) in enumerate(tiles):
        fade_in = (tiles[index - 1][1] - start) * scale if index > 0 else 0
        fade_out = (end - tiles[index + 1][0]) * scale if index + 1 < len(tiles) else 0
        alphas.append(_ramp((end - start) * scale, fade_in, fade_out))
    return alphas


def stitch(tile_frames, tiles_y, tiles_x, y_alphas, x_alphas, scale):
    """Blend the rendered tiles of one frame back into a single HWC uint8 frame."""
    height = tiles_y[-1][1] * scale
    width = tiles_x[-1][1] * scale
    pixels = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    index = 0
    for row, (y0, y1) in enumerate(tiles_y):
        for col, (x0, x1) in enumerate(tiles_x):
            alpha = (y_alphas[row][:, None] * x_alphas[col][None, :])[..., None]
            top, left = y0 * scale, x0 * scale
            region = (slice(top, top + (y1 - y0) * scale), slice(left, left + (x1 - x0) * scale))
            pixels[region] += tile_frames[index].astype(np.float32) * alpha
            weights[region] += alpha
            index += 1
    return np.clip(pixels / np.maximum(weights, 1e-6), 0, 255).round().astype(np.uint8)


def grid(width, height, cols, rows, overlap, stride):
    """The tile rectangles for a frame, row-major, plus their output-pixel weights."""
    tiles_x = axis_tiles(width, cols, overlap, stride)
    tiles_y = axis_tiles(height, rows, overlap, stride)
    cells = [(y0, y1, x0, x1) for (y0, y1) in tiles_y for (x0, x1) in tiles_x]
    return cells, tiles_y, tiles_x
