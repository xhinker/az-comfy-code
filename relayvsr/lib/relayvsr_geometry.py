#!/usr/bin/env python3
"""The spatial geometry implied by a RelayVSR native scale.

  x4 weights: latent stride 4, Wan backbone stride 8
  x2 weights: latent stride 8, Wan backbone stride 16
"""

# Ported from the RelayVSR reference implementation (github.com/kopperx/RelayVSR) and the
# ComfyUI node port (github.com/bbaudio-2025/Comfyui-Relay-VSR), both Apache-2.0; those in
# turn derive from LightX2V (github.com/ModelTC/LightX2V). Modified here: module layout,
# English comments, and wiring to ComfyUI core through relayvsr_load.py.

from dataclasses import dataclass

@dataclass(frozen=True)
class RelayVSRGeometry:
    scale: int

    def __post_init__(self):
        if self.scale not in (2, 4):
            raise ValueError("RelayVSR native scale must be 2 or 4")

    @property
    def latent_stride(self):
        return 16 // self.scale

    @property
    def wan_stride(self):
        return 2 * self.latent_stride

    @property
    def encoder_strides(self):
        # The three downsample strides of the Flash LR encoder (last stage is a 1x1 projection).
        return (2, 2, self.latent_stride // 4)

    def latent_grid(self, height, width):
        return height // self.latent_stride, width // self.latent_stride

    def wan_grid(self, height, width):
        return height // self.wan_stride, width // self.wan_stride

    def padded_size(self, height, width):
        alignment = self.wan_stride
        return height + (-height) % alignment, width + (-width) % alignment

    def output_size(self, sr_ratio, height, width):
        if self.scale == 2:
            if sr_ratio != 2:
                raise ValueError("RelayVSR x2 weights only support a 2x ratio")
        elif sr_ratio not in (2, 4):
            raise ValueError("RelayVSR x4 weights accept sr_ratio 2 or 4")
        target = (int(height * sr_ratio), int(width * sr_ratio))
        if min(target) <= 0:
            raise ValueError("RelayVSR output size must be positive")
        return target
