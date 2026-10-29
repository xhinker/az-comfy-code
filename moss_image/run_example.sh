#!/bin/bash
# Example runs for moss_image_gen.py.  Read README.md first: every weight loads
# from the local ComfyUI files (nothing is downloaded), and --memory decides how
# much VRAM it needs alongside whatever else is on the card.
set -euo pipefail
cd "$(dirname "$0")"

VENV=${VENV:-/home/andrewzhu/storage_1t_1/az_git_folder/az-comfy-code/.venv}
MEMORY=${MEMORY:-model}      # auto|full|model|sequential|stream|disk
GPU=${GPU:-0}                # 0 = the RTX 5090

run() { CUDA_VISIBLE_DEVICES=$GPU "$VENV/bin/python" moss_image_gen.py --memory "$MEMORY" "$@"; }

# 1. text to image, upstream defaults (1024x1024, 50 steps, CFG 3, layer skip 22,
#    empty negative).  Measured 136 s on the 5090 with --memory model.
run "A wide shot of a Kyoto alley in heavy rain, paper lanterns, wet reflections" --seed 42

# 2. JSON layout with rendered text at fixed positions (MOSS's headline feature)
run --layout layout_example.json --seed 7

# 3. cheap smoke test, worth running after any change to the loading code
run "A red apple on a white table" --height 512 --width 512 --steps 4 --out out/smoke_512.png

# 4. fits next to a loaded ComfyUI: leaf-level offload, measured 1.9 GiB peak
MEMORY=sequential run "A ceramic bowl of persimmons on a linen cloth, soft window light" --steps 20
