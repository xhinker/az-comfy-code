#!/usr/bin/env bash
# Worked ReCaVSR commands. Cheap smoke test first - the 4x path is heavy on VRAM.
# Run with the ComfyUI venv (see lib/recavsr_load.py); --gpu picks the card.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CV=/mnt/data_2t_3/github_repos/ComfyUI/venv_py312_comfyui/bin/python
IN="${1:-$HERE/inputs/demo_input.mp4}"
GPU="${GPU:-2}"                                # 0 is the card ComfyUI's UI is sitting on

echo "== 1. smoke test: 32 frames, int8 weights, 4x =="
"$CV" "$HERE/recavsr_upscale.py" "$IN" -o "$HERE/outputs/smoke_x4.mp4" \
      --scale 4 --weights int8 --max-frames 32 --gpu "$GPU"

echo "== 2. whole clip at 4x =="
"$CV" "$HERE/recavsr_upscale.py" "$IN" -o "$HERE/outputs/full_x4.mp4" --scale 4 --gpu "$GPU"

echo "== 3. same weights, 2x delivery (render at 4x, Lanczos down to 2x size) =="
"$CV" "$HERE/recavsr_upscale.py" "$IN" -o "$HERE/outputs/x4_to2.mp4" \
      --scale 4 --post-size 0.5 --max-frames 96 --gpu "$GPU"

echo "== 4. native 2x ratio (the reference's other setting) =="
"$CV" "$HERE/recavsr_upscale.py" "$IN" -o "$HERE/outputs/full_x2.mp4" --scale 2 --max-frames 96 --gpu "$GPU"

echo "== 5. colour drift, if the upscale shifts hue =="
"$CV" "$HERE/recavsr_upscale.py" "$IN" -o "$HERE/outputs/x4_wavelet.mp4" \
      --scale 4 --color-fix wavelet --max-frames 96 --gpu "$GPU"

echo "== 6. big frames on a 24 GB card: attention history in system RAM =="
"$CV" "$HERE/recavsr_upscale.py" "$IN" -o "$HERE/outputs/x4_kvmem.mp4" \
      --scale 4 --kv-offload --max-frames 64 --gpu "$GPU"

echo "== 7. the reference's own numbers, bit for bit (slower) =="
"$CV" "$HERE/recavsr_upscale.py" "$IN" -o "$HERE/outputs/x4_exact.mp4" \
      --scale 4 --kernels exact --max-frames 32 --gpu "$GPU"
