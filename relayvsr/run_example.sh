#!/usr/bin/env bash
# Worked RelayVSR commands. Cheap smoke test first - the 4x path is heavy on VRAM.
# Run with the ComfyUI venv on PATH-visible GPU (see relayvsr_load.py).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CV=/mnt/data_2t_3/github_repos/ComfyUI/venv_py312_comfyui/bin/python
IN="${1:-$HERE/inputs/demo_input.mp4}"

echo "== 1. smoke test: 32 frames, x4 weights, 4x =="
"$CV" "$HERE/relayvsr_upscale.py" "$IN" -o "$HERE/outputs/smoke_x4.mp4" \
      --scale 4 --sr-ratio 4 --max-frames 32

echo "== 2. whole clip at 4x =="
"$CV" "$HERE/relayvsr_upscale.py" "$IN" -o "$HERE/outputs/full_x4.mp4" --scale 4 --sr-ratio 4

echo "== 3. same weights, 2x delivery (still computes at 4x, then downsizes) =="
"$CV" "$HERE/relayvsr_upscale.py" "$IN" -o "$HERE/outputs/full_x4_to2.mp4" --scale 4 --sr-ratio 2

echo "== 4. native x2 weights (cheaper, 2x only) =="
"$CV" "$HERE/relayvsr_upscale.py" "$IN" -o "$HERE/outputs/full_x2.mp4" --scale 2 --sr-ratio 2

echo "== 5. 24 GB cards: stream the Wan blocks through CPU RAM =="
"$CV" "$HERE/relayvsr_upscale.py" "$IN" -o "$HERE/outputs/offload_x4.mp4" \
      --scale 4 --sr-ratio 4 --offload --chunk-len 8
