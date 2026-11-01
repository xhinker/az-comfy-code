#!/bin/bash
# Example runs for lib/h3_core.py - MiniMax H3 through ComfyUI's core modules, in-process.
# No ComfyUI server: the weights are read straight from the ComfyUI installation on this
# box (paths are listed at the top of lib/h3_load.py / h3_core_interactive.py).
#
#   ./run_example.sh              runs the smoke test only
#   ./run_example.sh 2            runs example 2
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-../.venv/bin/python}
run() { $PY lib/h3_core.py "$@"; }

case "${1:-1}" in
1)  # cheap smoke test - 22 frames at 480p, 4 steps. Good for checking the install.
    run "A red fox trotting through a snowy pine forest, snow crunching underfoot" \
        --width 832 --height 480 --length 22 --steps 4 --seed 7
    ;;
2)  # a real draft: 5 s at 480p, 20 steps, three seeds
    run "A barista pulls a double espresso, steam hissing, close shot, cafe ambience" \
        --width 832 --height 480 --length 124 --steps 20 --seed 11 --seeds 3
    ;;
3)  # native 768p final. ~5.2 s/it on a 5090, so 20 steps is ~100 s.
    run "A barista pulls a double espresso, steam hissing, close shot, cafe ambience" \
        --length 124 --steps 20 --seed 11
    ;;
4)  # first frame to video
    run "Slow push in, the lanterns flicker, fog drifts across the frame" \
        --mode i2v --image /path/to/first_frame.png --length 124 --steps 20
    ;;
5)  # reference to video: identity from an image, and <Picture 1> in the prompt labels it
    run "<Picture 1> is the subject. The subject turns to camera and speaks ..." \
        --mode ref2va --ref-image portrait.png --length 124 --steps 20
    ;;
6)  # sparse attention draft: ~1.7-2x faster, softens fine facial detail, so draft with it
    #   and re-run the keeper dense.
    run "A heron standing in shallow water, wind in the reeds" \
        --width 832 --height 480 --length 124 --steps 20 --sparse sol-attn --tau 1.3
    ;;
7)  # turbo LoRA: 4-8 steps instead of 20
    run "Neon rain on a night street, a taxi slides past, reflections on wet asphalt" \
        --length 124 --steps 6 --seed 5 \
        --lora minimax_h3/minimax_h3_turbo_4step_ckpt850_pruned_comfyui.safetensors:1.0
    ;;
esac
