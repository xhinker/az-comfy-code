#!/usr/bin/env bash
# Download the RelayVSR weights into relayvsr/models/ (git-ignored).
# Comfy-Org's repackage of kopper/RelayVSR - the same files a ComfyUI workflow uses.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE=https://huggingface.co/Comfy-Org/RelayVSR/resolve/main
mkdir -p "$HERE/models/diffusion_models" "$HERE/models/vae"
get() {  # url  dest
  [ -s "$2" ] && { echo "have $(basename "$2")"; return; }
  echo "getting $(basename "$2")"
  if command -v aria2c >/dev/null; then
    aria2c -x 16 -s 16 --file-allocation=none --continue=true -d "$(dirname "$2")" -o "$(basename "$2")" "$1"
  else
    curl -L -C - --retry 3 -o "$2" "$1"
  fi
}
SCALE="${1:-both}"                       # both | x4 | x2
case "$SCALE" in
  both|x4) get "$BASE/diffusion_models/relayvsr_x4_int8_convrot.safetensors"           "$HERE/models/diffusion_models/relayvsr_x4_int8_convrot.safetensors"
           get "$BASE/vae/relayvsr_flash_vae_x4_fp32.safetensors"                      "$HERE/models/vae/relayvsr_flash_vae_x4_fp32.safetensors" ;;&
  both|x2) get "$BASE/diffusion_models/relayvsr_x2_lightx2v_int8_convrot.safetensors"  "$HERE/models/diffusion_models/relayvsr_x2_lightx2v_int8_convrot.safetensors"
           get "$BASE/vae/relayvsr_flash_vae_x2_fp32.safetensors"                      "$HERE/models/vae/relayvsr_flash_vae_x2_fp32.safetensors" ;;
esac
echo "done:"; du -sh "$HERE"/models/*/*
