#!/usr/bin/env bash
# Download the ReCaVSR weights into ComfyUI's models/recavsr - the folder the ReCaVSR nodes read,
# so one copy of the files serves both a ComfyUI workflow and the scripts in this folder.
# Override the target with RECAVSR_MODELS=/some/path.
set -euo pipefail
DEST="${RECAVSR_MODELS:-/mnt/data_2t_3/github_repos/ComfyUI/models/recavsr}"
mkdir -p "$DEST"
BF16=https://huggingface.co/kopper/ReCaVSR/resolve/main
INT8=https://huggingface.co/PedroMarinhoDev/ReCaVSR-ComfyUI/resolve/main/recavsr
get() {  # url  dest-name
  local out="$DEST/$2"
  [ -s "$out" ] && { echo "have $2 ($(du -h "$out" | cut -f1))"; return; }
  echo "getting $2 <- $1"
  if command -v aria2c >/dev/null; then
    aria2c -x 16 -s 16 --file-allocation=none --continue=true -d "$DEST" -o "$2" "$1"
  else
    curl -L -C - --retry 3 -o "$out" "$1"
  fi
}
WHAT="${1:-all}"                          # all | core | int8 | bf16
case "$WHAT" in
  all|core) get "$BF16/prompt.safetensors"       prompt.safetensors
            get "$BF16/flashdecoder.safetensors" flashdecoder.safetensors ;;
esac
# one case per group: bash leaves the whole `case` at the first `;;`, so two groups sharing
# one case silently skip the second - which is what `all` used to do to the bf16 files.
case "$WHAT" in
  all|int8) get "$INT8/transformer_int8_convrot.safetensors" transformer_int8_convrot.safetensors ;;
esac
case "$WHAT" in
  all|bf16) get "$BF16/transformer.safetensors" transformer.safetensors
            get "$BF16/vae.safetensors"         vae.safetensors ;;
esac
echo "done:"; ls -la "$DEST"
