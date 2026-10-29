#!/usr/bin/env bash
# az-comfy-code environment setup — one command, no surprises.
#
#   ./setup.sh                 create/refresh .venv (CPython 3.13) + install everything
#   ./setup.sh --recreate      delete .venv first and build from scratch
#   PYTHON_VERSION=3.12 CUDA_INDEX=https://download.pytorch.org/whl/cu128 ./setup.sh
#
# Idempotent: a second run only installs what is missing (uv resolves from its cache).
# Written for andrewzhu-ubuntu (192.168.68.65): Ubuntu 22.04 has no python3.13 in apt and
# sudo needs a password, so the interpreter comes from uv's own managed CPython.
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
VENV="${VENV:-$ROOT/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu132}"
KERNEL_NAME="${KERNEL_NAME:-az-comfy-code-313}"

# pypi.nvidia.com (which hosts the CUDA wheels) is slow from this box: a single timeout
# partway through ~4 GB otherwise throws away the whole attempt.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}"

[ "${1:-}" = "--recreate" ] && { echo "removing $VENV"; rm -rf "$VENV"; }

echo "=== 1. uv (no sudo: it installs its own CPython toolchains) ==="
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version

echo "=== 2. CPython $PYTHON_VERSION + venv (--seed so the venv contains pip) ==="
# --seed matters: a plain `uv venv` has no pip, which silently breaks VS Code's
# "install ipykernel into the selected environment" step and any `pip install` you type.
uv python install "$PYTHON_VERSION"
[ -d "$VENV" ] || uv venv --python "$PYTHON_VERSION" --seed "$VENV"
"$VENV/bin/python" --version

echo "=== 3. PyTorch (cu132 wheels: RTX 5090 is sm_120, needs cu128 or newer) + the stack ==="
ok=0
for attempt in 1 2 3; do
  echo "--- attempt $attempt ---"
  if uv pip install --python "$VENV/bin/python" -r "$ROOT/requirements.txt" \
       --index-url https://pypi.org/simple \
       --extra-index-url "$CUDA_INDEX" \
       --index-strategy unsafe-best-match; then
    ok=1; break
  fi
  echo "attempt $attempt failed - retrying in 10 s"; sleep 10
done
[ "$ok" = 1 ] || { echo "install FAILED after 3 attempts"; exit 1; }

echo "=== 4. Jupyter support ==="
# Ctrl+Enter in a .py file runs in a Jupyter kernel, which is a SEPARATE selection from the
# status-bar interpreter - the venv must offer ipykernel and a kernelspec to be pickable.
uv pip install --python "$VENV/bin/python" ipykernel ipywidgets 2>&1 | tail -2
"$VENV/bin/python" -m ipykernel install --user \
  --name "$KERNEL_NAME" --display-name "Python ($(basename "$ROOT") $PYTHON_VERSION)" 2>&1 | tail -1

echo "=== 5. verify ==="
"$VENV/bin/python" - <<'PY'
import sys, pip, torch, torchvision, diffusers, transformers, ipykernel
print("python       :", sys.version.split()[0], "->", sys.executable)
print("pip          :", pip.__version__)
print("torch        :", torch.__version__, "| cuda runtime:", torch.version.cuda)
print("torchvision  :", torchvision.__version__)
print("diffusers    :", diffusers.__version__, "| transformers:", transformers.__version__)
print("ipykernel    :", ipykernel.__version__)
print("arch list    :", torch.cuda.get_arch_list())
if not torch.cuda.is_available():
    print("!! torch sees no CUDA device"); sys.exit(1)
print("cuda devices :", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("   gpu %d      : %s %s" % (i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i)))
a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
_ = (a @ a).sum(); torch.cuda.synchronize()
print("bf16 matmul  : ok")
PY
RC=$?
[ $RC -eq 0 ] || { echo "verification FAILED"; exit $RC; }

uv pip freeze --python "$VENV/bin/python" > "$ROOT/requirements.lock.txt"
echo
echo "Done."
echo "  activate :  source .venv/bin/activate        (check: echo \$VIRTUAL_ENV)"
echo "  run      :  cd moss_image && ../.venv/bin/python moss_image_gen.py \"a red apple\" --memory model"
echo "  VS Code  :  Ctrl+Shift+P -> 'Jupyter: Select Kernel' -> this .venv (that is what Ctrl+Enter obeys)"
