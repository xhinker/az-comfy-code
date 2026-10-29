# az-comfy-code

Python (non-node-graph) versions of the image / video generation pipelines we normally run
inside ComfyUI — plain scripts you can read, debug and Ctrl+Enter through. They load their
weights **from the existing ComfyUI installation on this machine**, so nothing is downloaded.

Current contents:

| Path | What it is |
|---|---|
| `moss_image/` | MOSS-Image-SFT through **diffusers** (pipeline vendored from the official model definition; see `moss_image/README.md` for the model-level detail) |
| `setup.sh` | one-command environment setup — see below |
| `requirements.txt` | the hand-maintained dependency list (what to install) |
| `requirements.lock.txt` | generated exact freeze of the working `.venv` (what was installed) |

## Quick start

```bash
./setup.sh
```

That is the whole thing. It is idempotent — re-run it any time and it only installs what is
missing. Useful variants:

```bash
./setup.sh --recreate                                   # throw .venv away and rebuild
PYTHON_VERSION=3.12 ./setup.sh                          # different interpreter
CUDA_INDEX=https://download.pytorch.org/whl/cu128 ./setup.sh   # different CUDA build
```

## What `setup.sh` does, and why each step is there

1. **Installs `uv`** (to `~/.local/bin`, no sudo). This box is Ubuntu 22.04: apt has no
   python3.13 and `sudo` needs a password, so the interpreter comes from uv's own managed
   CPython instead of the system packages.
2. **`uv python install 3.13` + `uv venv --python 3.13 --seed .venv`.** The `--seed` flag is
   not cosmetic: a plain `uv venv` contains **no pip**, which silently breaks VS Code's
   "install ipykernel into the selected environment" step and any `pip install` you type in
   an activated shell (it falls through `PATH` to `~/.local/bin/pip`, which here is pip 22.0.2
   for **Python 3.10** — it installs into your home directory without complaining).
3. **Installs `requirements.txt`** from PyPI plus the cu132 wheel index, with
   `UV_HTTP_TIMEOUT=180` and a 3-attempt retry loop. `pypi.nvidia.com` (which hosts the CUDA
   wheels) times out mid-download on this network roughly once a run; without the retry you
   lose ~4 GB of progress.
4. **Installs `ipykernel` + `ipywidgets` and registers a kernelspec**
   (`az-comfy-code-313`). Without these the venv cannot be chosen as a Jupyter kernel at all.
5. **Verifies**: prints python / torch / diffusers / transformers versions, the compiled CUDA
   arch list, every visible GPU, and runs a bf16 matmul on the GPU. Exits non-zero if torch
   cannot see CUDA. Then writes `requirements.lock.txt`.

## The environment it produces

```
.venv → CPython 3.13.13 (uv-managed), ~5.6 GB, 67 packages

torch                        2.15.0+cu132        cuda runtime 13.2
torchvision                  0.29.2+cu132        torchao 0.18.0+cu132   triton 3.9.0
arch list      sm_75 sm_80 sm_86 sm_90 sm_100 sm_120     ← sm_120 = RTX 5090
diffusers 0.41.0   transformers 5.20.2   accelerate 1.10.1   huggingface-hub 1.33.0
safetensors 0.8.0  tokenizers 0.23.3     sentencepiece 0.2.2
numpy 2.5.4        pillow 13.0.0         opencv-python-headless 5.0.0.93
ipykernel 7.4.0    ipywidgets
```

CUDA build choice: the RTX 5090 is `sm_120` (Blackwell), which needs **cu128 or newer**; the
box runs driver 610.57.04 with CUDA 12.9 and 13.0 toolkits installed, so the `cu132` wheels
are the current match. Override with `CUDA_INDEX=` if you move to another machine.

`kernels` is deliberately **not** installed: `kernels==0.10.0` pins `huggingface-hub<1.0`,
which is unsatisfiable next to the `huggingface-hub 1.33.0` this stack runs on. The MOSS code
never imports it.

## Running

```bash
source .venv/bin/activate
echo $VIRTUAL_ENV          # the reliable check — a venv name in your prompt is NOT activation
cd moss_image
python moss_image_gen.py "a red apple on a white ceramic table, soft studio light" --memory model
```

Or skip activation entirely, which removes a whole class of mistakes:

```bash
uv run --python .venv/bin/python moss_image/moss_image_gen.py --help
```

`--memory model` peaks at ~20 GiB; `--memory sequential` runs in 1.9 GiB beside a live
ComfyUI. `moss_image/run_example.sh` honours `VENV=` and `GPU=` environment variables.

## VS Code: the interpreter and the kernel are two different settings

This is the trap that costs people an hour:

- The **status-bar interpreter** governs IntelliSense, linting, terminal activation and the
  debugger.
- **Ctrl+Enter in a `.py` file** sends code to a **Jupyter kernel**, chosen separately with
  `Ctrl+Shift+P → Jupyter: Select Kernel`. Picking the interpreter does not change the kernel.

`.vscode/settings.json` in this repo pins the workspace to this venv:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "jupyter.interactiveWindow.creationMode": "perFile",
  "jupyter.notebookFileRoot": "${workspaceFolder}"
}
```

**Workspace settings beat machine settings.** Check for a stale remote default — on this box
`~/.vscode-server/data/Machine/settings.json` contained
`"python.defaultInterpreterPath": "/bin/python3.11"`, which overrode the status-bar pick and
launched the interactive kernel from Python 3.11. Edit it with
`Ctrl+Shift+P → Preferences: Open Remote Settings (JSON)`.

Also: `python.terminal.activateEnvironment` only affects terminals created **after** a reload,
and the kernel list is cached at startup — after changing any of this, run
`Developer: Reload Window`. Confirm what a cell really ran on with:

```python
import sys; print(sys.executable)
```

## Pitfalls on this machine

- **Weights are not in this repo.** `moss_image/` reads safetensors from absolute paths under
  `/mnt/data_2t_3/github_repos/ComfyUI/models/...`. Clone the code to another machine and those
  paths have to be repointed.
- **`convert_text_encoder.py` must run under ComfyUI's own venv** (`comfy.quant_ops` needs the
  native `comfy_aimdo` extension), not this one.
- **`Unable to import torchao Tensor objects`** at import time is harmless — torchao 0.18 moved
  the symbols diffusers 0.41 looks for, so diffusers' optional torchao path switches itself off.
- **`pypi.nvidia.com` timeouts** during install: just re-run `./setup.sh`, uv keeps what it got.
