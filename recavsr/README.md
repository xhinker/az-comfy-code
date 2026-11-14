# ReCaVSR — one-step streaming video super-resolution

**What it is.** ReCaVSR (kopperx/ReCaVSR, arXiv 2609.37831, USTC) is a Wan2.2 TI2V-5B video
upscaler that runs **one denoising step per block**: 21 frames, then 8 at a time. It stays
coherent without a full history by (a) feeding the **previous block's own latents** back in and
(b) letting **each transformer layer keep only the slice of past keys/values its learned router
chose** — the schedule is frozen at the end of training, so inference never evaluates a router.
It decodes with an **LR-conditioned FlashDecoder** (57 M params) instead of the Wan VAE: ~24x the
VAE's decode throughput at ~5% of its memory. Paper numbers: 21.20 FPS at 1080x1920 on one
A100-80GB, 15.16 GB peak, first output in 0.98 s.

**How this folder differs from `../relayvsr`.** RelayVSR had no usable ComfyUI node when we
ported it, so `relayvsr/lib` *is* the network. ReCaVSR does have one: **ComfyUI-ReCaVSR**
(PedroMarinhoDev/ComfyUI-ReCaVSR, in `ComfyUI/custom_nodes/`) runs the released weights on
ComfyUI's own ops, memory management and quantization, and reproduces the reference bit for bit
with `--kernels exact`. That pack is **GPL-3.0**, so its code stays in its own repo and is
*imported* here — nothing in this folder is a copy of it. What lives here is the driver: the
clip reader, the streaming loop, the memory switches, the CLI and the notebook cells.

## Layout

```
recavsr/
  lib/recavsr_load.py      paths, sys.path, weight check, load(), fit(), headroom(), free()
  lib/recavsr_media.py     probe(), BlockFrames (block-at-a-time decode), read_audio(), parse_size()
  lib/recavsr_engine.py    upscale(): the streaming loop, block by block
  recavsr_interactive.py   the same thing as notebook cells
  recavsr_upscale.py       the same thing as a CLI
  get_weights.sh           fetches the weights into ComfyUI/models/recavsr
  run_example.sh           worked commands
  inputs/ outputs/         git-ignored
```

## Weights

`get_weights.sh` (aria2c, 16 connections) puts them in **ComfyUI's `models/recavsr/`** — the same
folder the nodes read, so one copy serves both the UI and these scripts. Override with
`RECAVSR_MODELS=/path`.

| file | size | role |
|---|---|---|
| `transformer_int8_convrot.safetensors` | 5.98 GB | int8 ConvRot, ~1.8x the bf16 speed, ~45 dB from it |
| `transformer.safetensors` | 10.1 GB | the released bf16 |
| `prompt.safetensors` | 4.0 MB | the baked-in prompt embedding — there is no text encoder |
| `flashdecoder.safetensors` | 218 MB | the LR-conditioned decoder |
| `vae.safetensors` | 2.69 GB | optional, only for `--decoder vae` |

## Run

Always with the **ComfyUI venv** — `comfy` and the `recavsr` package resolve through the
`sys.path` inserts in `lib/recavsr_load.py`, not from installed packages:

```bash
/mnt/data_2t_3/github_repos/ComfyUI/venv_py312_comfyui/bin/python recavsr_upscale.py \
    inputs/demo_input.mp4 -o outputs/smoke_x4.mp4 --scale 4 --weights int8 --max-frames 32 --gpu 2
```

From Python: run the cells of `recavsr_interactive.py` top to bottom, then re-run cell 6 per
clip. `run_example.sh` has the whole set, including the `--post-size` and `--kv-offload` paths.

## Knobs that actually matter

- `--scale` — trained for 4; 2 and 1.5 are the reference's other settings. Sizes that are not a
  multiple of 32 are padded and cropped back.
- `--weights int8|bf16` — int8 is the one for a 24 GB card, and it is *faster*.
- `--kernels fast|exact` — `fast` uses ComfyUI's fused norm/rope/residual kernels; `exact` is the
  reference op sequence, bit for bit, and slower.
- `--kv-offload` — the attention history (66 planes of bf16 K/V over 30 blocks, ~0.8 MB per output
  token) lives in system RAM. That is what makes a big frame fit; it costs a transfer per step.
- `--post-size 0.5|720|1248x720` — render at 4x, then Lanczos down. This is how to compare a 4x
  pass against a 2x finish at the same final size (the same trick we used in RelayVSR).
- `--color-fix adain|wavelet` — a one-step generative upscale drifts in colour; both pull it back
  to the input.
- There is **no steps / cfg / denoise knob**. It is one step. Do not go looking for one.

## Measured on this box

RTX 5090 (32 GB), int8 ConvRot weights, `fast` kernels, `flashdecoder`, the 624x360 demo clip:

```
scale 4 -> 2496x1440   96 frames in 7.6 s = 12.6 fps
   block  1/11 (21 frames)  transformer 1407 ms   decode 390 ms   <- pays the triton compile
   blocks 2-11 (8 frames)   transformer  ~290 ms   decode  ~86 ms  <- steady state, 12.6 fps
scale 2 -> 1248x720    32 frames in 9.9 s
peak VRAM at 1440p: 15.8 GiB of the 31.4 GiB card; cell 3 drops it to 1.5
```

**Judge it per block, not per clip.** A 32-frame run measured 3.5 fps end to end earlier and that
number is misleading: the first block carries roughly a second of triton compilation, so short
clips look four times slower than the same pipeline at steady state.

`fit()` predicted 8.9 GiB for the 1440p step; weights 6.0 + activations landed at about 15.8 GiB,
so the estimate is the right order and the right shape: a 24 GB card runs 1440p with room, and
`--kv-offload` is what buys the step beyond that.

## Gotchas

- **Turn autograd off, or the step costs double.** The node pack never calls `no_grad` /
  `inference_mode` itself - inside the ComfyUI server, `execution.py` wraps node execution in
  `torch.inference_mode()`. Driven from a script it does not, so autograd builds a graph over
  every intermediate: 4x at 1440p OOM'd at 31 GB on a 32 GB card. With `inference_mode` the same
  run peaks at 15.8 GiB. `recavsr_engine.upscale()` enters it around the whole loop; if you call
  `VideoStream.step()` yourself, wrap it.
- **`fast` kernels need the ComfyUI venv.** `comfy` comes from the checkout either way, but
  comfy-kitchen comes from the interpreter's site-packages, and the repo's `.venv` (torch 2.15)
  carries one with no `window_attention` - the FlashDecoder's window op raises AttributeError.
  The driver detects that and drops to `exact`, which is right but far slower: 8 frames at
  1248x720 ran 0.93 fps on `.venv`/exact against 3.24 fps on the ComfyUI venv/fast. In VS Code,
  pick the ComfyUI venv as the kernel for these cells (Ctrl+Shift+P -> Python: Select
  Interpreter), since Ctrl+Enter follows the kernel, not the status bar.
- **Pylance needs `recavsr/lib`** in `python.analysis.extraPaths`, plus the node pack's root so
  `recavsr.loader` / `recavsr.session` resolve. Both are in the repo's `.vscode/settings.json`
  (git-ignored); without them every `import recavsr_*` in a cell is a red squiggle.
- **triton is not optional.** Without it both attention kernels fall back to PyTorch and get
  several times slower. The ComfyUI venv here has triton 3.7.1.
- **int8 needs compute capability 7.5 or newer**; below that the weights are dequantized on load.
  The `fast` kernels' window op wants 8.0 or newer (RTX 30 series and up).
- **Choose the card before torch starts**: `--gpu 2` sets `CUDA_VISIBLE_DEVICES`. Card 0 is where
  ComfyUI's own UI sits; leave it alone while the server is running.
- The node pack lives in `custom_nodes/`, so the ComfyUI **server has to be restarted** to see the
  nodes in the UI. These scripts do not care — they import the package directly.
- `--decoder vae` collects every latent frame and decodes at the end, so it loses the streaming
  property and holds more RAM.
