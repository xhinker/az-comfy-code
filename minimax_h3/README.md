# MiniMax H3 in-process on ComfyUI's core modules

H3 is MiniMax's open-weights omni model: one ~33B dense transformer denoises a single
packed sequence and produces **video and audio in the same pass** - 1-15 s at 24 fps,
768 px short edge natively, with the sound coming out of the same sampler as the picture.

This folder drives it from Python by importing ComfyUI's own `comfy` package and calling
the H3 node classes as plain functions. **No ComfyUI server, no workflow JSON, no HTTP.**
You get ComfyUI's engine - INT8 ConvRot kernels, DynamicVRAM staging, sparse attention,
LoRA loading - with each component in a variable you can hold, inspect and free.

Weights are read from the existing ComfyUI installation on this box; nothing is downloaded.

Why not diffusers: the H3 diffusers integration is Modular-Pipeline-only, and on consumer
cards the documented recipe is weight-only int8 plus streaming offload with **dense**
attention. The two things that make H3 usable here - ConvRot int8 and sparse attention -
are ComfyUI-side.

## Files

| File | What it is |
|---|---|
| `h3_load.py` | the DynamicVRAM bootstrap and one loader per component; VRAM helpers |
| `h3_core.py` | the four stages (encode / denoise / decode) over those components; also a CLI |
| `h3_core_interactive.py` | `# %%` cells, one component per cell - the notebook path |
Outputs land in `outputs/<script name>/` — one subfolder per generator (`h3_core/`, `h3_core_interactive/`, `minimax_h3_fast_t8/`), through `h3_core.out_dir_for(__file__)`.

| `run_example.sh` | seven worked CLI commands, cheap smoke test first |
| `minimax_h3_fast_t8.py` | `minimax_h3_5s_fast.json` (the ComfyUI workflow) converted to cells: 4-LoRA chain, T8 dual-clock sampler, sol-attn, AV decode |
| `outputs/` | generated mp4s (git-ignored) |

## Prerequisites

```bash
./setup.sh          # from the repo root; requirements.txt now carries the ComfyUI packages
```

- A ComfyUI checkout at `/mnt/data_2t_3/github_repos/ComfyUI` (path is the `COMFY` constant
  at the top of `h3_load.py`) with the H3 weights in its model folders.
- **The project `.venv`.** `comfy-aimdo` and `comfy-kitchen` are abi3 wheels and install on
  this venv's CPython 3.13; the core path is verified there. In a notebook, pick the
  `az-comfy-code-313` kernel (`Ctrl+Shift+P` -> "Notebook: Select Notebook Kernel").
- Nothing needs to be running. If the ComfyUI server happens to be up, it will be holding
  the GPU - stop it or use another card.

Weights on this box:

```
models/diffusion_models/minimax_h3/MiniMax-H3-Ref2VA_int8_convrot.safetensors   32 GB
models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors               15 GB
models/vae/minimax_h3/minimax_h3_video_vae_int8_convrot.safetensors            2.7 GB
models/vae/minimax_h3_audio_vae_fp32.safetensors                               576 MB
models/loras/minimax_h3/minimax_h3_turbo_4step_ckpt850_pruned_comfyui.safetensors
```

## Quick start

Notebook - run `h3_core_interactive.py` cell by cell. Cell 0 setup, cell 1 the model file
paths (edit them), cell 2 the engine, then one cell per component, then prompt and generate.

CLI:

```bash
../.venv/bin/python h3_core.py "A red fox trotting through a snowy pine forest" \
    --width 832 --height 480 --length 22 --steps 4 --seed 7
./run_example.sh 3          # native 1344x768, 20 steps
```

## Loaders

Each loader takes the parameters of the ComfyUI node it mirrors, and accepts either an
absolute path or a name relative to ComfyUI's model folders. Load one at a time.

| Call | Node | Parameters |
|---|---|---|
| `load_diffusion_model()` | UNETLoader | `unet_name`, `weight_dtype` (default / fp8_e4m3fn / fp8_e4m3fn_fast / fp8_e5m2), `disable_dynamic` || `load_diffusion_model()` | UNETLoader | `unet_name`, `weight_dtype` (default / fp8_e4m3fn / fp8_e4m3fn_fast / fp8_e5m2), `disable_dynamic` |
| *(all three loaders)* | — | `device` (`None`/`"default"`, `"cpu"`, `2`, `":2"`, `"cuda:2"`) and `keep=True` to pin past `free_all()`; `resident()` and `per_gpu()` show where things landed |
| `load_text_encoder()` | CLIPLoader | `clip_name`, `type="minimax"`, `device` (default / cpu) |
| `load_vae()` | VAELoader | `vae_name` - the same call for the video and the audio VAE |
| `load_lora(model, ...)` | LoraLoaderModelOnly | `lora_name`, `strength_model` |
| `sigma_shift()` | MiniMaxH3SigmaShift | `shift_video`, `shift_audio` (12.0 / 3.0 as trained; the FastH3 template uses 10.0 / 3.0) |
| `attention_backend()` | ModelAttentionBackend | `comfy kitchen attention` (INT8) or `pytorch attention` |
| `sparse_attention()` | BlockSparseAttention | `method` (sol-attn / vsa / sla), `tau`, `keep_percent`, `start_percent`, `end_percent`, `dense_blocks`, `min_tokens`, `sink_conditioning` |
| `free() / free_all() / status() / vram_gb()` | - | release components, print what is resident |

## What this box actually does

Loads, with DynamicVRAM staging (nothing is resident - `torch.cuda.memory_allocated()`
reads ~0 after loading all four):

```
transformer  32 GB int8     0.2 s   (32427 MB Staged)
text encoder 15 GB nvfp4    2.0 s   (14956 MB Staged)
video VAE     2.7 GB int8   0.1 s   ( 2677 MB Staged)
audio VAE   576 MB fp32     0.1 s   (  576 MB Staged)
all four                   3.0 s
```

Sampling, dense, one forward pass per step (H3 is guidance-distilled - there is no CFG):

| Geometry | Steps | Rate |
|---|---|---|
| 832x480, 22 frames | 4 | ~3 it/s |
| 832x480, 124 frames | 20 | ~3 it/s, ~61 s |
| 1344x768, 73 frames | 20 | ~5.2 s/it, ~103 s |

## Sharpness

Two settings do most of the damage. **Steps**: the checkpoint wants ~20; 4 steps gives a
soft, smeared frame that looks like a bad VAE (4-8 is fine with the turbo LoRA).
**Canvas**: the native short edge is 768 px - 1344x768, 1024x1024, 768x1344 - and 832x480
is inherently soft no matter what the sampler does. `length` snaps to the 17k+5 grid
(22, 39, 56, 73, 90, 107, 124), so asking for 62 gives 73 frames. Same prompt, 4 steps at
832x480 vs 20 steps at 1344x768: Laplacian variance 3936 -> 5800, frame std 53 -> 71.

## Pitfalls

- **`status()` must print `dynamic_vram=True aimdo=True`.** If it says `False`, ComfyUI
  silently falls back to loading everything resident and the video VAE will not fit. The
  bootstrap in `init_engine()` repeats what ComfyUI's `main.py` does:
  `comfy.memory_management.aimdo_enabled = True` and
  `comfy.model_patcher.CoreModelPatcher = ModelPatcherDynamic`; `comfy_aimdo.control.init()`
  has to run **before** any comfy import (`comfy_aimdo.host_buffer` snapshots `lib =
  control.lib` at import time); `init_devices()` takes bare device indices and returns
  `False` quietly on the wrong protocol.
- **Wrong interpreter** -> `ModuleNotFoundError: No module named 'comfy_aimdo'`.
  `init_engine()` re-raises it naming the venv to select.
- **The int8 video VAE is the default** because it is the one that fits on a 32 GB card.
  `VAE_VIDEO_FP16` is sharper but its decode activations do not fit next to the DynamicVRAM
  cache - it OOMs plain, tiled, with everything freed, and with a 2 GB headroom request.
- Never set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` with aimdo - it hooks the
  allocator.
- `h3_core_interactive.py` derives its import path from `__file__`, so it has to stay in
  this folder.
- There is no negative prompt and no guidance scale: `comfy.samplers.sample()` insists on a
  negative conditioning, so `denoise()` uses `Guider_Basic`, which is what BasicGuider does.

## Verification

- Per component, at INFO level: `Model <Name> prepared for dynamic VRAM loading. NNNN MB Staged.`
- `status()` after loading everything: `allocated=0.00 GB`, `loaded=[]`.
- `ffprobe` the mp4: expect `h264` video plus `aac` at 32 kHz (the audio comes from the same
  denoising pass), and a frame count on the 17k+5 grid.


## Canvas size: let the image decide (`h3_core.fit_size`)

H3's canvas rules are not optional and they are not documented in the model card - they live
in `comfy_extras/nodes_minimax_h3.py`:

- `CANVAS_MULTIPLE = 32` - both axes must be multiples of 32 (16x VAE spatial compression
  times the transformer's 2x2 patchify). A height of 860 gives a latent of 53 rows, which
  cannot tile; nothing validates it for you in the image-to-video node.
- `BASE_SHORT_EDGE = 768`, `MAX_PIXELS = 768 * 1344` - the trained canvas: 768 on the short
  edge, capped at ~1.03 Mpx.

`fit_size(image, long_side)` takes the **long side** as the setting, reads the image's own
aspect ratio, derives the short side, snaps both to the 32 grid, and pulls the area back
under the cap with a warning. `long_side=None` defers to ComfyUI's own `adapt_canvas`, which
does the opposite - fixes the short edge at 768 and grows the long side. It also prints the
latent shape and token count, which is what sol-attn's `min_tokens` gate keys off. `length`
snaps up to the model's `17k+5` frame grid on the way in (126 -> 141 frames), so the printed
latent depth is the real one, not the one you asked for.

Duration works the same way - set seconds and the frame count stops mattering. H3 only
accepts frame counts on its `17k+5` grid at 24 fps (5, 22, 39, ... 124, 141, spaced 17 frames
= 0.71 s apart). ComfyUI's `align_frame_count` always rounds **up**, which for a duration
input means 1.0 s becomes 1.62 s; `frames_for` picks the **nearest** grid point instead and
prints the snap, plus a note when you leave the trained ~124-362 frame (5-15 s) range.

```python
SECONDS = 5.0        # overrides LENGTH when set
LENGTH = H.frames_for(SECONDS)          # -> 124 frames = 5.17s
WIDTH, HEIGHT = H.fit_size(FIRST_FRAME, LONG_SIDE, length=LENGTH)
# fit_size: image 896x1440 (ratio 28:45) -> canvas 768x1280, 983040 px (cap 1032192)
#           | latent 42x80x48 = 161280 voxels, 40320 tokens after 2x2 patchify
```

## The fast-workflow conversion (`minimax_h3_fast_t8.py`)

Reproduces `ComfyUI/user/default/workflows/minimax_h3/minimax_h3_5s_fast.json` by calling the
same node implementations the server calls - including the third-party ones:

| workflow node | in-process call |
|---|---|
| UNETLoader (pruned int8 convrot base) | `h3_load.load_diffusion_model` |
| LoraLoaderModelOnly x4 (0.75 / 0.75 / 0.35 / 0.5) | `h3_load.load_lora`, chained |
| SolAttnMiniMax (T8 pack) | `sol_attn_minimax_v2.SolAttnMiniMax.execute` |
| MiniMaxH3DualClockSamplerT8 | `h3_t8.sampling.setup_dual_clock_sampling` -> model, sampler, sigmas |
| BasicGuider + SamplerCustomAdvanced | `Guider_Basic` + `guider.sample()` |
| MiniMaxH3AVDecodeT8 | `h3_t8.audio_ops.decode_av_latent` |
| CreateVideo + SaveVideo | `comfy_api` `VideoFromComponents.save_to` |

Two deviations, both forced by the environment rather than the graph:

- **`MiniMaxH3MemoryEfficientSageAttentionPatch` (KJ nodes) needs a source build of
  sageattention 2.2.0**, which PyPI does not carry. Two things bite: setup.py pins
  `-std=c++17` while torch 2.15 headers require C++20, and `cusparse.h` is absent from the
  system toolkit at `/usr/local/cuda-13.0` but present in the venv's pip cuda-toolkit 13.2,
  so `CPATH` must point there. Recipe in `requirements.txt`. When it works the run logs
  `Applying MiniMax H3 Memory Efficient Sage Attention Patch to all transformer blocks`
  followed by `[sol_attn] composed with 50 patched attention forward(s)`.
- **`torchaudio` is a manual install** (see `requirements.txt`): the T8 pack imports it, and
  the only cu132 build would otherwise downgrade torch.

Measured, at the workflow's own 768x1344 x 226 frames, 8 steps: ~10 s/it after warm-up,
9.42 s out, h264 + aac stereo 32 kHz, audio peak -8.9 dBFS. Sol-attn logs
`chaining onto an existing attention override`, and the dual-clock schedule comes out as
`1.0 0.988 0.973 0.952 0.923 0.878 0.8 0.632 0.0`.
