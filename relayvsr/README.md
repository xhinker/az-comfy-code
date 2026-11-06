# RelayVSR in plain Python, on ComfyUI's core modules

RelayVSR is a **real-world video super-resolution** system from *RelayVSR: Large-Small Model
Collaboration for Efficient Real-World Video Super-Resolution*
([arXiv 2609.37850](https://arxiv.org/abs/2609.37850), [project page](https://kopperx.github.io/RelayVSR/),
[weights](https://huggingface.co/Comfy-Org/RelayVSR)). This folder runs it from Python - no
ComfyUI server, no workflow JSON - the same way `minimax_h3/` runs H3.

**Why there is no node to call:** upstream ComfyUI has no RelayVSR support (checked master and
the release notes through v0.44.0, 2026-11-04 - nothing named relayvsr anywhere in `comfy/`,
`comfy_extras/` or the node registry). The Comfy-Org repo is only a repackage of the weights
into ComfyUI's folder layout. So the networks here are the reference implementation, ported
into `lib/`, and ComfyUI core is used for what it is actually better at: **INT8
conv-rotation kernels** (`comfy.quant_ops`) and device / memory management
(`comfy.model_management`).

## How the two models split the work

| | model | what it does | size |
|---|---|---|---|
| large | **Wan backbone** - Wan2.2-5B-shaped DiT (dim 3072, 30 layers, 24 heads x 128, 48-ch in/out, patch [1,2,2]) plus a `lq_proj` that injects the low-res pixels and local block-causal spatial attention | denoises **sparse keyframes** only (first and last frame of each window) and hands their latents over as references | 5.16 GB INT8 conv-rot (9.4 GB in bf16) |
| small | **Flash decoder** - Dual-Memory Video Transformer (dim 512, depth 10, 8 heads, 2 KV groups, 48-ch condition, variant S) | super-resolves **every** frame, reusing the keyframe references plus a previous-frame K/V cache | 128 MB fp32 |

It is **pixel-space and one-step distilled** (sigma 1 to 0, no CFG), so there is no VAE, no
CLIP and no text encoder: the prompt embedding is baked into the Wan checkpoint as
`positive_conditioning`. Comfy-Org files the Flash decoder under `models/vae/` because that is
where it sits in the graph - it replaces the video VAE, it is not one.

Because only the keyframes go through Wan, a long clip costs Flash time, not Wan time: 96 frames
at 624x360 -> 2496x1440 ran at ~3 fps on a 5090.

## Getting the best quality out of it

Measured on a 5090 with the official demo clip (624x360 -> 2496x1440, 48 frames). Proxies:
`sharp` = Laplacian variance, `hf` = mean high-frequency energy, `tremb` = frame-to-frame
change in that high-frequency band (shimmer/crawl), `delta` = mean pixel distance from the
plain bicubic upscale (how much the model is actually adding).

| setting | fps | sharp | hf | tremb | delta |
|---|---|---|---|---|---|
| x4 weights, sr_ratio 4, detail 1.0, chunk 16 | 3.2 | 105 | 0.90 | 1.04 | 4.62 |
| same, detail 0.8 | 3.2 | 91 | 0.80 | 0.92 | 3.75 |
| same, detail 0.6 | 3.2 | 80 | 0.70 | 0.79 | 2.84 |
| same, detail 0.0 (= bicubic) | 3.2 | 64 | 0.49 | 0.53 | 0.62 |
| chunk_len 8 | 2.8 | 110 | 0.95 | 1.11 | 4.99 |
| chunk_len 32 | 3.5 | 102 | 0.87 | 1.01 | 4.35 |

At the same 1248x720 output size: **x4 weights at sr_ratio 4 then a lanczos downscale** scored
sharp 417 / hf 2.09 / tremb 2.67, versus **native x2 weights at sr_ratio 2** at 270 / 1.91 / 2.25.
More detail, but also ~19% more shimmer - which is the trade in this model family.

What that means in practice:

- **`detail` (0-1, default 1.0)** is the quality knob: it scales the Flash decoder's residual
  before it is added to the bicubic upscale of each frame, so 0.0 is plain bicubic and 1.0 is
  the full model output. Drop it to 0.6-0.8 when the pass over-sharpens, invents texture the
  source cannot support, or the added fine detail crawls in motion. Same knob as ComfyUI core's
  VSR Decode node (`--detail` on the CLI, `DETAIL` in the interactive script).
- **`sr_ratio` and the weights are not the same decision.** The x4 weights can render at 4x and
  you can downscale yourself afterwards; that buys real detail at 2x output. `--downscale` does
  exactly that in one pass, on the way to the encoder:

  ```bash
  ./relayvsr_upscale.py clip.mp4 --scale 4 --sr-ratio 4 --downscale 720
  ```

  Three forms: `0.5` (factor of the model output, decimal point required), `720` (target height,
  width follows the aspect ratio), `1248x720` (exact size - note the demo clip's 2496x1440 is
  1.733:1, not 16:9, so an exact WxH can nudge the aspect). Lanczos, dimensions rounded to even
  for YUV420, and it only shrinks - growing the model output here would be bilinear nonsense.
  The x2 weights compute at 2x instead, which is 3-4x cheaper and steadier in motion; LightX2V
  ships them precisely so H3/Wan clips do not pay 4x pixel work for a 2x finish.
- **`chunk_len` is the fidelity/cost dial** (default 16 = a Wan reference every 15 frames).
  Shorter windows = more generative references = more detail, less drift, more time. The paper
  is explicit that keyframe errors propagate across the frames that follow, so a long window
  that starts on a bad reference keeps paying for it.
- **One-step distilled: do not go hunting for steps/CFG/denoise.** There are none. `seed` only
  re-rolls the reference latents; it rarely changes the look, but it is the first thing to try
  when one specific frame has an artifact.
- **The prompt is baked in.** `positive_conditioning` lives in the checkpoint; there is no text
  encoder and no way to steer "keep the bokeh" or "no extra grain" with words.
- **Feed it clean input.** It is a super-resolver, not a repair tool: heavy macroblocking and
  banding get upscaled into permanent structure. Deint/denoise/decdeblock upstream if the
  source is a bad rip, and keep the encode you feed it high quality (CRF 12-15 or a lossless
  intermediate).
- **Output encode is not the quality limit.** `CRF 18 / medium` costs under a second per 24
  frames against ~15 s of model time; go to CRF 16 or `veryslow` for free if you care about the
  file rather than the wall clock.
- **Colour drift:** the model can shift contrast/colour slightly. If you are matching against
  the source, a per-frame colour-match pass after this one is standard practice in this model
  family (SeedVR2/FlashVSR users do the same).

## Files

| File | What it is |
|---|---|
| `relayvsr_interactive.py` | `# %%` cells: input, settings, weights, engine, **load the diffusion model**, **load the Flash (VAE-side) model**, run, save the output video, free |
| `relayvsr_upscale.py` | the same pipeline as a CLI (`--scale`, `--sr-ratio`, `--chunk-len`, `--detail`, `--downscale`, `--offload`, `--codec`, `--crf`) |
| `lib/relayvsr_load.py` | the ComfyUI-core bootstrap: `COMFY` / `PYTHON` paths, weight resolution, `engine()`, `load_wan()`, `load_flash()`, `run()`, `save()`, `free()`, `vram_line()` |
| `lib/relayvsr_engine.py` | assembly + streaming inference: `load_wan()` / `load_flash()` / `restore(frames, seed, sr_ratio, chunk_len)` |
| `lib/relayvsr_wan.py` `lib/relayvsr_flash.py` | the two networks: pre / blocks / post, ring K/V caches, windowed attention |
| `lib/relayvsr_weights.py` | safetensors loading, the **key-alias layer**, INT8 `comfy_quant` -> `QuantizedTensor`, fp32 LayerNorm / channel RMSNorm |
| `lib/relayvsr_attention.py` `lib/relayvsr_rope.py` `lib/relayvsr_geometry.py` | clamped vs centred rectangular windows, 3D RoPE (Wan) and per-frame RoPE (Flash), stride maths |
| `lib/relayvsr_metadata.py` | reads the `format` / `config` tags out of the checkpoints and refuses mismatched x2/x4 pairs |
| `lib/relayvsr_media.py` | PyAV frame iterator + h264/h265 writer, then imageio-ffmpeg muxes the original audio back in |
| `get_weights.sh` | aria2c download into `models/` (`both` / `x4` / `x2`) |
| `models/` | the weights - **git-ignored**, see below |
| `outputs/` | generated mp4s - git-ignored |

## Prerequisites

```bash
./get_weights.sh                 # ~10.6 GB into relayvsr/models/ (git-ignored)
```

- **Kernel: either venv works** - the project `.venv` (py3.13 / torch 2.15.0+cu132, the
  workspace default) and the ComfyUI venv (py3.12 / torch 2.12.1) both ran the INT8 path end to
  end. `comfy` is not installed as a package: `relayvsr_load.py` puts
  `/mnt/data_2t_3/github_repos/ComfyUI` on `sys.path`, which is why the INT8 kernels are there.
- **VS Code / Pylance**: `relayvsr/lib` is in `python.analysis.extraPaths` in `.vscode/settings.json`
  (that file is git-ignored, so it is per-machine). Without it every `import relayvsr_*` shows
  "could not be resolved", because Pylance never executes the cell-0 `sys.path.insert`.
- Nothing has to be running. If the ComfyUI server is up it is holding the GPU.

Weights (Comfy-Org's repackage of `kopper/RelayVSR`, Apache-2.0 - the same files a ComfyUI
workflow loads):

```
models/diffusion_models/relayvsr_x4_int8_convrot.safetensors               5.16 GB
models/diffusion_models/relayvsr_x2_lightx2v_int8_convrot.safetensors      5.16 GB
models/vae/relayvsr_flash_vae_x4_fp32.safetensors                          128 MB
models/vae/relayvsr_flash_vae_x2_fp32.safetensors                          128 MB
```

Single-connection downloads from HF run ~0.9 MB/s on this box and the `hf` CLI stalled;
`aria2c -x 16 -s 16` (what `get_weights.sh` uses) pulled 5.2 GB in about two minutes.

## Run it

```bash
./run_example.sh                                    # cheap smoke test first
./relayvsr_upscale.py inputs/demo_input.mp4 --scale 4 --sr-ratio 4 --max-frames 96
./relayvsr_upscale.py clip.mp4 --scale 4 --sr-ratio 2 --offload      # 24 GB cards
./relayvsr_upscale.py clip.mp4 --scale 4 --sr-ratio 4 --downscale 720  # render 4x, save at 720p
./relayvsr_upscale.py clip.mp4 --scale 4 --sr-ratio 4 --downscale 0.5  # ...or half the model output
```

or open `relayvsr_interactive.py` and run the cells - cell 5 loads the Wan backbone on its own,
cell 6 the Flash decoder, so you can keep one loaded and swap the other.

## Measured on this box (RTX 5090 32 GB, bf16, no offload)

| clip | output | speed | peak CUDA memory |
|---|---|---|---|
| 624x360 x 32 frames, x4 weights, sr_ratio 4 | 2496x1440 | 2.7-3.0 fps (12.0 s) | **20.5 GB** |
| 624x360 x 96 frames, same | 2496x1440 | 2.88 fps (33.3 s) | same |
| 624x360 x 16 frames, **x2 weights**, sr_ratio 2 | 1248x720 | 9.1-10.2 fps (1.8 s) | **9.2 GB** |

Load times: Wan 1.1 s (INT8), Flash 0.1 s. Weights resident after loading: 5.5 GB - the rest of
that 20.5 GB is the Wan K/V cache and the Flash activations at 2496x1440 (the native x2
weights at half the output size peak at 9.2 GB instead), so a 24 GB card needs
`--offload` (or a smaller `--chunk-len`) at this size.

Sanity check that it is really generating detail rather than sharpening: Laplacian variance of
the output vs bicubic-upscaled input, per frame: **8.4 -> 81.9** and **6.7 -> 54.8**.

## Things that bit us

- **The Comfy-Org Wan file renames every parameter** relative to the official checkpoint:
  `attn1.to_q` -> `self_attn.q`, `condition_embedder.text_embedder.linear_1` ->
  `text_embedding.0`, `lq_proj.out_proj` -> `lq_proj_in.linear`, `prompt_embeds` ->
  `positive_conditioning`, `proj_out` -> `head.head`, and so on. `relayvsr_weights.WAN_KEY_ALIASES`
  resolves both spellings, so the official and the Comfy-Org bundles both load.
- **`lq_proj.norm*.weight` is stored flat `(C,)`** in the Comfy-Org file where the official one
  uses `(C,1,1)`. `channel_rms_norm` reshapes it - otherwise the gain broadcasts against the
  width axis and dies (the upstream node port has this bug on Comfy-Org weights).
- **PyAV wants a rational frame rate**: `add_stream(..., rate=16.0)` raises inside
  `to_avrational`; it gets a `Fraction`.
- `sr_ratio`: x2 weights only produce 2x. x4 weights accept 2 or 4 - with 2 they still compute
  at 4x internally and then downsample, so it costs the same.
- `chunk_len` is the reference interval + 1 (default 16). Bigger windows blend fewer seams and
  cost more VRAM.
- The Wan K/V cache is the memory hog at high resolution - use `--offload` (blocks stream from
  CPU RAM) on 24 GB cards.

## Lineage / license

Apache-2.0 throughout. `lib/relayvsr_{wan,flash,engine,weights,attention,rope,geometry,metadata}.py`
are ported from the [RelayVSR reference implementation](https://github.com/kopperx/RelayVSR) and
the [ComfyUI node port](https://github.com/bbaudio-2025/Comfyui-Relay-VSR) (which derive from
[LightX2V](https://github.com/ModelTC/LightX2V)), with the module layout, English comments, key
aliases and ComfyUI-core wiring changed here. New code in `relayvsr_load.py`, `relayvsr_media.py`,
`relayvsr_upscale.py` and `relayvsr_interactive.py` is this repo's.
