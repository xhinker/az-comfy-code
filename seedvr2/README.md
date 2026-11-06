# SeedVR2 video upscale on ComfyUI core nodes

SeedVR2 is in ComfyUI core (`comfy_extras/nodes_seedvr.py`, `comfy/ldm/seedvr/`, a
`SeedVR2` entry in `supported_models.py`, detection in `model_detection.py`) - no custom
node pack. This folder drives it from Python: no server, no workflow JSON.

| File | What it is |
|---|---|
| `lib/seedvr2_core.py` | the machinery as functions; every path and setting is passed in |
| `seedvr2_upscaler.py` | `# %%` cells: set `INPUT`, run |

## Chain

```
lanczos resize -> SeedVR2Preprocess -> VAEEncode(Tiled) -> SeedVR2TemporalChunk
  -> per chunk: SeedVR2Conditioning + KSampler -> SeedVR2TemporalMerge
  -> VAEDecode(Tiled) -> SeedVR2PostProcessing -> CreateVideo -> Video.save_to
```

Sampling is one euler step at cfg 1, denoise 1.0 - the official template's numbers, and the
model itself (adversarial one-step post-training). More steps re-noise a restored frame.

## Models

```
models/diffusion_models/seedvr2_7b_convrot_w4a8.safetensors   4.44 GB  (ussoewwin W4A8)
models/vae/seedvr2_ema_vae_fp16.safetensors                    478 MB  (numz/SeedVR2_comfyUI)
```

Both are found by key shape, not filename: `blocks.35.mlp.vid.proj_out.weight` with
out-features 3072 for the 7B, `decoder.up_blocks.2.upsamplers.0.upscale_conv.weight` for the
VAE. Verified here: core loads the W4A8 file as `NaDiT`, 36 blocks, `ModelType.FLOW`,
SeedVR2 latent format (16 channels).

## Contracts the graph hides

- `SeedVR2TemporalChunk`'s `chunking_mode` is a DynamicCombo: pass a dict
  `{"chunking_mode": "auto"}`, not a string.
- `SeedVR2TemporalMerge`'s `temporal_overlap` arrives as a **list** (it is fed by an output
  list): index `[0]` inside, so pass `[overlap]`.
- `VAEEncodeTiled` / `VAEDecodeTiled` live in `nodes.py`, not `comfy_extras/nodes_mask.py`.
- `SeedVR2Preprocess` pads H,W to multiples of 16 and the frame count to 4n+1 by repeating
  the last frame; `SeedVR2PostProcessing` trims back to the reference you give it, which must
  be the resized original **before** Preprocess.
- `SaveVideo` reads `cls.hidden` - headless, use `Video.save_to`.

## Two paths, one working

| path | entry point | status on a 32 GB card |
|---|---|---|
| **pack** (`upscale_pack`) | `SeedVR2LoadDiTModel` + `SeedVR2LoadVAEModel` + `SeedVR2VideoUpscaler` from the ussoewwin fork | **works**: 124 frames 416x576 -> 832x1152 in **58.7 s**, peak 19.8 GB |
| core (`upscale`) | `SeedVR2Preprocess/Conditioning/TemporalChunk/TemporalMerge/PostProcessing` + `KSampler` | short clips only: the VAE pins ~25 GiB and does not release it |

### How the memory wall was solved: BlockSwap

ComfyUI core has no BlockSwap module - DynamicVRAM is its nearest equivalent, and it pins the
DiT and keeps it pinned (`--vram-headroom 14` changed nothing measurable). The SeedVR2 pack
does have it, and that is the technique behind the working GUI workflow
`user/default/workflows/upscale/SeedVR2_Video_Upscaling_az.json` (`SeedVR2BlockSwap` with
`blocks_to_swap` over an offload device). So the Python version drives the pack's nodes:

```
blocks_to_swap=16, swap_io_components=True, offload_device="cpu"
encode_tiled=True, decode_tiled=True, tile 512 / overlap 64
```

Measured per phase, 124 frames at 832x1152 (pack's own debug counters):

| phase | time | VRAM allocated |
|---|---|---|
| VAE encoding | 12.6 s | 10.95 GB |
| DiT upscaling | 15.3 s | 7.75 GB (BlockSwap saved 6.91 GB, 80 swaps) |
| VAE decoding | 28.1 s | 19.78 GB |
| post-processing | 2.5 s | 5.09 GB |
| **total** | **58.7 s** | peak 19.78 GB / 26.73 GB reserved |

For the same job the MiniMax H3 + Tessera refine took 145-164 s.

## Contracts the graph hides

- `SeedVR2VideoUpscaler`'s `resolution` is the **shortest edge** of the output, not the
  longest and not a multiplier. Passing 1152 for a 416x576 clip asks for 1152x1592 and OOMs.
- The pack's loaders stamp their config with `get_executing_context().node_id`, which only
  exists under the graph executor. Headless, patch that symbol with a stub (`_StubContext`).
- `SeedVR2LoadVAEModel` returns **one** config dict feeding both encode and decode.
- `batch_size` is frames per temporal window (4n+1); 41 fit at 832x1152, 21 did not.
- Core-node path, if you use it: `SeedVR2TemporalChunk`'s `chunking_mode` is a dict
  `{"chunking_mode": "auto"}`; `SeedVR2TemporalMerge`'s `temporal_overlap` arrives as a
  **list**; `VAEEncodeTiled`/`VAEDecodeTiled` live in `nodes.py`; `SeedVR2Preprocess` pads
  H,W to 16 and frames to 4n+1, and `SeedVR2PostProcessing` trims to the reference you pass,
  which must be the resized original **before** Preprocess; `SaveVideo` needs `cls.hidden`,
  so headless use `Video.save_to`.

## Core-path ceiling, for the record

| geometry | result |
|---|---|
| 13 frames @ 624x864 (x1.5) | works end to end, 48 s |
| 25 frames @ 576x416 encode | 15.9 GiB peak |
| 5 frames @ 832x1152 encode | 12.0 GiB peak |
| 9 frames @ 832x1152 encode | OOM |
| 124 frames @ 832x1152, 1-frame segments | 53 of 124 segments, then OOM at 25.6 GiB live |

Core's own `memory_used_encode` estimate for those shapes is 1.1-2.8 GiB and frame-independent,
under-predicting by roughly 10x - worth reporting upstream, but the pack path makes it moot.
