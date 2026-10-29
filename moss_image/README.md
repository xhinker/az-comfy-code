# MOSS-Image-SFT in diffusers, from local ComfyUI files

Generates with MOSS-Image-SFT through diffusers **without downloading any
weights**. Everything comes off the ComfyUI installation on this machine.

## Why this directory exists

diffusers has no MOSS pipeline. Verified against the installed 0.41.0: no
`MossImagePipeline`, no MOSS transformer, no MOSS module - and the model card
itself loads the model with `custom_pipeline=`, which is Hub-hosted code rather
than core. So the pipeline is assembled here from the official model definition.

`moss_def/` holds that definition, vendored from `OpenMOSS-Team/MOSS-Image-SFT`:
**code and configuration only, no weights** (~40 KB total).

## Where each component comes from

| Component | Source | Load result |
|---|---|---|
| transformer | ComfyUI `models/diffusion_models/moss_image_sft/moss_image_sft_bf16.safetensors` | 463/463 tensors |
| text encoder | ComfyUI `models/text_encoders/moss_image_sft/qwen3.5_9b_bf16.safetensors` (converted, see below) | 759/759 |
| vae | ComfyUI `models/vae/moss_image_sft/flux2-vae.safetensors` | 251/251 |
| tokenizer | ComfyUI `comfy/text_encoders/qwen35_tokenizer` (vocab + merges) with the official chat template in `moss_def/tokenizer/` | 248053 vocab |
| scheduler | `moss_def/scheduler/scheduler_config.json` | shift 5.0 |

`load_into()` fails the run if any tensor in a file has nowhere to go, or if any
model tensor has no weights, so a silent partial load is not possible.

## Converting the text encoder (one time)

The only text encoder on this machine was `qwen3.5_9b_int8_convrot.safetensors`,
ComfyUI's private int8 format: `{"format": "int8_tensorwise", "convrot": true,
"convrot_groupsize": 256}` per layer. `convert_text_encoder.py` dequantises it
with **ComfyUI's own kernels** (`comfy_kitchen` `TensorWiseINT8Layout`, reached
through `comfy.quant_ops`), so the output is what ComfyUI computes at runtime
rather than a guess at it. It must run under ComfyUI's interpreter, because
`comfy.ops` needs ComfyUI's native `comfy_aimdo` extension:

```bash
/mnt/data_2t_3/github_repos/ComfyUI/venv_py312_comfyui/bin/python convert_text_encoder.py --self-test
/mnt/data_2t_3/github_repos/ComfyUI/venv_py312_comfyui/bin/python convert_text_encoder.py
```

Output: `qwen3.5_9b_bf16.safetensors` next to the original - 19.31 GB, 775
tensors, 33 seconds. CPU and CUDA backends agree bit for bit on the same layer.

## Name mapping, and why each one is needed

The ComfyUI files are not in the layout diffusers expects. Three separate
mappings, each in one function:

- **`split_fused_qkv`** - ComfyUI stores each block's q/k/v as one `proj.qkv`
  matrix; the official class has three `Linear`s. Splitting rows in order is
  what `qkv(x).chunk(3, -1)` does at runtime.
- **`to_base_model_keys`** - the checkpoint is the full Qwen3_5
  conditional-generation model. The pipeline only calls the base model for
  `last_hidden_state`, so the `model.` prefix is stripped and `lm_head` (logits)
  and `mtp` (multi-token prediction) are dropped: neither is reachable from
  `encode_prompt`.
- **`ldm_to_diffusers_vae_keys`** - the VAE is in original LDM/CompVis naming.
  Besides module names, two things are easy to get wrong: the decoder counts its
  up-blocks from the image end while diffusers counts from the latent end, so
  those indices are **reversed** (the `nin_shortcut` shapes only line up that way
  round), and LDM attention projections are 1x1 convolutions where diffusers uses
  `Linear`, so they are squeezed. The repo config also describes the reduced
  "small decoder"; this file carries the full 512-channel FLUX.2 decoder, which
  is the better one, so the width is taken from the file.

## Running

```bash
V=/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/samplecode_venv_p311_2
CUDA_VISIBLE_DEVICES=0 $V/bin/python moss_image_gen.py \
  "a red apple on a white ceramic table, soft studio light" --memory model
```

Upstream defaults are used unless overridden: 1024x1024, 50 steps, CFG 3,
scheduler shift 5, negative-branch layer skip 22, empty negative prompt,
dimensions multiples of 16 (the script refuses anything else).

## Memory modes

| `--memory` | Peak VRAM | Notes |
|---|---|---|
| `full` | ~30 GB | everything resident; needs a free 32 GB card |
| `model` | **19.9 GiB** | whole-model offload along `text_encoder->transformer->vae`; 1024x1024 at 50 steps in 136 s on the 5090 |
| `sequential` | **1.9 GB** | leaf-level offload; measured, runs beside a loaded ComfyUI |
| `stream` | lower than `model` | block streaming on top of model offload |
| `disk` | lower still | same, with weights staged in `.offload_cache/` |

`auto` picks `full` above 40 GiB free VRAM (upstream's threshold) and `model`
below it.

## Turbo mode: 8 steps instead of 50

```bash
CUDA_VISIBLE_DEVICES=0 $V/bin/python moss_image_gen.py "a bowl of persimmons" --turbo
```

`--turbo` forces 8 steps, CFG 1.0, no layer skip, and loads
`ComfyUI/models/loras/moss_image_sft/lora.safetensors` - lodestones' 8-step
data-space DMD distillation of MOSS-Image-SFT (rank 128, alpha 128, fp32).
Settings are the ones the authors recommend: Euler flow matching, shift 5,
strength 1.0, no CFG. Measured **17 s for 8 steps at 1024²**.

The LoRA is applied **unmerged, with the low-rank term in fp32**, on purpose: the
distillation delta is smaller than bf16's rounding step, so merging it into the
bf16 transformer discards most of the distillation. `--lora PATH` and
`--lora-strength` give the same mechanism for any kohya/ComfyUI adapter
(`diffusion_model.<module>.lora_down.weight` / `lora_up.weight` / `alpha`).

Turbo LoRAs trade quality and seed variety for speed. Community reports are mixed
and the effect shows on faces and fine texture; for a keeper, run the 50-step
path. Use one turbo LoRA at a time.

## Loading speed

The build used to take 76 s. Measured breakdown of that:

| Step | Time |
|---|---|
| read both weight files (29 GB, safetensors mmap) | **0.0 s** |
| copy the weights into the model | 2.4 s |
| random-initialise 14.6 B parameters | **70.5 s** |

So 93% of it was filling 14.6 billion parameters with random numbers that were
overwritten a moment later. That is also the difference from ComfyUI: it never
initialises weights it is about to replace, it maps the file and assigns into
the model, and it keeps weights in the file's own dtype (its int8 text encoder is
9.7 GB to touch, not 19.3 GB).

`skipping_random_init()` neutralises the random fills during construction and
`load_into()` zeroes anything the checkpoint does not supply, so nothing is left
holding uninitialised memory. Deterministic fills such as the RMSNorm `ones_`
still run. Build is now **5.7 s**.


## The tokenizer bug, and why every image was wrong

The first working version produced sharp, well-exposed, completely wrong images at
every step count, with and without the LoRA. Cause: the chat and thinking tokens in
the prompt template are **added tokens**, not vocabulary entries. ComfyUI's
`tokenizer_config.json` registers 33 of them with their ids; the official MOSS repo
relies on a prebuilt `tokenizer.json` instead, which was not vendored here. With the
official 1.1 KB config and no registry, those strings have no ids and are not in
`vocab.json` either, so the BPE encoder split them into literal characters:

    ComfyUI   12 tokens   [turn_open] user 
 beautiful girl [turn_close] 
 ...
    this repo 24 tokens   '<' '|' 'im' '_start' '|' '>' user 
 beautiful girl ...

The text encoder never saw a prompt, so the transformer denoised toward the mean of
its training distribution. Everything downstream - weights, VAE, scheduler - was
fine, which is exactly why the images looked like photographs.

`ensure_tokenizer()` now copies ComfyUI's `tokenizer_config.json` as authoritative,
applies the official chat template explicitly, and **fails the run if any control
token in the rendered template does not encode to exactly one id**. That check is the
point: "the template rendered" is not a test, "the template tokenises to the same ids
as the reference engine" is.

To compare against ComfyUI directly:

    cd /mnt/data_2t_3/github_repos/ComfyUI && ./venv_py312_comfyui/bin/python /tmp/tok_both.py comfy
    $V/bin/python /tmp/tok_both.py diffusers

## Sampler and scheduler, for the record

ComfyUI's `moss_image_sft_simple.json` runs `euler_cfg_pp` with the `beta` scheduler,
shift 5, 8 steps, CFG 1, LoRA at 1.0. This script uses the model authors' own
`MOSSImageScheduler` instead: a shifted grid snapped onto the training grid, stepped
with plain flow-matching Euler. Both are legitimate; the upstream one is the default
here because it is what the release pipeline ships.

## Interactive session

`moss_interactive.py` is a VS Code Python Interactive file (cells marked `# %%`) that
does the same thing in three steps: load the weights once, optionally apply a LoRA,
then generate. Cell 1 takes ~6 s; after that, edit `PROMPT` in cell 3 and re-run it -
each image costs only the denoising (27 s at 1024²/8 steps), with no reload.

Cell 2 is re-runnable: change `LORA_PATH` or `LORA_STRENGTH` and run it again to swap
adapters, or set `LORA_PATH = ""` to drop them. This works because `apply_lora` attaches
**forward hooks** instead of replacing modules, so the model's structure - and any
offload hooks accelerate put on it - survives being re-run. Verified: LoRA at 1.0,
swapped live to 0.6, then removed, three images, no reload, no errors.

`generate_image()` is the same function the command line path calls, so the two cannot
drift apart.

## Loading LoRAs from elsewhere

`apply_lora` accepts both naming dialects you will actually meet:

| source | factor keys | alpha |
|---|---|---|
| kohya / ComfyUI | `<module>.lora_down.weight`, `.lora_up.weight` | `.alpha` tensors |
| ai-toolkit, PEFT, OneTrainer (most Civitai downloads) | `<module>.lora_A.weight`, `.lora_B.weight` | none |

A file with no `.alpha` tensors falls back to `alpha = rank`, which is scale 1.0 and the
same convention ComfyUI uses, so a LoRA that behaves a certain way there behaves the same
here. If a file matches nothing, the error now says how many factors it found and shows
the first keys, so you can tell "wrong dialect" from "wrong network".

Note that a rank-1 slider is an even more extreme version of the bf16 problem above than
the distillation adapter already is - its contribution is smaller still, so run these in
fp16 or fp32 or you may see no effect at all.

## DiT precision: fp32 is the reference, fp16 is the fast option, bf16 is broken

Set `DIT_DTYPE` in `moss_image_gen.py`, or pass `--dit-dtype fp32|fp16|bf16`. Measured at
1024x1024, 8 steps, turbo LoRA, seed 42, against the fp32 output:

| DiT weights | step time | weights | mean pixel diff vs fp32 | pixels changed >16 |
|---|---|---|---|---|
| `fp32` (default) | 2.6 s | 18.9 GiB | reference | - |
| `fp16` | 1.09 s | 9.5 GiB | 5.38 / 255 | 5.0% |
| `bf16` | 1.15 s | 9.5 GiB | **41.41 / 255** | **69.6%** |

bf16 is not a rounding difference, it is a different picture: faces come out wrong, and
the earlier version of this README recommended it for speed. That was wrong. The axis is
mantissa width - fp16 has 11 bits, bf16 has 8 - and fp16 sits close to fp32 while bf16
does not. fp16 is the one to reach for when you want the speed.

The bf16 failure is the LoRA, not the model. With the adapter off, bf16 and fp32 differ by
3.04/255 at 8 steps - harmless. With it on, 41.41. The reason is arithmetic: measured over
all 252 injection points, the median LoRA contribution is 0.10% of the activation it edits
while bf16's spacing is 0.195%, so **145 of the 252 adapters are smaller than one bf16 unit
and round to zero on every pass**. The distillation is mostly absent, and 8 steps of a
schedule tuned for a fully distilled model produces wrong structure - faces first. fp16's
spacing is 0.049%, four times finer, and the adapter survives.

This is the upstream card's warning ("the distillation delta is smaller than bf16's
rounding step") read correctly: it is about the arithmetic, not only about merging into
weights. Applying the adapter unmerged in fp32 does not dodge it, because the sum is still
rounded to bf16 on the way back into the residual stream. `apply_lora` now warns if you
attach an adapter to a bf16 model.

`torch.autocast("cuda", bf16)` over fp32 weights is what the code asks for and it is fast,
but autocast caches bf16 copies of every weight beside the fp32 originals: 28.3 GiB for
this DiT, which will not fit a 32 GiB card alongside the text encoder. It is not used.

`--memory resident` keeps the DiT and VAE on the GPU between steps. It measured no
faster than `model` (31.2s vs 31.4s for 12 steps) because accelerate was not
re-transferring the module per step after all; it is kept because it avoids restaging
9.5 GiB between generations in the interactive loop.

## Choosing precision

```python
pipe, _ = moss.build_pipeline(dtype="bf16",              # "bf16" | "fp16" | "fp32"
                              vae_dtype=torch.float32,   # fp32 is the authors' choice
                              te_dtype=None)             # None = follow dtype
```

`dtype` sets the transformer and, unless overridden, the text encoder; activations inside
both follow the weights. Two things that are not obvious from the signature:

- **Text encoder precision is a visible knob**, not an implementation detail - it lands
  directly in the conditioning. `dtype="fp16"` alone (encoder included) produces a
  different picture than `dtype="fp16", te_dtype="bf16"`, and the latter is the combination
  checked against the fp32 reference. `moss_interactive.py` uses it.
- **`dtype="fp32"` clamps the encoder to fp16** and says so: 9B parameters in fp32 is
  36.6 GiB, which cannot be staged on this card even with the DiT offloaded.

The denoising state stays fp32 regardless - the vendored pipeline pins it with `.float()`,
and accumulating the sampler in fp32 is worth keeping. Bad values are rejected up front:
`dtype must be one of ['bf16', 'fp16', 'fp32'], got 'int8'`.

## Three-stage generation, and keeping the DiT resident

Generation is split into three functions so each stage can run alone, which is what the
interactive file uses:

```python
conditioning = moss.encode_text(pipe, PROMPT, negative=None)   # text -> embeds, TE -> CPU
latents = moss.generate_latent(pipe, conditioning, steps=8, cfg=1.0, seed=42,
                               skip_layers=[], keep_dit=True)  # the denoising loop only
image = moss.decode_latent(pipe, latents)                       # VAE -> PIL
```

`encode_text` hands the text encoder's ~19 GiB back when it finishes, so a session that
encodes once and denoises many times never pays for it twice. Pass `negative=""` there if
you intend to run with CFG, otherwise the pipeline has to bring the encoder back for the
negative branch.

`keep_dit` on `generate_latent` decides whether the 9.5 GiB DiT stays on the GPU after the
run. Two things about it are not obvious. diffusers' `maybe_free_model_hooks()`
*reinstalls* accelerate's hooks at the end of every `__call__`, so the residency flag has
to be applied after the call or it is silently undone - `generate_latent` does this. And
accelerate chains the hooks so a module's `pre_forward` offloads the one before it in the
chain, which means the VAE decode pushes the DiT out; `decode_latent` notices the caller
asked for residency and restores it. Measured, 8 steps at 1024² including decode:
13.9s, then 10.9s, then 10.2s once nothing has to be restaged.

`generate_image()` is now built from these three, so the CLI and the notebook share one
path - verified bit-identical at the same seed (mean absolute pixel difference 0.000).

## Troubleshooting

- **`No module named comfy_aimdo`** - the converter was run with the wrong
  interpreter. Use ComfyUI's venv, not the diffusers venv.
- **`size mismatch for decoder.conv_in.weight`** - the config's small-decoder
  channels were used; the script derives the width from the file, so this means
  a different VAE file is configured.
- **Images are sharp but ignore the prompt** - tokenizer added-tokens are missing;
  `ensure_tokenizer()` should have caught it. Compare token ids with ComfyUI's.
- **`Unable to import torchao Tensor objects`** - harmless here. torchao 0.18
  removed `torchao.float8.Float8InferRole` and `torchao.dtypes.affine_quantized_tensor`,
  which diffusers 0.41 still tries to import, so its optional torchao path switches
  itself off. Only matters for checkpoints serialised *with* torchao; these are plain
  bf16 safetensors.
- **`causal_conv1d is not installed`** - informational. Qwen3.5's hybrid
  attention falls back to a reference implementation; slower, still correct.
- **Text encoder loads 0/759** - the converted file is missing or stale; rerun
  the converter.
