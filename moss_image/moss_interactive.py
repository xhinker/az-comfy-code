# %% [markdown]
# # MOSS-Image-SFT, interactive
#
# Run the cells in order. Cell 1 loads the weights once (~6 s); after that, edit the
# prompt in cell 3 or the knobs in cell 4 and re-run - nothing is reloaded.
# Generation is split into three cells so each stage can be run and inspected alone:
# text -> conditioning, conditioning -> latents, latents -> pixels.

# %% 1. load: DiT + Qwen3.5 text & vision encoder + VAE + tokenizer
import os
import pathlib
import sys

GPU = "0"                       # the RTX 5090; set before torch is imported
os.environ.setdefault("CUDA_VISIBLE_DEVICES", GPU)

# Anchored to this file, not to the kernel's working directory and not to the old
# az_samples copy: sys.path and the output folder both follow this repo.
# (__file__ is missing in some interactive kernels, hence the cwd fallback.)
BASE = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path.cwd()
HERE = BASE                                   # import path for moss_image_gen
OUT_DIR = BASE / "outputs"                    # generated images land here (git-ignored)
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

import moss_image_gen as moss
print(f"moss_image from: {HERE}\nimages to:     {OUT_DIR}")

# fp16 for the DiT - the LoRA needs finer spacing than bf16 gives it - and bf16 for the
# text encoder, which is the pairing checked against the fp32 reference. dtype alone
# would move the encoder too, and encoder precision lands in the conditioning.
pipe, _ = moss.build_pipeline(dtype="bf16", te_dtype="bf16")
# "model" stages each component in and out; "sequential" (1.9 GiB) shares the card with ComfyUI
moss.set_memory_mode(pipe, "model", str(moss.DISK_CACHE))

# %% 2. LoRA - optional, and re-runnable.
# Change the path or the strength and run the cell again to swap adapters; the base
# weights are untouched either way. Set LORA_PATH = "" to run without any adapter.
# LORA_PATH = [str(moss.TURBO_LORA)]  # 8-step DMD distillation of MOSS-Image-SFT
# LORA_STRENGTH = [1.0]               # the authors' recommended strength
# moss.apply_lora(pipe.transformer, LORA_PATH, LORA_STRENGTH)

#%%
# Lists stack. Every adapter adds its own term to the module it edits, so the combined
# effect is the sum of the deltas and the order in the list does not matter. Keep the
# distillation card in the list whenever STEPS is small - it is what makes 8 steps work -
# and put style LoRAs after it at reduced strength. Re-running a cell replaces everything
# the previous call attached, so each cell states a complete stack.
LORA_PATH = [
    str(moss.TURBO_LORA),
    "/mnt/data_2t_3/github_repos/ComfyUI/models/loras/moss_image_sft/MOSS-Image-NSFW-Uncensored-LoRA-ComfyUI.safetensors"
]
LORA_STRENGTH = [0.9,0.75]     # sliders are strong; 0.3-0.5 is the useful range
moss.apply_lora(pipe.transformer, LORA_PATH, LORA_STRENGTH)
print('lora load done')

# %% 2b. compile the DiT once - one-time warmup, then ~1.6x on every denoise.
# Re-run this cell only if you change the LoRA stack above (the compile captures the
# adapters). Skip it for a single image - the warmup outweighs one fast denoise.
moss.compile_dit(pipe)

# %% 3. text -> conditioning. The text encoder's ~19 GiB is handed back at the end of
#     this cell, so the rest of the session runs without it. Re-run only when the prompt
#     changes; pass NEGATIVE = "" if you intend to run with CFG, so the negative branch
#     is encoded now and the encoder never has to come back.

prompt = """\
a flying cat
"""

NEGATIVE = None                     # "" if CFG > 1, otherwise None
conditioning = moss.encode_text(pipe, prompt, negative=NEGATIVE)

# %% 4. conditioning -> latents. The denoising loop alone, no VAE. Edit these and re-run
#     as often as you like; the prompt is not re-encoded.
HEIGHT,WIDTH        = 1680,1280
HEIGHT              = (HEIGHT//16)*16
WIDTH               = (WIDTH//16)*16                # both must be multiples of 16
STEPS, CFG, SEED    = 10, 1.0, 120                  # upstream non-distilled defaults: 50 steps, CFG 3
SKIP_LAYERS         = [] if CFG <= 1.0 else [22]    # the negative-branch skip only applies with CFG
KEEP_DIT            = True                          # leave the 9.5 GiB DiT on the GPU for the next run;
                                    # False releases it for whatever else needs the card

for i in range(5):
    SEED = SEED+i
    latents = moss.generate_latent(pipe, conditioning, height=HEIGHT, width=WIDTH, steps=STEPS,
                                cfg=CFG, seed=SEED, skip_layers=SKIP_LAYERS, keep_dit=KEEP_DIT)

    # 5. latents -> pixels. Cheap, so re-decode the same latents freely (a different
    #     seed in cell 4 gives a different image from the same conditioning).
    image = moss.decode_latent(pipe, latents, out=OUT_DIR / f"interactive_{SEED}.png")
    image
