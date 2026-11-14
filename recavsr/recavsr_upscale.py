#!/usr/bin/env python3
"""recavsr_upscale - the same pipeline as recavsr_interactive.py, as a CLI.

    ./recavsr_upscale.py INPUT -o OUT.mp4 --scale 4 --weights int8 --max-frames 96

Run it with the ComfyUI venv (see lib/recavsr_load.py):
    /mnt/data_2t_3/github_repos/ComfyUI/venv_py312_comfyui/bin/python recavsr_upscale.py ...
"""
import argparse
import math
import os
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "lib"))


def main():
    p = argparse.ArgumentParser(description="ReCaVSR one-step streaming video super-resolution")
    p.add_argument("input", help="input video")
    p.add_argument("-o", "--output", help="output mp4 (default: outputs/<stem>_x<scale>.mp4)")
    p.add_argument("--scale", type=float, default=4.0,
                   help="output size = input x scale. Trained for 4; 2 and 1.5 are the reference's others")
    p.add_argument("--weights", default="int8", choices=("int8", "bf16"),
                   help="int8 ConvRot: 6.0 GB, ~1.8x faster, ~45 dB from bf16. bf16: 10.1 GB")
    p.add_argument("--decoder", default="flashdecoder", choices=("flashdecoder", "vae"),
                   help="flashdecoder: the model's own, streaming, 25-40x faster. vae: the stock "
                        "Wan2.2 VAE, finer texture, needs vae.safetensors")
    p.add_argument("--kernels", default="fast", choices=("fast", "exact"),
                   help="fast: ComfyUI's fused norm/rope/residual kernels and retuned attention tiles; "
                        "exact: the reference op sequence, bit for bit its output")
    p.add_argument("--kv-offload", action="store_true",
                   help="keep the attention history in system RAM instead of VRAM: needed for big "
                        "frames on a 24 GB card, costs transfer time at every step")
    p.add_argument("--color-fix", default="none", choices=("none", "adain", "wavelet"),
                   help="match the output's colour back to the input; a one-step generative model drifts")
    p.add_argument("--encoder", default="auto", help="auto | hevc nvenc | h264 nvenc | h264")
    p.add_argument("--seed", type=int, default=42, help="seed of the single denoising step")
    p.add_argument("--max-frames", type=int, default=0, help="0 = whole clip")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--post-size", default="",
                   help="shrink the model output afterwards: 0.5, a target height like 720, or 1248x720. "
                        "Lanczos, after the model - so x4 rendered big then shrunk beats a smaller ratio")
    p.add_argument("--keep-latents", default="", help="also write the generated latents to this .pt path")
    p.add_argument("--gpu", default="", help="physical card to use, e.g. 2 (sets CUDA_VISIBLE_DEVICES)")
    args = p.parse_args()

    src = pathlib.Path(args.input)
    if not src.is_file():
        raise SystemExit(f"no such file: {src}")
    if args.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)      # before torch is imported

    import recavsr_load as L
    from recavsr_media import parse_size, probe

    info = probe(src)
    model_h = math.floor(info["height"] * args.scale + 0.5)
    model_w = math.floor(info["width"] * args.scale + 0.5)
    out_w, out_h = model_w, model_h            # what actually gets written, before padding
    post = None
    if args.post_size:
        w, h = parse_size(args.post_size, model_w, model_h)
        post, (out_w, out_h) = (h, w), (w, h)
    # the name carries the ratio AND the written resolution, so a 4x render shrunk to 2x size
    # cannot be mistaken for a native 2x pass when the two sit in the same folder
    name = f"{src.stem}_x{args.scale:g}_{out_w}x{out_h}.mp4"
    out = pathlib.Path(args.output) if args.output else BASE / "outputs" / name
    out.parent.mkdir(parents=True, exist_ok=True)

    e = L.engine()
    L.load(e, args.weights)
    if args.decoder == "vae":
        L.load_vae(e)
    from recavsr_engine import upscale as run
    stats = run(e, str(src), str(out), scale=args.scale, color_fix=args.color_fix, seed=args.seed,
                decoder=args.decoder, kernels=args.kernels, kv_offload=args.kv_offload,
                encoder=args.encoder, max_frames=args.max_frames, start_frame=args.start_frame,
                post_size=post, keep_latents=args.keep_latents)
    print(L.headroom())
    L.free(e, verbose=False)
    return stats


if __name__ == "__main__":
    main()
