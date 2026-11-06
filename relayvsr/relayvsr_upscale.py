#!/usr/bin/env python3
"""relayvsr_upscale - the same pipeline as relayvsr_interactive.py, as a CLI.

    ./relayvsr_upscale.py INPUT -o OUT.mp4 --scale 4 --sr-ratio 4 --max-frames 128

Run it with the ComfyUI venv (see relayvsr_load.py).
"""
import argparse
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "lib"))

import relayvsr_load as L


def main():
    p = argparse.ArgumentParser(description="RelayVSR video super-resolution")
    p.add_argument("input", help="input video (or use --input)")
    p.add_argument("-o", "--output", help="output mp4 (default: outputs/<stem>_x<ratio>.mp4)")
    p.add_argument("--scale", type=int, default=4, choices=(2, 4), help="which weights to load")
    p.add_argument("--sr-ratio", type=int, default=4, choices=(2, 4), help="output ratio")
    p.add_argument("--chunk-len", type=int, default=16, help="frames per streaming window")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--detail", type=float, default=1.0,
                   help="0-1, how much of the decoder's detail is added on top of the bicubic "
                        "upscale; 1 = full model output (ComfyUI core's VSR Decode knob)")
    p.add_argument("--max-frames", type=int, default=0, help="0 = whole clip")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    p.add_argument("--offload", action="store_true", help="park Wan blocks in CPU RAM (24 GB cards)")
    p.add_argument("--codec", default="h264", choices=("h264", "h265", "libx264", "libx265"))
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--downscale", default="",
                   help="shrink the saved video: 0.5 = half the model output, 720 = target height, "
                        "1248x720 = exact size. Lanczos, applied after the model - so x4 weights "
                        "at --sr-ratio 4 downscaled to a 2x finish keep more detail than x2 weights")
    p.add_argument("--preset", default="medium")
    args = p.parse_args()

    src = pathlib.Path(args.input)
    if not src.is_file():
        raise SystemExit(f"no such file: {src}")
    if args.scale == 2 and args.sr_ratio != 2:
        raise SystemExit("x2 weights only produce a 2x output")
    out = pathlib.Path(args.output) if args.output else BASE / "outputs" / f"{src.stem}_x{args.sr_ratio}.mp4"

    engine = L.engine(scale=args.scale, dtype=args.dtype, offload=args.offload)
    L.load_wan(engine)
    L.load_flash(engine)

    frames, plan = L.run(engine, str(src), sr_ratio=args.sr_ratio, chunk_len=args.chunk_len,
                         seed=args.seed, detail=args.detail, downscale=args.downscale or None,
                         max_frames=args.max_frames, start_frame=args.start_frame)
    model = f" (model rendered {plan['model_size']}, downscaled {plan['downscale']})" \
        if "model_size" in plan else ""
    print(f"{plan['in_size']} -> {plan['out_size']} @ {plan['fps']:.2f} fps, "
          f"native x{plan['native_scale']}, chunk {plan['chunk_len']}{model}")
    path, written, seconds = L.save(frames, out, fps=plan["fps"], codec=args.codec, crf=args.crf,
                                    preset=args.preset, audio_from=src if plan["has_audio"] else None)
    print(f"saved {written} frames in {seconds:.1f} s ({written / seconds:.2f} fps) -> {path}")
    L.free(engine, verbose=False)


if __name__ == "__main__":
    main()
