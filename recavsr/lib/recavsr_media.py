#!/usr/bin/env python3
"""Clip input for ReCaVSR: what the file is, and a reader that hands over one block at a time.

The model streams - 21 frames, then 8 at a time - so the reader decodes forward only and keeps
just the frames the current block still needs. A 1000-frame clip therefore costs disk, not RAM.
Audio is read once, whole, and handed to the writer for the mux.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
from fractions import Fraction

import av
import numpy as np


def probe(path) -> dict:
    """width, height, fps, frames, has_audio, codec of the first video stream."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        frames = int(stream.frames or 0)
        if not frames and container.duration:
            frames = int(round(container.duration / 1e6 * fps))
        if not frames:                                    # containers that lie about their count
            frames = _count_packets(path) or _count_decoded(path)
        return {"width": stream.width, "height": stream.height, "fps": fps, "frames": frames,
                "has_audio": bool(container.streams.audio), "codec": stream.codec_context.name}


def _count_packets(path) -> int:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                              "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, check=True).stdout.strip()
        return int(out)
    except Exception:
        return 0


def _count_decoded(path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(container.streams.video[0]))


class BlockFrames:
    """Forward-only decoder: get(start, stop) -> uint8 [n, h, w, 3]. Ask in order; the frames
    behind you are dropped as soon as they are no longer needed."""

    def __init__(self, path):
        self.path = str(path)
        info = probe(self.path)
        self.total, self.info = info["frames"], info
        self.fps = Fraction(info["fps"]).limit_denominator(10000) if info["fps"] else Fraction(24)
        self.container = av.open(self.path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.frames = (frame for frame in self.container.decode(self.stream))
        self.cache: list[np.ndarray] = []
        self.first = 0

    def get(self, start: int, stop: int) -> np.ndarray:
        while self.first + len(self.cache) < stop:
            frame = next(self.frames, None)
            if frame is None:
                break
            self.cache.append(frame.to_ndarray(format="rgb24"))
        if start > self.first:
            del self.cache[:start - self.first]
            self.first = start
        return np.stack(self.cache[start - self.first:stop - self.first])

    def close(self) -> None:
        self.container.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_audio(path, first_frame: int = 0, frames: int | None = None, fps: float = 0.0):
    """(samples [C, S] float32, rate) of the first audio stream, trimmed to the window that was
    rendered, or None when the clip has no audio. ReCaVSR carries audio through untouched."""
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp")
        chunks = [f.to_ndarray() for f in container.decode(stream) for f in resampler.resample(f)]
        if not chunks:
            return None
        samples, rate = np.concatenate(chunks, axis=1), stream.rate
        if first_frame and fps:
            samples = samples[:, int(first_frame / fps * rate):]
        if frames and fps:
            samples = samples[:, :int(frames / fps * rate)]
        return samples, rate


def parse_size(text: str, base_width: int, base_height: int) -> tuple[int, int]:
    """A post-model output size: "1248x720" exact, "720" a target height, "0.5" a factor."""
    text = text.strip().lower()
    if "x" in text:
        w, h = text.split("x", 1)
        return max(2, int(w)), max(2, int(h))
    if 0 < float(text) < 8 and any(c in text for c in "."):
        factor = float(text)
        return max(2, round(base_width * factor)), max(2, round(base_height * factor))
    height = int(text)
    return max(2, round(base_width * height / base_height)), height


def list_inputs(folder="inputs") -> list[pathlib.Path]:
    """Videos in a folder, sorted - what the interactive cell offers as INPUT."""
    root = pathlib.Path(folder)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir()
                  if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm", ".avi") and p.stat().st_size > 0)


def shrink(src, size: str = "0.5", dst=None, crf: int = 20, keep_original: bool = True) -> pathlib.Path:
    """Re-encode an already-rendered clip at a smaller size, without touching the model.

    `size` is "0.5" (half each way), "720" (a target height) or "1248x720" (exact) - the same
    syntax as --post-size. Lanczos on the pixels, then hevc: NVENC when it opens on this box,
    otherwise x265. Audio is stream-copied, so nothing about the sound changes.

    This is the cheap way to keep a 4x render: the model already did the work, and a shrink pass
    costs seconds on the encoder instead of another render. It is NOT the same as asking the
    model for a smaller ratio - 4x then shrunk keeps more detail than rendering at 2x.
    """
    src = pathlib.Path(src)
    if not src.is_file():
        raise SystemExit(f"shrink: no such file: {src}")
    if not shutil.which("ffmpeg"):
        raise SystemExit("shrink: ffmpeg is not on this kernel's PATH - the shrink pass is an "
                         "ffmpeg re-encode, it does not run through PyAV")
    info = probe(src)
    width, height = parse_size(str(size), info["width"], info["height"])
    width &= ~1                      # hevc wants even dimensions
    height &= ~1
    if (width, height) == (info["width"], info["height"]):
        print(f"shrink: {size} is already {info['width']}x{info['height']}, nothing to do")
        return src
    dst = pathlib.Path(dst) if dst else src.with_name(f"{src.stem}_to{width}x{height}.mp4")
    tmp = dst.with_suffix(".part.mp4")
    scale = f"scale={width}:{height}:flags=lanczos"
    encoders = [("hevc_nvenc", ["-rc", "vbr", "-cq", str(crf), "-b:v", "0", "-preset", "p5", "-tune", "hq"]),
                ("libx265", ["-crf", str(crf), "-preset", "medium"])]
    last = ""
    for codec, opts in encoders:
        cmd = (["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", scale, "-c:v", codec, *opts,
                "-pix_fmt", "yuv420p", "-tag:v", "hvc1", "-c:a", "copy", str(tmp)])
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 1024:
            break
        last = done.stderr.strip().splitlines()[-1] if done.stderr.strip() else codec + " failed"
        tmp.unlink(missing_ok=True)
    else:
        raise RuntimeError(f"no hevc encoder produced a file: {last}")
    out = probe(tmp)
    if out["frames"] and out["frames"] != info["frames"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"shrink lost frames: {info['frames']} -> {out['frames']}")
    before = src.stat().st_size          # read it before anything is removed
    tmp.replace(dst)
    if not keep_original and dst != src:
        src.unlink()
    print(f"shrink {info['width']}x{info['height']} -> {width}x{height}   "
          f"{before / 2**20:.0f} MB -> {dst.stat().st_size / 2**20:.0f} MB   ({dst.name})"
          + ("" if keep_original else "   big render removed"))
    return dst
