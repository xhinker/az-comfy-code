#!/usr/bin/env python3
"""Clip input for ReCaVSR: what the file is, and a reader that hands over one block at a time.

The model streams - 21 frames, then 8 at a time - so the reader decodes forward only and keeps
just the frames the current block still needs. A 1000-frame clip therefore costs disk, not RAM.
Audio is read once, whole, and handed to the writer for the mux.
"""
from __future__ import annotations

import pathlib
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
