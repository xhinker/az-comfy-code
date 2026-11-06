#!/usr/bin/env python3
"""Video in / video out for the RelayVSR scripts.

Frames come in as torch uint8 (3, H, W) - what relayvsr_engine.restore() wants - and go
out as HWC uint8 numpy arrays, which is what the encoder wants. Two passes are used for
the file: PyAV encodes the video (its bundled ffmpeg has libx264, the system ffmpeg on
this box does not), then imageio-ffmpeg muxes the original audio track back in without
re-encoding it. A clip with no audio, or an audio codec the container will not take,
just comes out silent.
"""
from __future__ import annotations

import pathlib
import shutil
from fractions import Fraction
import subprocess
import tempfile

import av
import numpy as np
import torch


def video_info(path) -> dict:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 24.0
        frames = stream.frames or 0
        return {
            "width": stream.width,
            "height": stream.height,
            "fps": fps,
            "frames": frames,
            "has_audio": bool(container.streams.audio),
        }


def iter_frames(path, max_frames: int = 0, start_frame: int = 0):
    """Yield torch uint8 (3, H, W) frames, in decode order."""
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        seen = 0
        emitted = 0
        for frame in container.decode(stream):
            if seen < start_frame:
                seen += 1
                continue
            arr = frame.to_ndarray(format="rgb24")
            yield torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
            seen += 1
            emitted += 1
            if max_frames and emitted >= max_frames:
                break
    finally:
        container.close()


class VideoWriter:
    """Encode HWC uint8 frames to mp4/webm; optionally mux the source audio back in."""

    def __init__(self, path, fps: float, codec: str = "h264", crf: int = 18,
                 preset: str = "medium", audio_from=None):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.codec = codec
        self.crf = int(crf)
        self.preset = preset
        self.audio_from = pathlib.Path(audio_from) if audio_from else None
        self.frames = 0
        self.size = None
        self._container = None
        self._stream = None

    def __enter__(self):
        return self

    def _open(self, width: int, height: int):
        self.size = (width, height)
        self._container = av.open(str(self.path), mode="w")
        options = {"crf": str(self.crf)}
        if self.codec in ("h264", "libx264"):
            options["preset"] = self.preset
        # av wants an exact rational frame rate; a bare float raises inside to_avrational.
        self._stream = self._container.add_stream(self.codec, rate=Fraction(self.fps).limit_denominator(1000))
        self._stream.width, self._stream.height = width, height
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = options

    def write(self, hwc_uint8):
        arr = np.ascontiguousarray(hwc_uint8, dtype=np.uint8)
        if self.size is None:
            self._open(arr.shape[1], arr.shape[0])
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self.frames += 1

    def close(self):
        if self._container is None:
            return
        if self._stream is not None:                 # don't mask the real error if _open failed
            for packet in self._stream.encode():
                self._container.mux(packet)
        self._container.close()
        self._container = None
        if self.audio_from is not None:
            self._mux_audio()

    def __exit__(self, *exc):
        self.close()
        return False

    # ------------------------------------------------------------------ audio
    def _mux_audio(self):
        """Copy the source audio track onto the encoded video. Re-encode only if needed."""
        ffmpeg = _ffmpeg_exe()
        if not ffmpeg:
            print("no ffmpeg available - the output stays silent")
            return
        tmp = tempfile.NamedTemporaryFile(suffix=self.path.suffix or ".mp4", delete=False, dir=str(self.path.parent))
        tmp.close()
        self.path.replace(tmp.name)
        base = [ffmpeg, "-y", "-loglevel", "error", "-i", tmp.name, "-i", str(self.audio_from),
                "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-shortest"]
        ok = subprocess.run(base + ["-c:a", "copy", str(self.path)], capture_output=True, text=True)
        if ok.returncode != 0:
            subprocess.run(base + ["-c:a", "aac", "-b:a", "192k", str(self.path)], capture_output=True, text=True)
        pathlib.Path(tmp.name).unlink(missing_ok=True)


def _ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")
