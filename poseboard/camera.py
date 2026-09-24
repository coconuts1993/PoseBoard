"""Camera capture thread: every frame carries a perf_counter timestamp; video can be
written while capturing (with a per-frame timestamp file: perf_counter ``t``, ``t_rel`` and
Unix ``t_unix``).

Video is written as MPEG-4 in a Matroska container (``.mkv``): unlike MP4, whose index is
only written when the file is closed, an MKV file stays readable (up to the last few seconds
still in the writer's buffers) if the program is killed, crashes or loses power. If the MKV writer cannot be opened, MJPG in AVI is
used instead."""

from __future__ import annotations

import csv
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

VIDEO_EXT = ".mkv"
FLUSH_INTERVAL_S = 1.0  # timestamp CSV is flushed about once per second


@dataclass
class Frame:
    index: int
    t: float
    image: np.ndarray


def configure_capture(cap, width: int | None = None, height: int | None = None,
                      fps: float | None = None, mjpg: bool = False) -> None:
    """Request frame rate, size and (optionally) the MJPG format.

    The order matters for DirectShow: with no FOURCC set it prefers uncompressed YUY2, which
    USB2 webcams deliver at only 5-10 fps at 720p/1080p. Setting the FOURCC re-opens the device
    with the current size, and any later size or fps change falls back to YUY2, so the FOURCC
    is set last."""
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if mjpg:
        want = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, want)
        got = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        if got != want:
            log.info("camera did not accept MJPG (FOURCC %#x); the frame rate may be limited", got)
    if width and height:
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
        if all(size) and size != (int(width), int(height)):
            log.warning("camera delivers %dx%d instead of the requested %dx%d", *size, width, height)


def open_capture(source: int | str, width: int | None = None, height: int | None = None,
                 fps: float | None = None) -> cv2.VideoCapture:
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    dshow = isinstance(source, int) and sys.platform.startswith("win")
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if dshow else cv2.VideoCapture(source)
    configure_capture(cap, width, height, fps, mjpg=dshow)
    return cap


def open_video_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    """Open a video writer at ``path`` (.mkv, MPEG-4), falling back to MJPG in .avi.
    Returns (writer, actual path); raises RuntimeError if no writer can be opened."""
    path = Path(path)
    tries = [(path, "mp4v")]
    if path.suffix.lower() != ".avi":
        tries.append((path.with_suffix(".avi"), "MJPG"))
    for p, fourcc in tries:
        w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*fourcc), float(fps), size)
        if w.isOpened():
            if p != path:
                log.warning("could not write %s; recording to %s (MJPG) instead", path.name, p.name)
            return w, p
        w.release()
    raise RuntimeError(f"Cannot write video file {path} (codec/backend not available, or the "
                       "folder is not writable)")


class CameraStream:
    """Reads one camera on a background thread. source can be a device index, a video file
    or a network stream URL."""

    stop_timeout_s = 2.0

    def __init__(self, name: str, source: int | str, width: int | None = None,
                 height: int | None = None, fps: float | None = None):
        self.name = name
        self.source = source
        self.req = (width, height, fps)
        self.fps = fps or 30.0  # nominal (reported by the camera once opened)
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._writer: cv2.VideoWriter | None = None
        self._ts_file = None
        self._ts_writer = None
        self._rec_index = 0
        self._rec_offset = 0.0
        self._rec_t0 = 0.0
        self._rec_size: tuple[int, int] | None = None  # (w, h) of the video being written
        self._last_flush = 0.0
        self.video_path: Path | None = None  # file being (or last) recorded
        self.dropped_frames = 0  # frames not written (size changed during the recording)
        self.error: str | None = None  # last read or recording error (None while frames arrive)
        self.measured_fps = 0.0
        self._is_file = isinstance(source, str) and not source.isdigit() and Path(source).exists()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        cap = open_capture(self.source, *self.req)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open camera {self.source}")
        f = cap.get(cv2.CAP_PROP_FPS)
        if f and 1 < f < 1000:
            self.fps = f
        self._cap = cap
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(cap, self._stop),
                                        name=f"cam-{self.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop capturing. The capture thread releases the device itself when it leaves
        ``read()``; a read that blocks (e.g. a stalled network stream) is never released under
        its feet."""
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=self.stop_timeout_s)
            if t.is_alive():
                log.warning("camera %s is still blocked in read(); it is released when the read "
                            "returns", self.name)
        self._thread = None
        self.stop_recording()
        self._cap = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def frame_age(self, now: float | None = None) -> float | None:
        """Seconds since the last good frame (None before the first one)."""
        f = self._latest
        if f is None:
            return None
        return (time.perf_counter() if now is None else now) - f.t

    def _run(self, cap: cv2.VideoCapture, stop: threading.Event) -> None:
        try:
            self._loop(cap, stop)
        finally:
            cap.release()  # the thread owns the capture: never released during a read

    def _loop(self, cap: cv2.VideoCapture, stop: threading.Event) -> None:
        idx = 0
        last = time.perf_counter()
        period = 1.0 / self.fps if self._is_file else 0.0
        while not stop.is_set():
            ok, img = cap.read()
            t = time.perf_counter()
            if stop.is_set():
                break
            if not ok:
                if self._is_file:  # loop video files for easier debugging
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.error = "Failed to read frame"
                time.sleep(0.01)
                continue
            if self.error == "Failed to read frame":
                self.error = None
            frame = Frame(idx, t, img)
            idx += 1
            dt = t - last
            last = t
            if dt > 0:
                self.measured_fps = 0.9 * self.measured_fps + 0.1 / dt if self.measured_fps else 1 / dt
            failed = None
            with self._lock:
                self._latest = frame
                if self._writer is not None:
                    try:
                        self._write_frame(img, t)
                    except Exception as e:  # noqa: BLE001  (disk full, removed drive, ...)
                        failed = e
            if failed is not None:
                log.error("camera %s: recording stopped: %s", self.name, failed)
                self.error = f"recording stopped: {failed}"
                self.stop_recording()
            if period:
                time.sleep(max(0.0, period - (time.perf_counter() - t)))

    def _write_frame(self, img: np.ndarray, t: float) -> None:
        # caller holds self._lock
        h, w = img.shape[:2]
        if (w, h) != self._rec_size:
            # The writer silently drops frames of another size: do not log a timestamp for them
            if not self.dropped_frames:
                log.warning("camera %s: frame size changed to %dx%d during the recording (video "
                            "is %dx%d); these frames are not saved", self.name, w, h,
                            *self._rec_size)
            self.dropped_frames += 1
            return
        self._writer.write(img)
        self._ts_writer.writerow([self._rec_index, f"{t:.6f}", f"{t - self._rec_t0:.6f}",
                                  f"{t + self._rec_offset:.6f}"])
        self._rec_index += 1
        if t - self._last_flush >= FLUSH_INTERVAL_S:
            self._ts_file.flush()
            self._last_flush = t

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    # -------------------------------------------------------------- recording
    def start_recording(self, video_path: Path, clock_offset_unix: float | None = None,
                        t0: float | None = None) -> Path:
        """Write video to ``video_path`` and frame times to ``<stem>_timestamps.csv``
        (``frame``, ``t`` = perf_counter, ``t_rel`` = t - t0, ``t_unix`` = t + clock_offset_unix).
        Pass the session's offset and t0 so all files share them; by default they are taken now.
        Returns the path of the video actually written (``.avi`` if the fallback was used)."""
        if clock_offset_unix is None:
            clock_offset_unix = time.time() - time.perf_counter()
        if t0 is None:
            t0 = time.perf_counter()
        video_path = Path(video_path)
        f = self.latest()
        if f is None:
            raise RuntimeError(f"Camera {self.name} has no frames yet")
        h, w = f.image.shape[:2]
        writer, video_path = open_video_writer(video_path, self.fps, (w, h))
        try:
            ts_file = open(video_path.with_name(video_path.stem + "_timestamps.csv"), "w", newline="")
        except OSError:
            writer.release()
            raise
        ts_writer = csv.writer(ts_file)
        ts_writer.writerow(["frame", "t", "t_rel", "t_unix"])
        with self._lock:
            self._writer, self._ts_file, self._ts_writer, self._rec_index = writer, ts_file, ts_writer, 0
            self._rec_offset = float(clock_offset_unix)
            self._rec_t0 = float(t0)
            self._rec_size = (w, h)
            self._last_flush = time.perf_counter()
            self.dropped_frames = 0
            self.video_path = video_path
        return video_path

    def stop_recording(self) -> None:
        with self._lock:
            writer, ts_file = self._writer, self._ts_file
            self._writer = self._ts_file = self._ts_writer = None
        if writer is not None:
            writer.release()
        if ts_file is not None:
            try:
                ts_file.close()
            except OSError as e:
                log.error("camera %s: cannot close the timestamp file: %s", self.name, e)
