"""相机采集线程：每帧带 perf_counter 时间戳，可边采集边写视频。"""

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


@dataclass
class Frame:
    index: int
    t: float
    image: np.ndarray


def open_capture(source: int | str, width: int | None = None, height: int | None = None,
                 fps: float | None = None) -> cv2.VideoCapture:
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    if isinstance(source, int) and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


class CameraStream:
    """后台线程读取一台相机。source 可以是设备序号、视频文件或网络流地址。"""

    def __init__(self, name: str, source: int | str, width: int | None = None,
                 height: int | None = None, fps: float | None = None):
        self.name = name
        self.source = source
        self.req = (width, height, fps)
        self.fps = fps or 30.0
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._writer: cv2.VideoWriter | None = None
        self._ts_file = None
        self._ts_writer = None
        self._rec_index = 0
        self.error: str | None = None
        self.measured_fps = 0.0
        self._is_file = isinstance(source, str) and not source.isdigit() and Path(source).exists()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._cap = open_capture(self.source, *self.req)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开相机 {self.source}")
        f = self._cap.get(cv2.CAP_PROP_FPS)
        if f and 1 < f < 1000:
            self.fps = f
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"cam-{self.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        self.stop_recording()
        if self._cap:
            self._cap.release()
        self._cap = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        idx = 0
        last = time.perf_counter()
        period = 1.0 / self.fps if self._is_file else 0.0
        while not self._stop.is_set():
            ok, img = self._cap.read()
            t = time.perf_counter()
            if not ok:
                if self._is_file:  # 视频文件循环播放，方便调试
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.error = "读取帧失败"
                time.sleep(0.01)
                continue
            frame = Frame(idx, t, img)
            idx += 1
            dt = t - last
            last = t
            if dt > 0:
                self.measured_fps = 0.9 * self.measured_fps + 0.1 / dt if self.measured_fps else 1 / dt
            with self._lock:
                self._latest = frame
                if self._writer is not None:
                    self._writer.write(img)
                    self._ts_writer.writerow([self._rec_index, f"{t:.6f}"])
                    self._rec_index += 1
            if period:
                time.sleep(max(0.0, period - (time.perf_counter() - t)))

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    # -------------------------------------------------------------- recording
    def start_recording(self, video_path: Path) -> None:
        f = self.latest()
        if f is None:
            raise RuntimeError(f"相机 {self.name} 还没有画面")
        h, w = f.image.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, float(self.fps), (w, h))
        ts_file = open(video_path.with_name(video_path.stem + "_timestamps.csv"), "w", newline="")
        ts_writer = csv.writer(ts_file)
        ts_writer.writerow(["frame", "t"])
        with self._lock:
            self._writer, self._ts_file, self._ts_writer, self._rec_index = writer, ts_file, ts_writer, 0

    def stop_recording(self) -> None:
        with self._lock:
            writer, ts_file = self._writer, self._ts_file
            self._writer = self._ts_file = self._ts_writer = None
        if writer is not None:
            writer.release()
        if ts_file is not None:
            ts_file.close()
