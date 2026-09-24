"""Synchronized recording: Wii force data, video from each camera (with per-frame
timestamps) and 3D pose, all on the same perf_counter clock. Every stream is optional:
pose-only (cameras, no board), Wii-only (board, no cameras) or both.

Every row carries ``t`` (perf_counter, s), ``t_rel`` (s since the recording start) and
``t_unix`` (wall-clock Unix time, s). ``t_unix = t + clock_offset_unix`` with the offset
measured once at the start, so it never jumps even if the system clock is adjusted; use it
to align recordings made by other programs or devices.

Output folder layout::

    recordings/20260924_153000_subject/
        session.json            calibration, board pose, device info, clock offset, start/stop times
        wii.csv                 ~100 Hz: four sensors in kg, total weight, COP (board + world coords)
        pose3d.csv              one row per pose frame: keypoint world coordinates + COM
        events.csv              event markers added during the recording (e.g. "sync" jumps)
        cam0.mkv, cam0_timestamps.csv
        fused.csv               written after stop: force data interpolated at pose timestamps + COM/COP
        summary.json            written after stop: static balance metrics
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from poseboard.calibration import CameraCalibration
from poseboard.camera import VIDEO_EXT, CameraStream
from poseboard.fusion import cop_to_world
from poseboard.geometry import BoardGeometry, BoardPose
from poseboard.pose.base import Pose3D
from poseboard.pose.com import center_of_mass
from poseboard.wii.device import ForceSample, ForceSource

log = logging.getLogger(__name__)

WII_HEADER = ["t", "t_rel", "t_unix", "TR_kg", "BR_kg", "TL_kg", "BL_kg", "total_kg",
              "cop_x_board", "cop_y_board", "cop_x_world", "cop_y_world", "cop_z_world"]
EVENTS_HEADER = ["t", "t_rel", "t_unix", "label"]
FLUSH_INTERVAL_S = 1.0  # CSV files are flushed about once per second (little loss on a crash)


def _f(v) -> str:
    return "" if v is None or not np.isfinite(v) else f"{v:.6f}"


def capture_clock_offset(max_wait_s: float = 0.05) -> tuple[float, float]:
    """Read ``time.perf_counter()`` and ``time.time()`` back to back; returns (t, t_unix).

    The pair is taken right after the wall clock ticks, so a coarse ``time.time()`` (about
    15.6 ms on older Windows Pythons) still gives an offset accurate to about a millisecond.
    """
    prev = time.time()
    deadline = time.perf_counter() + max_wait_s
    best: tuple[float, float, float] | None = None
    while True:
        a = time.perf_counter()
        u = time.time()
        b = time.perf_counter()
        if best is None or b - a < best[0]:
            best = (b - a, (a + b) / 2, u)
        if u != prev:  # the wall clock just ticked: u is fresh
            return (a + b) / 2, u
        if b > deadline:
            return best[1], best[2]
        prev = u


def on_console_close(callback):
    """Windows: call ``callback`` (in a system thread) when the console window is closed, or the
    user logs off / shuts down. The process is then ended within a few seconds without running
    any cleanup code, so ``callback`` must close files quickly (e.g. ``recorder.stop(analyze=False)``).
    Returns a handle for ``remove_console_close`` (None if not available)."""
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes

        @ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)
        def handler(ctrl_type):
            if ctrl_type in (2, 5, 6):  # CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT
                try:
                    callback()
                except Exception:  # noqa: BLE001
                    log.exception("closing the recording failed")
                return 1
            return 0  # Ctrl+C / Ctrl+Break: Python's own handling

        if not ctypes.windll.kernel32.SetConsoleCtrlHandler(handler, True):
            return None
        return handler  # keep a reference: the callback must stay alive
    except Exception:  # noqa: BLE001
        log.debug("console close handler not installed", exc_info=True)
        return None


def remove_console_close(handle) -> None:
    if handle is None:
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleCtrlHandler(handle, False)
    except Exception:  # noqa: BLE001
        pass


def iso_time(t_unix: float) -> str:
    """Local ISO 8601 time with timezone, e.g. 2026-09-24T15:30:00.123+08:00."""
    return datetime.fromtimestamp(t_unix).astimezone().isoformat(timespec="milliseconds")


class SessionRecorder:
    def __init__(self, root: str | Path = "recordings"):
        self.root = Path(root)
        self.folder: Path | None = None
        self.t0: float | None = None
        self.t0_unix: float | None = None
        self.clock_offset_unix: float | None = None
        self._force: ForceSource | None = None
        self._cams: list[CameraStream] = []
        self._board: BoardPose | None = None
        self._wii_file = self._wii_csv = None
        self._pose_file = self._pose_csv = None
        self._events_file = self._events_csv = None
        self._pose_names: list[str] | None = None
        self._pose_names_warned = False
        self._flushed = {"wii": 0.0, "pose": 0.0}
        self._lock = threading.Lock()
        self._post_thread: threading.Thread | None = None
        self.meta: dict = {}
        self.counts = {"wii": 0, "pose": 0, "events": 0}

    @property
    def recording(self) -> bool:
        return self.folder is not None

    def to_unix(self, t: float) -> float:
        """Convert a perf_counter time to Unix time using this session's fixed offset."""
        off = self.clock_offset_unix
        return t + (off if off is not None else time.time() - time.perf_counter())

    def _new_folder(self, name: str) -> Path:
        """Create a new, empty session folder: ``name``, or ``name_2``, ``name_3``, ... if a
        recording started in the same second already uses it."""
        self.root.mkdir(parents=True, exist_ok=True)
        folder, n = self.root / name, 2
        while True:
            try:
                folder.mkdir()
                return folder
            except FileExistsError:
                folder, n = self.root / f"{name}_{n}", n + 1

    def start(self, *, cams: list[CameraStream], calibrations: dict[str, CameraCalibration],
              force: ForceSource | None, board: BoardPose | None, geometry: BoardGeometry,
              pose_backend: str | None, subject: str = "", notes: str = "") -> Path:
        """Start recording. ``cams`` may be empty (Wii-only) and ``force`` may be None
        (pose/video only)."""
        if self.recording:
            raise RuntimeError("Already recording")
        t0, t0_unix = capture_clock_offset()
        start_dt = datetime.fromtimestamp(t0_unix)
        stamp = start_dt.strftime("%Y%m%d_%H%M%S")
        name = f"{stamp}_{subject}" if subject else stamp
        folder = self._new_folder(name)
        self._board, self._force, self._cams = board, force, list(cams)
        self.counts = {"wii": 0, "pose": 0, "events": 0}
        self._pose_names = None
        self._pose_names_warned = False
        self.t0, self.t0_unix = t0, t0_unix
        self.clock_offset_unix = t0_unix - t0
        self.meta = {
            "created": start_dt.isoformat(timespec="seconds"),
            "subject": subject,
            "notes": notes,
            "clock": "time.perf_counter (seconds); t_rel = t - t0; "
                     "t_unix = t + clock_offset_unix (Unix seconds, offset fixed at start)",
            "t0": t0,
            "t0_unix": t0_unix,
            "clock_offset_unix": self.clock_offset_unix,
            "start_time_iso": iso_time(t0_unix),
            "has_wii": force is not None,
            "has_pose": pose_backend is not None,
            "has_video": bool(cams),
            "camera_names": [c.name for c in cams],
            "world_frame": "checkerboard (Z up), meters",
            "board_geometry": geometry.to_dict(),
            "board_pose": None if board is None else board.to_dict(),
            "cameras": {n: c.to_dict() for n, c in calibrations.items()},
            "streams": [{"name": c.name, "source": str(c.source), "fps": c.fps,
                         "video": f"{c.name}{VIDEO_EXT}"} for c in cams],
            "force_source": None if force is None else force.info(),
            "force_source_history": [],
            "pose_backend": pose_backend,
        }

        started: list[CameraStream] = []
        try:
            for c, st in zip(cams, self.meta["streams"]):
                path = c.start_recording(folder / f"{c.name}{VIDEO_EXT}",
                                         clock_offset_unix=self.clock_offset_unix, t0=t0)
                st["video"] = Path(path).name
                started.append(c)
        except Exception:
            for c in started:
                c.stop_recording()
            raise
        self._write_meta(folder)
        with self._lock:
            self.folder = folder
            if force is not None:
                self._open_wii_csv()
        if force is not None:
            self.log_force_source(force, "start")
            force.add_listener(self._on_force)
        return folder

    def _write_meta(self, folder: Path) -> None:
        (folder / "session.json").write_text(json.dumps(self.meta, indent=2, ensure_ascii=False),
                                             encoding="utf-8")

    def _open_wii_csv(self) -> None:
        # caller holds self._lock
        if self._wii_csv is None and self.folder is not None:
            self._wii_file = open(self.folder / "wii.csv", "w", newline="")
            self._wii_csv = csv.writer(self._wii_file)
            self._wii_csv.writerow(WII_HEADER)

    def attach_force(self, force: ForceSource | None) -> None:
        """Switch the force source during a recording, e.g. when the board is connected (or
        reconnected as a new object) after recording started. Samples go to the same wii.csv."""
        if not self.recording or force is self._force:
            return
        old = self._force
        if old is not None:
            old.remove_listener(self._on_force)
        with self._lock:
            if self.folder is None:
                return
            self._force = force
            if force is not None:
                self._open_wii_csv()
                self.meta["has_wii"] = True
                self.meta["force_source"] = force.info()
        if force is not None:
            self.log_force_source(force, "attached")
            force.add_listener(self._on_force)

    def log_force_source(self, force: ForceSource | None, reason: str) -> None:
        """Append the force source's state (device, tare, COP threshold) to
        ``force_source_history`` in session.json, so the tare in effect for every part of
        wii.csv is known (e.g. after a reconnect or a different board)."""
        if force is None or not self.recording:
            return
        t = time.perf_counter()
        try:
            info = force.info()
        except Exception:  # noqa: BLE001
            log.exception("force source info failed")
            return
        entry = {"t_rel": t - self.t0, "t_unix": t + self.clock_offset_unix, "reason": reason,
                 "type": info.get("type"), "device": getattr(force, "device_key", None),
                 "tare_kg": info.get("tare_kg"), "min_total_kg": info.get("min_total_kg")}
        with self._lock:
            self.meta.setdefault("force_source_history", []).append(entry)

    # ------------------------------------------------------------ callbacks
    def _on_force(self, s: ForceSample) -> None:
        with self._lock:
            if self._wii_csv is None:
                return
            cw = (cop_to_world(self._board, s.cop_board)
                  if self._board is not None and np.all(np.isfinite(s.cop_board)) else [np.nan] * 3)
            self._wii_csv.writerow([f"{s.t:.6f}", f"{s.t - self.t0:.6f}",
                                    f"{s.t + self.clock_offset_unix:.6f}", *[_f(v) for v in s.kg],
                                    _f(s.total_kg), _f(s.cop_board[0]), _f(s.cop_board[1]),
                                    *[_f(v) for v in cw]])
            self.counts["wii"] += 1
            self._maybe_flush("wii", self._wii_file, s.t)

    def _maybe_flush(self, key: str, fh, t: float) -> None:
        # caller holds self._lock
        if t - self._flushed[key] >= FLUSH_INTERVAL_S:
            fh.flush()
            self._flushed[key] = t

    def _match_pose_names(self, pose: Pose3D) -> tuple[np.ndarray, np.ndarray]:
        """Keypoints/scores in the order of the pose3d.csv header (by name; NaN if missing)."""
        if list(pose.names) == self._pose_names:
            return pose.keypoints, pose.scores
        idx = {n: i for i, n in enumerate(pose.names)}
        kp = np.full((len(self._pose_names), 3), np.nan)
        sc = np.full(len(self._pose_names), np.nan)
        for j, n in enumerate(self._pose_names):
            i = idx.get(n)
            if i is not None:
                kp[j], sc[j] = pose.keypoints[i], pose.scores[i]
        if not self._pose_names_warned:
            self._pose_names_warned = True
            extra = [n for n in pose.names if n not in set(self._pose_names)]
            log.warning("pose keypoint names changed during the recording; pose3d.csv keeps its "
                        "columns (by name); not stored: %s", ", ".join(extra) or "-")
            self._write_event_locked(time.perf_counter(), "pose_keypoints_changed")
        return kp, sc

    def add_pose(self, pose: Pose3D) -> None:
        with self._lock:
            if self.folder is None:
                return
            if self._pose_csv is None:
                self._pose_names = list(pose.names)
                self._pose_file = open(self.folder / "pose3d.csv", "w", newline="")
                self._pose_csv = csv.writer(self._pose_file)
                head = ["t", "t_rel", "t_unix"]
                for n in self._pose_names:
                    head += [f"{n}_x", f"{n}_y", f"{n}_z", f"{n}_score"]
                head += ["com_x", "com_y", "com_z", "com_x_board", "com_y_board", "com_z_board"]
                self._pose_csv.writerow(head)
            row = [f"{pose.t:.6f}", f"{pose.t - self.t0:.6f}", f"{pose.t + self.clock_offset_unix:.6f}"]
            kps, scores = self._match_pose_names(pose)
            for p, s in zip(kps, scores):
                row += [_f(p[0]), _f(p[1]), _f(p[2]), _f(s)]
            com = center_of_mass(pose.keypoints, pose.names)
            if com is None:
                row += [""] * 6
            else:
                cb = (self._board.world_to_board.apply(com) if self._board is not None
                      else [np.nan] * 3)
                row += [_f(v) for v in com] + [_f(v) for v in cb]
            self._pose_csv.writerow(row)
            self.counts["pose"] += 1
            self._maybe_flush("pose", self._pose_file, pose.t)

    def add_event(self, label: str, t: float | None = None) -> dict | None:
        """Write an event marker (e.g. "sync" when the subject jumps) to events.csv.

        ``t`` is a perf_counter time (default: now). Thread-safe; returns the written row as a
        dict, or None when not recording.
        """
        if t is None:
            t = time.perf_counter()
        with self._lock:
            if self.folder is None:
                return None
            return self._write_event_locked(t, label)

    def _write_event_locked(self, t: float, label: str) -> dict:
        # caller holds self._lock. UTF-8 with a BOM, so Excel shows non-ASCII labels correctly.
        label = " ".join(str(label).split()) or "event"
        if self._events_csv is None:
            self._events_file = open(self.folder / "events.csv", "w", newline="", encoding="utf-8-sig")
            self._events_csv = csv.writer(self._events_file)
            self._events_csv.writerow(EVENTS_HEADER)
        ev = {"t": t, "t_rel": t - self.t0, "t_unix": t + self.clock_offset_unix, "label": label}
        self._events_csv.writerow([f"{ev['t']:.6f}", f"{ev['t_rel']:.6f}", f"{ev['t_unix']:.6f}", label])
        self._events_file.flush()  # markers are rare and valuable: do not lose them on a crash
        self.counts["events"] += 1
        return ev

    # ----------------------------------------------------------------- stop
    def stop(self, analyze: bool = True, background: bool = False) -> Path | None:
        """Stop recording, then post-process (``analyze``; fused.csv and summary.json): in a
        background thread with ``background`` (see ``post_processing``). Safe to call more than
        once (returns None)."""
        t_stop = time.perf_counter()
        with self._lock:
            folder, self.folder = self.folder, None
            if folder is None:
                return None
            files = (self._wii_file, self._pose_file, self._events_file)
            self._wii_file = self._wii_csv = self._pose_file = self._pose_csv = None
            self._events_file = self._events_csv = None
        if self._force is not None:
            self._force.remove_listener(self._on_force)
        for c in self._cams:
            c.stop_recording()
        for f in files:
            if f is not None:
                f.close()
        self.meta["t_stop"] = t_stop
        self.meta["t_stop_unix"] = t_stop + self.clock_offset_unix
        self.meta["stop_time_iso"] = iso_time(t_stop + self.clock_offset_unix)
        self.meta["duration_s"] = t_stop - self.t0
        self.meta["samples"] = dict(self.counts)
        self.meta["has_wii"] = files[0] is not None
        self.meta["has_pose"] = self.counts["pose"] > 0
        if self._force is not None:
            try:
                self.meta["force_source"] = self._force.info()
            except Exception:  # noqa: BLE001
                log.exception("force source info failed")
        self._write_meta(folder)
        if analyze:
            if background:
                self._post_thread = threading.Thread(target=self._post_process, args=(folder,),
                                                     name="post-processing")
                self._post_thread.start()
            else:
                self._post_process(folder)
        return folder

    @staticmethod
    def _post_process(folder: Path) -> None:
        try:
            from poseboard.analysis import analyze_session

            analyze_session(folder)
        except Exception:  # noqa: BLE001
            log.exception("post-processing failed")

    @property
    def post_processing(self) -> bool:
        """True while a background post-processing (``stop(background=True)``) is running."""
        t = self._post_thread
        return t is not None and t.is_alive()

    def wait_post_processing(self, timeout: float | None = None) -> bool:
        """Wait for a background post-processing; True when it has finished."""
        t = self._post_thread
        if t is not None:
            t.join(timeout)
        return not self.post_processing
