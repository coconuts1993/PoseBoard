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
        pose2d_cam0.csv         2D keypoints (pixels) of the subject in each processed frame of cam0
        pose2d_json/cam0/       optional: OpenPose-format JSON per pose, for Pose2Sim
                                (cam0_<set:06d>_keypoints.json; sets.csv maps set -> frames/times;
                                Calib.toml of the recorded cameras)
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

from poseboard.calibration import CameraCalibration, pose2sim_camera_order, save_pose2sim_toml
from poseboard.camera import VIDEO_EXT, CameraStream
from poseboard.fusion import cop_to_world
from poseboard.geometry import BoardGeometry, BoardPose
from poseboard.pose.base import Pose2D, Pose3D
from poseboard.pose.com import center_of_mass
from poseboard.pose.formats import FORMATS
from poseboard.wii.device import ForceSample, ForceSource

log = logging.getLogger(__name__)

WII_HEADER = ["t", "t_rel", "t_unix", "TR_kg", "BR_kg", "TL_kg", "BL_kg", "total_kg",
              "cop_x_board", "cop_y_board", "cop_x_world", "cop_y_world", "cop_z_world"]
EVENTS_HEADER = ["t", "t_rel", "t_unix", "label"]
POSE2D_JSON_DIR = "pose2d_json"
JSON_SETS_CSV = "sets.csv"  # in POSE2D_JSON_DIR: set number -> pose time, video frames, times
FLUSH_INTERVAL_S = 1.0  # CSV files are flushed about once per second (little loss on a crash)


def _f(v) -> str:
    return "" if v is None or not np.isfinite(v) else f"{v:.6f}"


def _json_default(o):
    """session.json: numpy values (e.g. in ``pose_info``) as plain numbers/lists, else text."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _fp(v, digits: int = 3) -> str:
    return "" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


def openpose_json(pose2d: Pose2D | None) -> dict:
    """OpenPose (1.3) keypoint JSON of one image: the subject as the only person (all keypoints of
    the format in ``pose_keypoints_2d`` as x, y, confidence; missing = 0, 0, 0), or no person."""
    people = []
    if pose2d is not None:
        vals: list[float] = []
        for (x, y), c in zip(np.asarray(pose2d.keypoints, np.float64),
                             np.asarray(pose2d.scores, np.float64)):
            if np.isfinite(x) and np.isfinite(y):
                vals += [round(float(x), 3), round(float(y), 3),
                         round(float(c), 4) if np.isfinite(c) else 0.0]
            else:
                vals += [0.0, 0.0, 0.0]
        people.append({"person_id": [-1], "pose_keypoints_2d": vals, "face_keypoints_2d": [],
                       "hand_left_keypoints_2d": [], "hand_right_keypoints_2d": [],
                       "pose_keypoints_3d": [], "face_keypoints_3d": [],
                       "hand_left_keypoints_3d": [], "hand_right_keypoints_3d": []})
    return {"version": 1.3, "people": people}


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
    """Writes one recording session (see the module docstring for the folder layout).

    ``save_pose2d``: write the subject's 2D keypoints of every processed camera frame to
    ``pose2d_<cam>.csv`` (default on). ``save_openpose_json``: also write them as OpenPose JSON,
    so the recording can be re-triangulated offline with Pose2Sim (needs ``save_pose2d``):

    * ``pose2d_json/<cam>/<cam>_<set:06d>_keypoints.json``: one "set" per pose with a new video
      frame of at least one recorded camera. Every recorded camera gets a file in every set
      (an empty ``people`` list when its frame for that pose was not new, not in its video or
      not processed), so the set numbers run 0, 1, 2, ... without gaps in every folder and
      the same number is the same pose in all of them, which is how Pose2Sim pairs the files.
      The cameras are those passed to ``start`` (without camera streams: the cameras of the
      first pose).
    * ``pose2d_json/sets.csv``: set -> pose time (t, t_rel, t_unix) and each camera's video
      frame number and capture time (empty when its file has no person).
    * ``pose2d_json/Calib.toml``: the recorded cameras with extrinsics, in the order in which
      Pose2Sim sorts ``<cam>_json`` folders (``pose2sim_camera_order``)."""

    def __init__(self, root: str | Path = "recordings", save_openpose_json: bool = False,
                 save_pose2d: bool = True):
        self.root = Path(root)
        self.save_openpose_json = bool(save_openpose_json)
        self.save_pose2d = bool(save_pose2d)
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
        self._flushed = {"wii": 0.0, "pose": 0.0, "pose2d": 0.0}
        # camera -> [file, csv writer, keypoint names of the header]
        self._pose2d: dict[str, list] = {}
        self._pose2d_on = True  # pose2d_<cam>.csv enabled for the running recording
        self._json = False  # OpenPose JSON enabled for the running recording
        self._json_last: dict[str, int] = {}  # camera -> last video frame written as JSON
        self._json_cams: list[str] | None = None  # cameras with a JSON folder (fixed per recording)
        self._json_missing_logged: set[str] = set()
        # camera -> perf_counter time when its video writer was open: frames captured later
        # are in the video (earlier ones, just after t0, may not be)
        self._video_started: dict[str, float] = {}
        self._json_skip_logged = False
        self._sets_file = self._sets_csv = None
        self._lock = threading.Lock()
        # wii.csv has its own lock, so the Wii reader thread (whose samples are stamped when they
        # are read) never waits for the pose / JSON / events output
        self._wii_lock = threading.Lock()
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
              pose_backend: str | None, subject: str = "", notes: str = "",
              pose_backend_key: str | None = None, keypoint_format: str | None = None,
              pose_info: dict | None = None, save_openpose_json: bool | None = None,
              save_pose2d: bool | None = None) -> Path:
        """Start recording. ``cams`` may be empty (Wii-only) and ``force`` may be None
        (pose/video only).

        ``pose_backend``: a label of the pose source (None: no pose); ``pose_backend_key``: the
        backend key (``poseboard.pose.detectors.BACKENDS``); ``keypoint_format``: its keypoint
        format key (else taken from the first pose); ``pose_info``: more details for
        session.json (e.g. ``MultiViewEstimator.info()``); ``save_pose2d`` and
        ``save_openpose_json`` override the recorder's settings for this recording."""
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
        self._pose2d = {}
        self._pose2d_on = bool(self.save_pose2d if save_pose2d is None else save_pose2d)
        self._json = self._pose2d_on and bool(
            self.save_openpose_json if save_openpose_json is None else save_openpose_json)
        self._json_skip_logged = False
        self._json_last = {}
        self._json_missing_logged = set()
        self._video_started = {}
        self._json_cams = pose2sim_camera_order([c.name for c in cams]) if cams else None
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
            "pose_backend_key": pose_backend_key,
            "keypoint_format": None,
            "pose2sim_model": None,
            "pose_info": pose_info,
            "pose2d": {"csv": "pose2d_<camera>.csv" if self._pose2d_on else None, "cameras": [],
                       "openpose_json": self._json,
                       "json_dir": POSE2D_JSON_DIR if self._json else None,
                       "json_files": ("<camera>/<camera>_<set:06d>_keypoints.json" if self._json
                                      else None),
                       "json_sets_csv": (f"{POSE2D_JSON_DIR}/{JSON_SETS_CSV}" if self._json
                                         else None),
                       "json_cameras": self._json_cams if self._json else None,
                       "calib_toml": None,
                       "json_sets_written": 0, "json_sets_skipped": 0},
        }
        self._set_format_locked(keypoint_format)
        if self._json:
            self._write_calib_toml(folder, cams, calibrations)

        started: list[CameraStream] = []
        try:
            for c, st in zip(cams, self.meta["streams"]):
                path = c.start_recording(folder / f"{c.name}{VIDEO_EXT}",
                                         clock_offset_unix=self.clock_offset_unix, t0=t0)
                self._video_started[c.name] = time.perf_counter()
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
            with self._wii_lock:
                self._open_wii_csv()
            self.log_force_source(force, "start")
            force.add_listener(self._on_force)
        return folder

    def _set_format_locked(self, key: str | None) -> None:
        # caller holds self._lock (or the recording is not running yet)
        if not key or self.meta.get("keypoint_format"):
            return
        self.meta["keypoint_format"] = key
        fmt = FORMATS.get(key)
        self.meta["pose2sim_model"] = fmt.pose2sim_model if fmt is not None else None

    def _write_calib_toml(self, folder: Path, cams: list[CameraStream],
                          calibrations: dict[str, CameraCalibration]) -> None:
        """pose2d_json/Calib.toml for Pose2Sim: the recorded cameras (all calibrations without
        camera streams) that have extrinsics, in Pose2Sim's folder order."""
        names = [c.name for c in cams] if cams else list(calibrations)
        calibs = [calibrations[n] for n in names if n in calibrations]
        missing = [n for n in names if n not in calibrations]
        notes = [f"Not exported (no calibration): {', '.join(missing)}"] if missing else []
        if not any(c.has_extrinsics for c in calibs):
            self.meta["pose2d"]["calib_toml_note"] = ("not written: no recorded camera has "
                                                     "extrinsics")
            return
        d = folder / POSE2D_JSON_DIR
        d.mkdir(parents=True, exist_ok=True)
        try:
            notes += save_pose2sim_toml(d / "Calib.toml", calibs)
        except Exception as e:  # noqa: BLE001  (never stop a recording for this)
            log.exception("writing Calib.toml failed")
            self.meta["pose2d"]["calib_toml_note"] = f"not written: {e}"
            return
        self.meta["pose2d"]["calib_toml"] = f"{POSE2D_JSON_DIR}/Calib.toml"
        self.meta["pose2d"]["calib_toml_cameras"] = pose2sim_camera_order(
            [c.name for c in calibs if c.has_extrinsics])
        if notes:
            self.meta["pose2d"]["calib_toml_note"] = "; ".join(notes)

    def _write_meta(self, folder: Path) -> None:
        (folder / "session.json").write_text(
            json.dumps(self.meta, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8")

    def _open_wii_csv(self) -> None:
        # caller holds self._wii_lock
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
                with self._wii_lock:
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
        # Wii reader thread: only the wii.csv lock, never the one of the pose output
        with self._wii_lock:
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
        # caller holds the lock of that file (self._wii_lock for wii.csv, else self._lock)
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
                head += ["com_x", "com_y", "com_z", "com_x_board", "com_y_board", "com_z_board",
                         "mode", "reproj_error_px"]
                self._pose_csv.writerow(head)
                self._set_format_locked(pose.format_key)
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
            row += [getattr(pose, "mode", "") or "", _fp(getattr(pose, "reproj_error_px", None))]
            self._pose_csv.writerow(row)
            self.counts["pose"] += 1
            self._maybe_flush("pose", self._pose_file, pose.t)
            if self._pose2d_on and (pose.per_camera_2d or getattr(pose, "camera_frames", None)):
                self._write_pose2d_locked(pose)

    # ------------------------------------------------------------ 2D keypoints
    def _pose2d_file(self, cam: str, names: list[str]) -> list:
        # caller holds self._lock
        entry = self._pose2d.get(cam)
        if entry is None:
            fh = open(self.folder / f"pose2d_{cam}.csv", "w", newline="")
            w = csv.writer(fh)
            head = ["t", "t_rel", "t_unix", "frame"]
            for n in names:
                head += [f"{n}_x", f"{n}_y", f"{n}_score"]
            w.writerow(head)
            entry = self._pose2d[cam] = [fh, w, list(names)]
            self.meta["pose2d"]["cameras"].append(cam)
        return entry

    @staticmethod
    def _pose2d_values(p2: Pose2D | None, head: list[str], names: list[str]) -> np.ndarray:
        """(H, 3) x, y, score in the order of the file's header ``head`` (NaN if missing)."""
        out = np.full((len(head), 3), np.nan)
        if p2 is None:
            return out
        kp = np.asarray(p2.keypoints, np.float64).reshape(-1, 2)
        sc = np.asarray(p2.scores, np.float64).reshape(-1)
        vals = np.column_stack([kp, sc])
        if len(vals) == len(head) and (len(vals) != len(names) or list(names) == head):
            return vals
        if len(vals) == len(names):  # map by keypoint name
            idx = {n: i for i, n in enumerate(names)}
            for j, n in enumerate(head):
                i = idx.get(n)
                if i is not None:
                    out[j] = vals[i]
            return out
        m = min(len(vals), len(head))
        out[:m] = vals[:m]
        return out

    def _write_pose2d_locked(self, pose: Pose3D) -> None:
        """pose2d_<cam>.csv rows (and OpenPose JSON files) for every camera processed for
        ``pose``; cameras without a subject get empty values / an empty people list."""
        # caller holds self._lock
        names = list(pose.names)
        frames = dict(getattr(pose, "camera_frames", None) or {})
        cams = list(frames) + [c for c in pose.per_camera_2d if c not in frames]
        entries: dict[str, tuple[float, int | None, Pose2D | None]] = {}
        for cam in cams:
            p2 = pose.per_camera_2d.get(cam)
            t_cf, idx_cf = frames.get(cam, (None, None))
            t = next(v for v in (getattr(p2, "t", None), t_cf, pose.t) if v is not None)
            idx = getattr(p2, "frame_index", None)
            if idx is None:
                idx = idx_cf
            if idx is not None and t < self.t0:
                idx = None  # a frame written by an earlier recording
            if p2 is not None and len(p2.keypoints) != len(names):
                head_names = [f"kp{i}" for i in range(len(p2.keypoints))]
            else:
                head_names = names
            fh, w, head = self._pose2d_file(cam, head_names)
            vals = self._pose2d_values(p2, head, names)
            row = [f"{t:.6f}", f"{t - self.t0:.6f}", f"{t + self.clock_offset_unix:.6f}",
                   "" if idx is None else int(idx)]
            for x, y, c in vals:
                row += [_fp(x), _fp(y), _fp(c, 4)]
            w.writerow(row)
            entries[cam] = (float(t), None if idx is None else int(idx), p2)
        if self._json and cams:
            self._write_json_set_locked(pose, entries)
        if pose.t - self._flushed["pose2d"] >= FLUSH_INTERVAL_S:
            for fh, _, _ in self._pose2d.values():
                fh.flush()
            if self._sets_file is not None:
                self._sets_file.flush()
            self._flushed["pose2d"] = pose.t

    def _write_json_set_locked(self, pose: Pose3D,
                               entries: dict[str, tuple[float, int | None, Pose2D | None]]) -> None:
        """One OpenPose JSON file per JSON camera for this pose (see the class docstring), or
        none at all when no camera has a new frame of its video (e.g. a pose computed from
        frames captured before the recording started)."""
        # caller holds self._lock
        if self._json_cams is None:  # no camera streams: the cameras of the first pose
            self._json_cams = pose2sim_camera_order(list(entries))
            self.meta["pose2d"]["json_cameras"] = list(self._json_cams)
        new: dict[str, tuple[float, int, Pose2D | None]] = {}
        for cam in self._json_cams:
            t, idx, p2 = entries.get(cam, (None, None, None))
            if idx is not None and idx > self._json_last.get(cam, -1):
                new[cam] = (t, idx, p2)
            elif (idx is None and t is not None and t >= self._video_started.get(cam, self.t0)
                  and cam not in self._json_missing_logged):
                # captured after its video writer was opened, but the frame is not in its video
                # (camera removed and added again, video writer failed, frame size changed)
                self._json_missing_logged.add(cam)
                log.warning("OpenPose JSON: frames of %s are not in its recorded video; its "
                            "JSON files have no person while this lasts", cam)
        if not new:
            self.meta["pose2d"]["json_sets_skipped"] += 1
            if not self._json_skip_logged:
                self._json_skip_logged = True
                log.info("OpenPose JSON not written for a pose without a new frame of the "
                         "recorded videos (e.g. captured just before the recording started)")
            return
        n = int(self.meta["pose2d"]["json_sets_written"])
        root = self.folder / POSE2D_JSON_DIR
        for cam in self._json_cams:
            d = root / cam
            d.mkdir(parents=True, exist_ok=True)
            p2 = new[cam][2] if cam in new else None
            (d / f"{cam}_{n:06d}_keypoints.json").write_text(
                json.dumps(openpose_json(p2), separators=(",", ":")), encoding="utf-8")
            if cam in new:
                self._json_last[cam] = new[cam][1]
        if self._sets_csv is None:
            self._sets_file = open(root / JSON_SETS_CSV, "w", newline="")
            self._sets_csv = csv.writer(self._sets_file)
            head = ["set", "t", "t_rel", "t_unix"]
            for cam in self._json_cams:
                head += [f"{cam}_frame", f"{cam}_t"]
            self._sets_csv.writerow(head)
        row = [n, f"{pose.t:.6f}", f"{pose.t - self.t0:.6f}",
               f"{pose.t + self.clock_offset_unix:.6f}"]
        for cam in self._json_cams:
            t, idx, _ = new.get(cam, (None, None, None))
            row += ["", ""] if idx is None else [idx, f"{t:.6f}"]
        self._sets_csv.writerow(row)
        self.meta["pose2d"]["json_sets_written"] = n + 1

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
            with self._wii_lock:
                wii_file = self._wii_file
                self._wii_file = self._wii_csv = None
            files = (wii_file, self._pose_file, self._events_file, self._sets_file,
                     *[e[0] for e in self._pose2d.values()])
            self._pose_file = self._pose_csv = None
            self._events_file = self._events_csv = None
            self._sets_file = self._sets_csv = None
            self._pose2d = {}
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
