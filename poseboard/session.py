"""Synchronized recording: Wii force data, video from each camera (with per-frame
timestamps) and 3D pose, all on the same perf_counter clock.

Output folder layout::

    recordings/20260924_153000_subject/
        session.json            calibration, board pose, device info, start/stop times
        wii.csv                 ~100 Hz: four sensors in kg, total weight, COP (board + world coords)
        pose3d.csv              one row per pose frame: keypoint world coordinates + COM
        cam0.mp4, cam0_timestamps.csv
        fused.csv               written after stop: force data interpolated at pose timestamps + COM/COP
        summary.json            written after stop: static balance metrics
"""

from __future__ import annotations

import csv
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from poseboard.calibration import CameraCalibration
from poseboard.camera import CameraStream
from poseboard.fusion import cop_to_world
from poseboard.geometry import BoardGeometry, BoardPose
from poseboard.pose.base import Pose3D
from poseboard.pose.com import center_of_mass
from poseboard.wii.device import ForceSample, ForceSource

log = logging.getLogger(__name__)

WII_HEADER = ["t", "t_rel", "TR_kg", "BR_kg", "TL_kg", "BL_kg", "total_kg",
              "cop_x_board", "cop_y_board", "cop_x_world", "cop_y_world", "cop_z_world"]


def _f(v) -> str:
    return "" if v is None or not np.isfinite(v) else f"{v:.6f}"


class SessionRecorder:
    def __init__(self, root: str | Path = "recordings"):
        self.root = Path(root)
        self.folder: Path | None = None
        self.t0: float | None = None
        self._force: ForceSource | None = None
        self._cams: list[CameraStream] = []
        self._board: BoardPose | None = None
        self._wii_file = self._wii_csv = None
        self._pose_file = self._pose_csv = None
        self._pose_names: list[str] | None = None
        self._lock = threading.Lock()
        self.meta: dict = {}
        self.counts = {"wii": 0, "pose": 0}

    @property
    def recording(self) -> bool:
        return self.folder is not None

    def start(self, *, cams: list[CameraStream], calibrations: dict[str, CameraCalibration],
              force: ForceSource | None, board: BoardPose | None, geometry: BoardGeometry,
              pose_backend: str | None, subject: str = "", notes: str = "") -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{stamp}_{subject}" if subject else stamp
        self.folder = self.root / name
        self.folder.mkdir(parents=True, exist_ok=True)
        self._board, self._force, self._cams = board, force, list(cams)
        self.counts = {"wii": 0, "pose": 0}
        self.t0 = time.perf_counter()
        self.meta = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "subject": subject,
            "notes": notes,
            "clock": "time.perf_counter (seconds); t_rel = t - t0",
            "t0": self.t0,
            "world_frame": "checkerboard (Z up), meters",
            "board_geometry": geometry.to_dict(),
            "board_pose": None if board is None else board.to_dict(),
            "cameras": {n: c.to_dict() for n, c in calibrations.items()},
            "streams": [{"name": c.name, "source": str(c.source), "fps": c.fps,
                         "video": f"{c.name}.mp4"} for c in cams],
            "force_source": None if force is None else force.info(),
            "pose_backend": pose_backend,
        }
        self._write_meta(self.folder)

        if force is not None:
            self._wii_file = open(self.folder / "wii.csv", "w", newline="")
            self._wii_csv = csv.writer(self._wii_file)
            self._wii_csv.writerow(WII_HEADER)
            force.add_listener(self._on_force)
        for c in cams:
            c.start_recording(self.folder / f"{c.name}.mp4")
        return self.folder

    def _write_meta(self, folder: Path) -> None:
        (folder / "session.json").write_text(json.dumps(self.meta, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")

    # ------------------------------------------------------------ callbacks
    def _on_force(self, s: ForceSample) -> None:
        with self._lock:
            if self._wii_csv is None:
                return
            cw = (cop_to_world(self._board, s.cop_board)
                  if self._board is not None and np.all(np.isfinite(s.cop_board)) else [np.nan] * 3)
            self._wii_csv.writerow([f"{s.t:.6f}", f"{s.t - self.t0:.6f}", *[_f(v) for v in s.kg],
                                    _f(s.total_kg), _f(s.cop_board[0]), _f(s.cop_board[1]),
                                    *[_f(v) for v in cw]])
            self.counts["wii"] += 1

    def add_pose(self, pose: Pose3D) -> None:
        with self._lock:
            if self.folder is None:
                return
            if self._pose_csv is None:
                self._pose_names = list(pose.names)
                self._pose_file = open(self.folder / "pose3d.csv", "w", newline="")
                self._pose_csv = csv.writer(self._pose_file)
                head = ["t", "t_rel"]
                for n in self._pose_names:
                    head += [f"{n}_x", f"{n}_y", f"{n}_z", f"{n}_score"]
                head += ["com_x", "com_y", "com_z", "com_x_board", "com_y_board", "com_z_board"]
                self._pose_csv.writerow(head)
            row = [f"{pose.t:.6f}", f"{pose.t - self.t0:.6f}"]
            for p, s in zip(pose.keypoints, pose.scores):
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

    # ----------------------------------------------------------------- stop
    def stop(self) -> Path | None:
        folder = self.folder
        if folder is None:
            return None
        if self._force is not None:
            self._force.remove_listener(self._on_force)
        for c in self._cams:
            c.stop_recording()
        with self._lock:
            for f in (self._wii_file, self._pose_file):
                if f is not None:
                    f.close()
            self._wii_file = self._wii_csv = self._pose_file = self._pose_csv = None
            self.folder = None
        self.meta["duration_s"] = time.perf_counter() - self.t0
        self.meta["samples"] = dict(self.counts)
        self._write_meta(folder)
        try:
            from poseboard.analysis import analyze_session

            analyze_session(folder)
        except Exception:  # noqa: BLE001
            log.exception("post-processing failed")
        return folder
