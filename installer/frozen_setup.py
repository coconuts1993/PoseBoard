"""Runtime setup shared by the frozen (PyInstaller) entry points.

* The models folder (``mediapipe_backend.MODEL_DIR``) is the writable ``models`` folder next to
  the executable: YOLO / MoveNet downloads, OpenPose weights (``models\\openpose\\body_25``) and
  other MediaPipe models go there, as the README describes. The MediaPipe model bundled with
  the executable (``_internal\\models``) is found through ``BUNDLED_MODEL_DIR``.
* Makes relative default paths (e.g. ``recordings/``) resolve next to the executable
  instead of wherever the process happened to be started from.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def app_dir() -> Path:
    """Folder that contains the executable (or the repo root when not frozen)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def configure() -> None:
    if not getattr(sys, "frozen", False):
        return
    from poseboard.pose import mediapipe_backend

    bundled = Path(getattr(sys, "_MEIPASS", app_dir())) / "models"
    if any(bundled.glob("pose_landmarker_*.task")):
        mediapipe_backend.BUNDLED_MODEL_DIR = bundled
    mediapipe_backend.MODEL_DIR = app_dir() / "models"
    os.chdir(app_dir())
