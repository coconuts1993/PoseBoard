"""Runtime setup shared by the frozen (PyInstaller) entry points.

* Points the MediaPipe backend at the pose model bundled with the executable, or at a
  writable ``models`` folder next to the executable if no model was bundled
  (it is then downloaded on first use).
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
        mediapipe_backend.MODEL_DIR = bundled
    else:
        mediapipe_backend.MODEL_DIR = app_dir() / "models"
    os.chdir(app_dir())
