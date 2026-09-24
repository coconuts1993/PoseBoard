"""MediaPipe Pose Landmarker backend (tasks API, mediapipe>=0.10).

``MediaPipePose`` is ``MultiViewEstimator`` with the MediaPipe detector
(``poseboard.pose.detectors.mediapipe_det``), kept for backward compatibility:

* Single camera: run PnP between MediaPipe's world landmarks (a hip-centered 3D skeleton
  in meters) and the 2D pixel keypoints to get the skeleton's position in the camera
  frame, then transform it to the world frame with the camera extrinsics.
  Depth accuracy is limited, but good enough to put the body skeleton and the balance
  board in the same coordinate frame.
* Multiple cameras (>= 2 with calibrated extrinsics): detect 2D keypoints in each camera,
  then triangulate with confidence weights (and outlier rejection) for higher accuracy.
* Frames are never mixed: if any camera has extrinsics, only cameras with extrinsics are used;
  without extrinsics only ``world_camera`` (the camera whose frame is the world frame).
* When no 3D is possible, ``process`` returns None (as before); use
  ``MultiViewEstimator(create_detector("mediapipe"))`` to get ``2d_only`` poses instead.

This module also owns the model files (``MODEL_DIR``, ``ensure_model``).
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from poseboard.pose.base import Pose2D
from poseboard.pose.detectors.base import Person2D
from poseboard.pose.formats import MEDIAPIPE33
from poseboard.pose.multiview import MultiViewEstimator, single_view_lift, world_view

__all__ = ["MP_NAMES", "MP_SKELETON", "MODEL_URLS", "MODEL_DIR", "MediaPipePose", "ensure_model",
           "model_path", "single_view_lift", "world_view"]

log = logging.getLogger(__name__)

MP_NAMES = list(MEDIAPIPE33.names)
MP_SKELETON = list(MEDIAPIPE33.skeleton)

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


def model_path(variant: str = "full") -> Path:
    return MODEL_DIR / f"pose_landmarker_{variant}.task"


def ensure_model(variant: str = "full", timeout: float = 30.0) -> Path:
    """Path of the model file, downloading it first if needed (``timeout`` seconds without data
    aborts the download). The error message names the URL and where to put the file."""
    path = model_path(variant)
    if path.exists():
        return path
    url = MODEL_URLS[variant]
    tmp = path.with_suffix(".part")
    log.info("downloading %s", url)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 16)
        tmp.replace(path)
    except Exception as e:  # noqa: BLE001
        try:
            tmp.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"Cannot download the MediaPipe model ({e}).\nDownload it on any computer from\n"
            f"  {url}\nand save it as\n  {path}\n(or copy that file from the packaged PoseBoard "
            "build or another PC).") from e
    return path


class MediaPipePose(MultiViewEstimator):
    name = "mediapipe"
    keypoint_names = MP_NAMES
    skeleton = MP_SKELETON
    format = MEDIAPIPE33
    provides_3d = True
    min_score = 0.5
    emit_2d_only = False  # no 3D possible -> None (the behaviour of earlier versions)

    def __init__(self, variant: str = "full", min_score: float = 0.5, **kwargs):
        from poseboard.pose.detectors.mediapipe_det import MediaPipeDetector

        kwargs.setdefault("emit_2d_only", False)
        super().__init__(MediaPipeDetector(model=variant), min_score=min_score, **kwargs)
        self.name = "mediapipe"
        self.model_path = self.detector.model_path

    def detect(self, cam_name: str, t: float, image: np.ndarray):
        """Return (Pose2D, world_landmarks (33, 3)) of the first person, or None."""
        people = self.detector.detect(image, t, cam_name)
        if not people:
            return None
        p = people[0]
        return Pose2D(p.keypoints, p.scores), p.keypoints_3d

    def _detect(self, cam_name: str, t: float, image: np.ndarray) -> list[Person2D]:
        d = self.detect(cam_name, t, image)  # subclasses (and tests) may override detect()
        if d is None:
            return []
        pose2d, world = d
        return [Person2D(pose2d.keypoints, pose2d.scores, keypoints_3d=world)]
