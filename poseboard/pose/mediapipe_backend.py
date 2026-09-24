"""MediaPipe Pose Landmarker backend (tasks API, mediapipe>=0.10).

* Single camera: run PnP between MediaPipe's world landmarks (a hip-centered 3D skeleton
  in meters) and the 2D pixel keypoints to get the skeleton's position in the camera
  frame, then transform it to the world frame with the camera extrinsics.
  Depth accuracy is limited, but good enough to put the body skeleton and the balance
  board in the same coordinate frame.
* Multiple cameras (>= 2 with calibrated extrinsics): detect 2D keypoints in each camera,
  then triangulate with confidence weights for higher accuracy.
* Frames are never mixed: if any camera has extrinsics, only cameras with extrinsics are used;
  without extrinsics only ``world_camera`` (the camera whose frame is the world frame).
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from poseboard.calibration import CameraCalibration
from poseboard.pose.base import Pose2D, Pose3D, PoseEstimator
from poseboard.pose.triangulation import triangulate_keypoints

log = logging.getLogger(__name__)

MP_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
]

MP_SKELETON = [
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_heel", "left_foot_index"), ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_heel"), ("right_heel", "right_foot_index"),
    ("right_ankle", "right_foot_index"), ("nose", "left_ear"), ("nose", "right_ear"),
]

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


class MediaPipePose(PoseEstimator):
    name = "mediapipe"
    keypoint_names = MP_NAMES
    skeleton = MP_SKELETON

    def __init__(self, variant: str = "full", min_score: float = 0.5):
        from mediapipe.tasks.python import BaseOptions, vision

        self._vision = vision
        self._base = BaseOptions
        self.model_path = str(ensure_model(variant))
        self.min_score = min_score
        self._landmarkers: dict[str, object] = {}
        self._last_ts: dict[str, int] = {}

    def _landmarker(self, cam_name: str):
        if cam_name not in self._landmarkers:
            v = self._vision
            opts = v.PoseLandmarkerOptions(
                base_options=self._base(model_asset_path=self.model_path),
                running_mode=v.RunningMode.VIDEO, num_poses=1)
            self._landmarkers[cam_name] = v.PoseLandmarker.create_from_options(opts)
            self._last_ts[cam_name] = -1
        return self._landmarkers[cam_name]

    def detect(self, cam_name: str, t: float, image: np.ndarray):
        """Return (Pose2D, world_landmarks(33,3)) or None."""
        import mediapipe as mp

        lm = self._landmarker(cam_name)
        ts = max(int(t * 1000), self._last_ts[cam_name] + 1)
        self._last_ts[cam_name] = ts
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        if not res.pose_landmarks:
            return None
        h, w = image.shape[:2]
        l2 = res.pose_landmarks[0]
        kp = np.array([[p.x * w, p.y * h] for p in l2])
        sc = np.array([p.visibility if p.visibility is not None else 1.0 for p in l2])
        world = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]])
        return Pose2D(kp, sc), world

    def process(self, frames, cams):
        dets: dict[str, tuple[Pose2D, np.ndarray]] = {}
        for name, (t, img) in frames.items():
            d = self.detect(name, t, img)
            if d is not None:
                dets[name] = d
        if not dets:
            return None
        per2d = {n: d[0] for n, d in dets.items()}

        name = world_view(cams, self.world_camera)
        if name is None:  # world = checkerboard: only cameras with extrinsics
            calibrated = [n for n in dets if n in cams and cams[n].has_extrinsics]
            if len(calibrated) >= 2:
                t_mean = float(np.mean([frames[n][0] for n in calibrated]))
                kps, conf = triangulate_keypoints([cams[n] for n in calibrated],
                                                  [dets[n][0].keypoints for n in calibrated],
                                                  [dets[n][0].scores for n in calibrated],
                                                  self.min_score)
                return Pose3D(t_mean, list(MP_NAMES), kps, conf, per2d)
            if not calibrated:
                return None  # only cameras without extrinsics see the person: other frame
            name = calibrated[0]
        if name not in dets:
            return None
        pose2d, world = dets[name]
        kps = single_view_lift(cams[name], pose2d, world, self.min_score)
        if kps is None:
            return None
        return Pose3D(float(frames[name][0]), list(MP_NAMES), kps, pose2d.scores.copy(), per2d)

    def close(self):
        for lm in self._landmarkers.values():
            lm.close()
        self._landmarkers.clear()


def world_view(cams: dict[str, CameraCalibration], world_camera: str | None = None) -> str | None:
    """None if the world frame is the checkerboard (some camera has extrinsics); otherwise the
    camera whose frame is the world frame: ``world_camera`` if given, else the first camera."""
    if any(c.has_extrinsics for c in cams.values()):
        return None
    if world_camera in cams:
        return world_camera
    return next(iter(cams), None)


def single_view_lift(cam: CameraCalibration, pose2d: Pose2D, body_pts: np.ndarray,
                     min_score: float = 0.5) -> np.ndarray | None:
    """Place a body-centered 3D skeleton in the world frame using PnP (single camera)."""
    mask = ((pose2d.scores >= min_score) & np.all(np.isfinite(body_pts), axis=1)
            & np.all(np.isfinite(pose2d.keypoints), axis=1))
    if mask.sum() < 6:
        return None
    obj = body_pts[mask].astype(np.float64)
    img = pose2d.keypoints[mask].astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, cam.K, cam.dist, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, img, cam.K, cam.dist, rvec, tvec, True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or tvec[2] <= 0:
        return None
    R_bc = cv2.Rodrigues(rvec)[0]
    pts_c = body_pts @ R_bc.T + tvec.ravel()
    if not cam.has_extrinsics:
        return pts_c
    return (pts_c - cam.t) @ cam.R  # camera -> world: R^T (X - t)
