"""MediaPipe Pose Landmarker as a ``Detector2D`` (tasks API, mediapipe>=0.10).

Gives the 33 BlazePose keypoints in pixels (score = landmark visibility) and MediaPipe's world
landmarks as ``Person2D.keypoints_3d``: a metric skeleton centred between the hips, which lets
``MultiViewEstimator`` place the person in the world with a single camera (PnP). The model
file (``models/pose_landmarker_<model>.task``) is downloaded on first use
(``poseboard.pose.mediapipe_backend.ensure_model``).
"""

from __future__ import annotations

import cv2
import numpy as np

from poseboard.pose.detectors.base import Detector2D, Person2D, keypoint_bbox
from poseboard.pose.formats import MEDIAPIPE33


class MediaPipeDetector(Detector2D):
    key = "mediapipe"
    label = "MediaPipe Pose Landmarker"
    format = MEDIAPIPE33
    provides_3d = True

    def __init__(self, model: str = "full", num_poses: int = 1,
                 min_pose_detection_confidence: float = 0.5,
                 min_pose_presence_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        super().__init__()
        from mediapipe.tasks.python import BaseOptions, vision

        from poseboard.pose import mediapipe_backend as mpb  # MODEL_DIR may be patched (frozen)

        if model not in mpb.MODEL_URLS:
            raise ValueError(f"MediaPipe model must be one of {', '.join(mpb.MODEL_URLS)}, "
                             f"not {model!r}")
        self._vision = vision
        self._base = BaseOptions
        self.model_path = str(mpb.ensure_model(model))
        self.num_poses = max(1, int(num_poses))
        self._conf = (float(min_pose_detection_confidence), float(min_pose_presence_confidence),
                      float(min_tracking_confidence))
        self.options = {"model": model, "num_poses": self.num_poses}
        self._landmarkers: dict[str, object] = {}
        self._last_ts: dict[str, int] = {}

    def _landmarker(self, cam_name: str):
        """One landmarker per camera: VIDEO mode tracks over time and needs increasing
        timestamps per stream."""
        if cam_name not in self._landmarkers:
            v = self._vision
            opts = v.PoseLandmarkerOptions(
                base_options=self._base(model_asset_path=self.model_path),
                running_mode=v.RunningMode.VIDEO, num_poses=self.num_poses,
                min_pose_detection_confidence=self._conf[0],
                min_pose_presence_confidence=self._conf[1],
                min_tracking_confidence=self._conf[2])
            self._landmarkers[cam_name] = v.PoseLandmarker.create_from_options(opts)
            self._last_ts[cam_name] = -1
        return self._landmarkers[cam_name]

    def detect(self, image_bgr: np.ndarray, t: float, cam_name: str) -> list[Person2D]:
        import mediapipe as mp

        lm = self._landmarker(cam_name)
        ts = max(int(t * 1000), self._last_ts[cam_name] + 1)
        self._last_ts[cam_name] = ts
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        if not res.pose_landmarks:
            return []
        h, w = image_bgr.shape[:2]
        people = []
        for i, l2 in enumerate(res.pose_landmarks):
            kp = np.array([[p.x * w, p.y * h] for p in l2], np.float64)
            sc = np.array([p.visibility if p.visibility is not None else 1.0 for p in l2],
                          np.float64)
            world = None
            if res.pose_world_landmarks and i < len(res.pose_world_landmarks):
                world = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[i]],
                                 np.float64)
            people.append(Person2D(kp, sc, bbox=keypoint_bbox(kp), score=float(np.mean(sc)),
                                   keypoints_3d=world))
        return people

    def close(self) -> None:
        for lm in self._landmarkers.values():
            lm.close()
        self._landmarkers.clear()


def create(model: str = "full", num_poses: int = 1, **kwargs) -> MediaPipeDetector:
    return MediaPipeDetector(model=model, num_poses=num_poses, **kwargs)
